"""Select and assess a second-view review rule without rescoring the test set."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from .calibration import temperature_scale, validate_labels, validate_probabilities
from .config import ARTIFACT_DIR, REPORT_DIR, RUN_DIR
from .data import sha256_file
from .review_guard import compare_policy, review_masks, select_support_threshold, summarize_review

V2_REPORTS = REPORT_DIR / "experiment_v2"
V2_RUNS = RUN_DIR / "experiment_v2"
OUTPUT = REPORT_DIR / "review_guard"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def checked_models():
    final = read_json(V2_REPORTS / "final_lock.json")
    main_dir = ARTIFACT_DIR / "v2"
    guard_dir = V2_RUNS / "word_logistic"
    main = read_json(main_dir / "metadata.json")
    guard = read_json(guard_dir / "metadata.json")
    if final["selected_release"] != "word_char_logistic" or not final["promoted"]:
        raise ValueError("Expected the locked v2 word+character release")
    if main["candidate_id"] != final["selected_release"] or guard["candidate_id"] != "word_logistic":
        raise ValueError("Candidate identity does not match the v2 lock")
    if main["labels"] != guard["labels"] or main["output_dim"] != guard["output_dim"]:
        raise ValueError("The two candidates do not share the same label order")
    if main["sha256"] != final["candidate_artifact_sha256"]["word_char_logistic"]:
        raise ValueError("Main artifact differs from the locked assessment candidate")
    if guard["sha256"] != final["candidate_artifact_sha256"]["word_logistic"]:
        raise ValueError("Word-only artifact differs from the matched assessment candidate")
    for directory, metadata in ((main_dir, main), (guard_dir, guard)):
        for name, digest in metadata["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError("Model files do not match the frozen v2 candidates")
    policy = final["review_policy"]
    if (main["review_threshold"] != policy["threshold"]
            or main.get("review_all", False) != policy["review_all"]
            or main["score_kind"] != final["score_kind"]
            or main["temperature"] != final["temperature"]):
        raise ValueError("Main score policy differs from the frozen v2 decision")
    if policy["review_all"] or policy["threshold"] != .75 or final["score_kind"] != "temperature":
        raise ValueError("Unexpected v2 score policy")
    return main, guard


def load_predictions(path, main):
    with np.load(path) as arrays:
        y = arrays["y"].copy()
        raw_main = arrays["word_char_logistic"].copy()
        word = arrays["word_logistic"].copy()
        folds = arrays["fold_ids"].copy() if "fold_ids" in arrays.files else None
    if y.ndim != 1 or raw_main.shape != word.shape or raw_main.shape != (len(y), main["output_dim"]):
        raise ValueError("Saved predictions do not match the locked labels")
    validate_labels(y, raw_main)
    validate_probabilities(word)
    if folds is not None and (folds.shape != y.shape or not np.issubdtype(folds.dtype, np.integer)
                              or set(folds.tolist()) != {1, 2, 3}):
        raise ValueError("Every development row must have a fold ID of 1, 2 or 3")
    return y, temperature_scale(raw_main, main["temperature"]), word, folds


def development():
    path = OUTPUT / "development_lock.json"
    if path.exists():
        raise FileExistsError("Review development selection is already locked")
    main, guard = checked_models()
    source = V2_RUNS / "development_oof.npz"
    y, scores, word, folds = load_predictions(source, main)
    if len(y) != 8001 or folds is None:
        raise ValueError("Unexpected development predictions")
    chosen = select_support_threshold(y, scores, word, main["review_threshold"])
    crossfold = []
    cross_review = np.zeros(len(y), dtype=bool)
    for fold in (1, 2, 3):
        fit, evaluate = folds != fold, folds == fold
        subchoice = select_support_threshold(y[fit], scores[fit], word[fit], main["review_threshold"])
        threshold = subchoice["support_threshold"] if subchoice else 0.0
        mask = review_masks(scores[evaluate], word[evaluate], main["review_threshold"], threshold)
        cross_review[evaluate] = mask["review"]
        crossfold.append({"fold": fold, "selection_examples": int(fit.sum()), "evaluation_examples": int(evaluate.sum()),
                          "support_threshold": subchoice["support_threshold"] if subchoice else None,
                          "comparison": compare_policy(y[evaluate], scores[evaluate], word[evaluate],
                                                       main["review_threshold"], threshold)})
    baseline_masks = review_masks(scores, word, main["review_threshold"], 0.0)
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Previously used v2 development out-of-fold predictions; policy selection on these rows is optimistic.",
            "source_oof_sha256": sha256_file(source), "main_artifact_sha256": main["sha256"],
            "guard_artifact_sha256": guard["sha256"], "label_count": main["output_dim"],
            "main_temperature": main["temperature"], "main_threshold": main["review_threshold"],
            "selection_rule": "Maximum additional accepted errors caught with <=15% extra reviews, >=200 accepted, no drop in accepted accuracy and >=10% error reduction; see docs/REVIEW_GUARD_PLAN.md.",
            "selected": chosen,
            "crossfold_diagnostic": {"folds": crossfold,
                                     "existing_pooled": summarize_review(y, baseline_masks["top"], baseline_masks["low_main"]),
                                     "guarded_pooled": summarize_review(y, baseline_masks["top"], cross_review),
                                     "caveat": "Thresholds are selected on other OOF folds, but their underlying model training sets overlap; the existing temperature was fitted on all development OOF rows."}}
    write_once(path, lock)
    print(json.dumps({"selected": chosen, "crossfold_pooled": lock["crossfold_diagnostic"]["guarded_pooled"]}, indent=2))


def assessment():
    output = OUTPUT / "final_lock.json"
    if output.exists():
        raise FileExistsError("Review assessment decision is already locked")
    main, guard = checked_models()
    development_lock = read_json(OUTPUT / "development_lock.json")
    if (development_lock["main_artifact_sha256"] != main["sha256"]
            or development_lock["guard_artifact_sha256"] != guard["sha256"]
            or development_lock["main_temperature"] != main["temperature"]
            or development_lock["main_threshold"] != main["review_threshold"]):
        raise ValueError("Candidate artifacts or score policy changed after threshold selection")
    chosen = development_lock["selected"]
    source = V2_RUNS / "assessment_probabilities.npz"
    y, scores, word, _ = load_predictions(source, main)
    if len(y) != 2002:
        raise ValueError("Unexpected assessment predictions")
    comparison = None
    promoted = False
    if chosen:
        threshold = chosen["support_threshold"]
        comparison = compare_policy(y, scores, word, main["review_threshold"], threshold)
        old, new = comparison["existing"], comparison["guarded"]
        promoted = bool(old["accepted_errors"] > 0 and new["accepted"] >= 200
                        and new["accepted_accuracy"] >= old["accepted_accuracy"]
                        and comparison["newly_reviewed_errors"] / old["accepted_errors"] >= .05
                        and comparison["additional_review_fraction"] <= .18)
    decision = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Previously viewed v2 development assessment; used as a predeclared release gate, not an independent test.",
                "source_assessment_sha256": sha256_file(source), "test_set_rescored": False,
                "main_artifact_sha256": main["sha256"], "guard_artifact_sha256": guard["sha256"],
                "selected_before_assessment": chosen["support_threshold"] if chosen else None,
                "promoted": promoted, "comparison": comparison,
                "release_rule": "At least 5% of previously accepted errors caught, no loss in accepted accuracy, >=200 accepted and <=18% additional review."}
    write_once(output, decision)
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "assessment"))
    args = parser.parse_args()
    {"development": development, "assessment": assessment}[args.stage]()


if __name__ == "__main__":
    main()
