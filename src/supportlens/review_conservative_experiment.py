"""Assess a conservative review guard; never reads the official test."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from .data import sha256_file
from .review_guard import (compare_policy, review_masks, select_conservative_threshold, summarize_review)
from .review_guard_experiment import OUTPUT, V2_RUNS, checked_models, load_predictions, read_json, write_once


def development():
    path = OUTPUT / "conservative_development_lock.json"
    if path.exists():
        raise FileExistsError("Conservative development selection is already locked")
    main, guard = checked_models()
    source = V2_RUNS / "development_oof.npz"
    y, scores, word, folds = load_predictions(source, main)
    if len(y) != 8001 or folds is None:
        raise ValueError("Unexpected development predictions")
    selected = select_conservative_threshold(y, scores, word, main["review_threshold"])
    cross_review = np.zeros(len(y), dtype=bool)
    per_fold = []
    for fold in (1, 2, 3):
        fit, held = folds != fold, folds == fold
        choice = select_conservative_threshold(y[fit], scores[fit], word[fit], main["review_threshold"])
        cutoff = choice["support_threshold"] if choice else 0.0
        masks = review_masks(scores[held], word[held], main["review_threshold"], cutoff)
        cross_review[held] = masks["review"]
        per_fold.append({"fold": fold, "selected_on_other_folds": choice["support_threshold"] if choice else None,
                         "comparison": compare_policy(y[held], scores[held], word[held], main["review_threshold"], cutoff)})
    base = review_masks(scores, word, main["review_threshold"], 0.0)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Previously used v2 development OOF predictions; cutoff fit and checked on these rows is optimistic.",
              "source_oof_sha256": sha256_file(source), "main_artifact_sha256": main["sha256"],
              "guard_artifact_sha256": guard["sha256"], "main_threshold": main["review_threshold"],
              "main_temperature": main["temperature"], "selected": selected,
              "selection_rule": "Lowest fixed-grid cutoff with at least 1000 accepted and 95% Wilson lower bound >=0.97; see docs/REVIEW_CONSERVATIVE_PLAN.md.",
              "crossfold_diagnostic": {"folds": per_fold,
                                       "existing_pooled": summarize_review(y, base["top"], base["low_main"]),
                                       "guarded_pooled": summarize_review(y, base["top"], cross_review),
                                       "caveat": "Underlying model training sets overlap and the existing temperature was fitted on all OOF rows."}}
    write_once(path, report)
    print(json.dumps({"selected": selected, "crossfold_pooled": report["crossfold_diagnostic"]["guarded_pooled"]}, indent=2))


def assessment():
    path = OUTPUT / "conservative_final_lock.json"
    if path.exists():
        raise FileExistsError("Conservative assessment decision is already locked")
    main, guard = checked_models()
    lock = read_json(OUTPUT / "conservative_development_lock.json")
    if (lock["main_artifact_sha256"] != main["sha256"] or lock["guard_artifact_sha256"] != guard["sha256"]
            or lock["main_temperature"] != main["temperature"]
            or lock["main_threshold"] != main["review_threshold"]):
        raise ValueError("Candidate artifacts or score policy changed after threshold selection")
    source = V2_RUNS / "assessment_probabilities.npz"
    y, scores, word, _ = load_predictions(source, main)
    if len(y) != 2002:
        raise ValueError("Unexpected assessment predictions")
    selected = lock["selected"]
    comparison, promoted = None, False
    if selected:
        comparison = compare_policy(y, scores, word, main["review_threshold"], selected["support_threshold"])
        old, new = comparison["existing"], comparison["guarded"]
        promoted = bool(old["accepted_errors"] > 0 and new["accepted"] >= 200
                        and new["accepted_accuracy"] >= old["accepted_accuracy"]
                        and comparison["newly_reviewed_errors"] / old["accepted_errors"] >= .15
                        and new["reviewed"] / len(y) <= .50)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Previously viewed development assessment; exploratory release gate, not an independent test.",
              "source_assessment_sha256": sha256_file(source), "test_set_rescored": False,
              "main_artifact_sha256": main["sha256"], "guard_artifact_sha256": guard["sha256"],
              "selected_before_assessment": selected["support_threshold"] if selected else None,
              "promoted": promoted, "comparison": comparison,
              "release_rule": "At least 15% of existing accepted errors caught, no accepted-accuracy drop, >=200 accepted and <=50% total review."}
    write_once(path, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "assessment"))
    args = parser.parse_args()
    {"development": development, "assessment": assessment}[args.stage]()


if __name__ == "__main__":
    main()
