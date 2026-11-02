"""Reproducible, CPU-only v2 comparison. See docs/EXPERIMENT_PLAN.md.

Run the stages in order: development, assessment, then previously-viewed-test.
No stage overwrites its evidence. Data and learned preprocessing remain split.
"""
import argparse
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits

from .calibration import (choose_review_policy, fit_temperature, multiclass_log_loss, reliability,
                          review_summary, temperature_scale)
from .config import DATA_DIR, DATA_SHA256, REPORT_DIR, ROOT, RUN_DIR, SEED, TORCH_CONFIG, VECTORIZER_CONFIG
from .data import load_data, normalized, sha256_file
from .torch_model import IntentNet, predict_probabilities, train_network

REPORTS = REPORT_DIR / "experiment_v2"
RUNS = RUN_DIR / "experiment_v2"
CANDIDATES = ("word_logistic", "word_char_logistic", "word_torch")
BASELINE = "word_logistic"
ASSESSMENT_SEED = 73
FOLD_SEED = 137
BOOTSTRAP_SEED = 2027
BOOTSTRAP_REPEATS = 1000


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_training_data():
    path = DATA_DIR / "train.csv"
    if sha256_file(path) != DATA_SHA256["train.csv"]:
        raise ValueError("Pinned official training data checksum does not match")
    frame = load_data(path)
    labels = sorted(frame.category.unique().tolist())
    if len(labels) != 77:
        raise ValueError("Expected the observed 77 labels")
    return frame, labels


def experiment_splits(frame):
    groups = frame.text.map(normalized)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=ASSESSMENT_SEED)
    dev_idx, assessment_idx = next(outer.split(frame.text, frame.category, groups))
    dev = frame.iloc[dev_idx]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=FOLD_SEED).split(
        dev.text, dev.category, dev.text.map(normalized)))
    all_labels = set(frame.category)
    if set(dev.category) != all_labels or set(frame.iloc[assessment_idx].category) != all_labels:
        raise ValueError("Outer split lost labels")
    if set(groups.iloc[dev_idx]) & set(groups.iloc[assessment_idx]):
        raise AssertionError("A normalized group crossed the assessment split")
    seen = np.zeros(len(dev_idx), dtype=int)
    for train_idx, val_idx in folds:
        seen[val_idx] += 1
        if set(dev.iloc[train_idx].category) != all_labels or set(dev.iloc[val_idx].category) != all_labels:
            raise ValueError("A development fold lost labels")
        if set(dev.iloc[train_idx].text.map(normalized)) & set(dev.iloc[val_idx].text.map(normalized)):
            raise AssertionError("A normalized group crossed a development fold")
    if not np.all(seen == 1):
        raise AssertionError("Every development row must be validated exactly once")
    return dev_idx, assessment_idx, folds


def make_vectorizer(name):
    word = TfidfVectorizer(**VECTORIZER_CONFIG, dtype=np.float32)
    if name != "word_char_logistic":
        return word
    character = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=18000,
                                min_df=2, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([("word", word), ("char", character)], transformer_weights={"word": 1.0, "char": 0.7})


def fit_candidate(name, texts, y, n_classes):
    start = time.perf_counter()
    vectorizer = make_vectorizer(name)
    x = vectorizer.fit_transform(texts)
    vectorize_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if name == "word_torch":
        model, losses = train_network(x, y, seed=SEED, n_classes=n_classes, **TORCH_CONFIG)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        info = {"training_loss": losses, "convergence_warnings": []}
    else:
        model = LogisticRegression(C=4.0, max_iter=500, solver="lbfgs", random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if messages:
            raise RuntimeError("Logistic regression did not converge: " + " | ".join(messages))
        parameter_bytes = model.coef_.nbytes + model.intercept_.nbytes
        info = {"iterations": model.n_iter_.tolist(), "convergence_warnings": messages}
    info.update({"features": int(x.shape[1]), "training_nonzero_entries": int(x.nnz),
                 "vectorizer_fit_seconds": vectorize_seconds, "model_fit_seconds": time.perf_counter() - start,
                 "model_parameter_bytes": int(parameter_bytes)})
    return vectorizer, model, info


def probabilities_for(name, vectorizer, model, texts):
    x = vectorizer.transform(texts)
    if name == "word_torch":
        return predict_probabilities(model, x)
    return model.predict_proba(x)


def report_metrics(y, p, labels, policy):
    pred = p.argmax(axis=1)
    top3 = np.argsort(-p, axis=1, kind="stable")[:, :min(3, len(labels))]
    ece, bins = reliability(y, p)
    pairs = Counter((labels[int(t)], labels[int(v)]) for t, v in zip(y, pred) if t != v)
    classification = classification_report(y, pred, labels=np.arange(len(labels)), target_names=labels,
                                           output_dict=True, zero_division=0)
    summary = {"examples": len(y), "supported_labels": len(set(y)),
               "accuracy": float(accuracy_score(y, pred)),
               "macro_f1": float(f1_score(y, pred, labels=np.arange(len(labels)), average="macro", zero_division=0)),
               "top3_accuracy": float(np.mean(np.any(top3 == np.asarray(y)[:, None], axis=1))),
               "log_loss": multiclass_log_loss(y, p), "ece_10_bins": ece, "reliability_bins": bins,
               "review": review_summary(y, p, policy),
               "common_confusions": [{"true_label": t, "predicted_label": v, "count": n}
                                     for (t, v), n in pairs.most_common(20)]}
    per_class = {label: classification[label] for label in labels}
    return summary, per_class


def macro_f1_fast(y, prediction, n_classes):
    cm = np.bincount(y * n_classes + prediction, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    true_positive = np.diag(cm)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.divide(2 * true_positive, denominator, out=np.zeros(n_classes), where=denominator != 0).mean())


def paired_bootstrap_delta(y, candidate, baseline, n_classes, repeats=BOOTSTRAP_REPEATS):
    """Paired class-stratified row bootstrap; approximate with shared fold training."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = [np.flatnonzero(y == label) for label in range(n_classes)]
    deltas = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in indices])
        deltas.append(macro_f1_fast(y[sample], candidate[sample], n_classes)
                      - macro_f1_fast(y[sample], baseline[sample], n_classes))
    low, high = np.quantile(deltas, [.025, .975])
    return {"lower_95": float(low), "upper_95": float(high), "repeats": repeats,
            "seed": BOOTSTRAP_SEED, "unit": "class-stratified rows, paired across models"}


def save_per_class(stage, name, per_class):
    path = REPORTS / f"{stage}_{name}_per_class.csv"
    pd.DataFrame.from_dict(per_class, orient="index").rename_axis("label").to_csv(path, lineterminator="\n")


def dev_stage():
    if (REPORTS / "development_lock.json").exists():
        raise RuntimeError("Development lock already exists; use a separate SUPPORTLENS_OUTPUT_ROOT to reproduce")
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    frame, labels = checked_training_data()
    dev_idx, assessment_idx, folds = experiment_splits(frame)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    label_ids = {label: index for index, label in enumerate(labels)}
    y = np.array([label_ids[label] for label in dev.category], dtype=np.int64)
    fold_ids = np.full(len(dev), -1, dtype=np.int64)
    for fold_number, (_, val_idx) in enumerate(folds, 1):
        fold_ids[val_idx] = fold_number
    assignment = pd.DataFrame({"row_id": np.arange(1, len(frame) + 1), "role": "assessment", "fold": 0})
    assignment.loc[dev_idx, "role"] = "development"
    assignment.loc[dev_idx, "fold"] = fold_ids
    assignment.to_csv(REPORTS / "assignments.csv", index=False, lineterminator="\n")
    oof, timing = {}, {}
    for name in CANDIDATES:
        oof[name] = np.zeros((len(dev), len(labels)), dtype=np.float64)
        timing[name] = []
        for number, (train_idx, val_idx) in enumerate(folds, 1):
            print(f"{name}: development fold {number}/3", flush=True)
            vectorizer, model, info = fit_candidate(name, dev.iloc[train_idx].text, y[train_idx], len(labels))
            start = time.perf_counter()
            oof[name][val_idx] = probabilities_for(name, vectorizer, model, dev.iloc[val_idx].text)
            info["validation_prediction_seconds"] = time.perf_counter() - start
            info["train_examples"], info["validation_examples"] = len(train_idx), len(val_idx)
            timing[name].append(info)
        if not np.allclose(oof[name].sum(axis=1), 1, atol=1e-5):
            raise AssertionError("Out-of-fold probabilities are incomplete")
    summaries, per_class, temperatures, policies = {}, {}, {}, {}
    for name in CANDIDATES:
        raw = oof[name]
        temperatures[name] = fit_temperature(y, raw)
        adjusted = temperature_scale(raw, temperatures[name])
        policies[name] = {"raw": choose_review_policy(y, raw), "temperature": choose_review_policy(y, adjusted)}
        summaries[name] = {}
        for kind, scores in (("raw", raw), ("temperature", adjusted)):
            summaries[name][kind], per_class[name] = report_metrics(y, scores, labels, policies[name][kind])
        save_per_class("development", name, per_class[name])
    baseline_pred = oof[BASELINE].argmax(axis=1)
    comparison = {}
    base_metrics = summaries[BASELINE]["raw"]
    for name in CANDIDATES:
        model_metrics = summaries[name]["raw"]
        gain = model_metrics["macro_f1"] - base_metrics["macro_f1"]
        interval = paired_bootstrap_delta(y, oof[name].argmax(axis=1), baseline_pred, len(labels))
        eligible = bool(name != BASELINE and gain >= .005 and interval["lower_95"] > 0
                        and model_metrics["accuracy"] >= base_metrics["accuracy"] - .002)
        comparison[name] = {"macro_f1_gain_vs_baseline": gain, "bootstrap_95": interval, "eligible": eligible}
    eligible = [name for name in CANDIDATES if comparison[name]["eligible"]]
    chosen = max(eligible, key=lambda name: summaries[name]["raw"]["macro_f1"]) if eligible else None
    if chosen == "word_torch" and "word_char_logistic" in eligible:
        if summaries["word_torch"]["raw"]["macro_f1"] - summaries["word_char_logistic"]["raw"]["macro_f1"] <= .002:
            chosen = "word_char_logistic"
    np.savez_compressed(RUNS / "development_oof.npz", y=y, fold_ids=fold_ids, **oof)
    save_json(RUNS / "labels.json", labels)
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "train_data_sha256": DATA_SHA256["train.csv"],
            "assignment_sha256": sha256_file(REPORTS / "assignments.csv"),
            "scope": "Three group-aware out-of-fold predictions on the development pool from official training data",
            "development_examples": len(dev), "assessment_examples": len(assessment_idx), "labels": len(labels),
            "split_seed": ASSESSMENT_SEED, "fold_seed": FOLD_SEED, "candidate_order": list(CANDIDATES),
            "candidate_configs": {"word": VECTORIZER_CONFIG, "char": {"analyzer": "char_wb", "ngram_range": [3, 5],
                                  "max_features": 18000, "min_df": 2, "sublinear_tf": True, "weight": .7},
                                  "logistic": {"C": 4, "solver": "lbfgs", "max_iter": 500}, "torch": TORCH_CONFIG},
            "selection_rule": "See docs/EXPERIMENT_PLAN.md; rank by out-of-fold macro F1 with prespecified eligibility and simplicity rules.",
            "selected_for_assessment_gate": chosen, "comparison_to_baseline": comparison,
            "temperature": temperatures, "review_policies": policies, "models": summaries, "fold_runs": timing,
            "versions": {"python": platform.python_version(), "platform": platform.platform(),
                         "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
            "notes": ["Temperature and review thresholds were fit on these out-of-fold labels; their development estimates can be optimistic.",
                      "The separate assessment affects a predeclared release gate and is therefore part of development, not an untouched final test."]}
    save_json(REPORTS / "development_lock.json", lock)
    print(json.dumps({"selected_for_assessment_gate": chosen, "comparison": comparison}, indent=2), flush=True)


def save_candidate(name, vectorizer, model, info, labels):
    folder = RUNS / name
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, folder / "vectorizer.joblib", compress=3)
    model_file = "model.pt" if name == "word_torch" else "model.joblib"
    if name == "word_torch":
        torch.save(model.state_dict(), folder / model_file)
    else:
        joblib.dump(model, folder / model_file, compress=3)
    metadata = {"artifact_version": 2, "candidate_id": name, "model": "torch" if name == "word_torch" else "baseline",
                "input_dim": info["features"], "output_dim": len(labels), "hidden_dim": TORCH_CONFIG["hidden_dim"],
                "labels": labels, "sklearn_version": sklearn.__version__,
                "sha256": {file: sha256_file(folder / file) for file in ("vectorizer.joblib", model_file)}}
    save_json(folder / "metadata.json", metadata)
    info["serialized_vectorizer_bytes"] = (folder / "vectorizer.joblib").stat().st_size
    info["serialized_model_bytes"] = (folder / model_file).stat().st_size
    return metadata


def load_candidate(name, labels):
    folder = RUNS / name
    metadata = read_json(folder / "metadata.json")
    if metadata["candidate_id"] != name or metadata["labels"] != labels:
        raise ValueError("Candidate label mapping mismatch")
    for filename, digest in metadata["sha256"].items():
        if sha256_file(folder / filename) != digest:
            raise ValueError("Candidate artifact checksum mismatch")
    vectorizer = joblib.load(folder / "vectorizer.joblib")
    if name == "word_torch":
        model = IntentNet(metadata["input_dim"], metadata["hidden_dim"], metadata["output_dim"])
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
    else:
        model = joblib.load(folder / "model.joblib")
        if list(model.classes_) != list(range(len(labels))):
            raise ValueError("Candidate classifier class ordering mismatch")
    return vectorizer, model, metadata


def assessment_stage():
    if (REPORTS / "final_lock.json").exists():
        raise RuntimeError("Assessment already exists; use a separate output root for a reproduction")
    lock = read_json(REPORTS / "development_lock.json")
    frame, labels = checked_training_data()
    if sha256_file(REPORTS / "assignments.csv") != lock["assignment_sha256"]:
        raise ValueError("Split assignment checksum mismatch")
    dev_idx, assessment_idx, _ = experiment_splits(frame)
    dev, assessment = frame.iloc[dev_idx], frame.iloc[assessment_idx]
    label_ids = {label: index for index, label in enumerate(labels)}
    y_dev = np.array([label_ids[label] for label in dev.category], dtype=np.int64)
    y = np.array([label_ids[label] for label in assessment.category], dtype=np.int64)
    results, probabilities, run_info, candidate_hashes = {}, {}, {}, {}
    for name in CANDIDATES:
        print(f"{name}: fit development pool and assess", flush=True)
        vectorizer, model, info = fit_candidate(name, dev.text, y_dev, len(labels))
        metadata = save_candidate(name, vectorizer, model, info, labels)
        candidate_hashes[name] = metadata["sha256"]
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            raw = probabilities_for(name, vectorizer, model, assessment.text)
            elapsed.append(time.perf_counter() - start)
        one = assessment.text.iloc[:1]
        single_times = []
        for _ in range(20):
            start = time.perf_counter()
            probabilities_for(name, vectorizer, model, one)
            single_times.append(time.perf_counter() - start)
        info.update({"assessment_batch_examples": len(assessment), "assessment_batch_seconds_3_runs": elapsed,
                     "single_prediction_median_ms_20_runs": float(np.median(single_times) * 1000)})
        run_info[name] = info
        probabilities[name] = raw
        adjusted = temperature_scale(raw, lock["temperature"][name])
        results[name] = {}
        for kind, scores in (("raw", raw), ("temperature", adjusted)):
            result, per_class = report_metrics(y, scores, labels, lock["review_policies"][name][kind])
            results[name][kind] = result
        save_per_class("assessment", name, per_class)
    np.savez_compressed(RUNS / "assessment_probabilities.npz", y=y, **probabilities)
    chosen = lock["selected_for_assessment_gate"]
    promoted = bool(chosen and results[chosen]["raw"]["macro_f1"] > results[BASELINE]["raw"]["macro_f1"]
                    and results[chosen]["raw"]["accuracy"] >= results[BASELINE]["raw"]["accuracy"] - .002)
    score_kind = None
    if promoted:
        raw, adjusted = results[chosen]["raw"], results[chosen]["temperature"]
        score_kind = "temperature" if adjusted["log_loss"] <= raw["log_loss"] + .01 and adjusted["ece_10_bins"] <= raw["ece_10_bins"] + .02 else "raw"
    final = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "assessment_examples": len(y),
             "selection_frozen_before_assessment": chosen, "promoted": promoted,
             "selected_release": chosen if promoted else "previous_v1", "score_kind": score_kind,
             "temperature": lock["temperature"][chosen] if score_kind == "temperature" else 1.0,
             "review_policy": lock["review_policies"][chosen][score_kind] if promoted else {"threshold": .35, "review_all": False},
             "candidate_artifact_sha256": candidate_hashes,
             "selection_rule": "Predeclared development candidate must beat matched baseline macro F1 on assessment and meet accuracy tolerance; otherwise retain v1.",
             "confidence_rule": "Use temperature only when assessment log loss and ECE satisfy the prespecified guard; threshold itself was chosen on development only.",
             "assessment_results": results, "assessment_run_resources": run_info,
             "assessment_scope": "Development release-confirmation split from official training data; it affects release and is not an untouched final test.",
             "test_seen_before_this_experiment": True}
    save_json(REPORTS / "final_lock.json", final)
    print(json.dumps({"selected_release": final["selected_release"], "score_kind": score_kind,
                      "review_policy": final["review_policy"]}, indent=2), flush=True)


def test_stage():
    output = REPORTS / "previously_viewed_test.json"
    if output.exists():
        raise RuntimeError("Exploratory test result already exists; do not repeatedly inspect the test")
    final = read_json(REPORTS / "final_lock.json")
    lock = read_json(REPORTS / "development_lock.json")
    path = DATA_DIR / "test.csv"
    if sha256_file(path) != DATA_SHA256["test.csv"]:
        raise ValueError("Pinned official test data checksum does not match")
    labels = read_json(RUNS / "labels.json")
    test = load_data(path, labels)
    ids = {label: i for i, label in enumerate(labels)}
    y = np.array([ids[label] for label in test.category], dtype=np.int64)
    output_models = {}
    for name in CANDIDATES:
        vectorizer, model, meta = load_candidate(name, labels)
        if meta["sha256"] != final["candidate_artifact_sha256"][name]:
            raise ValueError("Candidate changed after release lock")
        raw = probabilities_for(name, vectorizer, model, test.text)
        adjusted = temperature_scale(raw, lock["temperature"][name])
        output_models[name] = {}
        for kind, scores in (("raw", raw), ("temperature", adjusted)):
            result, per_class = report_metrics(y, scores, labels, lock["review_policies"][name][kind])
            output_models[name][kind] = result
        save_per_class("previously_viewed_test", name, per_class)
    from .predict import load_bundle
    vectorizer, model, meta = load_bundle(ROOT / "artifacts" / "previous_v1")
    if meta["labels"] != labels:
        raise ValueError("Previous v1 labels do not match")
    old = model.predict_proba(vectorizer.transform(test.text))
    output_models["previous_v1"], per_class = report_metrics(y, old, labels, {"threshold": .35, "review_all": False})
    save_per_class("previously_viewed_test", "previous_v1", per_class)
    save_json(output, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "examples": len(test),
                       "scope": "Full official test; previously viewed in v1. Exploratory comparison only.",
                       "test_previously_viewed": True, "selection_locked_before_test": final["selected_release"],
                       "reason": "One post-lock exploratory comparison of the fixed approaches; no tuning on test.",
                       "models": output_models})
    print(json.dumps({"selection_locked_before_test": final["selected_release"],
                      "examples": len(test), "test_previously_viewed": True}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "assessment", "previously-viewed-test"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"development": dev_stage, "assessment": assessment_stage, "previously-viewed-test": test_stage}[args.stage]()


if __name__ == "__main__":
    main()
