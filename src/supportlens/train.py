"""Train both candidates, compare the same validation rows, and lock the choice."""
import json
import platform
import shutil
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .config import (ARTIFACT_DIR, DATA_DIR, DATA_SHA256, REPORT_DIR, REVIEW_THRESHOLD, RUN_DIR,
                     SEED, SOURCE_BASE, SOURCE_COMMIT, TORCH_CONFIG, VECTORIZER_CONFIG)
from .data import frame_audit, load_data, overlap_audit, sha256_file, split_train_validation
from .torch_model import predict_probabilities, train_network


def metrics(y, probabilities):
    prediction = probabilities.argmax(axis=1)
    return {"examples": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0))}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    if (REPORT_DIR / "metrics.json").exists():
        raise RuntimeError("Test results already exist. Preserve this run before retraining; test data is no longer unseen.")
    for directory in (RUN_DIR, ARTIFACT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    if sha256_file(train_path) != DATA_SHA256["train.csv"]:
        raise ValueError("Official training CSV checksum does not match the pinned source")
    official_train = load_data(train_path)
    derived_train, validation = split_train_validation(official_train)
    labels = LabelEncoder().fit(official_train.category)
    y_train = labels.transform(derived_train.category)
    y_val = labels.transform(validation.category)
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG, dtype=np.float32)
    x_train = vectorizer.fit_transform(derived_train.text)
    x_val = vectorizer.transform(validation.text)
    print(f"TF-IDF train={x_train.shape}; validation={x_val.shape}; labels={len(labels.classes_)}", flush=True)
    baseline = LogisticRegression(C=4.0, max_iter=500, solver="lbfgs", random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        baseline.fit(x_train, y_train)
    warning_messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if warning_messages:
        raise RuntimeError("Baseline did not converge; adjust configuration before selection: " + " | ".join(warning_messages))
    baseline_metrics = metrics(y_val, baseline.predict_proba(x_val))
    print(f"baseline validation: {baseline_metrics}", flush=True)
    net, losses = train_network(x_train, y_train, seed=SEED, n_classes=len(labels.classes_), **TORCH_CONFIG)
    torch_metrics = metrics(y_val, predict_probabilities(net, x_val))
    print(f"torch validation: {torch_metrics}", flush=True)
    # Predeclared rule: higher validation macro F1 wins; exact ties go to the simpler baseline.
    winner = "torch" if torch_metrics["macro_f1"] > baseline_metrics["macro_f1"] else "baseline"
    joblib.dump(vectorizer, RUN_DIR / "vectorizer.joblib", compress=3)
    joblib.dump(baseline, RUN_DIR / "baseline.joblib", compress=3)
    torch.save(net.state_dict(), RUN_DIR / "torch.pt")
    label_names = labels.classes_.tolist()
    write_json(RUN_DIR / "labels.json", label_names)
    shutil.copyfile(RUN_DIR / "vectorizer.joblib", ARTIFACT_DIR / "vectorizer.joblib")
    model_file = "model.joblib" if winner == "baseline" else "model.pt"
    shutil.copyfile(RUN_DIR / ("baseline.joblib" if winner == "baseline" else "torch.pt"), ARTIFACT_DIR / model_file)
    metadata = {"artifact_version": 1, "model": winner, "labels": label_names,
                "input_dim": x_train.shape[1], "output_dim": len(label_names),
                "hidden_dim": TORCH_CONFIG["hidden_dim"], "review_threshold": REVIEW_THRESHOLD,
                "sklearn_version": sklearn.__version__,
                "sha256": {name: sha256_file(ARTIFACT_DIR / name) for name in ("vectorizer.joblib", model_file)}}
    write_json(ARTIFACT_DIR / "metadata.json", metadata)
    selection = {"locked_at_utc": datetime.now(timezone.utc).isoformat(),
                 "rule": "Higher validation macro F1; exact ties select baseline. No test metrics used.",
                 "selected": winner, "validation": {"baseline": baseline_metrics, "torch": torch_metrics},
                 "review_threshold": REVIEW_THRESHOLD, "threshold_rule": "Fixed in code before test evaluation; not tuned on test.",
                 "baseline": {"estimator": "LogisticRegression", "C": 4.0, "max_iter": 500, "solver": "lbfgs",
                              "iterations": baseline.n_iter_.tolist(), "convergence_warnings": warning_messages},
                 "torch": {**TORCH_CONFIG, "optimizer": "Adam", "loss": "CrossEntropyLoss", "device": "cpu", "epoch_training_loss": losses},
                 "tfidf": {**VECTORIZER_CONFIG, "dtype": "float32", "actual_features": x_train.shape[1]},
                 "artifact_sha256": metadata["sha256"]}
    write_json(REPORT_DIR / "selection.json", selection)
    manifest = {"source_commit": SOURCE_COMMIT, "source_urls": {name: f"{SOURCE_BASE}/{name}" for name in DATA_SHA256},
                "sha256": DATA_SHA256, "seed": SEED,
                "split": "First fold of shuffled 5-fold StratifiedGroupKFold over official training rows, grouped by case-folded, whitespace-normalized text.",
                "official_train": frame_audit(official_train), "derived_train": frame_audit(derived_train),
                "validation": frame_audit(validation), "derived_train_vs_validation": overlap_audit(derived_train, validation),
                "versions": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}}
    write_json(REPORT_DIR / "data_manifest.json", manifest)
    print(f"Selected {winner}; decision saved before test evaluation.", flush=True)


if __name__ == "__main__":
    main()
