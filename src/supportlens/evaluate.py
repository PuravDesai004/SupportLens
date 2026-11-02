"""Evaluate both locked candidates on the official test set, after validation selection."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import DATA_DIR, DATA_SHA256, REPORT_DIR, ROOT, RUN_DIR, TORCH_CONFIG
from .data import frame_audit, load_data, overlap_audit, sha256_file, split_train_validation
from .torch_model import IntentNet, predict_probabilities
from .train import write_json


def summarize(y_true, probabilities, labels, threshold):
    pred = probabilities.argmax(axis=1)
    score = probabilities.max(axis=1)
    accepted = score >= threshold
    report = classification_report(y_true, pred, labels=np.arange(len(labels)), target_names=labels,
                                   output_dict=True, zero_division=0)
    pairs = Counter((labels[int(t)], labels[int(p)]) for t, p in zip(y_true, pred) if t != p)
    result = {"examples": len(y_true), "supported_labels": len(set(y_true)),
              "accuracy": float(accuracy_score(y_true, pred)),
              "macro_f1": float(f1_score(y_true, pred, labels=np.arange(len(labels)), average="macro", zero_division=0)),
              "review_threshold": threshold, "review_count": int((~accepted).sum()),
              "accepted_count": int(accepted.sum()), "review_rate_of_all_examples": float((~accepted).mean()),
              "accuracy_among_accepted": float((pred[accepted] == y_true[accepted]).mean()) if accepted.any() else None,
              "common_confusions": [{"true_label": t, "predicted_label": p, "count": n} for (t, p), n in pairs.most_common(20)]}
    return result, report, pred, score, ~accepted


def sql_summary(csv_path, sql_path=ROOT / "sql" / "intent_summary.sql"):
    frame = pd.read_csv(csv_path)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("evaluation", frame)
        return con.execute(sql_path.read_text(encoding="utf-8")).df()
    finally:
        con.close()


def main(allow_rerun=False):
    metrics_path = REPORT_DIR / "metrics.json"
    previous = None
    if metrics_path.exists():
        if not allow_rerun:
            raise RuntimeError("Test results already exist. Use --allow-rerun only for a documented correctness rerun.")
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    selection = json.loads((REPORT_DIR / "selection.json").read_text(encoding="utf-8"))
    labels = json.loads((RUN_DIR / "labels.json").read_text(encoding="utf-8"))
    for name in ("train.csv", "test.csv"):
        if sha256_file(DATA_DIR / name) != DATA_SHA256[name]:
            raise ValueError(f"Official {name} checksum does not match pinned source")
    official_train = load_data(DATA_DIR / "train.csv", labels)
    test = load_data(DATA_DIR / "test.csv", labels)
    if set(test.category) != set(labels):
        raise ValueError("Official test label coverage differs from training labels")
    derived_train, validation = split_train_validation(official_train)
    manifest = json.loads((REPORT_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    manifest.update({"official_test": frame_audit(test), "derived_train_vs_test": overlap_audit(derived_train, test),
                     "validation_vs_test": overlap_audit(validation, test),
                     "official_train_vs_test": overlap_audit(official_train, test)})
    write_json(REPORT_DIR / "data_manifest.json", manifest)
    vectorizer = joblib.load(RUN_DIR / "vectorizer.joblib")
    x_test = vectorizer.transform(test.text)
    y_test = np.array([labels.index(label) for label in test.category], dtype=np.int64)
    baseline = joblib.load(RUN_DIR / "baseline.joblib")
    net = IntentNet(x_test.shape[1], TORCH_CONFIG["hidden_dim"], len(labels))
    net.load_state_dict(torch.load(RUN_DIR / "torch.pt", map_location="cpu", weights_only=True))
    probabilities = {"baseline": baseline.predict_proba(x_test), "torch": predict_probabilities(net, x_test)}
    all_results, all_reports, details = {}, {}, {}
    for name, scores in probabilities.items():
        result, report, pred, score, review = summarize(y_test, scores, labels, selection["review_threshold"])
        all_results[name], all_reports[name], details[name] = result, report, (pred, score, review)
    selected = selection["selected"]
    pred, score, review = details[selected]
    exported = pd.DataFrame({"row_id": np.arange(1, len(test) + 1), "true_label": test.category,
                             "predicted_label": [labels[i] for i in pred], "score": score,
                             "correct": pred == y_test, "review": review})
    csv_path = REPORT_DIR / "evaluation.csv"
    exported.to_csv(csv_path, index=False, float_format="%.8f", lineterminator="\n")
    summary = sql_summary(csv_path)
    summary.to_csv(REPORT_DIR / "intent_summary.csv", index=False)
    if (int(summary.examples.sum()) != len(exported) or int(summary.errors.sum()) != int((~exported.correct).sum())
            or int(summary.flagged_for_review.sum()) != int(exported.review.sum())):
        raise AssertionError("DuckDB summary does not reconcile with exported CSV")
    for name, report in all_reports.items():
        pd.DataFrame.from_dict({label: report[label] for label in labels}, orient="index").rename_axis("label").to_csv(
            REPORT_DIR / f"{name}_per_class.csv")
    shipped_test_seen = REPORT_DIR.resolve() != (ROOT / "reports").resolve() and (ROOT / "reports" / "metrics.json").exists()
    output = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "split": "Full official BANKING77 test split",
              "selected_before_test": selected, "models": all_results,
              "selected_csv_sha256": sha256_file(csv_path),
              "duckdb_reconciled_totals": {"examples": int(summary.examples.sum()), "errors": int(summary.errors.sum()),
                                           "flagged_for_review": int(summary.flagged_for_review.sum())},
              "test_previously_viewed": previous is not None or shipped_test_seen}
    if previous is not None:
        output["previous_test_metrics"] = previous["models"]
    write_json(metrics_path, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-rerun", action="store_true")
    main(parser.parse_args().allow_rerun)
