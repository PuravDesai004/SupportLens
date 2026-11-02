"""Export the locked selected model's development assessment without question text."""
import json

import numpy as np
import pandas as pd

from supportlens.calibration import temperature_scale
from supportlens.data import sha256_file
from supportlens.evaluate import sql_summary
from supportlens.experiment_v2 import REPORTS, RUNS, checked_training_data, experiment_splits


def main():
    final = json.loads((REPORTS / "final_lock.json").read_text(encoding="utf-8"))
    if not final["promoted"]:
        raise RuntimeError("V1 was retained; there is no promoted v2 assessment export")
    frame, labels = checked_training_data()
    _, assessment_idx, _ = experiment_splits(frame)
    selected = final["selected_release"]
    with np.load(RUNS / "assessment_probabilities.npz") as arrays:
        y = arrays["y"]
        scores = arrays[selected]
    if final["score_kind"] == "temperature":
        scores = temperature_scale(scores, final["temperature"])
    pred, confidence = scores.argmax(axis=1), scores.max(axis=1)
    policy = final["review_policy"]
    review = np.ones(len(y), dtype=bool) if policy["review_all"] else confidence < policy["threshold"]
    expected_y = np.array([labels.index(label) for label in frame.iloc[assessment_idx].category])
    if not np.array_equal(y, expected_y):
        raise ValueError("Assessment rows and saved predictions are misaligned")
    data = pd.DataFrame({"row_id": assessment_idx + 1, "true_label": [labels[i] for i in y],
                         "predicted_label": [labels[i] for i in pred], "score": confidence,
                         "correct": pred == y, "review": review})
    csv_path = REPORTS / "selected_assessment.csv"
    data.to_csv(csv_path, index=False, float_format="%.8f", lineterminator="\n")
    summary = sql_summary(csv_path)
    summary.to_csv(REPORTS / "selected_assessment_intent_summary.csv", index=False, lineterminator="\n")
    totals = {"examples": int(summary.examples.sum()), "errors": int(summary.errors.sum()),
              "flagged_for_review": int(summary.flagged_for_review.sum())}
    expected = final["assessment_results"][selected][final["score_kind"]]["review"]
    if totals != {"examples": expected["examples"], "errors": expected["total_errors"], "flagged_for_review": expected["reviewed"]}:
        raise AssertionError("DuckDB export totals differ from locked assessment")
    report = {"scope": "Selected model's development assessment; not an untouched final test",
              "candidate": selected, "score_kind": final["score_kind"], "csv_sha256": sha256_file(csv_path),
              "duckdb_reconciled_totals": totals, "contains_question_text": False}
    (REPORTS / "assessment_export.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
