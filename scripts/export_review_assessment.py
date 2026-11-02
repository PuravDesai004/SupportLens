"""Export the previously viewed assessment's old and new review decisions, without text."""
import json

import numpy as np
import pandas as pd

from supportlens.calibration import temperature_scale
from supportlens.config import ARTIFACT_DIR
from supportlens.data import sha256_file
from supportlens.experiment_v2 import checked_training_data, experiment_splits
from supportlens.review_guard import review_masks
from supportlens.review_guard_experiment import OUTPUT, V2_RUNS


def main():
    path = OUTPUT / "assessment_comparison.csv"
    report_path = OUTPUT / "assessment_export.json"
    if path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite review assessment evidence")
    decision = json.loads((OUTPUT / "conservative_final_lock.json").read_text(encoding="utf-8"))
    if not decision["promoted"]:
        raise RuntimeError("No conservative policy was promoted")
    meta = json.loads((ARTIFACT_DIR / "v3" / "metadata.json").read_text(encoding="utf-8"))
    development = json.loads((OUTPUT / "conservative_development_lock.json").read_text(encoding="utf-8"))
    if (meta["review_guard"]["support_threshold"] != decision["selected_before_assessment"]
            or meta["temperature"] != development["main_temperature"]
            or meta["review_threshold"] != development["main_threshold"]
            or meta.get("review_all", False)):
        raise ValueError("Release policy differs from the frozen review evidence")
    source = V2_RUNS / "assessment_probabilities.npz"
    if sha256_file(source) != decision["source_assessment_sha256"]:
        raise ValueError("Assessment predictions differ from the frozen review evidence")
    frame, labels = checked_training_data()
    _, assessment_idx, _ = experiment_splits(frame)
    with np.load(source) as arrays:
        y = arrays["y"].copy()
        main = temperature_scale(arrays["word_char_logistic"], meta["temperature"])
        word = arrays["word_logistic"].copy()
    if not np.array_equal(y, np.array([labels.index(label) for label in frame.iloc[assessment_idx].category])):
        raise ValueError("Assessment rows and saved predictions do not align")
    masks = review_masks(main, word, meta["review_threshold"], meta["review_guard"]["support_threshold"])
    output = pd.DataFrame({"row_id": assessment_idx + 1, "true_label": [labels[i] for i in y],
                           "predicted_label": [labels[i] for i in masks["top"]], "main_score": masks["main_score"],
                           "word_support": masks["word_support"], "correct": masks["top"] == y,
                           "previous_review": masks["low_main"], "weak_word_support": masks["weak_support"],
                           "current_review": masks["review"]})
    previous, current = decision["comparison"]["existing"], decision["comparison"]["guarded"]
    if (len(output) != current["examples"] or int((~output.correct).sum()) != current["total_errors"]
            or int(output.previous_review.sum()) != previous["reviewed"]
            or int((~output.correct & ~output.previous_review).sum()) != previous["accepted_errors"]
            or int(output.current_review.sum()) != current["reviewed"]
            or int((~output.correct & ~output.current_review).sum()) != current["accepted_errors"]):
        raise AssertionError("Assessment export does not reconcile with the frozen decision")
    output.to_csv(path, mode="x", index=False, float_format="%.8f", lineterminator="\n")
    result = {"scope": "Previously viewed development assessment, not an independent test",
              "csv_sha256": sha256_file(path), "rows": len(output), "previous_reviewed": previous["reviewed"],
              "current_reviewed": current["reviewed"], "accepted_errors": current["accepted_errors"],
              "contains_question_text": False, "test_set_rescored": False}
    with report_path.open("x", encoding="utf-8", newline="\n") as report:
        report.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
