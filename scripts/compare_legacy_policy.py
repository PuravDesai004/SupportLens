"""Describe the old fixed threshold on the already-scored assessment; no fitting or tuning."""
import json

import numpy as np

from supportlens.calibration import review_summary, temperature_scale
from supportlens.experiment_v2 import REPORTS, RUNS


def main():
    lock = json.loads((REPORTS / "final_lock.json").read_text(encoding="utf-8"))
    with np.load(RUNS / "assessment_probabilities.npz") as arrays:
        y = arrays["y"]
        baseline = review_summary(y, arrays["word_logistic"], {"threshold": .35, "review_all": False})
        selected = arrays[lock["selected_release"]]
        if lock["score_kind"] == "temperature":
            selected = temperature_scale(selected, lock["temperature"])
        current = review_summary(y, selected, lock["review_policy"])
    result = {"scope": "Post-lock descriptive comparison on the development assessment; not an independent test.",
              "note": "The baseline is retrained on the same development pool and uses the original v1 fixed raw-score threshold; this comparison did not affect selection.",
              "matched_word_baseline_original_raw_threshold_0_35": baseline,
              "selected_model_locked_policy": current}
    path = REPORTS / "legacy_policy_assessment.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing comparison")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
