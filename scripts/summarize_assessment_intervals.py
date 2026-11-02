"""Describe paired assessment differences without changing the frozen release decision."""
import json

import numpy as np

from supportlens.experiment_v2 import BASELINE, CANDIDATES, REPORTS, RUNS, macro_f1_fast, paired_bootstrap_delta


def main():
    path = REPORTS / "assessment_paired_intervals.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite assessment interval evidence")
    with np.load(RUNS / "assessment_probabilities.npz") as arrays:
        y = arrays["y"]
        baseline = arrays[BASELINE].argmax(axis=1)
        classes = arrays[BASELINE].shape[1]
        result = {}
        for name in CANDIDATES:
            pred = arrays[name].argmax(axis=1)
            result[name] = {"macro_f1_gain_vs_baseline": macro_f1_fast(y, pred, classes) - macro_f1_fast(y, baseline, classes),
                            "bootstrap_95": paired_bootstrap_delta(y, pred, baseline, classes)}
    report = {"scope": "Development assessment; used by the predeclared release gate, not an untouched final test.",
              "caveat": "Paired, class-stratified row bootstrap is descriptive and does not account for prior dataset exposure.",
              "models": result}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
