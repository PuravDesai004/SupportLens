"""Summarize the already generated out-of-fold predictions, without refitting."""
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from supportlens.experiment_v2 import CANDIDATES, REPORTS, RUNS


def main():
    path = REPORTS / "development_fold_metrics.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite development fold evidence")
    with np.load(RUNS / "development_oof.npz") as arrays:
        y, folds = arrays["y"], arrays["fold_ids"]
        result = {}
        for name in CANDIDATES:
            pred = arrays[name].argmax(axis=1)
            result[name] = []
            for fold in (1, 2, 3):
                selected = folds == fold
                result[name].append({"fold": fold, "examples": int(selected.sum()),
                                     "accuracy": float(accuracy_score(y[selected], pred[selected])),
                                     "macro_f1": float(f1_score(y[selected], pred[selected], average="macro", zero_division=0))})
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
