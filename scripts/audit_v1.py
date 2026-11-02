"""Inspect v1's original validation errors; public dataset examples only."""
import json
from collections import Counter

import numpy as np

from supportlens.config import DATA_DIR, DATA_SHA256, ROOT
from supportlens.data import load_data, sha256_file, split_train_validation
from supportlens.predict import load_bundle


def main():
    output = ROOT / "reports" / "experiment_v2" / "baseline_error_audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite the historical v1 error audit")
    train_path = DATA_DIR / "train.csv"
    if sha256_file(train_path) != DATA_SHA256["train.csv"]:
        raise ValueError("Unexpected training data checksum")
    data = load_data(train_path)
    _, validation = split_train_validation(data)
    vectorizer, model, meta = load_bundle(ROOT / "artifacts" / "previous_v1")
    scores = model.predict_proba(vectorizer.transform(validation.text))
    predictions = [meta["labels"][int(i)] for i in scores.argmax(axis=1)]
    mistakes = [(truth, pred, text) for truth, pred, text in zip(validation.category, predictions, validation.text)
                if truth != pred]
    pairs = Counter((truth, pred) for truth, pred, _ in mistakes)
    examples = []
    for (truth, pred), count in pairs.most_common(12):
        text = next(text for t, p, text in mistakes if (t, p) == (truth, pred))
        examples.append({"true_label": truth, "predicted_label": pred, "pair_count": count,
                         "public_dataset_example": text})
    result = {"scope": "Original v1 validation rows from the official training file, previously used for selection",
              "data_sha256": DATA_SHA256["train.csv"], "examples": len(validation), "mistakes": len(mistakes),
              "accuracy": float(np.mean(validation.category.to_numpy() == np.asarray(predictions))),
              "common_confusions": [{"true_label": t, "predicted_label": p, "count": n}
                                    for (t, p), n in pairs.most_common(20)],
              "representative_public_data_mistakes": examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as report:
        report.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
