"""Package the frozen main and word-only models as one checked local release."""
import hashlib
import json
import shutil

from .config import ARTIFACT_DIR, REPORT_DIR, RUN_DIR
from .data import sha256_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def label_hash(labels):
    return hashlib.sha256(json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    decision = read(REPORT_DIR / "review_guard" / "conservative_final_lock.json")
    if not decision["promoted"]:
        raise RuntimeError("The conservative guard did not pass its release gate")
    v2 = ARTIFACT_DIR / "v2"
    word = RUN_DIR / "experiment_v2" / "word_logistic"
    main_meta, word_meta = read(v2 / "metadata.json"), read(word / "metadata.json")
    if main_meta["sha256"] != decision["main_artifact_sha256"] or word_meta["sha256"] != decision["guard_artifact_sha256"]:
        raise ValueError("Model artifacts do not match the frozen review decision")
    if main_meta["labels"] != word_meta["labels"] or main_meta["output_dim"] != word_meta["output_dim"]:
        raise ValueError("The two models do not share a label order")
    for directory, meta in ((v2, main_meta), (word, word_meta)):
        for filename, digest in meta["sha256"].items():
            if sha256_file(directory / filename) != digest:
                raise ValueError("A frozen model file has changed")
    destination = ARTIFACT_DIR / "v3"
    if destination.exists():
        raise FileExistsError("The v3 release already exists; use a separate output root for reproduction")
    destination.mkdir(parents=True)
    files = ((v2 / "model.joblib", "model.joblib"), (v2 / "vectorizer.joblib", "vectorizer.joblib"),
             (word / "model.joblib", "review_model.joblib"), (word / "vectorizer.joblib", "review_vectorizer.joblib"))
    for source, name in files:
        shutil.copyfile(source, destination / name)
    main_meta["artifact_version"] = 3
    main_meta["sha256"] = {name: sha256_file(destination / name) for _, name in files}
    main_meta["review_guard"] = {"kind": "word_support", "candidate_id": "word_logistic",
                                 "support_threshold": decision["selected_before_assessment"],
                                 "model_file": "review_model.joblib", "vectorizer_file": "review_vectorizer.joblib",
                                 "input_dim": word_meta["input_dim"], "label_order_sha256": label_hash(word_meta["labels"]),
                                 "evidence": "reports/review_guard/conservative_final_lock.json"}
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(main_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    pointer = {"release": "v3", "metadata_sha256": sha256_file(metadata_path)}
    (ARTIFACT_DIR / "active.json").write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8", newline="\n")
    print("Activated the frozen second-view review guard; v1 and v2 artifacts remain unchanged.")


if __name__ == "__main__":
    main()
