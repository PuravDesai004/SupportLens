"""Copy the assessment-locked candidate into a versioned inference release."""
import json
import shutil

from .config import ARTIFACT_DIR, DATA_SHA256, REPORT_DIR, RUN_DIR
from .data import sha256_file


def main():
    final = json.loads((REPORT_DIR / "experiment_v2" / "final_lock.json").read_text(encoding="utf-8"))
    if not final["promoted"]:
        raise RuntimeError("The assessment gate retained v1; no candidate may be promoted")
    name = final["selected_release"]
    source = RUN_DIR / "experiment_v2" / name
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata["candidate_id"] != name or metadata["sha256"] != final["candidate_artifact_sha256"][name]:
        raise ValueError("Candidate does not match the pre-test release lock")
    for filename, digest in metadata["sha256"].items():
        if sha256_file(source / filename) != digest:
            raise ValueError("Candidate artifact checksum mismatch")
    destination = ARTIFACT_DIR / "v2"
    if destination.exists():
        raise RuntimeError("The v2 release already exists; use a separate output root for reproduction")
    destination.mkdir(parents=True)
    for filename in metadata["sha256"]:
        shutil.copyfile(source / filename, destination / filename)
    policy = final["review_policy"]
    metadata.update({"temperature": final["temperature"], "score_kind": final["score_kind"],
                     "review_threshold": policy["threshold"], "review_all": policy["review_all"],
                     "training_data_sha256": DATA_SHA256["train.csv"], "fit_scope": "8,001-row development pool",
                     "selection_evidence": "reports/experiment_v2/final_lock.json"})
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8", newline="\n")
    pointer = {"release": "v2", "metadata_sha256": sha256_file(destination / "metadata.json")}
    (ARTIFACT_DIR / "active.json").write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8", newline="\n")
    print(f"Activated {name} from the frozen assessment release; previous v1 files are unchanged.")


if __name__ == "__main__":
    main()
