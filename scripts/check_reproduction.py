"""Compare a separate development/assessment reproduction to the recorded run."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from supportlens.config import ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    original = ROOT / "reports" / "experiment_v2"
    replay = args.output_root / "reports" / "experiment_v2"
    first_dev, second_dev = read(original / "development_lock.json"), read(replay / "development_lock.json")
    first_final, second_final = read(original / "final_lock.json"), read(replay / "final_lock.json")
    names = first_dev["candidate_order"]
    fields = ("accuracy", "macro_f1", "top3_accuracy", "log_loss", "ece_10_bins")
    metrics = {}
    for name in names:
        metrics[name] = {}
        for stage, left, right in (("development", first_dev["models"][name], second_dev["models"][name]),
                                   ("assessment", first_final["assessment_results"][name], second_final["assessment_results"][name])):
            metrics[name][stage] = {field: {"original": left["raw"][field], "reproduced": right["raw"][field]}
                                    for field in fields}
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Same data and seeds, separate output root; reproduction is not an independent evaluation.",
              "test_rescored": (replay / "previously_viewed_test.json").exists(),
              "same_assignment_hash": first_dev["assignment_sha256"] == second_dev["assignment_sha256"],
              "same_candidate_artifact_hashes": first_final["candidate_artifact_sha256"] == second_final["candidate_artifact_sha256"],
              "same_temperatures": first_dev["temperature"] == second_dev["temperature"],
              "same_selected_release": first_final["selected_release"] == second_final["selected_release"],
              "same_review_policy": first_final["review_policy"] == second_final["review_policy"],
              "same_raw_metrics": all(abs(v["original"] - v["reproduced"]) < 1e-12
                                      for model in metrics.values() for stage in model.values() for v in stage.values()),
              "metrics": metrics}
    if not all(result[k] for k in ("same_assignment_hash", "same_candidate_artifact_hashes", "same_temperatures",
                                   "same_selected_release", "same_review_policy", "same_raw_metrics")):
        raise AssertionError("Reproduction differs from the recorded run; inspect before reporting equality")
    if result["test_rescored"]:
        raise AssertionError("This verification is intended to avoid rescoring the previously viewed test")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError("Refusing to overwrite an existing reproduction report")
        args.report.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
