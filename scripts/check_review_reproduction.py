"""Verify a separate review-policy reproduction without rescoring the test set."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from supportlens.config import ROOT
from supportlens.data import sha256_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    original = ROOT / "reports" / "review_guard"
    replay = args.output_root / "reports" / "review_guard"
    a, b = read(original / "conservative_final_lock.json"), read(replay / "conservative_final_lock.json")
    x, y = read(original / "assessment_export.json"), read(replay / "assessment_export.json")
    checks = {"same_cutoff": a["selected_before_assessment"] == b["selected_before_assessment"],
              "same_assessment_comparison": a["comparison"] == b["comparison"],
              "same_export_sha256": (sha256_file(original / "assessment_comparison.csv") == x["csv_sha256"]
                                      == sha256_file(replay / "assessment_comparison.csv") == y["csv_sha256"]),
              "same_v3_metadata_sha256": sha256_file(ROOT / "artifacts" / "v3" / "metadata.json")
              == sha256_file(args.output_root / "artifacts" / "v3" / "metadata.json"),
              "test_rescored": (b["test_set_rescored"]
                                  or (args.output_root / "reports" / "experiment_v2" / "previously_viewed_test.json").exists())}
    if not all(checks[k] for k in ("same_cutoff", "same_assessment_comparison", "same_export_sha256", "same_v3_metadata_sha256")) or checks["test_rescored"]:
        raise AssertionError("Review reproduction did not match, or the official test was rescored")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Same saved predictions and seeds in a separate output root; not an independent evaluation.",
              **checks}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError("Refusing to overwrite a reproduction report")
        args.report.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
