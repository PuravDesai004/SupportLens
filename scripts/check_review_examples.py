"""Compare releases on existing manually authored examples, not visitor inputs."""
import json

from supportlens.config import ROOT
from supportlens.predict import load_bundle, predict


def main():
    examples = json.loads((ROOT / "reports" / "experiment_v2" / "illustrative_predictions.json").read_text(encoding="utf-8"))["examples"]
    previous = load_bundle(ROOT / "artifacts" / "v2")
    current = load_bundle(ROOT / "artifacts" / "v3")
    rows = [{"kind": item["kind"], "manually_authored_example": item["manually_authored_example"],
             "previous": predict(item["manually_authored_example"], previous),
             "current": predict(item["manually_authored_example"], current)} for item in examples]
    report = {"scope": "Manually authored illustrations, not user inputs, labeled benchmarks or estimates of out-of-scope detection.",
              "examples": rows}
    path = ROOT / "reports" / "review_guard" / "illustrative_checks.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite illustrative evidence")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
