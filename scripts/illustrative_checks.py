"""Manually authored examples, not user inputs or labeled performance evidence."""
import argparse
import json
from pathlib import Path

from supportlens.config import ROOT
from supportlens.predict import load_bundle, predict

EXAMPLES = [
    ("in_domain", "My replacement card has not arrived yet."),
    ("in_domain", "I cannot remember the PIN for my card."),
    ("in_domain", "Why was I charged a fee for a cash withdrawal abroad?"),
    ("in_domain", "The same card payment appears twice on my statement."),
    ("ambiguous", "My card fails only in one particular shop."),
    ("out_of_scope", "Can you recommend a vegetarian pizza?"),
    ("out_of_scope", "What is the weather like on Mars?"),
    ("out_of_scope", "How do I cancel my Netflix account?"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "experiment_v2" / "illustrative_predictions.json")
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("Refusing to overwrite historical illustrative evidence")
    old = load_bundle(ROOT / "artifacts" / "previous_v1")
    current = load_bundle(ROOT / "artifacts" / "v2")
    rows = [{"kind": kind, "manually_authored_example": text,
             "previous_v1": predict(text, old), "selected_v2": predict(text, current)}
            for kind, text in EXAMPLES]
    result = {"scope": "Illustrative manually authored examples, not user-entered text, ground truth or performance estimates.",
              "examples": rows}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
