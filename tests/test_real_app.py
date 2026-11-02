from pathlib import Path
import hashlib
import json

import pytest
from streamlit.testing.v1 import AppTest

from supportlens.predict import load_bundle, predict
from supportlens.evaluate import sql_summary

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "metadata.json").exists(), reason="real artifact not available")


def test_real_bundle_has_compatible_output():
    bundle = load_bundle()
    result = predict("Where is my new card?", bundle)
    assert result["label"] in bundle[2]["labels"]
    assert len(bundle[2]["labels"]) == 77 and 0 <= result["score"] <= 1


def test_streamlit_app_real_artifact_and_invalid_input():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert any("enter a question" in warning.value.lower() for warning in app.warning)
    app.text_area[0].set_value("Where is my new card?")
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Suggested category" for metric in app.metric)


def test_shipped_evaluation_reconciles_with_recorded_evidence():
    csv_path = ROOT / "reports" / "evaluation.csv"
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == metrics["selected_csv_sha256"]
    summary = sql_summary(csv_path)
    totals = metrics["duckdb_reconciled_totals"]
    assert int(summary.examples.sum()) == totals["examples"]
    assert int(summary.errors.sum()) == totals["errors"]
    assert int(summary.flagged_for_review.sum()) == totals["flagged_for_review"]
