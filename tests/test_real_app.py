from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from supportlens.predict import load_bundle, predict
from supportlens.evaluate import sql_summary
from supportlens.calibration import temperature_scale

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "metadata.json").exists(), reason="real artifact not available")


def test_real_bundle_has_compatible_output():
    bundle = load_bundle()
    result = predict("Where is my new card?", bundle)
    assert result["label"] in bundle[2]["labels"]
    assert len(bundle[2]["labels"]) == 77 and 0 <= result["score"] <= 1


def test_active_release_matches_locked_model_and_scaled_inference():
    vectorizer, model, meta = load_bundle()
    lock = json.loads((ROOT / "reports" / "experiment_v2" / "final_lock.json").read_text(encoding="utf-8"))
    assert meta["candidate_id"] == lock["selected_release"]
    main_hashes = lock["candidate_artifact_sha256"][meta["candidate_id"]]
    assert {name: meta["sha256"][name] for name in main_hashes} == main_hashes
    guard_lock = json.loads((ROOT / "reports" / "review_guard" / "conservative_final_lock.json").read_text(encoding="utf-8"))
    assert meta["review_guard"]["support_threshold"] == guard_lock["selected_before_assessment"]
    assert meta["sha256"]["review_model.joblib"] == guard_lock["guard_artifact_sha256"]["model.joblib"]
    assert meta["sha256"]["review_vectorizer.joblib"] == guard_lock["guard_artifact_sha256"]["vectorizer.joblib"]
    assert vectorizer.transform([""]).shape[1] == model.n_features_in_ == meta["input_dim"]
    text = "My card has not arrived yet."
    raw = model.predict_proba(vectorizer.transform([text]))
    adjusted = temperature_scale(raw, meta["temperature"])[0]
    result = predict(text, (vectorizer, model, meta))
    assert result["score"] == pytest.approx(float(adjusted.max()))
    assert result["label"] == meta["labels"][int(np.argmax(adjusted))]
    guard_vectorizer, guard_model = meta["_loaded_review_guard"]
    word = guard_model.predict_proba(guard_vectorizer.transform([text]))[0]
    word_support = word[int(np.argmax(adjusted))]
    assert result["review"] == (result["score"] < meta["review_threshold"]
                                or word_support < meta["review_guard"]["support_threshold"])


def test_real_high_score_can_receive_a_second_view_review():
    text = "My replacement card has not arrived yet."
    previous = predict(text, load_bundle(ROOT / "artifacts" / "v2"))
    current = predict(text, load_bundle())
    assert current["score"] > .75 and current["label"] == previous["label"]
    assert current["score"] == pytest.approx(previous["score"])
    assert not previous["review"] and current["review"]
    assert "weak_word_support" in current["review_reasons"]


def test_streamlit_app_real_artifact_and_invalid_input():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert any("enter a question" in warning.value.lower() for warning in app.warning)
    app.text_area[0].set_value("Where is my new card?")
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Suggested category" for metric in app.metric)
    app.text_area[0].set_value("What is the weather like on Mars?")
    app.button[0].click().run()
    assert not app.exception
    assert any("review recommended" in warning.value.lower() and "75%" in warning.value for warning in app.warning)
    app.text_area[0].set_value("My replacement card has not arrived yet.")
    app.button[0].click().run()
    assert not app.exception
    assert any("second model" in warning.value.lower() and "high" in warning.value.lower() for warning in app.warning)


def test_streamlit_reload_uses_new_release_and_handles_broken_artifacts(tmp_path):
    shutil.copyfile(ROOT / "app.py", tmp_path / "app.py")
    artifacts = tmp_path / "artifacts"
    for release in ("v2", "v3"):
        shutil.copytree(ROOT / "artifacts" / release, artifacts / release)

    def activate(release):
        digest = hashlib.sha256((artifacts / release / "metadata.json").read_bytes()).hexdigest()
        (artifacts / "active.json").write_text(
            json.dumps({"release": release, "metadata_sha256": digest}), encoding="utf-8")

    activate("v2")
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=30).run()
    app.text_area[0].set_value("My replacement card has not arrived yet.")
    app.button[0].click().run()
    assert not app.exception and app.metric
    assert not any("second model" in warning.value.lower() for warning in app.warning)
    previous_score = next(metric.value for metric in app.metric if metric.label == "Model score")

    activate("v3")
    app.button[0].click().run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "Model score") == previous_score
    assert any("second model" in warning.value.lower() for warning in app.warning)

    (artifacts / "active.json").write_text("{broken json", encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception and not app.metric
    assert any("model files are missing or incompatible" in error.value.lower() for error in app.error)


def test_shipped_evaluation_reconciles_with_recorded_evidence():
    csv_path = ROOT / "reports" / "evaluation.csv"
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == metrics["selected_csv_sha256"]
    summary = sql_summary(csv_path)
    totals = metrics["duckdb_reconciled_totals"]
    assert int(summary.examples.sum()) == totals["examples"]
    assert int(summary.errors.sum()) == totals["errors"]
    assert int(summary.flagged_for_review.sum()) == totals["flagged_for_review"]


def test_selected_assessment_export_reconciles_with_locked_report():
    root = ROOT / "reports" / "experiment_v2"
    csv_path = root / "selected_assessment.csv"
    evidence = json.loads((root / "assessment_export.json").read_text(encoding="utf-8"))
    locked = json.loads((root / "final_lock.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == evidence["csv_sha256"]
    summary = sql_summary(csv_path)
    assert int(summary.examples.sum()) == evidence["duckdb_reconciled_totals"]["examples"]
    assert int(summary.errors.sum()) == evidence["duckdb_reconciled_totals"]["errors"]
    assert int(summary.flagged_for_review.sum()) == evidence["duckdb_reconciled_totals"]["flagged_for_review"]
    assert evidence["candidate"] == locked["selected_release"] and not evidence["contains_question_text"]


def test_review_assessment_export_matches_frozen_decisions():
    root = ROOT / "reports" / "review_guard"
    path = root / "assessment_comparison.csv"
    evidence = json.loads((root / "assessment_export.json").read_text(encoding="utf-8"))
    final = json.loads((root / "conservative_final_lock.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["csv_sha256"]
    data = pd.read_csv(path)
    assert len(data) == evidence["rows"] == 2002
    assert int(data.previous_review.sum()) == final["comparison"]["existing"]["reviewed"]
    assert int(data.current_review.sum()) == final["comparison"]["guarded"]["reviewed"]
    assert int((~data.correct & ~data.current_review).sum()) == final["comparison"]["guarded"]["accepted_errors"]
    assert "text" not in data.columns and not evidence["contains_question_text"]
    previous = pd.read_csv(ROOT / "reports" / "experiment_v2" / "selected_assessment.csv")
    assert data.row_id.tolist() == previous.row_id.tolist()
    assert data.predicted_label.tolist() == previous.predicted_label.tolist()
    np.testing.assert_allclose(data.main_score, previous.score, atol=1e-8)
