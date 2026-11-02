import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from supportlens.config import VECTORIZER_CONFIG
from supportlens.data import normalized, sha256_file, split_train_validation, validate_frame
from supportlens.evaluate import sql_summary, summarize
from supportlens.predict import ArtifactError, load_active_bundle, load_bundle, predict
from supportlens.torch_model import predict_probabilities, train_network


def test_data_contract_rejects_bad_rows_and_unknown_labels():
    with pytest.raises(ValueError, match="columns"):
        validate_frame(pd.DataFrame({"text": ["x"]}))
    with pytest.raises(ValueError, match="blank"):
        validate_frame(pd.DataFrame({"text": ["  "], "category": ["valid"]}))
    with pytest.raises(ValueError, match="identifiers"):
        validate_frame(pd.DataFrame({"text": ["hello"], "category": ["Bad Label"]}))
    with pytest.raises(ValueError, match="unknown"):
        validate_frame(pd.DataFrame({"text": ["hello"], "category": ["unknown"]}), ["known"])


def test_group_split_and_label_mapping_are_deterministic():
    records = [(f"{label} question {i}", label) for label in ("card", "transfer") for i in range(15)]
    records += [("  CARD  question 0 ", "card")]
    frame = pd.DataFrame(records, columns=["text", "category"])
    a, b = split_train_validation(frame)
    c, d = split_train_validation(frame)
    assert a.equals(c) and b.equals(d)
    assert set(a.text.map(normalized)).isdisjoint(set(b.text.map(normalized)))
    assert LabelEncoder().fit(frame.category).classes_.tolist() == ["card", "transfer"]


def test_validation_only_token_cannot_enter_vocabulary():
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG, dtype=np.float32)
    vectorizer.fit(["card delivery", "card delivery", "transfer waiting", "transfer waiting"])
    vectorizer.transform(["nevertrainword", "nevertrainword"])
    assert "nevertrainword" not in vectorizer.vocabulary_


def test_saved_bundle_predicts_and_checks_inputs(tmp_path):
    labels = ["card", "transfer"]
    vectorizer = TfidfVectorizer().fit(["card delivery", "card missing", "transfer pending", "transfer late"])
    x = vectorizer.transform(["card delivery", "card missing", "transfer pending", "transfer late"])
    model = LogisticRegression().fit(x, [0, 0, 1, 1])
    joblib.dump(vectorizer, tmp_path / "vectorizer.joblib")
    joblib.dump(model, tmp_path / "model.joblib")
    meta = {"model": "baseline", "input_dim": x.shape[1], "output_dim": 2, "labels": labels,
            "sklearn_version": sklearn.__version__, "review_threshold": 1.0,
            "sha256": {n: sha256_file(tmp_path / n) for n in ("vectorizer.joblib", "model.joblib")}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    bundle = load_bundle(tmp_path)
    result = predict("card delivery", bundle)
    assert result["label"] in labels and result["review"] and 0 <= result["score"] <= 1
    assert len(result["alternatives"]) == 1
    for bad in ("", "   ", None, 123, "x" * 2001):
        with pytest.raises(ValueError):
            predict(bad, bundle)
    (tmp_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


@pytest.mark.parametrize("change, message", [
    ({"sha256": {}}, "checksum"),
    ({"sha256": {"vectorizer.joblib": "0" * 64, "model.joblib": "0" * 64,
                 "../outside.joblib": "0" * 64}}, "checksum"),
    ({"labels": ["card", "card"]}, "distinct"),
    ({"review_threshold": float("nan")}, "threshold"),
    ({"temperature": float("nan")}, "Temperature"),
    ({"artifact_version": 3}, "requires its review guard"),
])
def test_invalid_bundle_metadata_is_rejected_before_deserialization(tmp_path, monkeypatch, change, message):
    meta = {"model": "baseline", "input_dim": 2, "output_dim": 2, "labels": ["card", "transfer"],
            "sklearn_version": sklearn.__version__, "review_threshold": .75,
            "sha256": {"vectorizer.joblib": "0" * 64, "model.joblib": "0" * 64}}
    meta.update(change)
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    def must_not_load(*args, **kwargs):
        pytest.fail("Invalid metadata reached artifact deserialization")

    monkeypatch.setattr(joblib, "load", must_not_load)
    with pytest.raises(ArtifactError, match=message):
        load_bundle(tmp_path)


@pytest.mark.parametrize("probabilities", [
    [[1.2, -.2]], [[.8, .8]], [[float("nan"), .5]], [[.8, .2], [.8, .2]], [.8, .2],
])
def test_unscaled_predictions_reject_invalid_probability_output(probabilities):
    class Vectorizer:
        def transform(self, texts):
            return texts

    class Model:
        def predict_proba(self, texts):
            return probabilities

    meta = {"model": "baseline", "labels": ["card", "transfer"], "temperature": 1.0,
            "review_threshold": .75}
    with pytest.raises(ArtifactError):
        predict("A generic question", (Vectorizer(), Model(), meta))
    with pytest.raises(ValueError) as error:
        predict("", (Vectorizer(), Model(), meta))
    assert not isinstance(error.value, ArtifactError)


def test_review_threshold_temperature_and_release_pointer(tmp_path):
    class Vectorizer:
        def transform(self, texts):
            return texts

    class Model:
        def predict_proba(self, texts):
            return np.array([[.6, .4]])

    meta = {"model": "baseline", "labels": ["card", "transfer"], "temperature": .5,
            "review_threshold": .60, "review_all": False}
    result = predict("A generic question", (Vectorizer(), Model(), meta))
    assert result["score"] == pytest.approx(.36 / (.36 + .16))
    meta["review_threshold"] = result["score"]
    assert not predict("A generic question", (Vectorizer(), Model(), meta))["review"]  # Equality is accepted.
    meta["review_threshold"] = result["score"] + .001
    assert predict("A generic question", (Vectorizer(), Model(), meta))["review"]
    meta["review_all"] = True
    assert predict("A generic question", (Vectorizer(), Model(), meta))["review"]
    (tmp_path / "active.json").write_text(json.dumps({"release": "../other", "metadata_sha256": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="release"):
        load_active_bundle(tmp_path)
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "active.json").write_text(json.dumps({"release": "v2", "metadata_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        load_active_bundle(tmp_path)


def test_review_guard_files_are_checked_before_loading(tmp_path):
    from supportlens.promote_review_guard import label_hash
    labels = ["card", "transfer"]
    texts = ["card delivery", "card missing", "transfer pending", "transfer late"]
    vectorizer = TfidfVectorizer().fit(texts)
    x = vectorizer.transform(texts)
    model = LogisticRegression().fit(x, [0, 0, 1, 1])
    for name, obj in (("vectorizer.joblib", vectorizer), ("model.joblib", model),
                      ("review_vectorizer.joblib", vectorizer), ("review_model.joblib", model)):
        joblib.dump(obj, tmp_path / name)
    files = ["vectorizer.joblib", "model.joblib", "review_vectorizer.joblib", "review_model.joblib"]
    meta = {"model": "baseline", "input_dim": x.shape[1], "output_dim": 2, "labels": labels,
            "sklearn_version": sklearn.__version__, "review_threshold": .75,
            "sha256": {name: sha256_file(tmp_path / name) for name in files},
            "review_guard": {"kind": "word_support", "vectorizer_file": "review_vectorizer.joblib",
                             "model_file": "review_model.joblib", "input_dim": x.shape[1],
                             "support_threshold": .475, "label_order_sha256": label_hash(labels)}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    bundle = load_bundle(tmp_path)
    assert predict("card delivery", bundle)["label"] in labels
    (tmp_path / "review_model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_metrics_and_duckdb_totals_reconcile(tmp_path):
    y = np.array([0, 0, 1, 1])
    p = np.array([[.9, .1], [.4, .6], [.2, .8], [.45, .55]])
    result, report, pred, score, review = summarize(y, p, ["card", "transfer"], .6)
    assert result["accuracy"] == pytest.approx(.75)
    assert result["macro_f1"] == pytest.approx((2 / 3 + .8) / 2)
    assert result["accepted_count"] == 3 and result["accuracy_among_accepted"] == pytest.approx(2 / 3)
    assert report["card"]["support"] == 2
    no_accepted = summarize(y, p, ["card", "transfer"], 1.0)[0]
    assert no_accepted["accepted_count"] == 0 and no_accepted["accuracy_among_accepted"] is None
    csv = tmp_path / "evaluation.csv"
    pd.DataFrame({"true_label": ["card", "card", "transfer", "transfer"], "correct": pred == y,
                  "review": review}).to_csv(csv, index=False)
    summary = sql_summary(csv)
    assert summary.examples.sum() == 4 and summary.errors.sum() == 1 and summary.flagged_for_review.sum() == 1


def test_cpu_torch_training_and_inference_fixture():
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(["card missing", "card late", "transfer waiting", "transfer late"])
    net, losses = train_network(x, np.array([0, 0, 1, 1]), hidden_dim=4, epochs=2,
                                batch_size=2, learning_rate=.01, seed=42, n_classes=2)
    probabilities = predict_probabilities(net, x, batch_size=2)
    assert len(losses) == 2 and probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
