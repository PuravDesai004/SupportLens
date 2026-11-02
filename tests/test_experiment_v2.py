import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from supportlens.data import normalized
from supportlens.experiment_v2 import (experiment_splits, fit_candidate, macro_f1_fast, make_vectorizer,
                                      paired_bootstrap_delta, probabilities_for, report_metrics)


def test_grouped_splits_are_repeatable_and_keep_duplicates_together():
    rows = [(f"{label} unique question number {i}", label) for label in ("card", "transfer") for i in range(30)]
    rows += [(" CARD unique question number 0 ", "card")]
    frame = pd.DataFrame(rows, columns=["text", "category"])
    dev, assessment, folds = experiment_splits(frame)
    dev2, assessment2, folds2 = experiment_splits(frame)
    np.testing.assert_array_equal(dev, dev2)
    np.testing.assert_array_equal(assessment, assessment2)
    assert set(frame.iloc[dev].text.map(normalized)).isdisjoint(set(frame.iloc[assessment].text.map(normalized)))
    for (train, valid), (train2, valid2) in zip(folds, folds2):
        np.testing.assert_array_equal(valid, valid2)
        assert set(frame.iloc[dev].iloc[train].text.map(normalized)).isdisjoint(
            set(frame.iloc[dev].iloc[valid].text.map(normalized)))


def test_hybrid_vocabulary_is_fit_only_on_training_text():
    vectorizer = make_vectorizer("word_char_logistic")
    vectorizer.fit(["card missing", "card missing", "transfer waiting", "transfer waiting"])
    vectorizer.transform(["zzqxxy unseen token", "zzqxxy unseen token"])
    parts = dict(vectorizer.transformer_list)
    assert "zzqxxy" not in parts["word"].vocabulary_
    assert "zzq" not in parts["char"].vocabulary_


def test_small_hybrid_training_and_metrics_work_without_full_data():
    texts = pd.Series(["card missing", "card missing", "card late", "card late",
                       "transfer pending", "transfer pending", "transfer late", "transfer late"])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    vectorizer, model, info = fit_candidate("word_char_logistic", texts, y, 2)
    probs = probabilities_for("word_char_logistic", vectorizer, model, texts)
    assert probs.shape == (8, 2) and info["convergence_warnings"] == []
    result, classes = report_metrics(y, probs, ["card", "transfer"], {"threshold": None, "review_all": True})
    assert result["examples"] == 8 and result["review"]["accepted"] == 0
    assert result["top3_accuracy"] == 1.0 and classes["card"]["support"] == 4


def test_paired_bootstrap_and_macro_f1_match_reference():
    y = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 1, 0])
    assert macro_f1_fast(y, pred, 2) == pytest.approx(f1_score(y, pred, average="macro"))
    first = paired_bootstrap_delta(y, pred, pred, 2, repeats=20)
    second = paired_bootstrap_delta(y, pred, pred, 2, repeats=20)
    assert first == second and first["lower_95"] == first["upper_95"] == 0
