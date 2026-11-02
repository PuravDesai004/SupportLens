import numpy as np
import pytest

from supportlens.calibration import (choose_review_policy, fit_temperature, multiclass_log_loss,
                                     reliability, review_summary, temperature_scale, wilson_lower)


def test_temperature_preserves_rank_and_probability_contract():
    p = np.array([[.7, .2, .1], [.1, .3, .6]])
    adjusted = temperature_scale(p, 1.7)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    np.testing.assert_array_equal(adjusted.argsort(axis=1), p.argsort(axis=1))
    with pytest.raises(ValueError):
        temperature_scale(p, 0)
    with pytest.raises(ValueError):
        temperature_scale(np.array([[.8, .8]]), 1)


def test_temperature_fit_reduces_in_sample_log_loss():
    p = np.array([[.55, .45], [.60, .40], [.45, .55], [.40, .60]])
    y = np.array([0, 0, 1, 1])
    temperature = fit_temperature(y, p)
    assert .25 <= temperature <= 4
    assert multiclass_log_loss(y, temperature_scale(p, temperature)) <= multiclass_log_loss(y, p)


@pytest.mark.parametrize("labels", [np.array([[0], [1]]), np.array([.9, 1.0]), np.array([0, 2])])
def test_metrics_reject_broadcast_or_truncated_class_labels(labels):
    p = np.array([[.8, .2], [.2, .8]])
    policy = {"threshold": .75, "review_all": False}
    for calculate in (multiclass_log_loss, reliability, choose_review_policy):
        with pytest.raises(ValueError, match="Labels"):
            calculate(labels, p)
    with pytest.raises(ValueError, match="Labels"):
        review_summary(labels, p, policy)


def test_calibration_rejects_empty_rows_and_invalid_selection_settings():
    with pytest.raises(ValueError):
        temperature_scale(np.empty((0, 2)), 1)
    with pytest.raises(ValueError):
        reliability([0], [[.9, .1]], bins=0)
    with pytest.raises(ValueError):
        choose_review_policy([0], [[.9, .1]], min_accepted=0)


def test_review_policy_and_reliability_have_correct_denominators():
    p = np.array([[.9, .1], [.8, .2], [.7, .3], [.2, .8]])
    y = np.array([0, 1, 0, 1])
    ece, bins = reliability(y, p, bins=2)
    assert ece == pytest.approx(abs(.8 - .75))
    assert sum(b["count"] for b in bins) == 4
    assert wilson_lower(0, 0) is None
    policy = {"threshold": .8, "review_all": False}
    summary = review_summary(y, p, policy)
    assert summary["accepted"] == 3 and summary["reviewed"] == 1
    assert summary["accuracy_among_accepted"] == pytest.approx(2 / 3)
    assert summary["accepted_errors"] == 1 and summary["total_errors"] == 1
    all_review = choose_review_policy(y, p, min_accepted=200)
    assert all_review["review_all"] and review_summary(y, p, all_review)["accuracy_among_accepted"] is None


def test_review_policy_selects_the_lowest_qualifying_grid_threshold():
    p = np.vstack([np.tile([.9, .1], (210, 1)), np.tile([.55, .45], (90, 1))])
    y = np.array([0] * 210 + [0] * 60 + [1] * 30)
    policy = choose_review_policy(y, p, min_accepted=200)
    assert policy["threshold"] == .575
    assert policy["development_accepted"] == 210
    assert policy["development_wilson_lower"] >= .95
