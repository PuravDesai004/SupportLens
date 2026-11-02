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
