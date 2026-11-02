import numpy as np
import pytest

from supportlens.review_guard import (compare_policy, review_masks, select_conservative_threshold,
                                     select_support_threshold)


def test_high_main_score_can_be_flagged_for_weak_second_view():
    main = np.array([[.96, .04], [.60, .40], [.96, .04]])
    word = np.array([[.20, .80], [.90, .10], [.80, .20]])
    result = review_masks(main, word, main_threshold=.75, support_threshold=.30)
    assert result["review"].tolist() == [True, True, False]
    assert result["low_main"].tolist() == [False, True, False]
    assert result["weak_support"].tolist() == [True, False, False]
    assert result["top"].tolist() == [0, 0, 0]


def test_support_is_for_the_main_label_not_the_second_models_favorite():
    main = np.array([[.9, .1]])
    word = np.array([[.3, .7]])
    result = review_masks(main, word, .75, .4)
    assert result["word_support"][0] == pytest.approx(.3)
    assert result["review"][0]
    assert not review_masks(main, word, .75, .3)["review"][0]
    with pytest.raises(ValueError, match="ordering"):
        review_masks(main, np.array([[.2, .3, .5]]), .75, .3)


def test_development_selection_obeys_review_budget_and_catches_errors():
    # Main predicts class 0 throughout. Low word support identifies most wrong rows.
    main = np.tile([.9, .1], (100, 1))
    y = np.array([1] * 10 + [0] * 90)
    word = np.vstack([np.tile([.1, .9], (8, 1)), np.tile([.2, .8], (2, 1)),
                      np.tile([.1, .9], (4, 1)), np.tile([.8, .2], (86, 1))])
    chosen = select_support_threshold(y, main, word, .75, max_added_fraction=.15, min_accepted=20)
    assert chosen is not None
    result = chosen["comparison"]
    assert result["newly_reviewed"] <= 15
    assert result["newly_reviewed_errors"] == 10
    assert result["guarded"]["accepted_errors"] == 0
    assert result["guarded"]["accepted_accuracy"] >= result["existing"]["accepted_accuracy"]
    assert compare_policy(y, main, word, .75, chosen["support_threshold"]) == result


def test_selection_retains_existing_policy_when_no_errors_can_be_caught():
    main = np.tile([.9, .1], (20, 1))
    word = np.tile([.2, .8], (20, 1))
    assert select_support_threshold(np.zeros(20, dtype=int), main, word, .75, min_accepted=1) is None


def test_conservative_selection_uses_the_lowest_qualifying_grid_cutoff():
    main = np.tile([.9, .1], (1000, 1))
    y = np.array([1] * 100 + [0] * 900)
    word = np.vstack([np.tile([.2, .8], (120, 1)), np.tile([.8, .2], (880, 1))])
    selected = select_conservative_threshold(y, main, word, .75, target_lower_bound=.97, min_accepted=500)
    assert selected["support_threshold"] == .225
    assert selected["comparison"]["guarded"]["accepted"] == 880
    assert selected["comparison"]["guarded"]["accepted_errors"] == 0
    assert selected["development_wilson_lower"] >= .97
