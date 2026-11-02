"""A general second-view review signal; it never changes the predicted label."""
import numpy as np
from .calibration import validate_labels, validate_probabilities, wilson_lower


def review_masks(main_probabilities, word_probabilities, main_threshold, support_threshold):
    """Compare both views on the main model's top label for each row."""
    main = np.asarray(main_probabilities, dtype=np.float64)
    word = np.asarray(word_probabilities, dtype=np.float64)
    if main.ndim != 2 or main.shape != word.shape or main.shape[1] < 2:
        raise ValueError("Both models must return the same two-dimensional class ordering")
    validate_probabilities(main)
    validate_probabilities(word)
    if not (0 <= main_threshold <= 1 and 0 <= support_threshold <= 1):
        raise ValueError("Review thresholds must be between zero and one")
    top = main.argmax(axis=1)
    main_score = main[np.arange(len(main)), top]
    word_support = word[np.arange(len(word)), top]
    low_main = main_score < main_threshold
    weak_support = word_support < support_threshold
    return {"top": top, "main_score": main_score, "word_support": word_support,
            "low_main": low_main, "weak_support": weak_support, "review": low_main | weak_support}


def summarize_review(y, predictions, review):
    y = np.asarray(y)
    predictions = np.asarray(predictions)
    review = np.asarray(review, dtype=bool)
    if y.ndim != 1 or y.shape != predictions.shape or y.shape != review.shape or len(y) == 0:
        raise ValueError("Labels, predictions and review decisions must align")
    correct = predictions == y
    accepted = ~review
    n_accepted = int(accepted.sum())
    return {"examples": len(y), "reviewed": int(review.sum()), "accepted": n_accepted,
            "accepted_accuracy": float(correct[accepted].mean()) if n_accepted else None,
            "accepted_errors": int((accepted & ~correct).sum()), "total_errors": int((~correct).sum()),
            "flagged_errors": int((review & ~correct).sum())}


def compare_policy(y, main_probabilities, word_probabilities, main_threshold, support_threshold):
    masks = review_masks(main_probabilities, word_probabilities, main_threshold, support_threshold)
    y = validate_labels(y, np.asarray(main_probabilities))
    old = masks["low_main"]
    new = masks["review"]
    correct = masks["top"] == y
    extra = new & ~old
    return {"existing": summarize_review(y, masks["top"], old),
            "guarded": summarize_review(y, masks["top"], new),
            "newly_reviewed": int(extra.sum()), "newly_reviewed_correct": int((extra & correct).sum()),
            "newly_reviewed_errors": int((extra & ~correct).sum()),
            "additional_review_fraction": float(extra.mean())}


def select_support_threshold(y, main_probabilities, word_probabilities, main_threshold,
                             max_added_fraction=.15, min_accepted=200, min_error_reduction=.10):
    """Select only from development predictions using the predeclared review budget."""
    options = []
    for step in range(37):
        threshold = round(.05 + .025 * step, 3)
        result = compare_policy(y, main_probabilities, word_probabilities, main_threshold, threshold)
        old, new = result["existing"], result["guarded"]
        if (old["accepted_errors"] > 0 and new["accepted"] >= min_accepted
                and result["additional_review_fraction"] <= max_added_fraction
                and new["accepted_accuracy"] >= old["accepted_accuracy"]
                and result["newly_reviewed_errors"] / old["accepted_errors"] >= min_error_reduction):
            options.append((threshold, result))
    if not options:
        return None
    threshold, result = max(options, key=lambda item: (item[1]["newly_reviewed_errors"],
                                                       -item[1]["newly_reviewed"], -item[0]))
    return {"support_threshold": threshold, "comparison": result, "eligible_cutoffs": len(options)}


def select_conservative_threshold(y, main_probabilities, word_probabilities, main_threshold,
                                  target_lower_bound=.97, min_accepted=1000):
    """Maximize coverage on a fixed grid subject to a development accuracy bound."""
    for step in range(37):
        threshold = round(.05 + .025 * step, 3)
        comparison = compare_policy(y, main_probabilities, word_probabilities, main_threshold, threshold)
        accepted = comparison["guarded"]["accepted"]
        errors = comparison["guarded"]["accepted_errors"]
        lower = wilson_lower(accepted - errors, accepted)
        if accepted >= min_accepted and lower is not None and lower >= target_lower_bound:
            return {"support_threshold": threshold, "development_wilson_lower": lower, "comparison": comparison}
    return None
