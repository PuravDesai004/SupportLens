"""Temperature scaling and review policy, independent of model training."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def validate_probabilities(probabilities):
    """Reject malformed model output before scoring, calibration or review."""
    p = np.asarray(probabilities)
    if (not np.issubdtype(p.dtype, np.number) or np.iscomplexobj(p)
            or p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError("Expected a nonempty matrix of finite probabilities between zero and one")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    return p


def validate_labels(y, probabilities):
    """Require one integer class ID per row; never truncate or broadcast labels."""
    y = np.asarray(y)
    if (y.ndim != 1 or len(y) != probabilities.shape[0]
            or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= probabilities.shape[1]).any()):
        raise ValueError("Labels must be a one-dimensional integer class ID per probability row")
    return y


def temperature_scale(probabilities, temperature):
    """Rescale class log-scores with one positive scalar; rankings are preserved."""
    p = validate_probabilities(np.asarray(probabilities, dtype=np.float64))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    logits = np.log(np.clip(p, 1e-15, 1.0)) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def multiclass_log_loss(y, probabilities):
    p = validate_probabilities(np.asarray(probabilities, dtype=np.float64))
    y = validate_labels(y, p)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean())


def fit_temperature(y, probabilities):
    """Fit only on out-of-fold development predictions, never the test set."""
    def objective(temperature):
        return multiclass_log_loss(y, temperature_scale(probabilities, temperature))

    result = minimize_scalar(objective, bounds=(0.25, 4.0), method="bounded", options={"xatol": 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization failed")
    return float(result.x)


def reliability(y, probabilities, bins=10):
    p = validate_probabilities(probabilities)
    y = validate_labels(y, p)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("The number of reliability bins must be a positive integer")
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    records, ece = [], 0.0
    for i in range(bins):
        mask = index == i
        count = int(mask.sum())
        avg_score = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(avg_score - accuracy)
        records.append({"lower": i / bins, "upper": (i + 1) / bins, "count": count,
                        "mean_score": avg_score, "accuracy": accuracy})
    return float(ece), records


def wilson_lower(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    rate = successes / total
    z2 = z * z
    return float((rate + z2 / (2 * total) - z * np.sqrt(rate * (1 - rate) / total + z2 / (4 * total**2)))
                 / (1 + z2 / total))


def choose_review_policy(y, probabilities, min_accepted=200, target_lower_bound=0.95):
    """Choose maximum accepted coverage on the predeclared grid, or review all."""
    probabilities = validate_probabilities(probabilities)
    y = validate_labels(y, probabilities)
    if (isinstance(min_accepted, bool) or not isinstance(min_accepted, (int, np.integer))
            or min_accepted < 1 or not 0 <= target_lower_bound <= 1):
        raise ValueError("Review selection requires a positive accepted count and a bound between zero and one")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    for step in range(37):
        threshold = round(0.05 + 0.025 * step, 3)
        accepted = confidence >= threshold
        count = int(accepted.sum())
        successes = int(correct[accepted].sum())
        lower = wilson_lower(successes, count)
        if count >= min_accepted and lower is not None and lower >= target_lower_bound:
            return {"threshold": threshold, "review_all": False, "development_accepted": count,
                    "development_accepted_correct": successes, "development_wilson_lower": lower}
    return {"threshold": None, "review_all": True, "development_accepted": 0,
            "development_accepted_correct": 0, "development_wilson_lower": None}


def review_summary(y, probabilities, policy):
    probabilities = validate_probabilities(probabilities)
    y = validate_labels(y, probabilities)
    if not policy["review_all"] and not 0 <= policy["threshold"] <= 1:
        raise ValueError("The review threshold must be between zero and one")
    pred = probabilities.argmax(axis=1)
    accepted = np.zeros(len(y), dtype=bool) if policy["review_all"] else probabilities.max(axis=1) >= policy["threshold"]
    correct = pred == y
    n_accepted = int(accepted.sum())
    errors = int((~correct).sum())
    return {"examples": len(y), "accepted": n_accepted, "reviewed": int((~accepted).sum()),
            "accepted_fraction": float(accepted.mean()) if len(y) else None,
            "accuracy_among_accepted": float(correct[accepted].mean()) if n_accepted else None,
            "accepted_errors": int((accepted & ~correct).sum()), "total_errors": errors,
            "errors_flagged": int((~accepted & ~correct).sum())}
