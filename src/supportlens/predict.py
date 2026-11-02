"""Load only the trusted, locally packaged artifacts; never accept visitor-supplied models."""
import json
from numbers import Real
from pathlib import Path
from pickle import UnpicklingError
import re
import joblib
import numpy as np
import sklearn
from .calibration import temperature_scale, validate_probabilities
from .config import ARTIFACT_DIR, MAX_INPUT_LENGTH
from .data import sha256_file
from .review_guard import review_masks


class ArtifactError(ValueError):
    """A broken model bundle, distinct from an invalid visitor question."""


ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError,
                   ImportError, EOFError, IndexError, UnpicklingError, RuntimeError)


def _valid_filename(name):
    return (isinstance(name, str) and name not in (".", "..")
            and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None)


def _finite_number(value, minimum, maximum=None):
    return (isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
            and value >= minimum and (maximum is None or value <= maximum))


def _validate_policy(meta):
    if not isinstance(meta.get("review_all", False), bool):
        raise ValueError("The review-all setting must be a boolean")
    if not meta.get("review_all", False) and not _finite_number(meta["review_threshold"], 0, 1):
        raise ValueError("The review threshold must be between zero and one")
    if not _finite_number(meta.get("temperature", 1.0), 0) or meta.get("temperature", 1.0) == 0:
        raise ValueError("Temperature must be positive and finite")
    guard = meta.get("review_guard")
    if guard is not None:
        if not isinstance(guard, dict) or guard.get("kind") != "word_support":
            raise ValueError("Unknown review guard type")
        if not _finite_number(guard["support_threshold"], 0, 1):
            raise ValueError("The support threshold must be between zero and one")
    elif meta.get("artifact_version") == 3:
        raise ValueError("The v3 release requires its review guard")


def load_active_bundle(directory=ARTIFACT_DIR):
    """Resolve the trusted project's release pointer, with v1 backward compatibility."""
    try:
        return _load_active_bundle(directory)
    except ArtifactError:
        raise
    except ARTIFACT_ERRORS as exc:
        raise ArtifactError(str(exc)) from exc


def _load_active_bundle(directory):
    directory = Path(directory)
    pointer_path = directory / "active.json"
    if not pointer_path.exists():
        return load_bundle(directory)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    release = pointer["release"]
    if not isinstance(release, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", release):
        raise ValueError("Invalid active release name")
    release_dir = directory / release
    if sha256_file(release_dir / "metadata.json") != pointer["metadata_sha256"]:
        raise ValueError("Active release metadata checksum mismatch")
    return load_bundle(release_dir)


def load_bundle(directory=None):
    if directory is None:
        return load_active_bundle(ARTIFACT_DIR)
    try:
        return _load_bundle(directory)
    except ArtifactError:
        raise
    except ARTIFACT_ERRORS as exc:
        raise ArtifactError(str(exc)) from exc


def _load_bundle(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    _validate_policy(meta)
    labels = meta["labels"]
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Artifact labels must be distinct nonempty strings")
    if any(type(meta[key]) is not int or meta[key] < 1 for key in ("input_dim", "output_dim")):
        raise ValueError("Artifact dimensions must be positive integers")
    if len(labels) != meta["output_dim"]:
        raise ValueError("Artifact labels do not match model metadata")
    if meta["model"] not in ("baseline", "torch"):
        raise ValueError("Unknown artifact model type")
    required = {"vectorizer.joblib", "model.joblib" if meta["model"] == "baseline" else "model.pt"}
    guard = meta.get("review_guard")
    if guard is not None:
        for key in ("vectorizer_file", "model_file"):
            if not _valid_filename(guard[key]):
                raise ValueError("Review guard file is not in the checked release")
            required.add(guard[key])
    checksums = meta["sha256"]
    if not isinstance(checksums, dict) or not required.issubset(checksums):
        raise ValueError("A required artifact checksum is missing")
    if sklearn.__version__ != meta["sklearn_version"]:
        raise ValueError(f"Expected scikit-learn {meta['sklearn_version']}; found {sklearn.__version__}")
    for name, expected in checksums.items():
        if not _valid_filename(name) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid artifact checksum entry")
    for name, expected in checksums.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    # FeatureUnion has multiple vocabularies; the transformed width works for both versions.
    if vectorizer.transform([""]).shape[1] != meta["input_dim"]:
        raise ValueError("Artifact vocabulary size does not match model metadata")
    if meta["model"] == "baseline":
        model = joblib.load(directory / "model.joblib")
        if model.n_features_in_ != meta["input_dim"] or list(model.classes_) != list(range(meta["output_dim"])):
            raise ValueError("Baseline model dimensions or label ordering do not match metadata")
    elif meta["model"] == "torch":
        import torch
        from .torch_model import IntentNet
        model = IntentNet(meta["input_dim"], meta["hidden_dim"], meta["output_dim"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
    if guard is not None:
        from .promote_review_guard import label_hash
        if guard["label_order_sha256"] != label_hash(meta["labels"]):
            raise ValueError("Review guard label ordering does not match")
        guard_vectorizer = joblib.load(directory / guard["vectorizer_file"])
        guard_model = joblib.load(directory / guard["model_file"])
        if (guard_vectorizer.transform([""]).shape[1] != guard["input_dim"]
                or guard_model.n_features_in_ != guard["input_dim"]
                or list(guard_model.classes_) != list(range(meta["output_dim"]))):
            raise ValueError("Review guard features or class ordering do not match")
        meta["_loaded_review_guard"] = (guard_vectorizer, guard_model)
    return vectorizer, model, meta


def predict(text, bundle):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a question.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Please keep the question to {MAX_INPUT_LENGTH} characters or fewer.")
    try:
        return _predict(text.strip(), bundle)
    except ARTIFACT_ERRORS as exc:
        raise ArtifactError(str(exc)) from exc


def _predict(text, bundle):
    vectorizer, model, meta = bundle
    _validate_policy(meta)
    x = vectorizer.transform([text])
    if meta["model"] == "baseline":
        probabilities = model.predict_proba(x)
    elif meta["model"] == "torch":
        from .torch_model import predict_probabilities
        probabilities = predict_probabilities(model, x)
    else:
        raise ValueError("Unknown artifact model type")
    probabilities = validate_probabilities(probabilities)
    if probabilities.shape != (1, len(meta["labels"])):
        raise ValueError("Model output does not match the saved labels")
    scores = probabilities[0]
    temperature = meta.get("temperature", 1.0)
    if temperature != 1.0:
        scores = temperature_scale(np.asarray(scores).reshape(1, -1), temperature)[0]
    top = np.argsort(-scores, kind="stable")[:min(3, len(scores))]
    reasons = []
    if meta.get("review_all", False):
        reasons.append("review_all")
    elif scores[top[0]] < meta["review_threshold"]:
        reasons.append("low_score")
    if meta.get("review_guard"):
        if "_loaded_review_guard" not in meta:
            raise ValueError("The review guard is missing from the loaded model bundle")
        guard_vectorizer, guard_model = meta["_loaded_review_guard"]
        word_scores = guard_model.predict_proba(guard_vectorizer.transform([text]))
        masks = review_masks(np.asarray(scores).reshape(1, -1), word_scores,
                             1.0 if meta.get("review_all", False) else meta["review_threshold"],
                             meta["review_guard"]["support_threshold"])
        if masks["weak_support"][0]:
            reasons.append("weak_word_support")
    return {"label": meta["labels"][int(top[0])], "score": float(scores[top[0]]),
            "review": bool(reasons), "review_reasons": reasons,
            "alternatives": [{"label": meta["labels"][int(i)], "score": float(scores[i])} for i in top[1:]]}
