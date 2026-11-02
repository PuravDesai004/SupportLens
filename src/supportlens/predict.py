"""Load only the trusted, locally packaged artifacts; never accept visitor-supplied models."""
import json
from pathlib import Path
import re
import joblib
import numpy as np
import sklearn
from .calibration import temperature_scale
from .config import ARTIFACT_DIR, MAX_INPUT_LENGTH
from .data import sha256_file
from .review_guard import review_masks


def load_active_bundle(directory=ARTIFACT_DIR):
    """Resolve the trusted project's release pointer, with v1 backward compatibility."""
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
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != meta["sklearn_version"]:
        raise ValueError(f"Expected scikit-learn {meta['sklearn_version']}; found {sklearn.__version__}")
    for name, expected in meta["sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    # FeatureUnion has multiple vocabularies; the transformed width works for both versions.
    if vectorizer.transform([""]).shape[1] != meta["input_dim"]:
        raise ValueError("Artifact vocabulary size does not match model metadata")
    if len(meta["labels"]) != meta["output_dim"]:
        raise ValueError("Artifact labels do not match model metadata")
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
    else:
        raise ValueError("Unknown artifact model type")
    guard = meta.get("review_guard")
    if guard:
        if guard.get("kind") != "word_support":
            raise ValueError("Unknown review guard type")
        from .promote_review_guard import label_hash
        if guard["label_order_sha256"] != label_hash(meta["labels"]):
            raise ValueError("Review guard label ordering does not match")
        for key in ("vectorizer_file", "model_file"):
            name = guard[key]
            if Path(name).name != name or name not in meta["sha256"]:
                raise ValueError("Review guard file is not in the checked release")
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
    vectorizer, model, meta = bundle
    x = vectorizer.transform([text.strip()])
    if meta["model"] == "baseline":
        scores = model.predict_proba(x)[0]
    else:
        from .torch_model import predict_probabilities
        scores = predict_probabilities(model, x)[0]
    if len(scores) != len(meta["labels"]) or not np.isfinite(scores).all():
        raise ValueError("Model output does not match the saved labels")
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
        word_scores = guard_model.predict_proba(guard_vectorizer.transform([text.strip()]))
        masks = review_masks(np.asarray(scores).reshape(1, -1), word_scores,
                             1.0 if meta.get("review_all", False) else meta["review_threshold"],
                             meta["review_guard"]["support_threshold"])
        if masks["weak_support"][0]:
            reasons.append("weak_word_support")
    return {"label": meta["labels"][int(top[0])], "score": float(scores[top[0]]),
            "review": bool(reasons), "review_reasons": reasons,
            "alternatives": [{"label": meta["labels"][int(i)], "score": float(scores[i])} for i in top[1:]]}
