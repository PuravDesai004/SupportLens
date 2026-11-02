"""Load only the trusted, locally packaged artifacts; never accept visitor-supplied models."""
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from .config import ARTIFACT_DIR, MAX_INPUT_LENGTH
from .data import sha256_file


def load_bundle(directory=ARTIFACT_DIR):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != meta["sklearn_version"]:
        raise ValueError(f"Expected scikit-learn {meta['sklearn_version']}; found {sklearn.__version__}")
    for name, expected in meta["sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    if len(vectorizer.vocabulary_) != meta["input_dim"]:
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
    top = np.argsort(-scores, kind="stable")[:min(3, len(scores))]
    return {"label": meta["labels"][int(top[0])], "score": float(scores[top[0]]),
            "review": bool(scores[top[0]] < meta["review_threshold"]),
            "alternatives": [{"label": meta["labels"][int(i)], "score": float(scores[i])} for i in top[1:]]}
