"""Validate public data, keep near-identical text groups together, and audit overlap."""
import hashlib
import re
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_DIR, DATA_SHA256, SEED, SOURCE_BASE


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv"):
        path = DATA_DIR / name
        if not path.exists():
            urlretrieve(f"{SOURCE_BASE}/{name}", path)
        digest = sha256_file(path)
        if digest != DATA_SHA256[name]:
            raise ValueError(f"Checksum mismatch for {path}: {digest}")
        print(f"{path}: sha256={digest}")


def normalized(text):
    return " ".join(text.casefold().split())


def validate_frame(frame, allowed_labels=None):
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError("Data must contain text and category columns")
    if frame.empty:
        raise ValueError("Data must contain at least one example")
    for col in ("text", "category"):
        if frame[col].isna().any() or not frame[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{col} contains missing or blank values")
    # The original data includes "Refund_not_showing_up" and "reverted_card_payment?".
    if not frame["category"].map(lambda x: bool(re.fullmatch(r"[A-Za-z0-9_]+\??", x))).all():
        raise ValueError("Labels must be BANKING77-style identifiers")
    if allowed_labels is not None and not set(frame.category).issubset(set(allowed_labels)):
        raise ValueError("Data contains an unknown label")
    return frame.reset_index(drop=True)


def load_data(path, allowed_labels=None):
    return validate_frame(pd.read_csv(path, keep_default_na=False), allowed_labels)


def split_train_validation(frame, seed=SEED):
    groups = frame.text.map(normalized)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame.text, frame.category, groups))
    train, val = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)
    if set(train.category) != set(frame.category) or set(val.category) != set(frame.category):
        raise ValueError("Split did not retain every label in both subsets")
    if set(train.text.map(normalized)) & set(val.text.map(normalized)):
        raise AssertionError("A normalized text group crossed the train/validation split")
    return train, val


def frame_audit(frame):
    norm = frame.text.map(normalized)
    grouped = pd.DataFrame({"normalized": norm, "category": frame.category}).groupby("normalized").category.nunique()
    counts = frame.category.value_counts()
    return {"rows": len(frame), "columns": list(frame.columns), "labels": int(frame.category.nunique()),
            "per_label_min": int(counts.min()), "per_label_max": int(counts.max()),
            "label_counts": dict(sorted((str(k), int(v)) for k, v in counts.items())),
            "exact_duplicate_rows_after_first": int(frame.duplicated(["text"]).sum()),
            "normalized_duplicate_rows_after_first": int(norm.duplicated().sum()),
            "normalized_groups_with_conflicting_labels": int((grouped > 1).sum())}


def overlap_audit(left, right):
    a = pd.DataFrame({"key": left.text.map(normalized), "label": left.category})
    b = pd.DataFrame({"key": right.text.map(normalized), "label": right.category})
    shared = set(a.key) & set(b.key)
    a_labels = a.groupby("key").label.agg(set)
    b_labels = b.groupby("key").label.agg(set)
    return {"shared_normalized_texts": len(shared), "left_rows_with_shared_text": int(a.key.isin(shared).sum()),
            "right_rows_with_shared_text": int(b.key.isin(shared).sum()),
            "shared_texts_with_conflicting_labels": sum(a_labels[k] != b_labels[k] for k in shared),
            "shared_exact_texts": len(set(left.text) & set(right.text))}


if __name__ == "__main__":
    download_data()
