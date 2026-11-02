"""Compare warm single-question inference times without saving question text."""
import json
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from supportlens.config import ROOT
from supportlens.predict import load_active_bundle, load_bundle, predict


def measure(bundle, text, repeats=100):
    for _ in range(5):
        predict(text, bundle)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(text, bundle)
        times.append((time.perf_counter() - start) * 1000)
    return {"repetitions": repeats, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main():
    previous = load_bundle(ROOT / "artifacts" / "v2")
    current = load_active_bundle(ROOT / "artifacts")
    with threadpool_limits(limits=4):
        sample = "The replacement card has not arrived yet."
        results = {"previous_v2": measure(previous, sample), "current_v3": measure(current, sample)}
    report = {"scope": "Warm local single-question inference on one manually authored sample; not a service guarantee.",
              "platform": platform.platform(), "processor": platform.processor(), "thread_limit": 4,
              "results": results, "question_text_saved": False,
              "guard_model_bytes": (ROOT / "artifacts" / "v3" / "review_model.joblib").stat().st_size,
              "guard_vectorizer_bytes": (ROOT / "artifacts" / "v3" / "review_vectorizer.joblib").stat().st_size}
    path = ROOT / "reports" / "review_guard" / "runtime.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite runtime evidence")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
