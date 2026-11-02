import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_ROOT = Path(os.environ.get("SUPPORTLENS_OUTPUT_ROOT", ROOT)).resolve()
RUN_DIR = OUTPUT_ROOT / "training_runs"
ARTIFACT_DIR = OUTPUT_ROOT / "artifacts"
REPORT_DIR = OUTPUT_ROOT / "reports"
SEED = 42
SOURCE_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
SOURCE_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_COMMIT}/banking_data"
DATA_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
REVIEW_THRESHOLD = 0.35  # Fixed before seeing any test results; not a calibration guarantee.
MAX_INPUT_LENGTH = 2000
VECTORIZER_CONFIG = {"ngram_range": (1, 2), "max_features": 6000, "min_df": 2, "sublinear_tf": True}
TORCH_CONFIG = {"hidden_dim": 64, "epochs": 10, "batch_size": 128, "learning_rate": 0.003}
