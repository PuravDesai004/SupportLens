# Architecture

## Training and evaluation

```text
data/train.csv (10,003 rows; pinned SHA-256)
  -> data.validate_frame + normalized-text group split, seed 42
  -> 8,002 derived train + 2,001 validation, each with 77 labels
  -> fit 6,000-column TF-IDF vocabulary on derived train only
  -> baseline logistic regression and CPU IntentNet (6000 -> 64 -> 77)
  -> compare validation macro F1; lock baseline and fixed 0.35 threshold
data/test.csv (3,080 rows) -> same fitted vectorizer -> both locked models
  -> reports/metrics.json, per-class CSVs, selected-model evaluation.csv
  -> sql/intent_summary.sql via DuckDB -> reconciled intent_summary.csv
```

`src/supportlens/data.py` checks the data contract, derives normalized groups, audits duplicates, and verifies downloaded files. `src/supportlens/train.py` fits the vectorizer, sorted `LabelEncoder` mapping, both models and validation metrics; it saves the choice before `src/supportlens/evaluate.py` reads test data for scoring. `src/supportlens/torch_model.py` converts only sparse batches to dense float32 arrays. The label mapping is fit on the official training labels and stored in artifact metadata; the held-out test never affects it. `reports/data_manifest.json` records source revision, checksums, actual distributions, overlap and versions.

The source has seven case-folded, whitespace-normalized train/test overlaps, including six derived training rows, with matching labels. The full official test remains the primary benchmark. The test was evaluated after the choice was locked; its results are now public and are not an untouched holdout for future iterations.

## Inference

```text
app.py: question -> supportlens.predict.predict
  -> artifacts/vectorizer.joblib -> 1 x 6000 sparse vector
  -> artifacts/model.joblib -> 1 x 77 scores
  -> artifacts/metadata.json maps sorted class IDs to labels
  -> top label, alternatives, and score < 0.35 review flag
```

The shipped selected model is scikit-learn logistic regression. `metadata.json` records model type, feature count, label order, scikit-learn version, threshold and SHA-256 checksums for the two model files. `load_bundle` validates the checksums, dimensions and class order before inference, and Streamlit caches the resulting bundle. The code can also load an `IntentNet` state dictionary when a future validation run selects it, provided the runtime includes PyTorch. The shipped app uses no visitor-provided serialized model.

If an artifact is missing, corrupt or incompatible, the app displays a setup error instead of inventing a prediction. joblib loading is safe only for artifacts from a trusted source; checksums detect accidental mismatch, not malicious replacement of both files and metadata. User-entered text is not logged or saved by application code, but provider logging is outside this project's control.

## Files and boundaries

- `app.py`: Streamlit interface, samples, prediction, measured evaluation and limitations.
- `src/supportlens/config.py`: fixed split/model settings and reproducible source revision.
- `src/supportlens/data.py`, `train.py`, `torch_model.py`, `evaluate.py`, `predict.py`: data, learning, evaluation and serving.
- `artifacts/`: only the inference files needed by the selected app model.
- `reports/`: recorded results, audit and selected-model CSV; `sql/`: the query that checks it.
- `tests/`: offline fixtures plus a Streamlit `AppTest` with the packaged artifact; `.github/workflows/tests.yml`: hosted offline test configuration.

The app is a single local Streamlit process, not a bank integration, API service, RAG system or production routing system. The owner requested a local-only demo.
