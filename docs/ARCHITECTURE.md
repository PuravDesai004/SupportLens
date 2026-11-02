# Architecture

## V2 experiment and release

```text
Official training CSV (10,003 rows, hash checked)
  -> normalized-text groups -> 8,001 development / 2,002 assessment (seed 73)
  -> 3 group-aware development folds (seed 137)
     -> fold-only TF-IDF -> word logistic / word+character logistic / PyTorch
     -> out-of-fold metrics, scalar temperature and review thresholds
  -> lock candidate and thresholds -> fit all candidates on development only
  -> assessment release gate -> freeze candidate, score method and threshold
  -> one exploratory comparison on the previously viewed official test
```

`src/supportlens/experiment_v2.py` implements the staged comparison. It saves `development_lock.json` before assessment and `final_lock.json` before the previously viewed test. Each stage refuses to overwrite its lock. The assessment is part of development because the fixed release gate uses it; no stage can turn prior test exposure into a new untouched holdout. The known seven normalized official train/test overlaps remain disclosed.

The chosen v2 inference model is logistic regression. Its `FeatureUnion` concatenates 6,000 word unigram/bigram features and 12,738 learned character 3–5-gram features, giving **18,738** sparse columns. The character feature block has weight 0.7. The model has 77 class outputs. All TF-IDF vocabularies and IDF weights are learned from the relevant training subset only. The class-name inventory is sorted and saved with each model.

`calibration.py` fits a single positive temperature on out-of-fold development probabilities. The selected temperature is approximately **0.51738**: applying softmax to log probabilities divided by this value sharpens scores without changing their rank. A 0.75 review threshold was chosen using the predeclared development rule. Assessment checks both ranking and the raw-versus-scaled log loss and ECE; it does not search for another threshold. Scores are still imperfect, especially for ambiguous or out-of-scope questions.

## Local inference

```text
app.py -> predict.load_active_bundle(artifacts/)
  -> active.json -> v2/metadata.json -> matching vectorizer.joblib + model.joblib
  -> user question -> 1 x 18,738 sparse TF-IDF row -> 1 x 77 raw probabilities
  -> fixed temperature -> top category + alternatives + 0.75 review flag
```

`active.json` records the release directory and metadata SHA-256. V2 metadata records the candidate ID, label order, feature count, exact scikit-learn version, temperature, review policy and model/vectorizer checksums. `predict.py` checks these files and dimensions before inference. Streamlit caches the bundle. Missing or mismatched files produce a setup error instead of a fabricated prediction. Hashes guard against accidental mismatch, not an attacker replacing both files and metadata; joblib artifacts must come from a trusted source.

The original root v1 inference files, `artifacts/previous_v1/`, `src/supportlens/train.py`, and old reports remain available for comparison. Calling `load_bundle()` resolves the active release; supplying `artifacts/previous_v1` explicitly loads the historical one. The prior v1 threshold and scores are not interchangeable with temperature-scaled v2 scores.

## Evidence and outputs

- `docs/EXPERIMENT_PLAN.md`: fixed comparison, selection, calibration and review rules.
- `reports/experiment_v2/development_lock.json`: pooled out-of-fold metrics, calibration bins, policies, timing and pre-assessment selection.
- `reports/experiment_v2/final_lock.json`: assessment results, resource measurements and pre-test release decision.
- `reports/experiment_v2/previously_viewed_test.json`: exploratory test comparison, explicitly labeled previously viewed.
- Per-class CSVs and common confusion pairs are included for each serious candidate. `selected_assessment.csv` contains no question text; `assessment_export.json` and the DuckDB summary reconcile its totals.
- `tests/`: fixture coverage for split grouping, fold-only features, calibration, threshold behavior, release metadata and app inference. `.github/workflows/tests.yml` runs offline tests.

The app does not log or save visitor-entered text in application code. It remains local and has no bank integration, external model service, RAG system, or production routing logic.
