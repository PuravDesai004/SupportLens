# Recorded run evidence

The source data were downloaded from the pinned official revision in `data_manifest.json` and verified against the SHA-256 values recorded there. Training and evaluation were run on Windows with Python 3.12.14, scikit-learn 1.6.1 and CPU PyTorch 2.6.0+cpu. Direct dependencies are pinned in the requirements files; exact floating-point reproduction can vary by platform.

The selection rule in `train.py` was fixed before official test evaluation: higher validation macro F1, with exact ties to the baseline. `selection.json` was written at 2026-11-02T17:46:14Z; `metrics.json` was written at 2026-11-02T17:46:36Z. Both models were trained on the official training data's derived training split. The baseline converged without a recorded warning. The test results have since been observed. The later addition of an optional output-directory setting did not alter training settings, fitted artifacts, or the recorded test results.

Commands from the project directory (PowerShell; environment setup is in the README):

```powershell
# Fixture checks before full training and test evaluation:
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -q --basetemp=.pytest_tmp
# 6 passed

$env:PYTHONPATH = 'src'
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -u -m supportlens.train
# 8,002 training rows, 2,001 validation rows, 6,000 features, 77 labels.
# Logistic regression selected on validation macro F1.
.\.venv\Scripts\python.exe -u -m supportlens.evaluate
# Both locked candidates evaluated once on the full official 3,080-row test split.

New-Item -ItemType Directory -Force -Path '.tmp' | Out-Null
$env:TEMP = (Resolve-Path '.tmp').Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
# 9 passed, including Streamlit AppTest with the real selected artifact and CSV integrity checks.
.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.
```

The initial data contract rejected two valid source label spellings and was corrected before training. Early fixture attempts also exposed a sandbox temporary-directory restriction; the project-local temporary directory resolved it. These failures did not involve test-set model scores.

A separate project-local environment installed only `requirements.txt`, had no PyTorch installed, passed `pip check`, loaded the selected artifact, and ran a Streamlit `AppTest`. For “Where is my new card?” the model predicted `card_arrival`, score 0.932915, with no review flag. This is one smoke-test example, not an independent accuracy estimate.

The app was also launched from that runtime-only environment on `127.0.0.1:8501` and exercised in a browser with the same question. The page displayed “Card arrival” and 93.3%. An observed dark-theme contrast problem was corrected by adding the project's explicit light theme and the result was checked again in the browser.

DuckDB reconciled the selected-model CSV to 3,080 rows, 382 errors and 624 review flags. The public test split has seven normalized overlaps with the official training split (six derived train, one validation), all with matching labels; the primary metrics retain all test rows. `metrics.json` records confusion pairs and whether test results had previously been viewed.

After the first push, a Git line-ending conversion was found to have changed the committed CSV bytes relative to the recorded hash. The original Windows CSV hash was `b1a627a03ff2c805d7e29a6926f122d93a678f6615a03889b4ba51c747c41814`. The same rows were converted to LF line endings, the recorded hash was updated, and both the exporter and `.gitattributes` now use LF; an integration test checks the hash and DuckDB totals. No model was retrained and the test set was not scored again for this correction.

The GitHub Actions workflow is configured in `.github/workflows/tests.yml`. [Hosted run 41495687880](https://github.com/PuravDesai004/SupportLens/actions/runs/41495687880) passed for commit `546328fce1650cf0f2c21f6544acccae60aeb76d`; check Actions for later commits. The owner requested a local-only demo, so no hosting account or public demo was created.
