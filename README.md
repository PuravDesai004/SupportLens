# SupportLens

SupportLens is an educational English-language banking-intent classifier. It suggests one of 77 categories for a support question, shows alternative categories, and flags low-scoring predictions for human review. It uses the public BANKING77 research dataset; it is not a bank service, does not access accounts or route tickets, and does not answer financial questions. Please do not enter account numbers, passwords, or other sensitive personal information.

Repository: [PuravDesai004/SupportLens](https://github.com/PuravDesai004/SupportLens). The app is **local only** at the owner's request; no hosted demo or hosting account is required. [GitHub Actions](https://github.com/PuravDesai004/SupportLens/actions) runs the offline tests. [Hosted run 41496473950](https://github.com/PuravDesai004/SupportLens/actions/runs/41496473950) passed on commit `20be076b02cea75de825bcd3480682924fee7fe3`; check Actions for the latest commit's result.

## Run the app on Windows

Python 3.12 was used for the recorded run. The available Python 3.14 installation was not used for this project. In PowerShell, from the project directory, use a Python 3.12 executable to create a project-local environment. On this machine the following path was used and verified:

```powershell
$Python312 = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $Python312 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On another machine, substitute the path to an installed Python 3.12 executable. If `py -3.12` is registered, `py -3.12 -m venv .venv` is an alternative. The app uses the included selected artifacts and does not download data, retrain, or need API keys at startup. Use only artifacts from a trusted source: joblib files can execute code when loaded.

A real local browser check: entering **“Where is my new card?”** returned **Card arrival**, a **93.3%** model score, and an above-threshold indication. This is one example, not an accuracy estimate.

To install training and test dependencies as well:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The sandbox used for the recorded local run required a project-local temporary folder. The exact successful command is documented in [the study guide](docs/PROJECT_STUDY_GUIDE.md#testing-and-ci).

## How it works

```text
Official train CSV -> grouped train / validation split
  -> fit TF-IDF on train only -> logistic regression + PyTorch candidates
  -> choose by validation macro F1 -> lock decision -> official test evaluation

Visitor question -> saved TF-IDF -> saved selected classifier -> top intents + review flag
```

The 6,000-feature TF-IDF vectorizer uses word unigrams and bigrams, `min_df=2`, and sublinear term frequency. Logistic regression is the selected model. A small CPU PyTorch network uses the same features, a 64-unit ReLU hidden layer and 77 output logits. Configuration and measured validation scores are in [reports/selection.json](reports/selection.json); details are in [architecture](docs/ARCHITECTURE.md).

## Data and attribution

The official source CSVs at revision [`9d081458ff52e53cf7e848f414e6e9344e4e6696`](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data) contain `text` and `category`: 10,003 training rows and 3,080 test rows across 77 labels. The train labels have 35–187 rows each; the official test has 40 per label. The derived split is 8,002 train and 2,001 validation rows, using the first shuffled five-fold `StratifiedGroupKFold` split with seed 42 and case-folded, whitespace-normalized text as groups. All 77 labels occur in both subsets.

No normalized text group crosses derived train and validation. Seven normalized texts occur in both official train and test (six in derived train and one in validation); their labels agree. There are no exact-string overlaps between official train and test. This overlap can make the full official test score slightly optimistic. The official test has not been silently filtered. Counts, distribution, duplicate checks, source URLs, checksums, and versions are recorded in [reports/data_manifest.json](reports/data_manifest.json).

Dataset: [BANKING77, curated by PolyAI](https://huggingface.co/datasets/PolyAI/banking77), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: Casanueva, Temčinas, Gerz, Henderson, and Vulić, [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807), 2020. SupportLens derives a validation split and numeric features from the published data; the included evaluation CSV contains labels and scores, not source text.

## Measured comparison

The rule was fixed before viewing test metrics: select the candidate with the higher validation macro F1; exact ties go to the simpler baseline. Both candidates used the same derived training and validation rows, and the same full official test rows after selection.

| Model | Validation accuracy / macro F1 (n=2,001) | Official test accuracy / macro F1 (n=3,080) |
| --- | --- | --- |
| TF-IDF + logistic regression **selected** | 87.96% / 0.8767 | 87.60% / 0.8761 |
| TF-IDF + PyTorch network | 87.86% / 0.8733 | 88.73% / 0.8878 |

The test ranking differs; the validation-based choice remains unchanged. Test metrics cover 77 supported labels. The baseline's fixed 0.35 review threshold flagged 624 of 3,080 test examples (20.26%); among the remaining 2,456 examples, accuracy was 94.95%. These are in-dataset results, not estimates of real bank performance. Scores are not necessarily calibrated, and a high score does not reliably detect an unfamiliar question. The threshold was fixed before test evaluation, not tuned on test data.

Open [reports/evaluation.csv](reports/evaluation.csv) in Excel: each row has a test row ID, true and predicted labels, score, correctness, and review flag. [reports/intent_summary.csv](reports/intent_summary.csv) was calculated by [sql/intent_summary.sql](sql/intent_summary.sql) using DuckDB; its totals reconcile to 3,080 rows, 382 errors, and 624 flags. Per-class precision, recall, F1 and support for both models, plus common confusion pairs, are in `reports/`.

## Reproduce the training run

The test set has already been viewed in this published project. Re-running it is a reproduction, not a new untouched holdout. The following commands download the pinned official CSVs, verify their SHA-256 hashes, and write new results to ignored `reproduction/`, preserving the shipped evidence and artifacts:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m supportlens.data
$env:SUPPORTLENS_OUTPUT_ROOT = "$PWD\reproduction"
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m supportlens.train
.\.venv\Scripts\python.exe -m supportlens.evaluate
Remove-Item Env:SUPPORTLENS_OUTPUT_ROOT
```

Install `requirements-dev.txt` first. Training is CPU-only. Fixed seeds and pinned direct dependencies help, but exact floating-point results may differ across operating systems or hardware. `supportlens.train` refuses to overwrite a directory containing test results. A deliberate same-directory correctness rerun of evaluation requires `--allow-rerun` and records prior metrics.

## Limits and next steps

The data has 77 in-domain labels and does not establish real ticket volumes, handling time, service levels, customer outcomes, or value. The classifier must choose a known label even for out-of-scope text. Human review, privacy controls, representative real-world evaluation and appropriate calibration would require separate design before any actual support use. The app does not print or persist visitor text in its own code; the hosting provider may retain logs. This project was built with coding-agent assistance and is intended to be understood and explained by its owner.
