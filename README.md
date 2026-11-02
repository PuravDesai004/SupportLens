# SupportLens

SupportLens is a **local educational demo** that suggests one of 77 BANKING77 intents for an English banking support question. It shows alternative categories and flags low-scoring predictions for review. It does not connect to a bank, answer banking questions, access accounts, or route tickets. Please do not enter account numbers, passwords, or other sensitive personal information.

The app remains local at the owner's request. The source is in [PuravDesai004/SupportLens](https://github.com/PuravDesai004/SupportLens); [GitHub Actions](https://github.com/PuravDesai004/SupportLens/actions) runs offline checks. The project was built and improved with coding-agent assistance.

## Quickstart on Windows

Python 3.12 was used for the measured runs. From this project folder, create an isolated environment with a Python 3.12 executable. The following executable path was verified on the owner's machine:

```powershell
$Python312 = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $Python312 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On another machine, substitute its Python 3.12 path; if registered, `py -3.12 -m venv .venv` is an alternative. The included trusted artifacts load without downloading data, retraining, PyTorch, API keys or a hosting account. Only load joblib artifacts from a trusted source; serialized files can execute code. To install training and test dependencies, use `python -m pip install -r requirements-dev.txt` in the environment.

## Current model and evidence

The active v3 release keeps the v2 word-and-character TF-IDF logistic classifier and its temperature-scaled scores. It flags a prediction if its main score is below **0.75**, or if a complementary word-only model assigns less than **0.475** raw support to the suggested label. The second score is an uncalibrated review signal; both models share training data and can be wrong together. The review check does not change the category, ranking or displayed score, and it cannot reliably detect every ambiguity or unfamiliar question.

The candidates were retrained on the same data: the original word-only logistic configuration, word-plus-character logistic regression, and the existing small PyTorch network. All learned text features were fit inside each training fold. On three pooled group-aware development folds (8,001 examples), word-plus-character logistic led the matched baseline on macro F1, 0.8734 versus 0.8472. Its paired, class-stratified bootstrap interval for the gain was approximately [0.0216, 0.0312]; this descriptive interval does not account for all dependence from overlapping fold training.

On the separate **development assessment** (2,002 examples, 77 labels), all three models were retrained without those assessment rows:

| Candidate | Accuracy | Macro F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: |
| Word-only logistic regression | 87.51% | 0.8736 | 96.60% |
| **Word + character logistic — selected** | **90.11%** | **0.9007** | **97.25%** |
| Word TF-IDF + PyTorch | 88.01% | 0.8776 | 96.75% |

The assessment was a predeclared release-confirmation gate and therefore also contributed to the release decision; it is not an untouched final test. The selected model's assessment log loss fell from 0.5800 to 0.3749 and 10-bin expected calibration error from 0.1967 to 0.0207 after temperature scaling. ECE depends on the chosen bins, and these results do not prove calibration on real-world questions.

At the previous 0.75-only threshold, **1,728 of 2,002** assessment predictions were accepted, with **95.25% accuracy among accepted**; 82 were errors. The new second-view policy accepted **1,343** with **96.87% accuracy among accepted**; 42 were errors. It reviewed **659** rather than 274 examples. Of the 385 newly reviewed examples, **40 were errors and 345 were correct**. This is a substantial workload tradeoff, and the policy still misses errors.

The word-support cutoff was selected using the 8,001 out-of-fold development predictions and a predeclared conservative accuracy target. The 2,002-row assessment had already been viewed and was used as a release gate, so the comparison is exploratory, not an independent test or a guarantee. The official test was not rescored for this review change. See [review guard evidence](reports/review_guard/RESULTS.md) and the [assessment comparison CSV](reports/review_guard/assessment_comparison.csv).

For context, the retrained word baseline with its original raw-score 0.35 policy accepted 1,592 of those assessment rows, of which 87 were errors. These policies use different score scales; none reliably catches every mistake.

After the model, temperature and threshold were frozen, all fixed approaches were scored once on the **previously viewed** official test. The original v1 artifact scored 87.60% accuracy / 0.8761 macro F1; v2 scored 89.84% / 0.8987 on the same 3,080 examples. This is an exploratory comparison on a test set already seen in the project, not a new independent holdout. See [the experiment results](reports/experiment_v2/RESULTS.md) for all candidates, calibration, per-class files, review counts, timing and limitations.

The owner's illustrative question still predicts `card_not_working` at 96.4%, but it is now **flagged for review** because the word-only model gives the category limited support. There is no example-specific text rule. The expected label remains unknown; a higher score does **not** prove this individual prediction is correct. Some clear questions are also flagged, and some ambiguous questions remain unflagged.

## Data and reproducibility

The public [BANKING77 dataset](https://huggingface.co/datasets/PolyAI/banking77), curated by PolyAI and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), is attributed to Casanueva, Temčinas, Gerz, Henderson and Vulić, [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807), 2020. The pinned source revision is [`9d081458ff52e53cf7e848f414e6e9344e4e6696`](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data). The files contain 10,003 official training and 3,080 official test rows with 77 labels. [data_manifest.json](reports/data_manifest.json) records source hashes, columns, label counts and duplicate checks.

V2 uses a first five-fold, normalized-text-group split (seed 73): 8,001 development and 2,002 assessment rows. The development pool uses three group-aware folds (seed 137). No normalized group crosses either split. Six normalized texts in the v2 development pool and one in its assessment also occur in the official test, all with matching labels. Those seven overlaps and prior test exposure limit the evidence. A new split does not erase prior exposure to the data.

[selected_assessment.csv](reports/experiment_v2/selected_assessment.csv) is the historical v2 score-only assessment; its [DuckDB summary](reports/experiment_v2/selected_assessment_intent_summary.csv) reconciles to 2,002 rows, 198 errors and 274 flags. The current [review comparison CSV](reports/review_guard/assessment_comparison.csv) records both review policies and contains row IDs and model results, **not question text**. The old v1 [evaluation CSV](reports/evaluation.csv), model reports, original code and artifacts under `artifacts/previous_v1/` are preserved.

To reproduce the v2 experiment in a new ignored output folder without overwriting the shipped evidence:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m supportlens.data
$env:SUPPORTLENS_OUTPUT_ROOT = "$PWD\reproduction\v2"
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m supportlens.experiment_v2 development
.\.venv\Scripts\python.exe scripts/summarize_development_folds.py
.\.venv\Scripts\python.exe -m supportlens.experiment_v2 assessment
.\.venv\Scripts\python.exe scripts/summarize_assessment_intervals.py
.\.venv\Scripts\python.exe -m supportlens.promote_v2
.\.venv\Scripts\python.exe scripts/export_v2_assessment.py
.\.venv\Scripts\python.exe -m supportlens.review_guard_experiment development
.\.venv\Scripts\python.exe -m supportlens.review_guard_experiment assessment
.\.venv\Scripts\python.exe -m supportlens.review_conservative_experiment development
.\.venv\Scripts\python.exe -m supportlens.review_conservative_experiment assessment
.\.venv\Scripts\python.exe -m supportlens.promote_review_guard
.\.venv\Scripts\python.exe scripts/export_review_assessment.py
Remove-Item Env:SUPPORTLENS_OUTPUT_ROOT
```

Install `requirements-dev.txt` first and use a fresh output folder. The commands do not rescore the already-viewed official test; doing so would not create a new independent test. Stages refuse to overwrite locked results. Runs are seeded and CPU-only, but floating-point results and speed can vary across hardware and operating systems. [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) records model-selection rules; [REVIEW_CONSERVATIVE_PLAN.md](docs/REVIEW_CONSERVATIVE_PLAN.md) records the review-policy follow-up.

## Tests and limitations

Run `python -m pytest -q` in the development environment. Tests cover grouped splits, fold-only features, calibration, thresholds, artifact checksums and label order, real model output, evaluation exports, and the Streamlit app. The [study guide](docs/PROJECT_STUDY_GUIDE.md) gives the exact sandbox command and an interview walkthrough. Check the [Actions page](https://github.com/PuravDesai004/SupportLens/actions) for the latest hosted test status.

This is a 77-class in-domain research-data demo. It has no reliable out-of-scope detector, and it does not establish ticket volumes, handling time, customer outcomes, or real-bank performance. Manually authored examples are illustrative only. The app code does not print or persist visitor-entered questions; if someone separately hosts it, their provider may retain logs. A real bank would require representative data, privacy controls, monitoring, calibrated uncertainty assessment and an appropriate human workflow.
