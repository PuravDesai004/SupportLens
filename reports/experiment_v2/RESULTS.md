# SupportLens v2: experiment and release evidence

## Scope and preserved baseline

The original v1 baseline, vectorizer, labels and 0.35 policy are preserved in `artifacts/previous_v1/`; the root v1 files and prior `reports/` remain intact. Their hashes, dimensions and class order were checked, and nine pre-change tests passed. No preprocessing, model-file or label-order mismatch was found. The original audit and representative **public dataset** validation errors are in `BASELINE_AUDIT.md` and `baseline_error_audit.json`.

The protocol in `docs/EXPERIMENT_PLAN.md` was committed as `8c50dcb` before new comparison training. The development selection and policies were locked in `457ef5c`; the assessment and release decision were locked in `5fd852d` before the exploratory test. The original official test had already been viewed in v1. The v2 test was scored once after that lock; no model, temperature, or threshold was changed in response to its scores.

## Fair design

From the official 10,003 training rows, a normalized-text-group, stratified split with seed 73 produced 8,001 development and 2,002 assessment rows. Three group-aware development folds use seed 137. Every development row receives an out-of-fold prediction, all labels occur in each fold, no normalized group crosses a split, and each vectorizer is fitted only on that fold's training text. `assignments.csv` records the row IDs and fold allocation; no question text is in that file.

The baseline and both candidates were refit on the same development pool without assessment rows before assessment. The old v1 artifact was **not** used as an independent comparator on that new split. The assessment is part of development because the predeclared gate uses it to decide the release and raw-versus-scaled policy. This split does not erase previous project exposure to the dataset. The official train and test share seven normalized texts with matching labels: six in the v2 development pool and one in the assessment.

Candidates were the v1 word TF-IDF/logistic configuration, the same word features plus character-within-word 3–5-grams and logistic regression, and the original word TF-IDF/64-unit PyTorch network. Character fragments were proposed to help with wording variation, not to target the owner's question. No external embedding, paid service, GPU or new dependency was used.

## Classification results

All values below are measured on the same rows within each comparison. Per-fold results, per-class precision/recall/F1/support, and frequent directional confusions are in the accompanying JSON and CSV files.

| Candidate | Development OOF n=8,001: accuracy / macro F1 / top-3 | Development assessment n=2,002: accuracy / macro F1 / top-3 |
| --- | --- | --- |
| Word logistic | 85.04% / 0.8472 / 94.91% | 87.51% / 0.8736 / 96.60% |
| **Word + character logistic** | **87.54% / 0.8734 / 96.24%** | **90.11% / 0.9007 / 97.25%** |
| Word + PyTorch | 85.86% / 0.8571 / 95.69% | 88.01% / 0.8776 / 96.75% |

Word+character logistic led the baseline in all three development folds. Its pooled macro-F1 gain was 0.0262, with an approximate paired, class-stratified row-bootstrap 95% interval [0.0216, 0.0312]. Assessment gain was 0.0271 [0.0172, 0.0379]. These are descriptive intervals: rows are resampled within labels, folds share training data, and earlier data exposure is not accounted for. The candidate met the predeclared eligibility and assessment gates, so it was promoted.

After the release was fixed, the full **previously viewed** official test gave:

| Fixed model | Accuracy | Macro F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: |
| Original v1 artifact | 87.60% | 0.8761 | 96.27% |
| Matched word logistic | 87.53% | 0.8756 | 96.53% |
| Selected word+character logistic | 89.84% | 0.8987 | 97.21% |
| Word + PyTorch | 88.54% | 0.8858 | 97.14% |

These test figures are exploratory on previously viewed data, not a new untouched benchmark. A representative assessment error remains `card_arrival` -> `card_delivery_estimate` (4 examples), with other confusion between reverted and declined payments and related exchange-rate labels. A model can struggle where labels overlap or the question omits context.

## Calibration and review behavior

A single temperature was fitted for each candidate on its out-of-fold development predictions. For the selected model, `T=0.5173768764`; this sharpens the model distribution without changing class ranking. Calibration fitted and measured on the same development predictions can look optimistic. On the separate assessment:

| Candidate | Raw log loss -> scaled | Raw 10-bin ECE -> scaled | Scaled-policy accepted / reviewed | Accuracy among accepted | Accepted errors |
| --- | --- | --- | --- | --- | --- |
| Word logistic | 0.7631 -> 0.4482 | 0.2598 -> 0.0193 | 1,554 / 448 | 96.20% | 59 |
| Word + character logistic | 0.5800 -> 0.3749 | 0.1967 -> 0.0207 | 1,728 / 274 | 95.25% | 82 |
| Word + PyTorch | 0.4315 -> 0.4250 | 0.0359 -> 0.0233 | 1,652 / 350 | 95.04% | 82 |

Each accepted/reviewed denominator is 2,002. ECE uses 10 equal-width confidence bins; the bin counts, average scores and observed accuracies are in `final_lock.json`. Lower log loss and ECE here do not establish calibration on other populations. The selected model's development-derived threshold is 0.75. Its assessment had 198 total errors: 116 were flagged and **82 were accepted**. The matched baseline's more conservative scaled policy flagged 191 of 250 errors and accepted 59 errors. V2 makes fewer total errors and accepts more questions, while its accepted accuracy here is lower. The policy's Wilson-bound target is not a guarantee.

A post-lock descriptive check also applied the original v1 **raw-score 0.35** policy to the matched retrained word baseline on the same assessment. It accepted 1,592 examples (94.54% correct), including 87 errors, and flagged 163 of its 250 errors. The v2 policy accepted 1,728 (95.25% correct), including 82 errors, and flagged 116 of 198 errors. The v2 policy accepted more rows and had fewer accepted errors in this sample, but caught a smaller fraction of all its errors. The policies use different score scales; this check did not affect model or threshold selection. Details are in `legacy_policy_assessment.json`.

On the previously viewed test, selected v2 accepted 2,629 of 3,080 examples, of which 2,543 were correct (96.73%); 451 were reviewed and 86 errors were accepted. The original v1 policy accepted 2,456, of which 2,332 were correct (94.95%); 624 were reviewed and 124 errors were accepted. These are test-set observations only and must not be presented as operational outcomes.

## Owner example and illustrative checks

For the user's illustrative question, v1 returned `card_not_working` 29.6%, `declined_card_payment` 7.9%, and `top_up_failed` 4.2%, with review. V2 returns `card_not_working` 96.4%, `declined_card_payment` 1.1%, and `top_up_failed` 0.4%, without review. The v2 raw top score before temperature scaling was 49.1%. Its wording is ambiguous and no verified expected label was supplied. A higher score alone is not evidence that this particular answer is correct; the app warns that even high scores can be wrong.

`illustrative_predictions.json` contains separately authored in-domain, ambiguous and out-of-scope examples. They are not labeled benchmark data. One PIN-related example produces a counterintuitive high-scoring label, and the dataset contains PIN-related examples under multiple labels; this underscores the limits of confidence and label semantics. The out-of-scope examples are only illustrations and do not measure out-of-scope detection. The app does not log or persist visitor text.

## Resource and verification notes

On the recorded Windows CPU run, 16 logical processors were visible and numerical thread pools were limited to four. The selected model learned 6,000 word and 12,738 character features; it used about 11.5 MB for classifier coefficient/intercept arrays and its compressed model and vectorizer were 8,468,195 and 161,263 bytes. Development-pool fitting took about 0.85 s for features plus 9.74 s for the classifier; a 2,002-row batch took about 0.20 s and the median of 20 single-question predictions was about 6.0 ms. The matched word baseline's corresponding single median was about 1.6 ms and compressed model was 3,481,995 bytes. These are machine-specific measurements, not peak-memory or service guarantees. Full measurements for all candidates are in `final_lock.json`.

The active release's metadata, hashes, feature width, label order, calibration and review threshold are checked during loading. The selected assessment CSV excludes question text; DuckDB reconciles 2,002 rows, 198 errors and 274 flags. The code and tests verify the export checksum across Windows and Linux. The local app was smoke-tested using the real v2 artifact, including an above-threshold example and a below-threshold review message; an isolated app copy also ran without the dataset, training outputs, PyTorch or API keys. The local suite passed **20 tests**, and `pip check` reported no broken requirements.

The development and assessment stages were repeated in a separate ignored `reproduction/verification` folder using the same data and seeds. Assignment hashes, candidate artifact hashes, temperatures, selection, review policy and reported raw metrics matched exactly on this machine; the previously viewed test was **not** rescored. This is a reproducibility check, not an independent evaluation. `reproduction_check.json` records the comparison. See the README and study guide for commands; check GitHub Actions for the status of the exact revision under review.

## Limitations

These results concern one English research dataset, not real customer outcomes. Seven normalized train/test overlaps, prior test viewing, fold dependence, imperfect labels, and use of assessment as a release gate limit generalization. The classifier always chooses a known label, and neither temperature scaling nor thresholding reliably identifies every out-of-domain question. No production, fairness, privacy, regulatory, or real-bank deployment claim is made.
