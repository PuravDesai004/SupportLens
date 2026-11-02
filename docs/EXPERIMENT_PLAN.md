# Experiment protocol: classification, score reliability and review

This protocol was written before fitting the new comparison models. It is separate from the historical v1 run. The initial repository commit is `4a93df4b850e0adc19dee7b60544ea613ea24b77`; the original inference files are also preserved in `artifacts/previous_v1/`. The original reports, training code and configuration remain in place.

## What has already been seen

The v1 model fit its vocabulary and weights on 8,002 rows of the 10,003-row official training file and used the remaining 2,001 for its validation-based choice. Both candidates were then scored on all 3,080 official test rows. The test is **previously viewed**, not a fresh holdout. Seven normalized train/test overlaps were already observed. This project has also viewed the owner's illustrative question; it is not a labeled test item and is not used to choose configurations or thresholds. No inference code logs visitor questions.

## Fixed splits and candidates

Check the pinned official training CSV hash first. Use the first fold of shuffled five-fold `StratifiedGroupKFold` with seed 73 as a development assessment set. Case-folded, whitespace-normalized texts are groups; all 77 classes must occur on each side and no group may cross. The other four-fifths form the development pool. On that pool use three shuffled, group-aware, stratified folds with seed 137. Every development example must receive one out-of-fold prediction. Fit vocabularies, IDF weights and classifiers anew on each fold's training rows. Retrain **every** candidate, including the v1 baseline configuration, on the same development pool before assessment; never compare the old deployed artifact against this new assessment as if it were independent.

The following configurations are fixed before comparison:

1. **word baseline:** v1 word TF-IDF (unigrams/bigrams, 6,000 features, `min_df=2`, sublinear TF, float32) plus logistic regression `C=4`, `lbfgs`, `max_iter=500`.
2. **word + character logistic:** the same word features concatenated with `char_wb` 3–5-grams (maximum 18,000, `min_df=2`, sublinear TF, float32), weighted 0.7 relative to word features, and the same logistic regression settings. Character fragments can represent spelling and wording variations. There is no rule for the owner's question.
3. **word + PyTorch:** the existing 6,000 word features and existing CPU `6000 -> 64 -> 77` ReLU network, 10 epochs, batch size 128, Adam learning rate 0.003, seed 42.

Use no GPU, external model download, paid API, or new required package. Record convergence warnings, wall-clock time, feature counts and coefficient/checkpoint sizes; measure batch inference speed on the same assessment rows and record the measurement environment. These describe practical costs, not a formal hardware benchmark.

## Selection, calibration and review policy

The primary selection metric is pooled out-of-fold macro F1; report accuracy, top-3 accuracy, per-class precision/recall/F1/support, confusion pairs, log loss and 10-bin expected calibration error (ECE). The candidate with the highest development macro F1 is eligible only if its gain over the matched baseline is at least 0.005, its accuracy is no more than 0.002 below baseline, and the lower bound of a seeded 95% paired, class-stratified bootstrap interval for its macro-F1 gain is positive. This interval is an approximate descriptive check: folds share training data and duplicate groups are rare but not independent bootstrap units. Otherwise retain v1. Exact metric ties favor the simpler baseline. If the best candidate is PyTorch but an eligible logistic candidate is within 0.002 macro F1, prefer logistic to avoid a large extra runtime dependency.

Fit one scalar temperature per candidate using only out-of-fold development scores and labels, minimizing multiclass log loss over a bounded positive interval. Temperature rescales log scores and preserves rank; it may improve probability quality but cannot itself improve top-1 or top-3 classification. Calibration fitted and measured on the same out-of-fold predictions can look optimistic, so report raw and adjusted reliability on the separate assessment too.

For both raw and temperature-adjusted scores, select the lowest threshold on a fixed 0.05–0.95 grid in 0.025 steps that accepts at least 200 examples and has a 95% Wilson lower bound of at least 0.95 for accepted accuracy on development out-of-fold predictions. The lowest qualifying threshold accepts the most rows on that grid. If none qualifies, use an explicit review-all fallback. This is a conservative development rule, **not a guarantee** of 95% future accuracy. Report accepted/reviewed counts, accepted accuracy and accepted errors, including the zero-accepted case.

Write the candidate ranking, temperatures and both threshold options to a lock file **before** assessment. Assessment is a predeclared release-confirmation gate, so it is part of the overall development process rather than an untouched final test: promote only if the preselected candidate beats the matched baseline's assessment macro F1 and is within 0.002 of its accuracy. If not, leave v1 deployed. For a promoted candidate, use adjusted scores only if their assessment log loss is at most 0.01 worse and ECE at most 0.02 worse than raw; otherwise use its precomputed raw-score threshold. This gate does not search for a new threshold. Record the outcome and freeze the deployment choice before any optional test scoring.

After selection and deployment configuration are frozen, score the locked approaches once on the **previously viewed** full official test for an exploratory comparison. Do not change the model, threshold, or calibration in response to test results. Record the prior test exposure and any rerun reason. A new split improves the fairness of this comparison but cannot undo earlier exposure to the dataset. No external labeled dataset is assumed.

## Qualitative checks

First inspect the v1 model's mistakes on its original validation subset and representative public-data examples to identify error patterns. After selection, inspect the owner's question and a small set of varied, manually authored examples only as illustrative behavior. Do not treat any such example as a measured accuracy, hardcode a label, change weights, or infer a verified true label for ambiguous wording.
