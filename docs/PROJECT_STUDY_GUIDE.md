# SupportLens study guide: the current system

This guide is a self-contained explanation of the code and measured v2 run. SupportLens was built with coding-agent assistance; explain what you understand rather than claiming to have typed every line. The original [v1 study guide](PROJECT_STUDY_GUIDE_V1.md), its reports and its model files are preserved as historical evidence.

## 1. The story and a one-minute explanation

A support agent might read an English banking question and need to identify its topic. SupportLens demonstrates a model that **suggests** one of 77 research-data intent labels. It offers alternatives and a review indication. It does not connect to a bank, answer the question, access accounts, or operate a real queue.

An interview-sized explanation:

> I started with a TF-IDF and logistic-regression baseline on BANKING77. I grouped normalized duplicate questions, retrained the baseline and two candidates on the same development folds, and fit text features inside each fold. A word-plus-character logistic model led on macro F1. I fitted a temperature using out-of-fold predictions, then added a conservative review check using a matched word-only model. The check can flag a category even when the main score is high. I measured its extra review workload on a previously viewed assessment and kept the app local. The old test was already viewed, and neither confidence nor the second check is a guarantee.

## 2. Follow the actual flow

```text
Public training CSV -> grouped development / assessment split
  -> 3 group-aware development folds -> fit features inside each fold
  -> compare baseline, word+character logistic, and PyTorch
  -> fit temperature and review policies using out-of-fold scores
  -> lock candidate -> retrain all on development -> assessment release gate
  -> freeze release -> one exploratory comparison on previously viewed test

Visitor text -> active release pointer -> saved word+character TF-IDF
  -> logistic probabilities -> temperature scaling -> top intents + score check
  -> matched word-only model -> support for the main category -> extra review check
```

The important files are `src/supportlens/data.py` (data validation and grouping), `experiment_v2.py` (staged experiment), `torch_model.py` (network), `calibration.py` (temperature and review policy), `promote_v2.py` (locked release), `predict.py` (inference), and `app.py` (interface). `docs/EXPERIMENT_PLAN.md` was committed before the comparison; `reports/experiment_v2/` holds the measured evidence.

## 3. Dataset, splits and exposure

The data is [BANKING77 by PolyAI](https://huggingface.co/datasets/PolyAI/banking77), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Casanueva, Temčinas, Gerz, Henderson and Vulić, [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807), 2020. The pinned original CSVs contain `text` and `category`; their hashes and per-label counts are in `reports/data_manifest.json`.

| Data | Rows | Use |
| --- | ---: | --- |
| Official training file | 10,003 | Source for the experiment's development and assessment |
| V2 development pool | 8,001 | Three group-aware folds; model and threshold candidates |
| V2 development assessment | 2,002 | Fit no model or vocabulary on these rows; predeclared release gate |
| Official test | 3,080 | Previously viewed in v1; one v2 post-lock exploratory comparison |

All sets cover the same 77 labels. The assessment split is the first fold of a five-fold `StratifiedGroupKFold` with seed 73; the development folds use seed 137. Group keys are case-folded text with normalized whitespace. No group crosses a relevant split. Each development row receives one prediction from a model that did not fit that row or its normalized group.

The assessment influences whether a candidate is released and whether its precomputed raw or temperature policy is used, so it is part of the overall development process—not a pristine final test. The official test was already observed during v1. Six v2 development texts and one assessment text overlap with official test texts after normalization; their labels agree. A new split improves this comparison's fairness but does not erase earlier exposure or these overlaps.

## 4. Word and character TF-IDF

A vocabulary is learned from training text. TF-IDF gives a weight to a term based on its frequency in a question and its frequency across the fitted corpus. Most features in one row are zero, so the representation is **sparse**.

The current model concatenates:

- Up to 6,000 word unigrams and bigrams, `min_df=2`, sublinear term frequency.
- Up to 18,000 character-within-word 3–5-grams, with the same `min_df` and sublinear setting. The character block has weight 0.7.

The actual release learned 6,000 word and 12,738 character features: **18,738 columns**. For illustration, a word such as `cards` can contribute fragments like `car`, `ard`, and `rds`. Similar spellings may share fragments even when an entire word or phrase differs. This is not the model understanding the meaning of every phrase.

In `experiment_v2.make_vectorizer`, `FeatureUnion` joins the word and character transformers. `fit_candidate` calls `fit_transform` only on the fold's training text; `probabilities_for` calls `transform` on the held-out text. Fitting vocabulary or IDF on held-out examples would leak information. The tests include a token that appears only in validation and check it never enters either fitted vocabulary.

## 5. Logistic regression and its baseline

Multiclass logistic regression learns a coefficient vector for each label. It combines a sparse TF-IDF row with those coefficients, then outputs one probability-like value per class. V2 used `C=4`, `lbfgs`, and at most 500 iterations; its full development fit converged in 32 iterations without a recorded convergence warning. Larger `C` means weaker regularization, not higher confidence.

The comparison baseline used the **same v1 configuration**—word unigrams/bigrams, 6,000 features and logistic regression—but was retrained in every v2 fold and on the v2 development pool. The original deployed v1 artifact might have trained on some v2 assessment examples, so using that old artifact as the assessment comparator would be unfair. The original artifact is still saved in `artifacts/previous_v1/` for historical and illustrative comparisons.

## 6. PyTorch foundations

The comparison also retrained the original CPU network. Its input is a float32 tensor `[batch, 6000]`; its targets are integer `torch.long` class IDs of shape `[batch]`. The network is `Linear(6000, 64) -> ReLU -> Linear(64, 77)`. Its output contains **raw logits**. `CrossEntropyLoss` handles log-softmax internally; do not apply softmax before this loss.

In `torch_model.py`, one training update is:

```python
inputs = torch.from_numpy(x[ids].toarray().astype(np.float32, copy=False))
targets = torch.as_tensor(y[ids], dtype=torch.long)
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Only one sparse batch is converted to dense data. The forward pass calculates logits and loss, backpropagation calculates gradients, and Adam updates weights. The run used up to 128 examples per batch, 10 epochs, learning rate 0.003 and a fixed seed. Softmax is applied at inference. The PyTorch candidate is useful to study, but it did not win this v2 comparison and is not required to run the active app.

## 7. Comparing classification quality

Accuracy is correct predictions divided by all examples. Precision for a label asks how many predictions of that label were correct; recall asks how many actual examples of that label were found. F1 combines precision and recall. **Macro F1** gives each label equal weight. Top-3 accuracy asks whether the true label appears among the model's three highest-ranked labels.

| Candidate | Development OOF accuracy / macro F1 | Assessment accuracy / macro F1 | Assessment top-3 |
| --- | --- | --- | --- |
| Word logistic | 85.04% / 0.8472 | 87.51% / 0.8736 | 96.60% |
| Word + character logistic | **87.54% / 0.8734** | **90.11% / 0.9007** | **97.25%** |
| Word + PyTorch | 85.86% / 0.8571 | 88.01% / 0.8776 | 96.75% |

Development metrics pool 8,001 out-of-fold predictions; assessment has 2,002 examples. The word+character model beat the matched baseline in all three development folds. The pooled development macro-F1 gain was 0.0262, with an approximate paired bootstrap interval [0.0216, 0.0312]. Its assessment gain was 0.0271, with an approximate interval [0.0172, 0.0379]. The intervals are descriptive: row resampling does not account for all shared training-fold dependence or earlier data exposure.

Selection rules were recorded before fitting: a candidate needed a meaningful development gain and a positive paired bootstrap lower bound, then had to pass the assessment gate. The gate used the matched baseline retrained without assessment data. The candidate and threshold were frozen before the previously viewed test comparison.

On that **previously viewed** 3,080-row test, the original v1 artifact scored 87.60% accuracy / 0.8761 macro F1 and v2 scored 89.84% / 0.8987. These are exploratory results on a set already seen in the project, not a new independent estimate. Per-class CSVs and common confusion pairs are in `reports/experiment_v2/`. For example, the selected model confused `card_arrival` with `card_delivery_estimate` four times on the assessment; similar labels and incomplete wording make this difficult.

## 8. Confidence, calibration and review

A model's largest output is a **score**, not automatically a calibrated probability. If many predictions assigned a score near 0.8 are correct about 80% of the time, those predictions are better calibrated for that population. **Log loss** penalizes assigning low probability to the true label; lower is better. The reported **10-bin expected calibration error (ECE)** compares average confidence and observed accuracy in score bins; its value depends on the binning and sample.

`calibration.fit_temperature` chooses one positive scalar using only development out-of-fold scores and labels. `temperature_scale` applies:

```text
scaled probabilities = softmax(log(raw probabilities) / temperature)
```

The selected scalar is approximately 0.51738. Values below 1 sharpen the distribution; values above 1 soften it. The same positive scaling is applied to every class, so the ranking does not change. On the separate assessment, v2's log loss went from 0.5800 to 0.3749 and ECE from 0.1967 to 0.0207. Calibration measured on the same out-of-fold predictions used to fit the scalar may look optimistic; the assessment is a separate check, though it also informs the release gate. Neither proves calibration for real bank traffic.

The development rule chooses the lowest threshold on a fixed grid with at least 200 accepted examples and a 95% Wilson lower bound of at least 0.95 for accepted accuracy. If no threshold qualifies, the implementation can flag every prediction. The chosen v2 threshold is **0.75**. It is a policy selected from development data, not a promise.

The earlier 0.75-only policy accepted 1,728 and reviewed 274 of the 2,002 assessment examples; 82 accepted predictions were wrong. The current policy also asks whether a matched **word-only** model gives the main model's suggested label at least 0.475 raw support. If not, it recommends review even when the main score is high. This is a second view trained on the same data, not an independent human reviewer and not a calibrated probability of correctness.

The current policy accepted 1,343 and reviewed 659; **42 accepted predictions were wrong**. Of the 385 newly reviewed, 40 were errors and 345 were correct. Accepted accuracy rose from 95.25% to 96.87% on this previously viewed assessment, while review volume increased substantially. These comparisons use an assessment already used in earlier model selection and this release gate; they are not guarantees. The raw word-support threshold and temperature-scaled main threshold use different score scales. Coverage and accuracy must be reported with denominators.

The 0.475 cutoff is the lowest value on a fixed grid that achieved a 95% Wilson lower bound of at least 97% for accepted accuracy on the development out-of-fold predictions, with at least 1,000 accepted examples. That target was not met as a guarantee on assessment: its observed accepted accuracy was 96.87%. An earlier, less conservative development-selected cutoff of 0.35 did not flag the owner's illustrative question; the follow-up plan was recorded before choosing the stricter cutoff. See `docs/REVIEW_CONSERVATIVE_PLAN.md` and `reports/review_guard/RESULTS.md` for the complete exploratory history.

## 9. The owner's example and other qualitative checks

The original model returned `card_not_working` at 29.6%, then `declined_card_payment` at 7.9% and `top_up_failed` at 4.2%; it flagged the result. Its word vectorizer recognized 14 of 17 analyzed unigram/bigram occurrences; several location-related bigrams were absent, while words about a card not working contributed to the top logit. The wording could refer to a general card fault, a declined payment or a location-specific acceptance problem. There is no verified true label for this example.

The main model returns the same top label at **96.4%** after temperature scaling, followed by `declined_card_payment` at 1.1% and `top_up_failed` at 0.4%. Its unscaled top score was about 49.1%. The current release **flags it for review** because the word-only model's raw support for that category is about 43.5%, below 47.5%. The rule is general, not a text match. The decision to investigate a stricter policy was prompted by the example; its cutoff was selected on development data and the example had no label and was not used in the threshold search. This higher score is not proof that the category is correct.

`reports/experiment_v2/illustrative_predictions.json` contains other manually authored examples, clearly labeled as illustrative, including out-of-scope questions. A PIN-related example received a counterintuitive high-scoring category. Those examples are not labeled benchmark data and cannot estimate real-world accuracy or out-of-scope detection. The app does not save visitor-entered questions.

## 10. Artifacts and inference

`artifacts/active.json` points to `artifacts/v3/` and records the metadata checksum. V3 contains the saved main `FeatureUnion` and classifier **plus** the matched word-only vectorizer and classifier. Metadata records label mapping, temperature, both review thresholds, dimensions and file hashes. `predict.load_bundle()` resolves the active pointer; `load_bundle(path)` loads a specifically requested trusted release. The loader checks scikit-learn version, hashes, feature widths and class ordering. These checks detect mismatched files, but joblib is safe to load only from a trusted source.

In `predict.py`, inference obtains the main model's probabilities, applies the fixed temperature and maps the highest indices to labels. It also runs the word-only model and checks its raw probability for the main label. Either a low main score or weak word-only support triggers review; the main ranking and displayed score are unchanged. `app.py` explains which signal caused review, caches the bundle, rejects blank or very long input, and does not download data or train at startup. The demo stays local at the owner's request.

The matched assessment run used a 16-logical-processor Windows machine with thread pools limited to four. The main model had 18,738 features, about 11.5 MB of coefficient/intercept arrays and an 8.47 MB serialized classifier; its vectorizer was about 0.16 MB. The word-only review files add about 3.54 MB compressed. A later warm single-question measurement on one manually authored example was about 3.4 ms before and 4.7 ms after the guard, on that machine. These are illustrative measurements, not service-level promises or peak-memory measurements.

## 11. Tests, SQL and CI

The offline suite checks data contracts, grouped splits, fold-only vocabulary, label order, temperature math and ranking, threshold and review-all behavior, saved file hashes, real release inference, a tiny CPU PyTorch training run, DuckDB reconciliation and Streamlit `AppTest`. Install `requirements-dev.txt` first. In the recorded Windows sandbox, the exact command was:

```powershell
New-Item -ItemType Directory -Force -Path '.tmp' | Out-Null
$env:TEMP = (Resolve-Path '.tmp').Path
$env:TMP = $env:TEMP
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
```

`selected_assessment.csv` is the historical score-only export: DuckDB reconciles it to 2,002 examples, 198 errors and 274 flags. The current `reports/review_guard/assessment_comparison.csv` contains old and new flags, 659 current flags, row IDs and scores, but no question text. `reports/evaluation.csv` is the historical v1 test export. GitHub Actions is configured for Python 3.12; inspect the repository's Actions page for the actual run on the commit being discussed.

## 12. Reproduction and practical decisions

The README gives the full staged commands using a fresh ignored `reproduction/v2` output directory. `development` writes the candidate lock; `assessment` fits all candidates without assessment rows and writes the release lock; `promote_v2` copies the exact locked candidate files. The review experiments then select cutoffs on OOF development predictions, assess them on previously viewed assessment predictions, and `promote_review_guard` packages the checked pair. The standard reproduction commands do not rescore the already-seen official test. Seeds and pinned direct dependencies help, but results can vary by hardware and OS.

Why these choices? Word TF-IDF and logistic regression are fast, sparse and explainable. Character n-grams can cover spelling variations but add features and inference cost. A small PyTorch network teaches gradient-based learning without a GPU, at the cost of an additional runtime dependency if deployed. Temperature scaling improves score reliability in this assessment but changes score magnitudes and is not an out-of-scope detector. A review threshold trades human workload against accepted errors. The app is local and simple, not evidence of a production cloud platform.

## 13. Interview questions

1. **Why split before fitting TF-IDF?** Otherwise held-out words and their document frequencies affect the features learned by the model.
2. **Why keep duplicate groups together?** The same normalized question on both sides could make validation artificially easy.
3. **Why retrain the baseline for v2 assessment?** The old artifact may have trained on those rows; all candidates must fit the same development data.
4. **What does macro F1 emphasize?** It averages per-label F1 equally, so common labels do not dominate the result.
5. **What is top-3 accuracy?** The fraction of examples whose correct label is among the three highest-ranked outputs.
6. **Does temperature scaling improve classification?** It preserves rankings; it can change score calibration and review decisions, not top-1 or top-3 accuracy.
7. **Does the current review check catch every error?** No. It left 42 assessment errors accepted, and the assessment was previously viewed. The 0.75 threshold and word-support check are not guarantees.
8. **Why not call the official test untouched?** Its results were already examined in v1 and there are seven normalized train/test overlaps.
9. **Why are logits passed directly to cross entropy?** The PyTorch loss incorporates the appropriate log-softmax internally.
10. **Does 96.4% on the owner's example prove it is correct?** No. Its true label is unverified, the wording is ambiguous, and calibration is population-dependent. The second-view check now flags it.
11. **What if an artifact is missing or mismatched?** The loader checks checksums, dimensions, class order and version, and the app shows a setup error.
12. **What business impact was measured?** None. These are research-data metrics, not actual support volumes, time saved or customer outcomes.

## 14. Hands-on exercises

1. Open `selected_assessment.csv` in Excel and group by `true_label`; compare total errors and flags with `selected_assessment_intent_summary.csv` and the SQL query.
2. Trace a generic question through `predict.py` and check the score vector has 77 values and uses the active metadata's temperature and label order.
3. On a **new experiment**, compare a different character-feature setting using only development folds. Record the decision rule first; do not tune against the already-viewed test.
4. Add an illustrative out-of-scope question, observe its label and review flag, and explain why one example cannot measure out-of-scope detection.

## 15. Limits and next steps

The data represents one English banking-intent research task, not a bank's live population. Some intent labels overlap, some questions lack context, the test was previously viewed, and there are seven normalized train/test overlaps. The classifier always chooses one of the 77 labels. A high score can be wrong or out of scope, and the illustrative PIN example demonstrates this risk. Before any real use, separate work would need representative and consented data, privacy and access controls, monitoring, subgroup assessment where appropriate, calibration on current traffic, and a validated human-review workflow. No such production claim is made here.
