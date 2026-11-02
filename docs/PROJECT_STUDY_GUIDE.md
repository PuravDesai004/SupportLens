# SupportLens study guide

This guide describes the files and measured run in this repository. It is designed to be read without the original coding session. The project was built with coding-agent assistance; in an interview, explain the design and what you understand rather than claiming to have typed every line yourself.

## 1. The story

Imagine a support agent reading a question such as “Where is my new card?” A useful first step might be to suggest a category, such as `card_arrival`, for a person to review. SupportLens demonstrates that narrow classification problem using published research data. It does **not** connect to a bank, answer the question, access accounts, route a ticket, or demonstrate a business saving. The intended user is a learner or reviewer exploring an ML workflow, not a bank customer seeking service.

## 2. A one-minute explanation

“I built a small intent-classification demo around the public BANKING77 dataset. I kept the official test split separate, grouped normalized duplicate texts when making a validation split, and fit a TF-IDF vocabulary only on the derived training rows. I compared logistic regression with a small PyTorch network using validation macro F1. Logistic regression narrowly won validation, so I saved it with its vectorizer and label mapping. I then evaluated both fixed candidates on the official test and built a Streamlit interface with a review flag. The model is a research-data demo; its scores are not guaranteed probabilities or a reliable out-of-scope detector.”

## 3. End-to-end flow

```text
TRAINING: public train CSV -> validate -> group split -> fit TF-IDF on train
          -> fit both models -> validation comparison -> save selected files
TEST:     public test CSV -> saved TF-IDF -> both fixed models -> metrics + CSV
APP:      typed question -> saved TF-IDF -> selected model -> category + score
```

Read `src/supportlens/data.py`, then `train.py`, `torch_model.py`, `evaluate.py`, `predict.py`, and finally `app.py`. `config.py` holds the seed and configuration. During inference a question is transformed by the saved vectorizer; the model produces one score per known intent, and `metadata.json` translates the highest-scoring index back to a label.

## 4. Dataset and split

The source is [BANKING77 by PolyAI](https://huggingface.co/datasets/PolyAI/banking77), cited as Casanueva et al., [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807) (2020). The dataset card lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We use the source CSVs at revision `9d081458ff52e53cf7e848f414e6e9344e4e6696`, containing `text` and `category`. The actual run found:

| Split | Rows | Labels | Notes |
| --- | ---: | ---: | --- |
| Official train | 10,003 | 77 | 35–187 examples per label |
| Derived train | 8,002 | 77 | Used to fit vocabulary and models |
| Validation | 2,001 | 77 | Used to select the model; 4–40 per label |
| Official test | 3,080 | 77 | 40 per label; used after selection |

The derived split is the first fold of a shuffled five-fold `StratifiedGroupKFold`, seed 42. Its groups are text after case-folding and whitespace normalization, so identical groups stay together; the split has zero shared normalized texts between derived train and validation. Source label spellings are preserved, including `Refund_not_showing_up` and `reverted_card_payment?`.

The official train and test contain seven overlapping normalized texts, with matching labels: six in derived train and one in validation. No exact string is shared. Familiar wording can make official test performance somewhat optimistic. The primary score still uses all 3,080 official test examples. The data does not tell us real support volume, arrival times, handling costs, or outcomes. Full label distributions, duplicate checks, source URLs, hashes, and versions are in `reports/data_manifest.json`.

## 5. TF-IDF and leakage

The vocabulary is a list of terms learned from the derived training text. A sparse row records weights only for terms that occur in one question; most of its 6,000 entries are zero. TF-IDF combines how often a term occurs in a document with how uncommon it is across training documents. The code uses unigrams and bigrams, `min_df=2`, `max_features=6000`, sublinear term frequency, scikit-learn's default normalization, and float32 features.

**Illustration, not a measured row:** suppose training questions contain “card late” and “card arrived.” A vocabulary could include `card`, `late`, `arrived`, `card late`, and `card arrived`. A question containing `late` gets a nonzero weight for that term; a word appearing in almost every training question tends to carry less distinguishing weight. The exact weights depend on the fitted corpus and normalization.

Fitting the vocabulary on validation or test text would let information about those sets influence features, even without labels. The code calls `fit_transform` on train and only `transform` on validation and test. A fixture test checks that a token seen only in validation does not enter the vocabulary.

## 6. The logistic regression baseline

Despite its name, multiclass logistic regression is a classifier. It learns a weight vector for each label and calculates class scores from the TF-IDF row. Its `predict_proba` returns one value per class, with the largest mapped to the proposed intent. This is a useful baseline because sparse text features work naturally with a linear model, it trains quickly, and the decision rule is easier to explain than a deep network.

The recorded model used `C=4.0`, `solver="lbfgs"`, `max_iter=500`, and converged in 45 iterations without a recorded convergence warning. Larger `C` means weaker regularization; it does not mean “confidence.” The deployed model is trained on the derived train subset, not retrained on validation after selection, so its reported evaluation corresponds to the saved artifact.

## 7. PyTorch foundations

- A **tensor** is a typed array. Each batch of features becomes a float32 tensor of shape `[batch, 6000]`; target class IDs are `torch.long` with shape `[batch]`.
- `IntentNet` has `Linear(6000, 64)`, `ReLU`, and `Linear(64, 77)`. The result has shape `[batch, 77]` and contains raw **logits**, not probabilities.
- `CrossEntropyLoss` compares logits with integer target IDs. It incorporates the relevant log-softmax operation; do not apply softmax first for this loss.
- A **forward pass** computes logits and loss. `loss.backward()` computes gradients. `optimizer.step()` changes weights using Adam; `zero_grad()` clears old gradients first.
- A **batch** is a subset of examples; an **epoch** visits the derived training rows once. The run used batches up to 128, 10 epochs, Adam at learning rate 0.003, CPU, and a 64-unit hidden layer. It densifies only one sparse batch at a time.
- At inference, softmax turns logits into scores summing to one. Those values still need calibration evidence before treating them as reliable probabilities.

## 8. Walk through real training code

In `train.py` (comments added here for explanation), the split is already made before fitting features:

```python
vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG, dtype=np.float32)
x_train = vectorizer.fit_transform(derived_train.text)
x_val = vectorizer.transform(validation.text)
```

The first line creates the configured transformer. The second learns vocabulary and inverse-document-frequency weights from 8,002 training questions and produces an `8002 x 6000` sparse matrix. The third reuses those learned weights for 2,001 validation questions; it does not refit.

In `torch_model.py`, the essential update is:

```python
inputs = torch.from_numpy(x[ids].toarray().astype(np.float32, copy=False))
targets = torch.as_tensor(y[ids], dtype=torch.long)
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Only the indexed batch is converted to dense input. The class IDs must be integers, not one-hot vectors. `model(inputs)` calls the two linear layers and ReLU; loss compares every row's 77 logits against the corresponding class ID. Backpropagation computes gradients, and Adam updates the parameters. The saved training loss fell from about 4.0175 in epoch 1 to 0.0558 in epoch 10. That is **training loss**, not proof that the model generalizes.

`train.py` compares validation scores and writes `reports/selection.json`; `evaluate.py` is a separate command run afterward. Candidate training files are kept locally in ignored `training_runs/`; only the selected model, vectorizer and metadata are needed for the app.

## 9. Evaluation and the measured result

Accuracy is correct predictions divided by examples. For each class, precision asks “of predictions of this class, how many were correct?” and recall asks “of examples truly in this class, how many were found?” F1 is the harmonic mean of precision and recall. **Macro F1** averages the per-class F1 values equally, so a common label does not dominate the comparison. The per-class CSVs include support, the number of true examples for each class.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 87.96% | 0.8767 | 87.60% | 0.8761 |
| PyTorch | 87.86% | 0.8733 | 88.73% | 0.8878 |

Validation uses 2,001 rows; test uses the same 3,080 official test rows for both models. The predeclared rule selected the higher validation macro F1; an exact tie would select the simpler baseline. Therefore the baseline remains selected even though PyTorch scores higher on test. Choosing again after seeing test scores would make the test part of selection. There were no convergence warnings in the baseline run. `reports/selection.json` and `reports/metrics.json` record the decision and results.

For the selected baseline, `score < 0.35` means “review recommended.” This threshold was fixed before test scoring. Of 3,080 test examples, 624 (20.26%) were flagged. Of the 2,456 not flagged, 2,332 were correct, or 94.95%. Those denominators matter; the app does not claim every accepted prediction is correct. The code returns `null` for accuracy among accepted examples if there are none. A high score can still be wrong or assigned to an out-of-scope input.

An example confusion in this run: seven test examples labeled `wrong_exchange_rate_for_cash_withdrawal` were predicted as `card_payment_wrong_exchange_rate`. That suggests a useful error-analysis question, not a claim about real ticket frequency. The first test evaluation is recorded as not previously viewed; future reproductions have already-seen test results and should be described that way.

## 10. Artifacts, app, and deployment

`artifacts/vectorizer.joblib`, `artifacts/model.joblib`, and `artifacts/metadata.json` belong together. Metadata records 77 sorted labels, 6,000 input features, model type, review threshold, version, and checksums. `predict.load_bundle` checks compatibility and hashes before loading the trusted joblib files. A hash mismatch or missing file yields an error; it does not make up an answer. Checksums alone do not make an untrusted pickle safe.

Streamlit caches the bundle and applies it to the question. The app offers sample questions, rejects empty and very long inputs, shows top alternatives and limitations, and requests that visitors omit sensitive information. It does not print or save visitor text in its own code. The owner requested a local-only demo, so no public app was deployed. A future public host may retain logs, and a public demo would not prove production infrastructure or an integration with a bank.

## 11. Testing and CI

Install `requirements-dev.txt`, then run `python -m pytest -q`. Tests cover bad data, a validation-only token, deterministic grouping, label order, artifact reload and tampering, input and review behavior, understandable toy metrics, DuckDB reconciliation, a tiny CPU PyTorch training run, and the real packaged model through Streamlit `AppTest`.

The recorded Windows sandbox required its temporary directory to stay in the project. The successful command was:

```powershell
New-Item -ItemType Directory -Force -Path '.tmp' | Out-Null
$env:TEMP = (Resolve-Path '.tmp').Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
```

GitHub Actions is configured in `.github/workflows/tests.yml` for Python 3.12 and offline tests. The [first hosted run](https://github.com/PuravDesai004/SupportLens/actions/runs/41495687880) passed for commit `546328fce1650cf0f2c21f6544acccae60aeb76d`. A workflow file by itself is not proof that a later commit passed; inspect the repository's Actions page for the run and commit under review.

## 12. Design decisions and alternatives

- **TF-IDF + logistic regression:** fast, sparse, understandable and a strong comparison point. It misses context beyond its word features.
- **Small PyTorch model:** shows tensors, backpropagation and nonlinear layers without a GPU. It adds training complexity and did not win the validation rule in this run.
- **Group-aware validation:** reduces normalized duplicate leakage between derived train and validation, although the official test still has seven overlaps and group stratification is approximate.
- **Macro F1:** gives each intent equal weight. Accuracy is also reported so the two views can be compared.
- **Streamlit:** a small Python interface that can load local artifacts without a separate API. It is not evidence of production cloud architecture.
- **DuckDB + CSV:** makes per-intent counts reproducible and easy to inspect in Excel; it does not imply a live analytics warehouse.

For a real bank, separate work would include representative and consented data, privacy and access controls, calibrated uncertainty, evaluation of unfamiliar requests and fairness across relevant groups, operational monitoring, human workflows, governance, and ongoing review. This repository does not establish those controls.

## 13. Interview practice

**Beginner**

1. *What is an intent?* A category describing what a question is about, such as card arrival; it is not the answer to the question.
2. *What is a feature?* A numeric input to the model. Here each TF-IDF column represents a learned word or word pair.
3. *Why use a validation set?* To make model choices without using the final test results for those choices.
4. *What does an epoch mean?* One pass through the training examples, in batches.

**Intermediate**

5. *Why not fit TF-IDF on all the data?* That would let information from held-out text influence the vocabulary and term weights.
6. *Why macro F1?* It weights each label equally and reveals a different view from overall accuracy. It can be noisy for labels with few validation examples.
7. *Why raw logits for cross entropy?* PyTorch's loss applies the appropriate log-softmax internally; a prior softmax changes the computation.
8. *Is 0.8 a calibrated 80% chance?* Not necessarily. It is a model score; calibration requires additional assessment on suitable held-out data.
9. *Why not automatically switch to PyTorch after test?* Test results are meant to evaluate the validation-selected choice, not choose it again.

**Project-specific**

10. *How was leakage addressed?* TF-IDF was fit on derived train only and normalized groups were kept together between derived train and validation. Seven normalized official train/test overlaps remain disclosed.
11. *How are the labels kept aligned?* A sorted `LabelEncoder` maps training labels to IDs; the ordered labels and model dimensions are saved and checked at inference.
12. *What runs in the demo?* The local app serves logistic regression. The owner requested a local-only demo; the PyTorch candidate remains training code and evaluation evidence.
13. *What should you say about impact?* State the measured dataset results and limits. There is no evidence of real customer use or a time or cost saving.

## 14. Hands-on exercises

1. Open `reports/evaluation.csv` in Excel. Create a pivot with `true_label` as rows and counts of `correct` or `review` as values. Compare totals to `reports/intent_summary.csv` and the SQL query.
2. In a separate reproduction directory, change a PyTorch training setting, run training, and compare **validation** macro F1. Record the setting and explain the tradeoff. The official test has already been observed and is not a fresh final exam; do not present another score as untouched.
3. Add a small unit test with a whitespace-only input and an unfamiliar nonempty question. Check the first is rejected and observe, without claiming an out-of-scope guarantee, that the second is still assigned a known label.
4. Trace one sample through `predict.py` and check that the number of scores equals the number of labels. Never inspect an untrusted joblib file by loading it.

## 15. Limits and next steps

These results describe one public English banking-intent dataset and one split. The dataset's known overlap, small validation support for some labels, fixed uncalibrated review threshold, and mandatory known-class output limit generalization. A future project could evaluate a properly collected out-of-domain set, validate calibration, assess subgroup behavior where appropriate, and test a human-review workflow with consent and privacy safeguards. None of those results is claimed here.
