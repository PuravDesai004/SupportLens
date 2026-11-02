# Second-view review: evidence and limitations

## What changed

The active v3 release preserves the v2 main classifier, temperature, predicted label, alternatives and displayed score. It adds a matched word-only logistic model as a **review signal**. A question is flagged if the main temperature-scaled score is below 0.75 **or** the word-only model gives the main model's top label less than 0.475 raw support. The word-only score is not presented as a calibrated probability of correctness. The models share training data and can fail together.

The app explains when the second model caused the review. Its privacy notice is unchanged, and application code does not log or persist visitor-entered text. The original v1 and v2 artifacts remain available. V3 metadata checks hashes, feature dimensions and label ordering for both models.

## How the policy was chosen

The first plan, committed before evaluation as `35eff4b`, chose a general support cutoff from 8,001 saved out-of-fold development predictions while capping extra review. It selected **0.35** (`2a890fd`), and the previously viewed 2,002-row development assessment passed its predeclared gate (`7b4ca83`). That rule would still not flag the owner's illustrative question. It was not deployed.

This qualitative failure prompted a follow-up, more conservative policy. The follow-up plan was committed as `11efa49` before its cutoff was chosen. Its rule picked the lowest fixed-grid support cutoff whose accepted development subset had at least 1,000 examples and a 95% Wilson lower bound of at least 0.97 for accuracy. It selected **0.475** (`7c02bc0`) using development predictions only. The owner's question was not a labeled data point, part of the grid search, or matched by any text rule. The assessment release decision was locked as `892c3ed` before the example was checked again.

Both policies used the same main classifier and matched word-only model, trained on the same development pool. The assessment was already viewed and used in earlier model selection, and it was reused as a release gate here. It is **not a fresh independent test**. The original official test was not rescored for this change. Iterating after seeing the illustrative example and reusing the assessment limits what can be concluded; the numerical cutoff nevertheless comes from the fixed aggregate development rule, not that example.

## Review tradeoff on the previously viewed assessment

The 2,002 rows include 198 main-model classification errors. The main model's category and score do not change between policies.

| Policy | Reviewed | Accepted | Accuracy among accepted | Accepted errors | Errors flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous main-score-only rule | 274 | 1,728 | 95.25% | 82 | 116 |
| First, 0.35 support rule | 463 | 1,539 | 96.10% | 60 | 138 |
| **Current, 0.475 support rule** | **659** | **1,343** | **96.87%** | **42** | **156** |

Relative to the previous rule, the current guard sent **385 additional** examples to review: **40 were errors and 345 were correct**. This reduces accepted errors on these rows but materially increases review workload. It still left **42 errors unflagged**. The conservative 97% Wilson-bound target applied to development predictions did not guarantee 97% observed accepted accuracy on assessment. No real-world error rate, queue volume or time saving is inferred.

On the 8,001 out-of-fold development rows, the current guard reviewed 3,472 and accepted 4,529, of which 104 were errors; the score-only policy reviewed 1,490 and accepted 6,511, of which 288 were errors. The cutoff was selected and evaluated on those same development predictions, so that estimate is optimistic. Cross-fold diagnostic summaries are saved in `conservative_development_lock.json`, but their underlying model training sets overlap and the existing temperature was fit on all OOF rows.

`assessment_comparison.csv` provides row IDs, true/predicted labels, scores and old/new flags, **without question text**. `assessment_export.json` records its checksum and reconciled totals. The historical `reports/experiment_v2/` CSVs and metrics remain unchanged.

## Illustrative behavior

After the policy was frozen, the owner's example retained `card_not_working` at 96.4%, but is now flagged because the word-only model's raw support for that category is about 43.5%, below the 47.5% cutoff. The expected label is unknown, so this does not prove the classification correct. No full user-entered question is stored by the app or in this report.

`illustrative_checks.json` contains separately authored examples. A seemingly straightforward card-delivery example is newly flagged, illustrating the extra reviews. A PIN-related example with a counterintuitive high-scoring category remains unflagged, illustrating that the guard does **not** catch every ambiguity. The small out-of-scope examples are illustrations, not a detection benchmark.

## Verification and cost

The local suite passed **28 tests**, covering general high-score/weak-support behavior, cutoff boundaries, fixed-grid selection, missing or mismatched guard files, label alignment, real artifact inference, CSV reconciliation and the local Streamlit app. An isolated app copy ran with no dataset, training outputs, PyTorch or API keys. The assessment export is text-free. A separate same-data reproduction matched the cutoff, assessment comparison, CSV checksum and v3 metadata checksum without rescoring the official test; `reproduction_check.json` records this as reproducibility evidence, not an independent evaluation.

On one Windows CPU run with numerical threads limited to four, a warm single-question median was about 3.4 ms for v2 and 4.7 ms with v3 (100 repetitions on one manually authored sample). The guard adds 3,481,995 bytes for its compressed classifier and 59,943 bytes for its vectorizer. These are local measurements, not peak-memory or service-level guarantees. See the README for reproducible stages and GitHub Actions for the status of the exact commit.
