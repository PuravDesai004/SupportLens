# Review guard experiment protocol

This protocol is written before selecting the new review rule. The aim is to find questions for which the current word-and-character classifier is confident after temperature scaling but a complementary word-only classifier gives the suggested label little support. This is a general signal; the owner's illustrative question is not used to fit or select the rule. Neither model is trained on user-entered questions, and the app must not log them.

## Data and exposure

Reuse the saved v2 out-of-fold probabilities for the 8,001 development rows: each word-only and word+character prediction was made by a model trained without that row's normalized-text group. The models use the same labels and folds. The current temperature and 0.75 threshold were already selected from this development data, so estimates on those rows are optimistic for a new policy.

The 2,002-row v2 assessment was not used to fit either full-development model or its features, but it has already been viewed and used in the v2 release gate. Treat it as a **previously viewed development assessment**, not as an independent new test. The official BANKING77 test has also been viewed; do not score it again for this change. The prior release and reports remain available.

## Candidate and fixed selection rule

Keep the current main classifier, temperature and 0.75 score threshold unchanged. The added signal is the raw word-only model probability assigned to the main model's top label. Both models were trained on the same data, so they are complementary views rather than statistically independent reviewers. The word-only score is not assumed calibrated. Review a question if the existing score rule flags it **or** this word-only support is below a development-selected cutoff.

On the 8,001 out-of-fold predictions only, search support cutoffs 0.05 through 0.95 in increments of 0.025. Permit at most 15 additional reviewed examples per 100 total examples, relative to the current policy. Require at least 200 accepted examples, no decrease in accepted accuracy, and at least a 10% relative reduction in errors among accepted predictions. Of qualifying cutoffs, choose the one that catches the most formerly accepted errors; break ties by fewer additional reviews, then the lower cutoff. If none qualifies, retain the existing review policy. The fixed 15-per-100 budget is a tradeoff selected for this experiment, not a measured business cost.

Freeze the cutoff before checking the assessment or the owner's illustrative question. On the previously viewed development assessment, promote the rule only if it catches at least 5% of the errors the existing policy accepted, does not reduce accepted accuracy, accepts at least 200 examples, and adds no more than 18 reviews per 100 total examples. Otherwise retain the prior policy and report the result. Report how many newly reviewed examples are correct and incorrect; flagging more questions is not automatically an improvement.

The supplementary inference artifact will be the word-only logistic candidate retrained on the same 8,001-row development pool during v2. Its hash and label order must match the existing locked v2 candidate record. It is not the older v1 artifact, which may have trained on some assessment rows. Test the owner's question only after the cutoff and release decision are locked. Check additional manually authored questions, including out-of-scope examples, without making accuracy or detection-rate claims from them. No review system can infer every ambiguity or reliably detect all out-of-scope requests.
