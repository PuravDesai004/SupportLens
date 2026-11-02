-- One row per true BANKING77 intent in the selected model's exported test CSV.
SELECT true_label, COUNT(*) AS examples,
       SUM(CASE WHEN correct THEN 0 ELSE 1 END) AS errors,
       SUM(CASE WHEN review THEN 1 ELSE 0 END) AS flagged_for_review
FROM evaluation
GROUP BY true_label
ORDER BY examples DESC, true_label;
