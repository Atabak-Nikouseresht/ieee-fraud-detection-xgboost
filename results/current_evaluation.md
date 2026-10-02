# Current corrected evaluation — IEEE-CIS transaction tables

This is an actual full supplied-file run, not synthetic evidence. All displayed decimals are rounded from `current_evaluation.json`; that file is authoritative. No raw rows, IDs, predictions or trained models are published.

## Data and provenance

- Labeled transaction rows: 590,540; unlabeled competition test rows: 506,691.
- Raw/transformed predictors: 392/392; target `isFraud`, excluded identifier `TransactionID`.
- Frauds: 20,663; prevalence: 3.499001%.
- Execution revision: `7148defa53109f6e47cc6993d6fbc4bd55d9c3ef`; clean worktree: `True`.
- Created UTC: `2026-10-02T22:51:03.575182+00:00`; runtime: 204.817 seconds; warnings: none.
- Python: 3.11.16; versions: `{'numpy': '2.4.6', 'pandas': '3.0.6', 'scikit-learn': '1.9.1', 'xgboost': '3.2.0'}`.

| File | SHA-256 |
|---|---|
| `train_transaction.csv` | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| `test_transaction.csv` | `2a8e51f1d335a86025d2b7f45beb9b78d0ab1edd726ef531d8b71a8a0065c011` |

Files were already extracted in the user-supplied Downloads location. All five CSVs matched their original ZIP member bytes by SHA-256. File dimensions, readable structure, unique IDs and schema alignment show no obvious sampling/truncation. This is not an independently signed official authenticity check. Identity files and sample submission were present but deliberately unused. Original downloads were not modified.

## Validation

| Partition | Rows | Frauds | TransactionDT min | TransactionDT max |
|---|---|---|---|---|
| model_train | 354,324 | 11,988 | 86400 | 8745772 |
| calibration | 118,108 | 4,611 | 8745798 | 12192842 |
| evaluation | 118,108 | 4,064 | 12192900 | 15811131 |

Chronological 60/20/20 with strict boundaries and timestamp ties retained. The competition test time range follows the full labeled file. Stratified feasibility was checked, but no secondary empirical split was trained. No verified entity key exists; card tuples remain proxies. Preprocessing/scaling and both estimators fit on model-training rows only. Calibration decision precedes final metrics, but initial full-file label validation means this is not a physically sealed-label experiment.

## Metrics

| Metric | Logistic Regression | XGBoost | XGBoost minus baseline |
|---|---|---|---|
| roc_auc | 0.805437 | 0.887546 | +0.082108 |
| average_precision | 0.173031 | 0.480197 | +0.307167 |
| precision | 0.325390 | 0.770548 | +0.445158 |
| recall | 0.262057 | 0.332185 | +0.070128 |
| f1 | 0.290309 | 0.464237 | +0.173927 |
| brier | 0.040453 | 0.023365 | -0.017089 |
| log_loss | 0.497000 | 0.097501 | -0.399499 |

Final evaluation prevalence: 3.440918%. Precision/recall/F1 use threshold 0.5. Lower Brier/log loss is better; higher ranking/recall/precision is better in their stated contexts.

Confusion matrices, rows=true and columns=predicted, ordered [[TN, FP], [FN, TP]]:
- Logistic Regression: `[[111836, 2208], [2999, 1065]]`.
- XGBoost: `[[113642, 402], [2714, 1350]]`.

XGBoost adds substantial ranking and constrained-review value on this cohort. Its AP gain is not a guarantee outside the evaluated population. The linear model converged, but its extreme probabilities and top-budget yield are poor; no tuning was undertaken to conceal that. Temporal drift, representation constraints and extrapolation are possible explanations, not causally established diagnoses.

## Fixed review-capacity scenarios

| Model | Budget | Alerts | Frauds captured | False positives | Precision | Recall | Score cutoff |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.5% | 591 | 4 | 587 | 0.006768 | 0.000984 | 1.000000 |
| Logistic Regression | 1.0% | 1,182 | 34 | 1,148 | 0.028765 | 0.008366 | 1.000000 |
| Logistic Regression | 2.0% | 2,363 | 724 | 1,639 | 0.306390 | 0.178150 | 0.806987 |
| XGBoost | 0.5% | 591 | 548 | 43 | 0.927242 | 0.134843 | 0.965198 |
| XGBoost | 1.0% | 1,182 | 1,029 | 153 | 0.870558 | 0.253199 | 0.771379 |
| XGBoost | 2.0% | 2,363 | 1,551 | 812 | 0.656369 | 0.381644 | 0.339076 |

Ceiling-rounded budgets and deterministic original-row ties. The top-0.5% linear cutoff is exactly 1.0; tied high scores are not a reliable investigation policy here. Batch score cutoffs do not guarantee a future alert rate. No staffing or monetary loss/prevention assumption is made.

## Calibration

Calibration-partition XGBoost ECE: **0.005008**. The predeclared trigger (>0.02) was not exceeded; sample support was sufficient. Therefore **no sigmoid calibration was fitted**. Raw and selected final XGBoost metrics are identical. No post-hoc calibrator was chosen from final outcomes.

| Model | Final Brier | Final log loss | Final quantile-bin ECE |
|---|---|---|---|
| Logistic Regression | 0.040453 | 0.497000 | 0.035347 |
| XGBoost | 0.023365 | 0.097501 | 0.004580 |

ECE is bin-dependent; Brier mixes discrimination and calibration. Low aggregate error does not establish adequate high-risk calibration. Reliability-bin counts, positives, probability means and fraud fractions are retained for both models in JSON and the executed notebook.

| XGBoost final reliability bin | Rows | Frauds | Mean probability | Observed fraud fraction |
|---|---|---|---|---|
| 1 | 11,811 | 28 | 0.001937 | 0.002371 |
| 2 | 11,811 | 36 | 0.003385 | 0.003048 |
| 3 | 11,811 | 40 | 0.004758 | 0.003387 |
| 4 | 11,810 | 55 | 0.006377 | 0.004657 |
| 5 | 11,811 | 87 | 0.008307 | 0.007366 |
| 6 | 11,811 | 110 | 0.010780 | 0.009313 |
| 7 | 11,810 | 181 | 0.014564 | 0.015326 |
| 8 | 11,811 | 289 | 0.021754 | 0.024469 |
| 9 | 11,811 | 579 | 0.040852 | 0.049022 |
| 10 | 11,811 | 2,659 | 0.253016 | 0.225129 |

## Feature contribution

| Transformed feature | Normalized total gain |
|---|---|
| `numeric__V258` | 0.101292 |
| `numeric__C14` | 0.047517 |
| `numeric__C1` | 0.047279 |
| `numeric__C13` | 0.040484 |
| `numeric__TransactionDT` | 0.035483 |
| `numeric__card1` | 0.032471 |
| `numeric__V294` | 0.025387 |
| `numeric__TransactionAmt` | 0.023413 |
| `numeric__C8` | 0.021317 |
| `numeric__card2` | 0.021140 |

Total gain is associational, not causal. V258 accounts for about one tenth of total gain; it is not a target/ID copy found by the mechanical audit. Its anonymized semantics and point-in-time construction remain uncertain, as do masked C/D/M/V fields generally. TransactionDT represents chronological/cohort information and remains an allowed predictor, not TransactionID. Influential uncertain features constrain operational interpretation, not an excuse to relabel them verified safe.

## Actual-data leakage and related-observation review

See `leakage_review.json` for aggregate method/results and input hashes. Mechanical checks passed; uncertain feature timing remains explicitly unresolved.

| Check | Finding |
|---|---|
| Target / TransactionID copies and suspicious predictor names | None detected by runner; excluded from fitted predictors |
| Numeric target-complement scan | None across all 392 predictors / 590,540 rows; not a general target-encoding detector |
| Train / calibration / evaluation indices and IDs | Zero pairwise overlaps; every labeled row assigned once |
| Full predictor duplicates including TransactionDT | 3 repeated rows excluding first; zero cross-partition duplicate groups |
| Predictor duplicates excluding TransactionDT | 1,503 repeated rows excluding first; 463 cross-partition groups; 192 evaluation rows match training |
| Complete card1..6 tuples | 113,953 / 115,926 complete evaluation tuples match training (98.298052%); unverified attribute proxies, not established identities |
| Top-feature exact nonconstant affine-ID check | None among V258, C14, C1, C13, TransactionDT, card1; arbitrary nonlinear proxies not excluded |
| V258 / C-family construction and timing | Uncertain; no point-in-time window contract verified |
| Label adjudication / maturation | Unknown; no invented embargo |

Raw-cell duplicate fingerprint candidates were reread and compared exactly. Their definition is raw-cell equality, not universal numeric-normalized equality. No deduplication, grouping redesign, sampling or model retuning followed these findings.

The high proxy recurrence means this is **not** evidence of unseen-entity generalization. V258 missingness shifts from 74.714103% training to 82.777627% calibration and 82.647238% evaluation, supporting a descriptive cohort-shift caveat, not a causal diagnosis. No serious confirmed target/ID leakage was identified; passing these checks does not prove all anonymized fields are available at scoring time.

## Historical boundary

Historical ROC-AUC `0.9421992865326172` belongs to the previous preprocessing workflow. Current corrected temporal ROC-AUC is **0.887546**. Different preprocessing/splits mean the numbers are not an apples-to-apples performance-change estimate. No score-recovery tuning or Kaggle submission was performed.

## Limitations

- No field-level decision-time availability or fraud-label maturation contract; no blanket no-leakage claim or arbitrary purge interval.
- Repeated proxy relationships may cross temporal boundaries; no entity-disjoint guarantee.
- One final temporal cohort, no bootstrap confidence intervals or secondary empirical robustness model run.
- Linear baseline uses compact ordinal encoding and numeric treatment of numeric card codes, not an optimized categorical representation.
- Baseline top-budget and probability performance is poor; high overall AUC does not rescue it.
- XGBoost calibration-to-final ROC-AUC/AP degradation and highest-decile overprediction show cohort sensitivity.
- No identity enrichment, deployment, confirmed monetary savings, leaderboard evidence or production-readiness claim.
- Full supplied-file processing and byte matching establish integrity consistency, not independent official provenance certification.

## Reproduction

Use the unchanged execution code and locked environment; raw CSVs must remain external. Choose a new output filename because completed evidence is protected.
```bash
uv sync --locked --no-dev --extra notebook
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run --no-sync python evaluation.py --data-dir /path/to/authorized/ieee --output /path/to/new_evaluation.json --split temporal --n-jobs 4 --bootstrap-replicates 0 --acknowledge-uncertain-feature-timing
uv run --no-sync python scripts/validate_results.py /path/to/new_evaluation.json
```

The variable prefix is Bash syntax. No sampling, source/dtype changes, parameter changes or identity merge were needed. Host memory was under pressure during execution, but the full run completed; a precise peak-memory trace was not captured.

## Interview questions this project should survive

| Question | Evidence-backed answer |
|---|---|
| Why is ROC-AUC insufficient? | The baseline has AUC 0.805437 but captures only 4 frauds among its top 591 alerts. Overall ranking does not specify tight-budget precision. |
| Why AP? | Fraud is a minority; XGBoost AP 0.480197 versus baseline 0.173031 describes positive retrieval beyond AUC. |
| Why the primary split? | Train on earlier transactions and evaluate later ones; timestamp ties and strict ranges are verified. |
| Why can random splitting overstate applicability? | It mixes chronology and related observations; no random-score winner was selected. |
| What changes under imbalance? | Evaluation fraud prevalence is 3.440918%; accuracy alone would conceal missed fraud. Precision, recall and capacity are needed. |
| Why not only threshold 0.5? | It yields a model-specific alert count, not a staffing budget. Fixed top-budget scenarios answer a different question. |
| How does limited capacity matter? | At 1,182 alerts, XGBoost captures 1,029 frauds versus 34 for the baseline; budgets are descriptive, not staffing guarantees. |
| What does calibration mean? | Predicted probabilities should agree with observed rates; reliability bins are measured, and the predeclared recalibration trigger did not fire. |
| What is the baseline? | Untuned scaled unweighted Logistic Regression on the same training-fitted representation; convergence passed. |
| Why did XGBoost improve? | The observed ranking/review gains are measured. Nonlinear interactions are a modeling explanation, not an isolated causal proof; the linear representation and extrapolation constrain the comparator. |
| Which features contribute most? | V258, C14, C1, C13 and TransactionDT lead total gain; masked meanings and timing remain uncertain. |
| How was leakage checked? | IDs/target excluded, copies/suspicious names refused, train/test IDs disjoint, temporal boundaries strict, preprocessing training-only; opaque feature timing is not verified. |
| What does the model not prove? | Future unseen-entity performance, monetary prevention, operational intervention, leaderboard rank or feature causality. |
| Why not production-ready? | No scoring-time availability/label-maturity contract, deployment tests, staffing/cost policy or prospective monitoring evidence. |
