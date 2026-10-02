# Phase 2 evaluation protocol — completed empirical execution

## Evidence boundary

The predeclared protocol below was executed successfully on the full local IEEE-CIS transaction files. See `../results/current_evaluation.json` for authoritative current metrics/provenance and `../results/current_evaluation.md` for interpretation. Restricted data remains external to this checkout. Historical results and synthetic checks are not current empirical evidence. The notebook now replays only current aggregate outputs; its former demonstration is retained in Git history.

`results/current_evaluation.json` is a status/provenance record. A blocked record has no dataset counts, empirical metrics, calibration conclusion, operational results, or feature ranking. A successful run must be explicitly identified as full IEEE transaction-table evaluation, not a sampled or synthetic substitute. Do not overwrite a completed result merely to accommodate another run.

## Data and access

Required files are `train_transaction.csv` and `test_transaction.csv`, obtained by an authorized user from the official IEEE-CIS Fraud Detection competition. The transaction training table supplies `isFraud`; `TransactionID` is a row identifier, never a predictor. The unlabeled competition test table is used to check the input contract and provenance, not as a scored validation set. `train_identity.csv` and `test_identity.csv` are not required or merged by this transaction-only study.

Keep restricted source files outside Git. The runner does not download data, accept competition rules, upload submissions, or publish raw rows/IDs/predictions. A user must verify current competition access/use/publication conditions on the official rules page; this protocol is not legal permission or a substitute for rule acceptance. Only compact aggregate evaluation/provenance evidence should be considered for publication, after checking the applicable conditions.

## Leakage review before fitting

Check unique/nonmissing IDs and binary target, train/test predictor alignment and ID overlap, excluded target/ID fields, exact target-copy and ID-copy predictors, duplicate feature rows, train-fitted preprocessing, and disjoint partitions. Record time coverage and repeated card-field proxies when present. Card attribute combinations are not verified customer/account IDs. Do not invent an entity group or equate shared card attributes with a shared person.

Anonymized feature definitions do not by themselves establish decision-time availability. Classify uncertain timing explicitly, including opaque aggregate/count/duration/derived fields. The real-data operation requires acknowledgment of this uncertainty and remains a retrospective benchmark, not a validated online fraud-intervention model. Acknowledgment is not evidence that a suspicious field is safe. Definite leakage must stop execution; uncertain availability must constrain interpretation. Review influential fields again before publishing results.

## Validation design

Three alternatives are considered, not exhaustively optimized:

- **Stratified holdout:** conventional same-distribution comparison; deterministic seed 42. Repeated entities and later transactions in training can inflate applicability to future scoring. It is an available alternative, not an automatically executed robustness search.
- **Temporal holdout:** planned primary if `TransactionDT` exists and supports valid strict ordering. Allocate approximately earliest 60% to model training, next 20% to calibration assessment, and final 20% to evaluation. Keep equal timestamps together; actual row fractions can differ. Require strictly separated time ranges, disjoint indices, and both classes in every partition. Do not silently fall back to random splitting when chronology fails.
- **Group-aware holdout:** not justified merely by card attribute fields. Requires a documented entity identifier with a defensible use case. No arbitrary card-tuple grouping is executed.

These are pre-data decisions. Actual feasibility, time boundaries, prevalence, repeated-proxy overlap and duplicate statistics cannot be claimed until the authorized files are inspected. Model-fit, calibration and final-evaluation rows together account for the full labeled training file: no silent small-sample substitution.

## Fair baseline and XGBoost

Reuse the corrected `build_preprocessor`: median numeric imputation, training-fitted categorical imputation/ordinal encoding, explicit unknown-category code, ID/target exclusion. Fit learned preprocessing only on model-training rows. Validation/calibration data must not set medians, vocabularies, scaling or model parameters.

The simple baseline is unweighted Logistic Regression with training-only scaling of the same processed feature matrix. It is deliberately not tuned. Ordinal codes have arbitrary nominal geometry for a linear model; this makes the baseline understandable but not an optimized categorical linear benchmark. Disclose that limitation when comparing it with a tree model. Do not publish an unconverged baseline as a completed comparison.

XGBoost retains the established configuration: 500 estimators, depth 6, learning rate 0.05, row/column subsampling 0.8, histogram tree method, seed 42 and ROC-AUC evaluation objective. A bounded CPU thread count is a runtime control, not a leaderboard search. Any further feasibility change must be explicit in provenance. No identity enrichment, broad tuning, additional model families, cloud infrastructure or SHAP dependency is introduced.

## Metrics and investigation capacity

Report ROC-AUC and Average Precision as ranking metrics; AP is the step-weighted precision-recall summary, not interchangeable with every trapezoidal PR-area calculation. Report prevalence to contextualize class imbalance. At threshold 0.5 report precision, recall, F1 and confusion matrix; accuracy is not a primary success measure.

Use fixed illustrative review budgets of top 0.5%, 1% and 2% of final evaluation transactions for both models, with deterministic ties. Report actual review count/rate, frauds captured, false positives, precision and recall. These are descriptive offline capacity scenarios, not optimized deployment thresholds. A score cutoff from this batch may not reproduce capacity under distribution drift. An investigation team should choose a budget using real staffing, service levels, fraud exposure and independently validated policy constraints. Do not invent loss/prevention costs or equate transaction amount with preventable fraud loss.

Optional row bootstrap intervals are conditional on the evaluated cohort and an IID-row assumption; they do not resolve time drift or repeated-entity dependence. Default omission is explicit, not a fabricated certainty claim. No repeated split search is required.

## Probability quality and calibration

Measure Brier score, log loss and quantile reliability bins. Brier score combines reliability, resolution and outcome uncertainty; lower Brier alone does not prove better calibration. Sparse/low-risk bins and imbalance can conceal high-risk errors. Report bin counts, positives, mean predicted probability and observed fraud fraction; an ECE summary is bin-dependent and is not a production adequacy certificate.

Use the middle partition to measure XGBoost calibration before final-outcome evaluation. A predeclared ECE flag of 0.02 is an illustrative exploratory trigger, not an industry standard. If material miscalibration is flagged and sufficient calibration observations exist, fit one sigmoid calibrator on a frozen, already fitted classifier using only the calibration partition. Otherwise record why no recalibration was attempted. Compare raw/calibrated final ranking and probability metrics, including possible ranking changes/ties. Do not pick a calibrator or retune a trigger using final outcomes. Ranking-only review prioritization does not inherently require calibrated probabilities; probability-based costs do.

## Interpretation and limitations

Use normalized XGBoost total-gain importance aligned to the actual transformed feature names, including dropped/all-missing-feature behavior. Total gain is predictive model contribution, not causation, and may favor features with more available splits or redistribute importance among correlated proxies. Do not assert that a feature causes fraud. Unknown measurement timing, cohort shift, proxy dependencies and incomplete feature semantics remain limitations even if all mechanical leakage checks pass.

No benchmark score establishes hosted deployment, operational effectiveness, production readiness, monetary savings, regulatory suitability, future population performance or leaderboard evidence.

## Provenance and reproducibility

Use the committed locked Python environment. Record actual Git execution SHA and clean/dirty state, source-file hashes, Python/library versions, UTC evaluation timestamp, input file names/hashes, full-file counts/prevalence, split method/seed/boundaries/index hashes, parameters, calibration decision, runtime and aggregate results. Hashes establish byte/source identity, not lawfulness or empirical correctness by themselves. Never infer an empirical run from the presence of a JSON file; validate its status and dataset kind.

CI uses small synthetic fixtures and schema/parser checks only. Restricted files and row-level derived artifacts must stay out of Git and GitHub Actions. Current README/notebook result claims were updated only after successful full-data execution.

### Authorized local execution

From the repository root, keep both CSVs in a separate authorized data directory and run:

```bash
uv sync --locked --no-dev --extra notebook
uv run --no-sync python scripts/verify_environment.py
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python evaluation.py --data-dir /path/to/authorized/ieee --output /path/to/new_evaluation.json --split temporal --n-jobs 4 --bootstrap-replicates 0 --acknowledge-uncertain-feature-timing
uv run --no-sync python scripts/validate_results.py /path/to/new_evaluation.json
```

`--overwrite` permits replacing the committed **blocked** status record only; completed evidence is protected. For another completed run choose a new output filename. Missing CSVs produce exit code **2**, `BLOCKED ON DATA`, and null metrics/counters. Other failed checks refuse publication. The parser verifies structural consistency, not truth of input provenance or completeness against an independently verified official release. The runner reads all rows of the supplied training CSV; a user must establish that these are the original full official files, not renamed samples. Windows paths should use native forward slashes, for example `D:/datasets/ieee`.

Calibration's exploratory trigger is ECE > 0.02 with at least 1,000 calibration rows and 30 examples in **each** class. It commits to the sigmoid recipe before final metrics, not to an empirically proven improvement. Final raw and selected XGBoost metrics are both retained regardless of whether recalibration improves them. Feature timing remains uncertain. No full-data result is established by the synthetic test run.

## Primary references

- Official competition data and rules: https://www.kaggle.com/competitions/ieee-fraud-detection/data and https://www.kaggle.com/competitions/ieee-fraud-detection/rules . Access conditions require user verification; a dynamic/blocked response is not verified rules text.
- scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html . Explains reliability diagrams, the Brier decomposition caveat and disjoint data for frozen-estimator calibration.
- scikit-learn calibration API: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html . Runtime compatibility is verified against this project's pinned installed version, not assumed from the latest docs.
- XGBoost importance API: https://xgboost.readthedocs.io/en/stable/python/python_api.html . Defines `total_gain`; the latest page may document a newer version than the project pin.
