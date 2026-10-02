# IEEE-CIS Fraud Detection — temporal baseline comparison

[![CI](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml)

A completed transaction-only, chronological fraud-classification study comparing fixed Logistic Regression and XGBoost. Restricted source data stays outside Git; the notebook presents aggregate evidence from the actual run, not another training workflow.

## Current corrected evaluation

Full labeled transaction CSV: **590,540 rows**, **392 predictors**, **20,663 frauds** (3.499001%). Primary split: earliest **354,324** model-training rows, next **118,108** calibration rows, latest **118,108** final evaluation rows. All time boundaries are strict; no timestamp ties cross them.

| Model | ROC-AUC | Average Precision (AP) |
|---|---|---|
| Logistic Regression | 0.805437 | 0.173031 |
| XGBoost | 0.887546 | 0.480197 |

Reported metrics are rounded to **6 decimal places** from [the authoritative JSON](results/current_evaluation.json); AP is the precision-recall summary, not every trapezoidal PR-area definition.

At the fixed top-1% review budget, XGBoost generates **1,182 alerts**, captures **1,029 frauds**, and yields **0.870558 precision** and **0.253199 recall**. These are offline batch scenarios, not validated staffing policies or loss savings. The linear baseline performs poorly at the tightest budgets despite its overall ROC-AUC; this is reported rather than tuned away.

XGBoost final Brier = **0.023365**, log loss = **0.097501**, quantile-bin ECE = **0.004580**. Calibration-partition ECE was **0.005008**, below the predeclared **0.02** trigger, so sigmoid was **not fitted**. Low overall ECE does not certify high-risk or production calibration.

**Main limitation:** anonymized feature decision-time availability and label maturation remain uncertain; repeated card-attribute proxies can occur across temporal boundaries. This is a retrospective temporal benchmark, not deployment or unseen-customer validation.

See [full current tables and interpretation](results/current_evaluation.md), [the analytical notebook](Fraud_Detection_IEEE.ipynb), and [the predeclared protocol](docs/phase2-evaluation.md).

## Historical results — separate evidence

The historical ROC-AUC `0.9421992865326172` belongs to the **previous preprocessing workflow**, not this corrected evaluation. Its validation conditions differ, so it is not an apples-to-apples temporal comparison. The historical ROC-AUC is retained, while the lower current result is published without score-recovery tuning. No Kaggle leaderboard score is claimed. Earlier notebook code remains in Git history; current notebook outputs replay the current aggregate artifact only.

## Method and data contract

`evaluation.py` reads external `train_transaction.csv` and `test_transaction.csv`. `TransactionID` is excluded from predictors; `isFraud` is training-only. Unique/nonmissing IDs, binary target, matching predictor schemas, disjoint train/test IDs, suspicious outcome names and exact target/ID copies are checked. The unlabeled competition test table (506,691 rows) supplies contract/provenance checks, not performance labels.

The model-training / calibration / final-evaluation design is approximately 60/20/20 chronologically by `TransactionDT`. A random stratified option is implemented but not selected for its score. No unverified card-tuple entity grouping or identity-table merge is used.

Existing numeric median imputation and categorical imputation/ordinal encoding are fitted only on model-training rows; unseen categories map to -1. Logistic Regression also uses training-only scaling (C=1, max_iter=1000, seed 42). Ordinal nominal geometry and dtype-based numeric card codes limit the linear comparator; it is not an optimized one-hot baseline.

XGBoost retains 500 estimators, depth 6, learning rate 0.05, row/column subsampling 0.8, histogram method, seed 42 and four CPU threads. No broad tuning, identity enrichment or alternate-score selection was performed. Calibration selection precedes final metrics. Full-file label validation occurs earlier: this is not a physically sealed-label holdout.

## Included evidence and source

- `evaluation.py` — the full-data evaluation entry point, provenance capture and evidence publication gates.
- `fraud_detection.py` — unchanged shared data contract and train-fitted preprocessing.
- `results/current_evaluation.json` — full-precision current metrics, source/data hashes, splits, parameters and limitations.
- `results/current_evaluation.md` — human-readable current results and interview answers.
- `Fraud_Detection_IEEE.ipynb` — executed compact analytical narrative from the aggregate result artifact; no raw rows or retraining.
- `docs/phase2-evaluation.md` — methodology, access conditions and reproduction procedure.
- `pyproject.toml`, `uv.lock`, `requirements.txt`, `.python-version` — existing locked environment and generated hash-pinned runtime export.

## Reproduce

Obtain the original files through the [IEEE-CIS competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data), following its access/use/publication conditions. Keep both transaction CSVs **outside the repository**. `train_identity.csv`, `test_identity.csv` and `sample_submission.csv` are deliberately unused.

Use Python **3.11.16** and the existing uv lock:

```bash
uv sync --locked --no-dev --extra notebook
uv run --no-sync python scripts/verify_environment.py
uv run --with pip --no-sync python -m pip check
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run --no-sync python evaluation.py --data-dir /path/to/authorized/ieee --output /path/to/new_evaluation.json --split temporal --n-jobs 4 --bootstrap-replicates 0 --acknowledge-uncertain-feature-timing
uv run --no-sync python scripts/validate_results.py /path/to/new_evaluation.json
```

The environment-variable prefix is Bash syntax; use equivalent shell-specific syntax elsewhere. Choose a **new output filename**: completed evidence cannot be overwritten. The actual run used the full supplied dataset, took **204.817 seconds**, and produced no warnings; runtime and memory vary by host. Exact execution revision: `7148defa53109f6e47cc6993d6fbc4bd55d9c3ef`. Hashes identify bytes, not legal permission or independently signed official authenticity.

To inspect or replay the notebook's aggregate outputs:

```bash
uv run --no-sync jupyter notebook Fraud_Detection_IEEE.ipynb
```

## Tests and CI

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/validate_results.py
uv run --no-sync python -m compileall -q fraud_detection.py evaluation.py scripts tests
uvx pip-audit -r requirements.txt
```

CI verifies small/synthetic fitting, analytical boundaries, result schema, notebook/README consistency, provenance and dependencies. It never obtains or trains on restricted competition data. Synthetic fixtures are not empirical results.

## Limitations

- Opaque feature timing and label maturation are unresolved; no blanket absence-of-leakage claim or arbitrary embargo.
- Temporal validation is not verified entity-disjoint validation; repeated proxies remain possible.
- One final chronological cohort; no bootstrap interval or secondary empirical split was run.
- Logistic Regression extreme-score precision and probability quality are poor; its representation is a deliberately constrained comparator.
- Reliability is bin-dependent; the final highest XGBoost decile overpredicts observed fraud on average.
- No identity enrichment, monetary fraud-loss claim, submission/leaderboard evidence, deployment or production-readiness claim.
- Competition data, raw rows, identifiers, per-row predictions and trained models are not redistributed.

## Author

Atabak Nikouseresht · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
