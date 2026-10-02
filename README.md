# IEEE-CIS Fraud Detection — XGBoost baseline

[![CI](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml)

A notebook-based fraud-classification study using IEEE-CIS transaction data. The notebook demonstrates a stratified train/validation split, training-only preprocessing, XGBoost training, and held-out ROC-AUC, average precision, threshold-based precision/recall/F1, and confusion matrix.

> **Evidence:** The historical ROC-AUC `0.9421992865326172` belongs to the previous preprocessing workflow. Notebook outputs are cleared; that value is not a result of the corrected workflow. No corrected full-data metric or Kaggle leaderboard score is claimed. The competition dataset is not included.

## Included files

- `Fraud_Detection_IEEE.ipynb` — data validation, preparation, training, validation, and submission-file workflow.
- `fraud_detection.py` — shared transaction-table contract and production preprocessing used by the notebook and dataset-free smoke tests.
- `pyproject.toml` and `uv.lock` — direct dependencies and a cross-platform, fully resolved lock.
- `requirements.txt` — generated, hash-pinned export of the runtime lock for pip-based consumers; regenerate it from `uv.lock`, do not hand-edit it.
- `.python-version` — exact interpreter patch version used for this reproducibility baseline.

## Method and input contract

The notebook reads `train_transaction.csv` and `test_transaction.csv` from the repository root. It requires `TransactionID` in both tables and `isFraud` in training only; IDs must be present and unique, the target must contain only 0/1, and train/test predictor columns must match (column order may differ). It models the transaction table only. The optional `train_identity.csv` and `test_identity.csv` competition files are neither required nor merged; identity features are outside this baseline's scope.

After validation, the notebook separates the target and IDs, then makes an 80/20 stratified split using `random_state=42`. Median numerical imputation and most-frequent categorical imputation plus ordinal encoding are fitted only on the training partition. Unseen validation/test categories map to `-1`. The fixed-width integer codes are compact tree inputs, not meaningful category rankings. XGBoost uses 500 estimators, depth 6, learning rate 0.05, row/column subsampling of 0.8, and ROC-AUC as its evaluation metric.

Metrics are computed on the held-out validation split; threshold 0.5 is descriptive and is not tuned to a real operating cost. The final notebook cell writes `submission.csv` only when the notebook is run; no model or generated output is committed.

## Reproduce

Access the original data through the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and follow its access and use terms. The data is not redistributed here. Put `train_transaction.csv` and `test_transaction.csv` in the repository root. Identity CSVs are not needed by this implementation.

Install Python **3.11.16** (the exact version is recorded in `.python-version`) and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-dev
uv run --no-sync python scripts/verify_environment.py
uv run --no-sync python -m pip check
```

`uv.lock` contains exact versions and hashes for all resolved packages across supported platforms. `pyproject.toml` pins direct runtime dependencies; `requirements.txt` is a generated pip-compatible export. To refresh dependencies intentionally, edit direct pins, run `uv lock`, regenerate with `uv export --locked --no-dev --format requirements-txt --no-emit-project -o requirements.txt`, then audit and test the full lock. Do not update only one of these files.

To run the notebook, install Jupyter separately in a notebook-capable environment (it is an authoring tool, not a model runtime dependency), then launch from the repository root:

```bash
uv pip install jupyter
uv run --no-sync jupyter notebook Fraud_Detection_IEEE.ipynb
```

## Tests and security audit

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m compileall -q fraud_detection.py scripts tests
uvx pip-audit -r requirements.txt
```

The dataset-free tests exercise the actual validation and preprocessing module through XGBoost fit, predict, predict-proba, and held-out metrics. They also check notebook JSON and Python syntax, saved-output provenance, the data contract, and direct dependency coverage. CI installs only the locked runtime environment, checks dependency consistency, runs the tests and syntax compilation, and audits the resolved requirements. It does not train on the unavailable competition data.

## Limitations

- No competition data is included or downloaded. A full-data training run and corrected full-data metrics remain unverified.
- The archived historical score is not evidence for the corrected preprocessing.
- Identity-table enrichment, threshold optimization, and Kaggle submission upload are outside scope.
- The baseline is not evidence of deployment performance.

## Author

Atabak Nikouseresht · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
