# IEEE-CIS Fraud Detection — XGBoost baseline

[![CI](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Atabak-Nikouseresht/ieee-fraud-detection-xgboost/actions/workflows/ci.yml)

A notebook-based fraud-classification study using the IEEE-CIS transaction data. It demonstrates a stratified train/validation split, training-only categorical preprocessing, XGBoost training, and held-out evaluation using ROC-AUC, average precision, threshold-based precision/recall/F1, and a confusion matrix.

> **Scope:** This repository contains a notebook and dependency list, not a packaged application. The historical saved validation ROC-AUC of `0.9421992865326172` was produced by the previous preprocessing workflow; its saved notebook outputs were cleared and it is not a result of the corrected methodology. No corrected validation score or Kaggle leaderboard score is claimed; a fresh run requires the original competition files and compatible package versions.

## What is included

- `Fraud_Detection_IEEE.ipynb` — data preparation, model fitting, validation, and submission-file workflow.
- `requirements.txt` — Python package requirements.

## Method at a glance

The notebook reads `train_transaction.csv` and `test_transaction.csv` from the working directory, separates the `isFraud` target, and uses an 80/20 stratified split with `random_state=42`. Missing numerical values are imputed with training-partition medians; missing categorical values are imputed with training-partition most-frequent values and then ordinal encoded by a preprocessor fitted only on the training partition. Unseen validation/test values map to `-1`. This fixed-width encoding avoids expanding the matrix by category cardinality; integer codes are compact tree inputs and do not represent a natural or economic ordering. The XGBoost model uses 500 estimators, depth 6, learning rate 0.05, row and column subsampling of 0.8, and ROC-AUC as its evaluation metric.

ROC-AUC, average precision, and threshold-based precision/recall/F1 plus confusion matrix are computed on the held-out validation split only. The 0.5 threshold is descriptive, not optimized for a real operating cost. These are not Kaggle leaderboard scores or evidence of deployment performance. No corrected metric is stated until the repaired notebook is run against the competition data.

## Reproduce the notebook

1. Obtain the transaction data through the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and follow its access and use terms. The data is not redistributed here.
2. From the repository root, place `train_transaction.csv` and `test_transaction.csv` beside the notebook.
3. Create an environment and install the listed requirements:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install jupyter
   ```

4. Launch Jupyter and run the notebook:

   ```bash
   jupyter notebook Fraud_Detection_IEEE.ipynb
   ```

Run from the repository root so the notebook's relative CSV paths resolve. The final cells create predictions for the competition test set; a Kaggle submission requires the competition's prescribed format and a separate upload.

## Tests and CI

From the repository root, run the dataset-free checks:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same notebook-structure, syntax, data-path, dependency-name, and saved-output-provenance checks on pushes, pull requests, and manual dispatch. It does not retrain the model without the competition data.

## Limitations

- The competition data is not included, so a clean-environment training rerun has not been verified from this repository alone.
- The validation ROC-AUC has not been recomputed after the preprocessing repair; no score is presented as current evidence.
- The notebook's original package versions were not recorded, and direct dependencies remain unpinned.
- The reported model configuration is a baseline, not evidence of deployment performance.

## Author

Atabak Nikouseresht · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
