# IEEE-CIS Fraud Detection — XGBoost baseline

A notebook-based fraud-classification study using the IEEE-CIS transaction data. It demonstrates categorical preprocessing, a stratified train/validation split, XGBoost training, and ROC-AUC evaluation on an imbalanced dataset.

> **Scope:** This repository contains a notebook and dependency list, not a packaged application. Kaggle competition leaderboard scores previously stated in this README are omitted because no submission record or leaderboard evidence is included here. The committed notebook output records a held-out validation ROC-AUC of **0.9422**; this is a local split result, not a Kaggle leaderboard score.

## What is included

- `Fraud_Detection_IEEE.ipynb` — data preparation, model fitting, validation, and submission-file workflow.
- `requirements.txt` — Python package requirements.

## Method at a glance

The notebook reads `train_transaction.csv` and `test_transaction.csv` from the working directory, separates the `isFraud` target, encodes categorical columns, and uses an 80/20 stratified split with `random_state=42`. The XGBoost model uses 500 estimators, depth 6, learning rate 0.05, row and column subsampling of 0.8, and ROC-AUC as its evaluation metric. The saved notebook output reports 472,432 training rows, 118,108 validation rows, and ROC-AUC 0.942199 on the validation split.

The validation result is evidence from the notebook's stored output. Re-running may require the original competition files and compatible package versions; this repository does not include the dataset or a dependency lockfile.

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

## Limitations

- The competition data is not included, so a clean-environment rerun has not been verified from this repository alone.
- The recorded validation score is from one random stratified split and should not be interpreted as evidence of deployment performance.
- The README does not claim leaderboard performance without verifiable submission evidence.
- There are no committed tests, CI workflow, or dependency lockfile.

## Author

Atabak Nikouseresht · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
