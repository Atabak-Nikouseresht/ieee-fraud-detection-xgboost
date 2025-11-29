# IEEE-CIS Fraud Detection – XGBoost Baseline (AUC: 0.929 Public, 0.906 Private)

This repository contains a clean and reproducible baseline solution for the 
IEEE-CIS Fraud Detection competition on Kaggle.  
The goal is to classify online transactions as fraudulent or legitimate using 
high-dimensional, highly imbalanced data.

---

## Results

| Metric        | Score  |
|---------------|--------|
| Public AUC    | **0.929** |
| Private AUC   | **0.906** |

This performance was achieved using a tuned XGBoost model trained on transaction-level data.

---

## What This Project Includes
- Loading and preprocessing `train_transaction.csv` and `test_transaction.csv`
- Handling imbalanced data
- Categorical encoding using pandas
- Train/validation split with `train_test_split`
- Training an optimized XGBoost classifier
- Evaluating model performance with ROC-AUC
- Generating a Kaggle-ready `submission.csv`

---

## Repository Structure

Fraud_Detection_IEEE.ipynb # Main notebook (end-to-end workflow)
requirements.txt # Main dependencies
README.md # This file


The dataset is **not included** due to Kaggle terms.  
Please download it manually from the competition page.

---

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt


The dataset is **not included** due to Kaggle terms.  
Please download it manually from the competition page.

---

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Open the notebook:
```
jupyter notebook Fraud_Detection_IEEE.ipynb
```
3. Place the downloaded Kaggle CSV files in the same directory as the notebook:
```
train_transaction.csv
test_transaction.csv
```
4. Run all notebook cells to reproduce training, AUC results, and submission file.

## Model Summary

Algorithm: XGBoost Classifier

Key hyperparameters:

n_estimators=500

max_depth=6

learning_rate=0.05

subsample=0.8

colsample_bytree=0.8

eval_metric="auc"

Evaluation:

Validation AUC around 0.94

Kaggle Leaderboard:

Public: 0.929

Private: 0.906

Author

Atabak Nikouseresht

FinTech & Risk Analytics

GitHub: github.com/Atabak-Nikouseresht

LinkedIn: linkedin.com/in/atabak-nikouseresht

Email: atabak.nikouseresht@gmail.com
