"""Shared data validation and preprocessing for the IEEE-CIS notebook."""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class CompetitionFrames:
    train_ids: pd.Series
    test_ids: pd.Series
    target: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    feature_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]


def _validate_columns(frame: pd.DataFrame, label: str) -> None:
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{label} contains duplicate column names: {duplicates}")


def _validate_ids(values: pd.Series, label: str) -> None:
    if values.isna().any():
        raise ValueError(f"{label} contains missing TransactionID values")
    if values.duplicated().any():
        raise ValueError(f"{label} TransactionID values must be unique")


def validate_competition_frames(train: pd.DataFrame, test: pd.DataFrame) -> CompetitionFrames:
    """Validate competition train/test frames and return aligned model inputs."""
    _validate_columns(train, "train_transaction.csv")
    _validate_columns(test, "test_transaction.csv")
    for label, frame in (("train_transaction.csv", train), ("test_transaction.csv", test)):
        if "TransactionID" not in frame.columns:
            raise ValueError(f"{label} is missing required TransactionID column")
        _validate_ids(frame["TransactionID"], label)
    if "isFraud" not in train.columns:
        raise ValueError("train_transaction.csv is missing required isFraud target")
    if "isFraud" in test.columns:
        raise ValueError("test_transaction.csv must not contain isFraud target")

    target = train["isFraud"]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("isFraud target must contain only non-missing 0 and 1 values")

    feature_columns = tuple(column for column in train.columns if column not in {"TransactionID", "isFraud"})
    test_feature_columns = tuple(column for column in test.columns if column != "TransactionID")
    if set(feature_columns) != set(test_feature_columns):
        missing = sorted(set(feature_columns) - set(test_feature_columns))
        unexpected = sorted(set(test_feature_columns) - set(feature_columns))
        raise ValueError(f"Train/test feature columns do not match (missing={missing}, unexpected={unexpected})")
    # Match training order even when the CSV columns are arranged differently.
    train_features = train.loc[:, feature_columns].copy()
    test_features = test.loc[:, feature_columns].copy()
    categorical = tuple(train_features.select_dtypes(include=["object", "category", "string"]).columns)
    return CompetitionFrames(
        train_ids=train["TransactionID"].copy(),
        test_ids=test["TransactionID"].copy(),
        target=target.copy(),
        train_features=train_features,
        test_features=test_features,
        feature_columns=feature_columns,
        categorical_columns=categorical,
    )


def build_preprocessor(feature_columns, categorical_columns):
    """Construct the training-fitted imputation and fixed-width encoding pipeline."""
    categorical_columns = list(categorical_columns)
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")
