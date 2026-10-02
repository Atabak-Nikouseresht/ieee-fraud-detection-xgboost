import unittest

import pandas as pd

from fraud_detection import build_preprocessor, validate_competition_frames


class CompetitionDataContractTests(unittest.TestCase):
    def valid_frames(self):
        train = pd.DataFrame({
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 1, 0, 1],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "ProductCD": ["W", "C", "W", "R"],
        })
        test = pd.DataFrame({
            "TransactionID": [5, 6],
            "TransactionAmt": [50.0, 60.0],
            "ProductCD": ["W", "new"],
        })
        return train, test

    def test_valid_frames_return_ids_target_and_feature_frames(self):
        train, test = self.valid_frames()
        contract = validate_competition_frames(train, test)
        self.assertEqual(contract.train_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(contract.test_ids.tolist(), [5, 6])
        self.assertEqual(contract.target.tolist(), [0, 1, 0, 1])
        self.assertEqual(contract.feature_columns, ("TransactionAmt", "ProductCD"))

    def test_rejects_missing_target_and_required_ids(self):
        train, test = self.valid_frames()
        with self.assertRaisesRegex(ValueError, "isFraud"):
            validate_competition_frames(train.drop(columns="isFraud"), test)
        with self.assertRaisesRegex(ValueError, "TransactionID"):
            validate_competition_frames(train, test.drop(columns="TransactionID"))

    def test_rejects_invalid_target_and_duplicate_or_null_ids(self):
        train, test = self.valid_frames()
        invalid_target = train.copy()
        invalid_target.loc[0, "isFraud"] = 2
        with self.assertRaisesRegex(ValueError, "0 and 1"):
            validate_competition_frames(invalid_target, test)
        duplicate_ids = train.copy()
        duplicate_ids.loc[1, "TransactionID"] = 1
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_competition_frames(duplicate_ids, test)
        null_ids = test.copy()
        null_ids.loc[0, "TransactionID"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_competition_frames(train, null_ids)

    def test_rejects_feature_schema_mismatch_and_duplicate_columns(self):
        train, test = self.valid_frames()
        with self.assertRaisesRegex(ValueError, "feature columns"):
            validate_competition_frames(train, test.rename(columns={"ProductCD": "other"}))
        duplicate_columns = pd.DataFrame([[1, 0, 1, 2]], columns=["TransactionID", "isFraud", "x", "x"])
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            validate_competition_frames(duplicate_columns, test)

    def test_production_preprocessor_handles_missing_and_unseen_categories(self):
        train, test = self.valid_frames()
        contract = validate_competition_frames(train, test)
        features = build_preprocessor(contract.feature_columns, contract.categorical_columns)
        transformed_train = features.fit_transform(contract.train_features)
        transformed_test = features.transform(contract.test_features)
        self.assertEqual(transformed_train.shape, (4, 2))
        self.assertEqual(transformed_test.shape, (2, 2))
        self.assertEqual(transformed_test[1, 1], -1)


if __name__ == "__main__":
    unittest.main()
