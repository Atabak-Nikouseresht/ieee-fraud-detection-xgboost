import ast
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "Fraud_Detection_IEEE.ipynb"
README = REPOSITORY / "README.md"


def output_text(notebook):
    chunks = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            chunks.extend(output.get("text", []))
            for value in output.get("data", {}).values():
                if isinstance(value, list):
                    chunks.extend(value)
                elif isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


class NotebookPortfolioChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.readme = README.read_text(encoding="utf-8")
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        cls.sources = "\n".join(cls.code_cells)
        cls.outputs = output_text(cls.notebook)

    def test_notebook_json_and_code_cells_are_valid(self):
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertTrue(self.code_cells)
        for number, source in enumerate(self.code_cells):
            ast.parse(source, filename=f"notebook-cell-{number}")

    def test_documented_dataset_inputs_match_notebook(self):
        self.assertIn('pd.read_csv("train_transaction.csv")', self.sources)
        self.assertIn('pd.read_csv("test_transaction.csv")', self.sources)
        self.assertIn("place `train_transaction.csv` and `test_transaction.csv` beside the notebook", self.readme)
        self.assertFalse((REPOSITORY / "train_transaction.csv").exists())
        self.assertFalse((REPOSITORY / "test_transaction.csv").exists())

    def test_stale_validation_output_is_not_presented_as_current_evidence(self):
        self.assertEqual(self.outputs.strip(), "")
        self.assertIn("historical saved validation ROC-AUC", self.readme)
        self.assertIn("previous preprocessing workflow", self.readme)
        self.assertIn("No corrected validation score", self.readme)

    def test_readme_does_not_claim_competition_leaderboard_score(self):
        self.assertNotRegex(self.readme, re.compile(r"(?:public|private)\s+(?:leaderboard\s+)?score\s*[:=]\s*\d", re.I))

    def test_requirements_cover_direct_external_imports(self):
        requirements = {
            line.split("#", 1)[0].strip().lower().replace("_", "-")
            for line in (REPOSITORY / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        mapping = {
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "xgboost": "xgboost",
        }
        for module, distribution in mapping.items():
            with self.subTest(module=module):
                self.assertIn(distribution, requirements)

    def test_training_uses_train_fitted_categorical_preprocessing(self):
        self.assertIn("preprocessor.fit_transform(X_train)", self.sources)
        self.assertIn("preprocessor.transform(X_val)", self.sources)
        self.assertIn("preprocessor.transform(test_features)", self.sources)
        self.assertIn("handle_unknown=\"use_encoded_value\"", self.sources)
        self.assertIn("unknown_value=-1", self.sources)
        self.assertNotIn(".cat.codes", self.sources)

    def test_split_is_deterministic_and_metrics_are_contextualized(self):
        self.assertIn("random_state=42", self.sources)
        self.assertIn("roc_auc_score(y_val, val_preds)", self.sources)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("validation", markdown.lower())
        self.assertIn("ROC-AUC", markdown)

    def test_train_fitted_ordinal_preprocessor_runs_with_real_xgboost(self):
        preprocessing_source = next(
            source for source in self.code_cells
            if "preprocessor = ColumnTransformer(" in source
        )
        X_train = pd.DataFrame({
            "amount": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
            "kind": ["A", "B", "A", None, "C", "B"],
        })
        X_val = pd.DataFrame({"amount": [20.0], "kind": ["never-seen"]})
        test_features = pd.DataFrame({"amount": [40.0], "kind": ["also-new"]})
        namespace = {
            "np": np,
            "X_train": X_train,
            "X_val": X_val,
            "test_features": test_features,
            "ColumnTransformer": ColumnTransformer,
            "SimpleImputer": SimpleImputer,
            "Pipeline": Pipeline,
            "OrdinalEncoder": OrdinalEncoder,
        }

        exec(compile(preprocessing_source, str(NOTEBOOK), "exec"), namespace)

        encoded_train = namespace["X_train_processed"]
        encoded_validation = namespace["X_val_processed"]
        encoded_test = namespace["test_processed"]
        self.assertIsInstance(encoded_train, np.ndarray)
        self.assertEqual(encoded_train.shape, (len(X_train), len(X_train.columns)))
        self.assertEqual(encoded_train.shape[1], encoded_validation.shape[1])
        self.assertEqual(encoded_train.shape[1], encoded_test.shape[1])
        encoder = namespace["preprocessor"].named_transformers_["categorical"].named_steps["ordinal"]
        self.assertEqual(encoder.categories_[0].tolist(), ["A", "B", "C"])
        self.assertEqual(encoded_validation[0, 1], -1)

        model = XGBClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=0.1,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(encoded_train, [0, 1, 0, 1, 0, 1])
        self.assertEqual(model.predict(encoded_validation).shape, (1,))
        probabilities = model.predict_proba(encoded_test)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_early_stopping_callback_is_not_imported_unused(self):
        self.assertNotIn("EarlyStopping", self.sources)


if __name__ == "__main__":
    unittest.main()
