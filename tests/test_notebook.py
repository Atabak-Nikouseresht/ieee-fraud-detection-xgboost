import ast
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
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
        self.assertIn('results/current_evaluation.json', self.sources)
        self.assertIn('validate_result(result)', self.sources)
        self.assertNotIn('pd.read_csv', self.sources)
        self.assertIn('outside the repository', self.readme)
        self.assertFalse((REPOSITORY / "train_transaction.csv").exists())
        self.assertFalse((REPOSITORY / "test_transaction.csv").exists())
        self.assertFalse((REPOSITORY / "train_identity.csv").exists())
        self.assertFalse((REPOSITORY / "test_identity.csv").exists())

    def test_stale_validation_output_is_not_presented_as_current_evidence(self):
        result = json.loads((REPOSITORY / 'results/current_evaluation.json').read_text())
        self.assertNotIn('0.9421992865326172', self.outputs)
        for metrics in result['metrics'].values():
            self.assertIn(f"{metrics['roc_auc']:.6f}", self.outputs)
            self.assertIn(f"{metrics['average_precision']:.6f}", self.outputs)
        self.assertIn("historical ROC-AUC", self.readme)
        self.assertIn("previous preprocessing workflow", self.readme)
        self.assertIn("Current corrected evaluation", self.readme)

    def test_readme_does_not_claim_competition_leaderboard_score(self):
        self.assertNotRegex(self.readme, re.compile(r"(?:public|private)\s+(?:leaderboard\s+)?score\s*[:=]\s*\d", re.I))

    def test_requirements_cover_direct_external_imports(self):
        requirements = {
            line.split("==", 1)[0].split(";", 1)[0].strip().lower().replace("_", "-")
            for line in (REPOSITORY / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }
        mapping = {
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "xgboost": "xgboost",
        }
        for module, distribution in mapping.items():
            with self.subTest(module=module):
                self.assertIn(distribution, requirements)

    def test_training_uses_train_fitted_categorical_preprocessing(self):
        runner = (REPOSITORY / 'evaluation.py').read_text(encoding='utf-8')
        self.assertIn('preprocessor.fit_transform(features.iloc[a])', runner)
        self.assertIn('preprocessor.transform(features.iloc[b])', runner)
        self.assertIn('preprocessor.transform(features.iloc[c])', runner)
        self.assertNotIn('.fit(', self.sources)
        production = (REPOSITORY / "fraud_detection.py").read_text(encoding="utf-8")
        self.assertIn('handle_unknown="use_encoded_value"', production)
        self.assertIn("unknown_value=-1", production)
        self.assertNotIn(".cat.codes", self.sources + production)

    def test_production_pipeline_smoke_reaches_xgboost_predictions_and_metrics(self):
        from fraud_detection import build_preprocessor, validate_competition_frames

        train = pd.DataFrame({
            "TransactionID": range(8),
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1],
            "amount": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "kind": ["A", "B", "A", None, "C", "B", "A", "C"],
        })
        test = pd.DataFrame({"TransactionID": [8, 9], "amount": [9.0, 10.0], "kind": ["new", "A"]})
        data = validate_competition_frames(train, test)
        X_train, X_val, y_train, y_val = train_test_split(
            data.train_features, data.target, test_size=0.25,
            stratify=data.target, random_state=42,
        )
        preprocessor = build_preprocessor(data.feature_columns, data.categorical_columns)
        train_matrix = preprocessor.fit_transform(X_train)
        validation_matrix = preprocessor.transform(X_val)
        test_matrix = preprocessor.transform(data.test_features)
        model = XGBClassifier(
            n_estimators=5, max_depth=2, learning_rate=0.1, n_jobs=1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        model.fit(train_matrix, y_train)
        labels = model.predict(validation_matrix)
        probabilities = model.predict_proba(validation_matrix)[:, 1]
        self.assertEqual(labels.shape, (len(y_val),))
        self.assertEqual(probabilities.shape, (len(y_val),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.isfinite(roc_auc_score(y_val, probabilities)))
        self.assertTrue(np.isfinite(average_precision_score(y_val, probabilities)))
        self.assertEqual(model.predict_proba(test_matrix).shape, (len(test), 2))

    def test_split_is_deterministic_and_metrics_are_contextualized(self):
        runner = (REPOSITORY / 'evaluation.py').read_text()
        self.assertIn('random_state=42', runner)
        self.assertIn('strict', self.readme.lower())
        self.assertIn('temporal', self.readme.lower())
        self.assertIn('review_budgets', self.sources)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("validation", markdown.lower())
        self.assertIn("ROC-AUC", markdown)

    def test_early_stopping_callback_is_not_imported_unused(self):
        self.assertNotIn("EarlyStopping", self.sources)


if __name__ == "__main__":
    unittest.main()
