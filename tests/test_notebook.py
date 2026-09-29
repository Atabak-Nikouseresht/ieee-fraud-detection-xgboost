import ast
import json
import re
import unittest
from pathlib import Path


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

    def test_validation_score_is_traceable_to_saved_notebook_output(self):
        self.assertRegex(self.outputs, r"Validation ROC-AUC:\s*0\.9421992865")
        self.assertIn("0.9422", self.readme)
        self.assertIn("local split result", self.readme)

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


if __name__ == "__main__":
    unittest.main()
