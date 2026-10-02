"""Protect publication boundaries without requiring competition data."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_results.py"
RESTRICTED_NAMES = {
    "train_transaction.csv", "test_transaction.csv", "train_identity.csv",
    "test_identity.csv", "sample_submission.csv", "submission.csv",
}


class PublicationBoundaryTests(unittest.TestCase):
    def test_result_validator_reports_missing_file_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(missing)],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Result validation failed", result.stderr)
        self.assertNotIn("schema verified", result.stdout)

    def test_result_validator_rejects_malformed_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"status": "completed", "roc_auc": 1.0}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(path)],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Result validation failed", result.stderr)

    def test_git_tracks_no_competition_dataset_or_submission(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT,
        ).decode("utf-8").split("\0")
        forbidden = [name for name in tracked if Path(name).name in RESTRICTED_NAMES]
        self.assertEqual(forbidden, [], "Restricted/row-level competition files must not be published")


if __name__ == "__main__":
    unittest.main()
