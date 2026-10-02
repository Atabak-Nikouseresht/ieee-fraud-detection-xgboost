"""Keep current portfolio claims tied to aggregate empirical evidence."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class CurrentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / 'results/current_evaluation.json').read_text())
        self.readme = (ROOT / 'README.md').read_text()
        self.notebook = json.loads((ROOT / 'Fraud_Detection_IEEE.ipynb').read_text())

    def test_readme_current_numbers_are_derived_from_evidence(self):
        from evaluation import validate_result
        validate_result(self.record)
        self.assertEqual(self.record['status'], 'completed')
        self.assertEqual(self.record['dataset_kind'], 'ieee')
        self.assertIn('## Current corrected evaluation', self.readme)
        for model, label in [('logistic_regression', 'Logistic Regression'), ('xgboost', 'XGBoost')]:
            m = self.record['metrics'][model]
            expected = f"| {label} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} |"
            self.assertIn(expected, self.readme)
        review = self.record['metrics']['xgboost']['review_budgets'][1]
        self.assertIn(f"{review['alerts']:,} alerts", self.readme)
        self.assertIn(f"{review['tp']:,} frauds", self.readme)
        self.assertIn('0.9421992865326172', self.readme)
        self.assertNotIn('full-data metrics remain unverified', self.readme)
        self.assertIn('6 decimal places', self.readme)

    def test_notebook_uses_current_artifact_without_raw_data_or_retraining(self):
        code='\n'.join(''.join(c.get('source',[])) for c in self.notebook['cells'] if c['cell_type']=='code')
        self.assertIn('validate_result(result)', code)
        self.assertIn('results/current_evaluation.json', code)
        self.assertNotIn('pd.read_csv',code)
        self.assertNotIn('.fit(',code)
        self.assertNotIn('submission.to_csv',code)
        outputs='\n'.join(''.join(o.get('text',[])) for c in self.notebook['cells'] for o in c.get('outputs',[]))
        for m in self.record['metrics'].values():
            self.assertIn(f"{m['roc_auc']:.6f}",outputs)
            self.assertIn(f"{m['average_precision']:.6f}",outputs)
        self.assertNotIn('0.9421992865326172',outputs)

    def test_execution_source_provenance_matches_normalized_git_content(self):
        import hashlib, subprocess
        r=self.record['provenance']
        self.assertTrue(r['worktree_clean'])
        for name, digest in r['source_sha256'].items():
            execution=subprocess.check_output(['git','show',f"{r['git_sha']}:{name}"],cwd=ROOT)
            local=(ROOT/name).read_bytes()
            self.assertEqual(execution.replace(b'\r\n',b'\n'),local.replace(b'\r\n',b'\n'))
            normalized=execution.replace(b'\r\n',b'\n')
            possible_bytes=(normalized,normalized.replace(b'\n',b'\r\n'))
            self.assertIn(digest,{hashlib.sha256(value).hexdigest() for value in possible_bytes})

if __name__=='__main__': unittest.main()
