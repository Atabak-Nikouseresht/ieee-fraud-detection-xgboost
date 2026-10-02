"""Narrow executable evaluation contract tests; all fixtures are synthetic."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
import copy
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation as ev


class EvaluationTests(unittest.TestCase):
    def test_temporal_boundaries_do_not_split_ties(self):
        frame = pd.DataFrame({'TransactionDT': np.repeat(np.arange(20), 2), 'isFraud': [0, 1] * 20})
        parts = ev.make_split(frame, 'temporal')
        self.assertEqual([len(p) for p in parts], [24, 8, 8])
        for a, b in zip(parts, parts[1:]):
            self.assertLess(frame.iloc[a].TransactionDT.max(), frame.iloc[b].TransactionDT.min())
        with self.assertRaises(ValueError):
            ev.make_split(frame.assign(TransactionDT=1), 'temporal')
        with self.assertRaises(ValueError):
            ev.make_split(frame.assign(isFraud=0), 'temporal')

    def test_metrics_reliability_and_stable_capacity(self):
        result = ev.metrics(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]))
        self.assertEqual(result['roc_auc'], 1.0)
        self.assertAlmostEqual(result['brier'], .025)
        self.assertEqual(result['confusion_matrix'], [[2, 0], [0, 2]])
        self.assertEqual(sum(b['count'] for b in result['reliability']['bins']), 4)
        ties = ev.review_budgets(np.array([0, 1, 1, 0]), np.full(4, .5))
        self.assertEqual(ties[0]['alerts'], 1)
        self.assertEqual(ties[0]['tp'], 0)
        self.assertEqual(ties[0]['threshold'], .5)
        indexed = ev.review_budgets(np.array([0, 1, 1, 0]), np.full(4, .5), row_indices=np.array([5, 1, 9, 3]))
        self.assertEqual(indexed[0]['tp'], 1)

    def fixture(self, directory):
        rng = np.random.default_rng(42)
        n = 300
        frame = pd.DataFrame({'TransactionID': np.arange(1000, 1000+n),
                              'TransactionDT': np.arange(n), 'isFraud': np.arange(n) % 2,
                              'amount': rng.normal(size=n), 'ProductCD': ['W', 'C', 'R'] * 100})
        frame.to_csv(Path(directory) / 'train_transaction.csv', index=False)
        test = frame.drop(columns='isFraud').iloc[:20].copy()
        test.TransactionID += 10000
        test.TransactionDT += 1000
        test.to_csv(Path(directory) / 'test_transaction.csv', index=False)
        return frame

    def test_real_xgboost_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            self.fixture(d)
            record = ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1, bootstrap_replicates=3)
            ev.validate_result(record)
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['dataset_kind'], 'synthetic')
            self.assertEqual(record['row_counts']['train_csv'], 300)
            self.assertEqual(record['dataset_summary']['predictor_count'], 3)
            self.assertEqual(record['dataset_summary']['target_prevalence'], .5)
            self.assertEqual(record['dataset_summary']['target'], 'isFraud')
            self.assertEqual(set(record['metrics']), {'logistic_regression', 'xgboost'})
            self.assertEqual(record['provenance']['versions']['scikit-learn'], '1.9.1')
            self.assertEqual(len(record['provenance']['source_sha256']['evaluation.py']), 64)
            text = json.dumps(record, allow_nan=False)
            for forbidden in ('"predictions"', '"train_ids"', '"raw_rows"'):
                self.assertNotIn(forbidden, text)
            self.assertTrue(record['calibration']['decision_before_final_metrics'])
            self.assertEqual(record['bootstrap']['replicates_completed'], 3)
            with self.assertRaises(ValueError):
                ev.run_evaluation(Path(d))

    def test_missing_data_blocked_record_and_output_policy(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            out = Path(d) / 'blocked.json'
            self.assertEqual(ev.main(['--data-dir', d, '--output', str(out)]), 2)
            record = json.loads(out.read_text())
            ev.validate_result(record)
            self.assertEqual(record['status'], 'blocked')
            self.assertIsNone(record['metrics'])
            self.assertIsNone(record['row_counts'])
            self.assertEqual(record['publication_status'], 'BLOCKED ON DATA')
            with self.assertRaises(ValueError):
                ev.validate_result({**record, 'metrics': {'fake': .9}})
            ev.write_result(out, record, overwrite=True)
            with self.assertRaises(FileExistsError):
                ev.write_result(out, record)

    def test_known_leakage_fails_before_model_fit(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            frame = self.fixture(d)
            for column, values in [('copy_y', frame.isFraud), ('copy_id', frame.TransactionID), ('post_outcome_score', np.linspace(0, 1, len(frame)))]:
                leaked = frame.assign(**{column: values})
                leaked.to_csv(Path(d) / ev.FILES[0], index=False)
                test = pd.read_csv(Path(d) / ev.FILES[1])
                test[column] = 0
                test.to_csv(Path(d) / ev.FILES[1], index=False)
                with patch.object(ev.XGBClassifier, 'fit', side_effect=AssertionError('fit forbidden')):
                    with self.assertRaisesRegex(ValueError, 'forbidden'):
                        ev.run_evaluation(Path(d), dataset_kind='synthetic')
                self.fixture(d)

    def test_chunked_validation_schema_ids_and_target(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            frame = self.fixture(d)
            testpath = Path(d) / ev.FILES[1]
            test = pd.read_csv(testpath)
            test.iloc[::-1, ::-1].to_csv(testpath, index=False)
            loaded, info = ev.load_data(Path(d), chunksize=3)
            self.assertEqual(len(loaded), 300)
            self.assertEqual(info['rows'], 20)
            test.loc[19, 'TransactionID'] = test.loc[0, 'TransactionID']
            test.to_csv(testpath, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                ev.load_data(Path(d), chunksize=3)
            self.fixture(d)
            test = pd.read_csv(testpath)
            test.loc[0, 'TransactionID'] = frame.TransactionID.iloc[0]
            test.to_csv(testpath, index=False)
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                ev.load_data(Path(d), chunksize=3)
            self.fixture(d)
            frame.loc[0, 'isFraud'] = 2
            frame.to_csv(Path(d) / ev.FILES[0], index=False)
            with self.assertRaisesRegex(ValueError, 'binary'):
                ev.load_data(Path(d))

    def test_sigmoid_frozen_estimator_branch_real_fit(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            self.fixture(d)
            # Test-only threshold controls force the API path on a small fixture.
            with patch.object(ev, 'CALIBRATION_MIN_ROWS', 10), patch.object(ev, 'CALIBRATION_MIN_CLASS', 1):
                record = ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)
            self.assertIn('FrozenEstimator', record['calibration']['method'])
            comparison = record['calibration']['final_comparison']
            self.assertEqual(set(comparison), {'raw_xgboost', 'selected_xgboost'})
            self.assertEqual(comparison['selected_xgboost'], record['metrics']['xgboost'])
            self.assertIn('roc_auc', comparison['raw_xgboost'])
            self.assertIn('brier', comparison['raw_xgboost'])
            self.assertLessEqual(record['calibration']['decision_at_utc'], record['final_metrics_opened_at_utc'])
            self.assertEqual(record['bootstrap']['replicates_completed'], 0)
            self.assertIsNone(record['bootstrap']['intervals'])

    def test_schema_rejects_corrupt_completed_evidence(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            self.fixture(d)
            record = ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)
            for change in ('auc', 'empty', 'hash', 'timestamp', 'raw', 'partitions'):
                bad = copy.deepcopy(record)
                if change == 'auc':
                    bad['metrics']['xgboost']['roc_auc'] = 2
                elif change == 'empty':
                    bad['metrics']['xgboost'] = {}
                elif change == 'hash':
                    bad['provenance']['source_sha256']['evaluation.py'] = 'wrong'
                elif change == 'timestamp':
                    bad['final_metrics_opened_at_utc'] = '2000-01-01T00:00:00+00:00'
                elif change == 'raw':
                    bad['raw_rows'] = [1]
                else:
                    bad['split']['partitions'] = {}
                with self.subTest(change=change), self.assertRaises(ValueError):
                    ev.validate_result(bad)
            out = Path(d) / 'completed.json'
            ev.write_result(out, record)
            with self.assertRaises(FileExistsError):
                ev.write_result(out, record, overwrite=True)
            proc = subprocess.run([sys.executable, str(ev.ROOT / 'evaluation.py'), '--data-dir', d,
                                   '--output', str(Path(d) / 'cli.json'), '--dataset-kind', 'synthetic',
                                   '--split', 'stratified', '--n-jobs', '1'], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            ev.validate_result(json.loads((Path(d) / 'cli.json').read_text()))

    def test_training_only_preprocessor_and_baseline_convergence_gate(self):
        from sklearn.exceptions import ConvergenceWarning
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            frame = self.fixture(d)
            frame.loc[180:, 'ProductCD'] = 'unseen'
            frame.to_csv(Path(d) / ev.FILES[0], index=False)
            built = []
            original = ev.build_preprocessor
            def tracked(*args):
                result = original(*args)
                built.append(result)
                return result
            with patch.object(ev, 'build_preprocessor', side_effect=tracked):
                ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)
            encoder = built[0].named_transformers_['categorical'].named_steps['ordinal']
            self.assertNotIn('unseen', encoder.categories_[0])
            def unconverged(*args, **kwargs):
                import warnings
                warnings.warn('synthetic convergence failure', ConvergenceWarning)
            with patch.object(ev.LogisticRegression, 'fit', side_effect=unconverged):
                with self.assertRaises(ConvergenceWarning):
                    ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)

    def test_publication_lock_prevents_concurrent_replacement(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            output = Path(d) / 'record.json'
            record = ev.blocked_result(Path(d))
            ev.write_result(output, record)
            before = output.read_bytes()
            lock = output.with_name(output.name + '.lock')
            lock.write_text('another writer')
            with self.assertRaises(FileExistsError):
                ev.write_result(output, record, overwrite=True)
            self.assertEqual(output.read_bytes(), before)
            lock.unlink()
            ev.write_result(output, record, overwrite=True)
            self.assertFalse(lock.exists())

    def test_schema_requires_raw_calibrated_comparison(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            self.fixture(d)
            record = ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)
            bad = copy.deepcopy(record)
            del bad['calibration']['final_comparison']
            with self.assertRaises(ValueError):
                ev.validate_result(bad)
            bad = copy.deepcopy(record)
            bad['calibration']['final_comparison']['raw_xgboost']['roc_auc'] = 2
            with self.assertRaises(ValueError):
                ev.validate_result(bad)

    def test_schema_malformed_payload_always_valueerror(self):
        with tempfile.TemporaryDirectory(dir=str(ev.ROOT.parent)) as d:
            record = ev.blocked_result(Path(d))
            for value in (None, [], {}, {**record, 'provenance': []}, {**record, 'created_at_utc': None}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    ev.validate_result(value)
            self.fixture(d)
            completed = ev.run_evaluation(Path(d), dataset_kind='synthetic', n_jobs=1)
            completed['metrics']['xgboost']['confusion_matrix'] = [[30, 30], [0, 0]]
            completed['metrics']['xgboost']['prevalence'] = 0
            completed['metrics']['xgboost']['reliability']['bins'] = [{'count': 60, 'positives': 0, 'probability_mean': 0, 'fraud_fraction': 0}]
            completed['metrics']['xgboost']['reliability']['ece'] = 0
            with self.assertRaises(ValueError):
                ev.validate_result(completed)


if __name__ == '__main__':
    unittest.main()
