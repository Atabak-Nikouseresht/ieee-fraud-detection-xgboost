"""Research-only IEEE-CIS evaluation. No download, tuning, or deployment."""
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import warnings

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from fraud_detection import build_preprocessor

ROOT = Path(__file__).resolve().parent
FILES = ('train_transaction.csv', 'test_transaction.csv')
CALIBRATION_MIN_ROWS = 1000
CALIBRATION_MIN_CLASS = 30
CALIBRATION_ECE_FLAG = .02


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(data_dir):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    state = git('status', '--porcelain')
    return {'captured_at_utc': utc_now(), 'git_sha': git('rev-parse', 'HEAD'),
            'worktree_clean': None if state is None else state == '',
            'worktree_dirty': None if state is None else state != '',
            'python': platform.python_version(),
            'versions': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost')},
            'source_sha256': {name: sha256_file(ROOT / name) for name in ('evaluation.py', 'fraud_detection.py')},
            'data_sha256': {name: sha256_file(data_dir / name) if (data_dir / name).is_file() else None for name in FILES}}


def load_data(data_dir, chunksize=100000):
    """Read full train once; inspect test header and only ID/time in chunks."""
    headers = {}
    for name in FILES:
        with open(data_dir / name, newline='', encoding='utf-8-sig') as stream:
            headers[name] = next(csv.reader(stream))
        if len(headers[name]) != len(set(headers[name])):
            raise ValueError('Duplicate CSV column names')
    train_header, test_header = (headers[name] for name in FILES)
    if not {'TransactionID', 'isFraud'} <= set(train_header) or 'TransactionID' not in test_header:
        raise ValueError('Missing required ID or target')
    features = [c for c in train_header if c not in ('TransactionID', 'isFraud')]
    if 'isFraud' in test_header or set(features) != set(test_header) - {'TransactionID'}:
        raise ValueError('Train/test feature schema mismatch')
    frame = pd.read_csv(data_dir / FILES[0])
    if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
        raise ValueError('Invalid or duplicate train TransactionID')
    if frame.isFraud.isna().any() or not frame.isFraud.isin([0, 1]).all() or set(frame.isFraud) != {0, 1}:
        raise ValueError('Target requires both nonmissing binary classes')
    train_ids = set(frame.TransactionID)
    seen, count, tmin, tmax, missing_time = set(), 0, None, None, 0
    selected = ['TransactionID'] + (['TransactionDT'] if 'TransactionDT' in test_header else [])
    with pd.read_csv(data_dir / FILES[1], usecols=selected, chunksize=chunksize) as reader:
        for chunk in reader:
            ids = chunk.TransactionID
            if ids.isna().any() or ids.duplicated().any() or seen.intersection(ids) or train_ids.intersection(ids):
                raise ValueError('Duplicate, missing, or overlapping test TransactionID')
            seen.update(ids)
            count += len(chunk)
            if 'TransactionDT' in chunk:
                times = pd.to_numeric(chunk.TransactionDT, errors='coerce')
                valid = times[np.isfinite(times)]
                missing_time += len(times) - len(valid)
                if len(valid):
                    lo, hi = float(valid.min()), float(valid.max())
                    tmin = lo if tmin is None else min(tmin, lo)
                    tmax = hi if tmax is None else max(tmax, hi)
    if not count:
        raise ValueError('Test CSV must be nonempty')
    return frame, {'rows': count, 'time_min': tmin, 'time_max': tmax,
                   'time_missing_or_nonfinite': missing_time if 'TransactionDT' in test_header else None,
                   'validation': 'header alignment and ID/time-only chunked scan; no competition test labels'}


def leakage_audit(frame):
    features = frame.drop(columns=['TransactionID', 'isFraud'])
    forbidden = []
    suspicious = []
    for name in features:
        col = features[name]
        lower = name.lower()
        if any(token in lower for token in ('target', 'label', 'fraud', 'transactionid', 'prediction', 'post_outcome')):
            suspicious.append(name)
        target_copy = col.equals(frame.isFraud) or (col.notna().all() and np.array_equal(col.to_numpy(), frame.isFraud.to_numpy()))
        id_copy = col.notna().all() and np.array_equal(col.to_numpy(), frame.TransactionID.to_numpy())
        if target_copy or id_copy or lower in ('target', 'label', 'fraud', 'transaction_id', 'transactionid', 'isfraud'):
            forbidden.append(name)
    if forbidden or suspicious:
        raise ValueError('Known forbidden or suspicious target/ID/post-outcome features require investigation: ' + ', '.join(sorted(set(forbidden + suspicious))))
    return {'known_forbidden_features': [], 'suspicious_feature_names': suspicious,
            'feature_timing': 'unknown for anonymized variables; availability at prediction time is not verified',
            'leakage_absence_claim': False, 'duplicate_predictor_rows': int(features.duplicated().sum()),
            'card_proxies_present': [c for c in features if c in [f'card{i}' for i in range(1, 7)]],
            'group_split': 'not executed: card1..6 are unverified proxies, not verified entity IDs'}


def paired_bootstrap(y, baseline, candidate, replicates):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    result = {'requested_replicates': replicates, 'seed': 42, 'replicates_completed': 0,
              'replicates_skipped_single_class': 0, 'intervals': None,
              'interpretation': 'conditional IID row resampling of fixed predictions; not entity/temporal/causal confidence intervals'}
    if not replicates:
        return result
    rng = np.random.default_rng(42)
    samples = {name: [] for name in ('roc_auc_difference', 'average_precision_difference', 'brier_difference')}
    for _ in range(replicates):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            result['replicates_skipped_single_class'] += 1
            continue
        for name, fn in zip(samples, (roc_auc_score, average_precision_score, brier_score_loss)):
            samples[name].append(float(fn(y[idx], candidate[idx]) - fn(y[idx], baseline[idx])))
        result['replicates_completed'] += 1
    result['intervals'] = {name: {'lower': float(np.quantile(values, .025)), 'upper': float(np.quantile(values, .975)),
                                 'level': .95, 'direction': 'xgboost minus logistic_regression'}
                           for name, values in samples.items() if values}
    return result


def run_evaluation(data_dir, split='temporal', n_jobs=4, bootstrap_replicates=0,
                   acknowledge_uncertain_feature_timing=False, dataset_kind='ieee'):
    data_dir = Path(data_dir)
    if dataset_kind not in ('ieee', 'synthetic') or n_jobs < 1 or bootstrap_replicates < 0:
        raise ValueError('Invalid dataset kind or runtime controls')
    if dataset_kind == 'ieee' and not acknowledge_uncertain_feature_timing:
        raise ValueError('Real run requires --acknowledge-uncertain-feature-timing')
    started = time.perf_counter()
    prov = provenance(data_dir)
    frame, test_info = load_data(data_dir)
    audit = leakage_audit(frame)
    audit['uncertain_feature_timing_acknowledged'] = acknowledge_uncertain_feature_timing
    feasibility = {}
    for strategy in ('temporal', 'stratified'):
        try:
            assessed = make_split(frame, strategy)
            feasibility[strategy] = {'feasible': True, 'counts': [len(p) for p in assessed], 'reason': None}
        except ValueError as exc:
            feasibility[strategy] = {'feasible': False, 'counts': None, 'reason': str(exc)}
    a, b, c = make_split(frame, split)
    features = frame.drop(columns=['TransactionID', 'isFraud'])
    categorical = list(features.select_dtypes(include=['object', 'category', 'string']).columns)
    preprocessor = build_preprocessor(features.columns, categorical)
    warning_records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter('error', ConvergenceWarning)
        train_x = preprocessor.fit_transform(features.iloc[a])
        calib_x = preprocessor.transform(features.iloc[b])
        baseline = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=42))
        baseline.fit(train_x, frame.iloc[a].isFraud.to_numpy())
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=.05, subsample=.8,
                            colsample_bytree=.8, tree_method='hist', random_state=42,
                            eval_metric='auc', n_jobs=n_jobs, device='cpu', objective='binary:logistic')
        xgb.fit(train_x, frame.iloc[a].isFraud.to_numpy())
        calib_y = frame.iloc[b].isFraud.to_numpy()
        calib_p = xgb.predict_proba(calib_x)[:, 1]
        calibration = {'criterion': {'ece_greater_than': CALIBRATION_ECE_FLAG, 'minimum_rows': CALIBRATION_MIN_ROWS,
                                    'minimum_per_class': CALIBRATION_MIN_CLASS},
                       'before': metrics(calib_y, calib_p, b), 'method': None,
                       'decision_before_final_metrics': True,
                       'limitations': 'Exploratory ECE flag, not calibration adequacy; sparse bins and severe imbalance can mask error. Brier mixes discrimination and calibration.'}
        counts = np.bincount(calib_y.astype(int), minlength=2)
        sufficient = len(b) >= CALIBRATION_MIN_ROWS and counts.min() >= CALIBRATION_MIN_CLASS
        flag = calibration['before']['reliability']['ece'] > CALIBRATION_ECE_FLAG
        selected_model = xgb
        if sufficient and flag:
            selected_model = CalibratedClassifierCV(FrozenEstimator(xgb), method='sigmoid', n_jobs=n_jobs)
            selected_model.fit(calib_x, calib_y)
            calibration['method'] = 'sigmoid FrozenEstimator; calibration partition only'
            calibration['decision'] = 'fit one predeclared sigmoid'
        else:
            calibration['decision'] = 'no calibration: insufficient samples' if not sufficient else 'no calibration: exploratory flag not exceeded; adequacy not established'
        calibration['decision_at_utc'] = utc_now()
        # No final predictions, metrics, or target-array extraction occurs before this lock.
        final_open = utc_now()
        final_x = preprocessor.transform(features.iloc[c])
        final_y = frame.iloc[c].isFraud.to_numpy()
        baseline_p = baseline.predict_proba(final_x)[:, 1]
        candidate_p = selected_model.predict_proba(final_x)[:, 1]
        results = {'logistic_regression': metrics(final_y, baseline_p, c), 'xgboost': metrics(final_y, candidate_p, c)}
        raw_metrics = results['xgboost'] if selected_model is xgb else metrics(final_y, xgb.predict_proba(final_x)[:, 1], c)
        calibration['final_comparison'] = {'raw_xgboost': raw_metrics, 'selected_xgboost': results['xgboost']}
        warning_records.extend({'category': w.category.__name__, 'message': str(w.message)} for w in caught)
    names = preprocessor.get_feature_names_out().tolist()
    gains = xgb.get_booster().get_score(importance_type='total_gain')
    total_gain = sum(gains.values())
    importance = [{'feature': name, 'total_gain': float(gains.get(f'f{i}', 0)),
                   'normalized_total_gain': float(gains.get(f'f{i}', 0) / total_gain) if total_gain else 0.}
                  for i, name in enumerate(names)]
    parts = {}
    for name, idx in zip(('model_train', 'calibration', 'evaluation'), (a, b, c)):
        times = frame.iloc[idx].TransactionDT if 'TransactionDT' in frame else None
        parts[name] = {'rows': len(idx), 'positive_count': int(frame.iloc[idx].isFraud.sum()),
                       'index_sha256': hashlib.sha256(np.asarray(idx, dtype='<i8').tobytes()).hexdigest(),
                       'time_min': float(times.min()) if times is not None and np.isfinite(times).all() else None,
                       'time_max': float(times.max()) if times is not None and np.isfinite(times).all() else None}
    record = {'schema_version': 1, 'status': 'completed', 'dataset_kind': dataset_kind,
              'publication_status': 'SYNTHETIC CHECK ONLY' if dataset_kind == 'synthetic' else 'FULL EMPIRICAL RUN',
              'created_at_utc': utc_now(), 'provenance': prov,
              'row_counts': {'train_csv': len(frame), 'test_csv': test_info['rows']},
              'dataset_summary': {'name': 'IEEE-CIS Fraud Detection transaction tables' if dataset_kind == 'ieee' else 'Synthetic test fixture',
                                  'source': 'Kaggle competition; authorized local files' if dataset_kind == 'ieee' else 'generated test fixture',
                                  'target': 'isFraud', 'id_field': 'TransactionID', 'predictor_count': len(features.columns),
                                  'transformed_feature_count': len(names), 'positive_count': int(frame.isFraud.sum()),
                                  'negative_count': int(len(frame) - frame.isFraud.sum()), 'target_prevalence': float(frame.isFraud.mean()),
                                  'scope': 'full supplied transaction training CSV, no sampling; identity tables not merged',
                                  'access_conditions': 'User must verify current access/use/publication rules; byte hashes do not establish authorization'},
              'split': {'strategy': split, 'seed': 42, 'requested_fractions': [.6, .2, .2], 'partitions': parts,
                        'feasibility': feasibility, 'default_temporal_pending_actual_time_inspection': True,
                        'time_units': 'TransactionDT native units; no inferred calendar origin',
                        'train_test_time': test_info,
                        'test_strictly_after_train': bool(test_info['time_min'] > frame.TransactionDT.max())
                            if test_info['time_min'] is not None and 'TransactionDT' in frame and np.isfinite(frame.TransactionDT).all() else None},
              'leakage_audit': audit, 'calibration': calibration, 'final_metrics_opened_at_utc': final_open,
              'metrics': results, 'model_params': {'xgboost': {key: ('NaN missing-value sentinel' if isinstance(value, float) and np.isnan(value) else value)
                                                             for key, value in xgb.get_params().items()},
                       'logistic_regression': baseline[-1].get_params()},
              'preprocessing': {'fit_partition': 'model_train only', 'transformed_feature_names': names,
                                'baseline_scaler_fit_partition': 'model_train only',
                                'categorical_geometry': 'ordinal encodings impose arbitrary distances for logistic regression'},
              'importance': {'kind': 'normalized total_gain', 'features': sorted(importance, key=lambda r: -r['total_gain']),
                             'limitations': 'Associational, not causal; feature availability unknown; suspicious names in leakage_audit'},
              'bootstrap': paired_bootstrap(final_y, baseline_p, candidate_p, bootstrap_replicates),
              'warnings': warning_records, 'runtime_seconds': time.perf_counter() - started,
              'limitations': ['No verified entity grouping; repeated-card dependence possible',
                              'Final holdout not used for tuning, thresholds or review-budget selection',
                              'Train CSV loaded fully; label checks and split feasibility inspect labels before final metric lock',
                              'No row-level records, IDs or predictions persisted; competition test outcomes unavailable']}
    validate_result(record)
    return record


def blocked_result(data_dir, dataset_kind='ieee', reason=None):
    return {'schema_version': 1, 'status': 'blocked', 'dataset_kind': dataset_kind,
            'publication_status': 'BLOCKED ON DATA', 'created_at_utc': utc_now(),
            'reason': reason or 'Required local CSV files unavailable; no automatic download',
            'expectations': {'required_files': list(FILES), 'real_run_acknowledgment_required': True,
                             'full_train_no_sampling': True, 'default_split': 'temporal pending TransactionDT inspection'},
            'provenance': provenance(Path(data_dir)), 'row_counts': None, 'metrics': None,
            'split': None, 'calibration': None, 'importance': None, 'bootstrap': None}


def validate_result(record):
    """Public parser contract: every schema/integrity failure is ValueError."""
    try:
        return _validate_result(record)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed result structure: ' + str(exc)) from exc


def _validate_result(record):
    """Fail closed on missing evidence and prohibited row-level payload keys."""
    required = {'schema_version', 'status', 'dataset_kind', 'publication_status', 'created_at_utc',
                'provenance', 'row_counts', 'metrics', 'split', 'calibration', 'importance', 'bootstrap'}
    if not isinstance(record, dict) or not required <= record.keys() or record['schema_version'] != 1:
        raise ValueError('Invalid result schema')
    if record['dataset_kind'] not in ('ieee', 'synthetic') or record['status'] not in ('blocked', 'completed'):
        raise ValueError('Invalid result kind/status')
    stamp = datetime.fromisoformat(record['created_at_utc'])
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('Timestamp must be UTC')
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {'predictions', 'probabilities', 'raw_rows', 'train_ids', 'test_ids', 'transaction_ids', 'row_indices'}:
                    raise ValueError('Restricted row-level output')
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(record)
    json.dumps(record, allow_nan=False)
    def check_hash(value, length=64, nullable=False):
        if nullable and value is None:
            return
        if not isinstance(value, str) or len(value) != length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid provenance/index hash')
    prov = record['provenance']
    check_hash(prov['git_sha'], 40, nullable=True)
    for filename in ('evaluation.py', 'fraud_detection.py'):
        check_hash(prov['source_sha256'][filename])
    for filename in FILES:
        check_hash(prov['data_sha256'][filename], nullable=record['status'] == 'blocked')
    for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost'):
        if not isinstance(prov['versions'][name], str) or not prov['versions'][name]:
            raise ValueError('Missing runtime version')
    if prov['worktree_clean'] not in (True, False, None) or prov['worktree_dirty'] not in (True, False, None):
        raise ValueError('Invalid worktree flags')
    if prov['worktree_clean'] is not None and prov['worktree_dirty'] != (not prov['worktree_clean']):
        raise ValueError('Inconsistent worktree flags')
    if record['status'] == 'blocked':
        if record['publication_status'] != 'BLOCKED ON DATA' or any(record[k] is not None for k in ('row_counts', 'metrics', 'split', 'calibration', 'importance', 'bootstrap')):
            raise ValueError('Blocked evidence must contain null metrics and counters')
    else:
        expected = 'SYNTHETIC CHECK ONLY' if record['dataset_kind'] == 'synthetic' else 'FULL EMPIRICAL RUN'
        if record['publication_status'] != expected or not isinstance(record['metrics'], dict) or set(record['metrics']) != {'logistic_regression', 'xgboost'}:
            raise ValueError('Invalid completed evidence')
        if not record['calibration']['decision_before_final_metrics']:
            raise ValueError('Calibration decision must precede final metrics')
        locked = datetime.fromisoformat(record['calibration']['decision_at_utc'])
        opened = datetime.fromisoformat(record['final_metrics_opened_at_utc'])
        if locked.utcoffset() is None or opened.utcoffset() is None or locked > opened:
            raise ValueError('Final metrics opened before calibration decision')
        parts = record['split']['partitions']
        if set(parts) != {'model_train', 'calibration', 'evaluation'}:
            raise ValueError('Invalid partitions')
        for part in parts.values():
            check_hash(part['index_sha256'])
            if type(part['rows']) is not int or not 0 < part['positive_count'] < part['rows']:
                raise ValueError('Both classes required in every partition')
        if record['split']['strategy'] == 'temporal':
            if not (parts['model_train']['time_max'] < parts['calibration']['time_min']
                    and parts['calibration']['time_max'] < parts['evaluation']['time_min']):
                raise ValueError('Temporal boundary overlap')
        elif record['split']['strategy'] != 'stratified':
            raise ValueError('Invalid split strategy')
        if sum(p['rows'] for p in parts.values()) != record['row_counts']['train_csv']:
            raise ValueError('Partition counts do not reconcile')
        if record['row_counts']['test_csv'] <= 0:
            raise ValueError('Invalid test row count')
        def check_metrics(values, n):
            for key in ('roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'prevalence', 'brier'):
                if not isinstance(values[key], (float, int)) or not 0 <= values[key] <= 1:
                    raise ValueError('Metric outside permitted range: ' + key)
            if values['threshold'] != .5 or values['log_loss'] < 0:
                raise ValueError('Invalid probability/threshold metric')
            cm = values['confusion_matrix']
            if len(cm) != 2 or any(len(row) != 2 for row in cm) or any(type(v) is not int or v < 0 for row in cm for v in row) or sum(map(sum, cm)) != n:
                raise ValueError('Confusion matrix counts do not reconcile')
            positives = cm[1][0] + cm[1][1]
            if not 0 < positives < n:
                raise ValueError('Metrics require both observed classes')
            if not np.isclose(values['prevalence'], positives / n):
                raise ValueError('Prevalence inconsistent with confusion matrix')
            rel = values['reliability']
            bins = rel['bins']
            if not bins or sum(b['count'] for b in bins) != n or sum(b['positives'] for b in bins) != positives:
                raise ValueError('Reliability counts do not reconcile')
            for bin_ in bins:
                if type(bin_['count']) is not int or bin_['count'] <= 0 or not 0 <= bin_['positives'] <= bin_['count']:
                    raise ValueError('Invalid reliability bin counts')
                if not 0 <= bin_['probability_mean'] <= 1 or not np.isclose(bin_['fraud_fraction'], bin_['positives'] / bin_['count']):
                    raise ValueError('Invalid reliability probability/fraction')
            ece = sum(b['count'] * abs(b['probability_mean'] - b['fraud_fraction']) for b in bins) / n
            if not np.isclose(rel['ece'], ece):
                raise ValueError('ECE does not reconcile')
            budgets = values['review_budgets']
            if len(budgets) != 3:
                raise ValueError('Three fixed review budgets required')
            for b, fraction in zip(budgets, (.005, .01, .02)):
                if b['budget_fraction'] != fraction or b['alerts'] != int(np.ceil(n * fraction)) or b['tp'] + b['fp'] != b['alerts']:
                    raise ValueError('Review capacity/counts do not reconcile')
                if not np.isclose(b['precision'], b['tp'] / b['alerts']) or not np.isclose(b['recall'], b['tp'] / positives) or not 0 <= b['threshold'] <= 1:
                    raise ValueError('Invalid review metrics')
        for values in record['metrics'].values():
            check_metrics(values, parts['evaluation']['rows'])
        check_metrics(record['calibration']['before'], parts['calibration']['rows'])
        comparison = record['calibration']['final_comparison']
        if set(comparison) != {'raw_xgboost', 'selected_xgboost'} or comparison['selected_xgboost'] != record['metrics']['xgboost']:
            raise ValueError('Invalid raw/selected calibration comparison')
        check_metrics(comparison['raw_xgboost'], parts['evaluation']['rows'])
        summary = record['dataset_summary']
        n = record['row_counts']['train_csv']
        if summary['target'] != 'isFraud' or summary['id_field'] != 'TransactionID' or summary['positive_count'] + summary['negative_count'] != n:
            raise ValueError('Dataset summary does not reconcile')
        if not np.isclose(summary['target_prevalence'], summary['positive_count'] / n) or summary['positive_count'] != sum(p['positive_count'] for p in parts.values()):
            raise ValueError('Dataset prevalence does not reconcile')
        if summary['predictor_count'] < 1 or summary['transformed_feature_count'] != len(record['preprocessing']['transformed_feature_names']):
            raise ValueError('Invalid predictor counts')
        if record['leakage_audit']['known_forbidden_features'] or record['leakage_audit']['leakage_absence_claim']:
            raise ValueError('Known leakage or unsupported absence claim')
        if record['dataset_kind'] == 'ieee' and not record['leakage_audit']['uncertain_feature_timing_acknowledged']:
            raise ValueError('Real run lacks timing acknowledgment')
        bootstrap = record['bootstrap']
        if bootstrap['replicates_completed'] + bootstrap['replicates_skipped_single_class'] != bootstrap['requested_replicates']:
            raise ValueError('Bootstrap counts do not reconcile')
    return True


def write_result(path, record, overwrite=False):
    """Serialize cooperating writers; publish new paths exclusively.

    A stale lock after a crash fails closed and requires operator inspection.
    Out-of-band filesystem edits do not participate in this writer protocol.
    """
    validate_result(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(lock_fd)
        existed = path.exists()
        if existed:
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous.get('status') != 'blocked' or not overwrite:
                raise FileExistsError('Existing evidence protected; only blocked records may be replaced')
            validate_result(previous)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False, suffix='.json.tmp') as stream:
            temporary = stream.name
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if existed:
            os.replace(temporary, path)
            temporary = None
        else:
            # Hard-link publication fails if a destination appears; never clobber.
            os.link(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/current_evaluation.json')
    parser.add_argument('--split', choices=('temporal', 'stratified'), default='temporal')
    parser.add_argument('--n-jobs', '--n-jobs4', type=int, default=4, nargs='?', const=4)
    parser.add_argument('--bootstrap-replicates', type=int, default=0)
    parser.add_argument('--acknowledge-uncertain-feature-timing', action='store_true')
    parser.add_argument('--dataset-kind', choices=('ieee', 'synthetic'), default='ieee')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        missing = [name for name in FILES if not (args.data_dir / name).is_file()]
        if missing:
            record = blocked_result(args.data_dir, args.dataset_kind, 'Missing local required files: ' + ', '.join(missing))
            write_result(args.output, record, args.overwrite)
            print('BLOCKED ON DATA: local CSV files required; metrics remain null')
            return 2
        record = run_evaluation(args.data_dir, args.split, args.n_jobs, args.bootstrap_replicates,
                                args.acknowledge_uncertain_feature_timing, args.dataset_kind)
        write_result(args.output, record, args.overwrite)
        print(record['publication_status'] + ': ' + str(args.output))
        return 0
    except (ValueError, FileExistsError, OSError, ConvergenceWarning) as exc:
        print('Evaluation refused: ' + str(exc))
        return 2


def make_split(frame, strategy='temporal'):
    """Return positional indices; temporal ties stay wholly in one partition."""
    indices = np.arange(len(frame))
    if strategy == 'temporal':
        if 'TransactionDT' not in frame or not np.isfinite(frame.TransactionDT).all():
            raise ValueError('Temporal split requires finite TransactionDT')
        times = frame.TransactionDT.to_numpy()
        order = np.argsort(times, kind='stable')
        starts = np.flatnonzero(np.r_[True, np.diff(times[order]) != 0])
        if len(starts) < 3:
            raise ValueError('Temporal split requires at least three distinct timestamps')
        a = int(starts[np.argmin(abs(starts[1:-1] - .6 * len(frame))) + 1])
        candidates = starts[starts > a]
        b = int(candidates[np.argmin(abs(candidates - .8 * len(frame)))])
        parts = (order[:a], order[a:b], order[b:])
    elif strategy == 'stratified':
        a, rest = train_test_split(indices, test_size=.4, stratify=frame.isFraud, random_state=42)
        b, c = train_test_split(rest, test_size=.5, stratify=frame.iloc[rest].isFraud, random_state=42)
        parts = (a, b, c)
    else:
        raise ValueError('Unknown split strategy')
    if any(set(frame.iloc[p].isFraud) != {0, 1} for p in parts):
        raise ValueError('Both classes required in every partition')
    return parts


def reliability(y, p, n_bins=10):
    """Quantile edges; equal probabilities are never split between bins."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    labels = np.searchsorted(edges[1:-1], p, side='right')
    bins = []
    for label in np.unique(labels):
        mask = labels == label
        bins.append({'count': int(mask.sum()), 'positives': int(y[mask].sum()),
                     'probability_mean': float(p[mask].mean()), 'fraud_fraction': float(y[mask].mean())})
    return {'method': 'quantile_unique_edges', 'requested_bins': n_bins, 'bins': bins,
            'ece': float(sum(b['count'] * abs(b['probability_mean'] - b['fraud_fraction']) for b in bins) / len(y))}


def review_budgets(y, p, row_indices=None):
    y, p = np.asarray(y), np.asarray(p)
    row_indices = np.arange(len(y)) if row_indices is None else np.asarray(row_indices)
    order = np.lexsort((row_indices, -p))
    table = []
    for budget in (.005, .01, .02):
        capacity = int(np.ceil(len(y) * budget))
        selected = order[:capacity]
        tp = int(y[selected].sum())
        table.append({'budget_fraction': budget, 'alerts': capacity, 'tp': tp, 'fp': capacity - tp,
                      'precision': float(tp / capacity), 'recall': float(tp / y.sum()),
                      'threshold': float(p[selected[-1]]),
                      'interpretation': 'descriptive only; stable evaluation-row ties, not deployable policy'})
    return table


def metrics(y, p, row_indices=None):
    from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                                 recall_score, f1_score, confusion_matrix, brier_score_loss, log_loss)
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if set(y) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Metrics require both classes and finite probabilities in [0,1]')
    prediction = p >= .5
    return {'roc_auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'threshold': .5, 'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction)), 'f1': float(f1_score(y, prediction)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'prevalence': float(y.mean()), 'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])), 'reliability': reliability(y, p),
            'review_budgets': review_budgets(y, p, row_indices)}


if __name__ == '__main__':
    raise SystemExit(main())
