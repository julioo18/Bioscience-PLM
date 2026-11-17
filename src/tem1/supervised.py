"""CPU-only nested Ridge benchmark on frozen TEM-1 features and position folds."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from .data import ASSAY_ID, parse_mutation, sha256, write_json

VECTOR_COLUMNS = ('emb_mut_pos', 'mean_local_5', 'mean_local_10', 'mean_local_20', 'mean_pooled')
REPRESENTATIONS = VECTOR_COLUMNS + ('emb_mut_pos_delta', 'emb_mut_pos_delta_mean_pooled')
ZERO_SHOT = ('zeroshot_wt', 'zeroshot_masked')
ALPHA_GRID = 10.0 ** np.arange(-2, 6, 0.5)
CHECKPOINT = 'facebook/esm2_t30_150M_UR50D'
REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
DEFAULT_OUTPUT = Path('results/supervised/baseline')
RAW_MEMBER = f'DMS_ProteinGym_substitutions/{ASSAY_ID}.csv'


@dataclass(frozen=True)
class InputPaths:
    features: Path = Path('features/esm2_150m/esm_variant_features.parquet')
    folds: Path = Path('data/processed/stiffler_single_folds.csv')
    wt: Path = Path('features/esm2_150m/esm_wt.npz')
    archive: Path = Path('data/raw/proteingym_substitutions.zip')
    masked_dir: Path = Path('results/zero_shot/esm2/150m')

    def files(self):
        return [self.features, self.folds, self.wt, self.archive,
                *[self.masked_dir / name for name in
                  ('predictions.csv', 'metrics.json', 'run.json', 'assay_metadata.json', 'input_manifest.json')]]


@dataclass
class PreparedData:
    rows: pd.DataFrame
    matrices: dict[str, np.ndarray]
    wt_hidden: np.ndarray
    validation: dict

    def representation(self, name):
        if name in ('emb_mut_pos_delta', 'emb_mut_pos_delta_mean_pooled'):
            embedding = self.matrices['emb_mut_pos']
            delta = embedding - self.wt_hidden[self.rows['position'].to_numpy() - 1]
            blocks = [embedding, delta]
            if name == 'emb_mut_pos_delta_mean_pooled':
                blocks.append(self.matrices['mean_pooled'])
            return np.concatenate(blocks, axis=1)
        return self.matrices[name]


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')


def align_table(frame, ids, name):
    """Reorder by variant key only after enforcing exact, unique set equality."""
    require_columns(frame, ['mutant'], name)
    if frame['mutant'].isna().any() or frame['mutant'].duplicated().any():
        raise ValueError(f'{name}: missing or duplicate mutant IDs')
    if len(frame) != len(ids) or set(frame['mutant']) != set(ids):
        raise ValueError(f'{name}: variant set differs; no rows may be excluded')
    return frame.set_index('mutant').loc[list(ids)].reset_index()


def finite_values(values, name):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: non-finite values')
    return array


def same_values(values, expected, name, tolerance=0):
    actual = finite_values(values, name)
    if not np.allclose(actual, expected, atol=tolerance, rtol=0):
        raise ValueError(f'{name}: values disagree with the reference table')


def prepare_tables(folds, features, raw, masked, reference, wt_hidden, wt_logits,
                   expected_count=4996, hidden_size=640):
    """Validate all inputs and align them to the frozen folds table, without filtering."""
    require_columns(folds, ['mutant', 'position', 'fold', 'wt_aa', 'mut_aa', 'DMS_score'], 'folds')
    if len(folds) != expected_count or folds['mutant'].duplicated().any() or folds['mutant'].isna().any():
        raise ValueError(f'folds: require {expected_count} unique variants')
    folds = folds.copy().reset_index(drop=True)
    for column in ('position', 'fold'):
        values = finite_values(folds[column], column)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{column}: require integer values')
        folds[column] = values.astype(np.int64)
    if set(folds['fold']) != set(range(5)):
        raise ValueError('folds: require labels 0..4')
    if not folds.groupby('position')['fold'].nunique().eq(1).all():
        raise ValueError('a position belongs to more than one fold')
    y = finite_values(folds['DMS_score'], 'DMS_score')
    ids = folds['mutant']
    features = align_table(features, ids, 'features')
    masked = align_table(masked, ids, 'masked predictions')
    raw = align_table(raw, ids, 'raw assay')
    require_columns(features, ['position', 'fold', 'zeroshot', *VECTOR_COLUMNS], 'features')
    require_columns(masked, ['position', 'wild_type', 'alternate', 'DMS_score', 'esm2_score'], 'masked predictions')
    require_columns(raw, ['mutated_sequence', 'DMS_score'], 'raw assay')
    same_values(raw['DMS_score'], y, 'raw targets', 1e-12)
    same_values(masked['DMS_score'], y, 'masked targets', 1e-12)
    for column in ('position', 'fold'):
        same_values(features[column], folds[column].to_numpy(), f'features {column}')
    same_values(masked['position'], folds['position'].to_numpy(), 'masked positions')
    if 'DMS_score' in features:
        same_values(features['DMS_score'], y, 'feature targets', 1e-12)
    for i, row in enumerate(folds.itertuples()):
        variant = parse_mutation(row.mutant, reference)
        if (variant.position, variant.wild_type, variant.alternate) != (row.position, row.wt_aa, row.mut_aa):
            raise ValueError(f'{row.mutant}: fold annotation mismatch')
        m = masked.iloc[i]
        if (m['wild_type'], m['alternate']) != (row.wt_aa, row.mut_aa):
            raise ValueError(f'{row.mutant}: masked annotation mismatch')
        p = row.position - 1
        if raw.iloc[i]['mutated_sequence'] != reference[:p] + row.mut_aa + reference[p + 1:]:
            raise ValueError(f'{row.mutant}: raw sequence mismatch')
    hidden = finite_values(wt_hidden, 'WT hidden states')
    logits = finite_values(wt_logits, 'WT logits')
    if hidden.shape != (len(reference), hidden_size) or logits.shape != (len(reference), 33):
        raise ValueError('WT arrays have unexpected shape')
    matrices = {}
    for column in VECTOR_COLUMNS:
        try:
            matrix = np.stack(features[column].to_numpy()).astype(np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError(f'{column}: malformed vectors') from error
        if matrix.shape != (expected_count, hidden_size) or not np.isfinite(matrix).all():
            raise ValueError(f'{column}: incorrect shape or non-finite vectors')
        matrices[column] = matrix
    z_wt = finite_values(features['zeroshot'], 'WT-marginals score')
    # Canonical residue IDs in the ESM-2 sequence vocabulary (not ESM-C).
    aa_ids = {aa: i + 4 for i, aa in enumerate('LAGVSERTIDPKQNFYMHWC')}
    positions = folds['position'].to_numpy() - 1
    expected_wt = (logits[positions, [aa_ids[a] for a in folds['mut_aa']]]
                   - logits[positions, [aa_ids[a] for a in folds['wt_aa']]])
    wt_error = float(np.max(np.abs(z_wt - expected_wt)))
    if wt_error > 1e-2:
        raise ValueError('WT score disagrees with saved FP16 WT logits')
    folds['DMS_score'] = y
    folds['zeroshot_wt'] = z_wt
    folds['zeroshot_masked'] = finite_values(masked['esm2_score'], 'masked-marginals score')
    counts = []
    for k, group in folds.groupby('fold'):
        counts.append(dict(fold=int(k), n_variants=len(group), n_positions=int(group['position'].nunique())))
    validation = dict(n_variants=len(folds), n_positions=int(folds['position'].nunique()),
                      fold_counts=counts, alignment='exact variant sets; joined by mutant; folds-table order',
                      target_tolerance=1e-12, no_rows_excluded=True, one_fold_per_position=True,
                      wt_score_max_deviation_from_fp16_logits=wt_error,
                      wt_score_tolerance=1e-2, feature_width=hidden_size)
    return PreparedData(folds, matrices, hidden, validation)


def load_inputs(paths=InputPaths()):
    run = json.loads((paths.masked_dir / 'run.json').read_text())
    metrics = json.loads((paths.masked_dir / 'metrics.json').read_text())
    metadata = json.loads((paths.masked_dir / 'assay_metadata.json').read_text())
    if (run['status'] != 'complete' or metrics['scope'] != 'full_assay'
            or metrics['assay_id'] != ASSAY_ID or metrics['n_variants'] != 4996
            or metadata['DMS_id'] != ASSAY_ID or int(metadata['seq_len']) != 286
            or int(metadata['DMS_number_single_mutants']) != 4996
            or run['model']['model_id'] != CHECKPOINT
            or run['model']['resolved_revision'] != REVISION
            or run['model']['scoring_method'] != 'masked_marginals'):
        raise ValueError('masked baseline metadata does not identify the completed ESM-2 150M assay')
    with zipfile.ZipFile(paths.archive) as archive:
        raw_bytes = archive.read(RAW_MEMBER)
        import io
        raw = pd.read_csv(io.BytesIO(raw_bytes), float_precision='round_trip')
    import hashlib
    manifest = json.loads((paths.masked_dir / 'input_manifest.json').read_text())
    if hashlib.sha256(raw_bytes).hexdigest() != manifest['sha256'][f'{ASSAY_ID}.csv']:
        raise ValueError('raw assay checksum differs from masked baseline input')
    with np.load(paths.wt, allow_pickle=False) as wt:
        if str(wt['checkpoint_name']) != CHECKPOINT:
            raise ValueError('WT artifact checkpoint differs from ESM-2 150M')
        data = prepare_tables(pd.read_csv(paths.folds, float_precision='round_trip'),
                              pd.read_parquet(paths.features), raw,
                              pd.read_csv(paths.masked_dir / 'predictions.csv', float_precision='round_trip'),
                              metadata['target_seq'], wt['wt_hidden'], wt['wt_logits'])
    if data.validation['n_positions'] != 263:
        raise ValueError('expected 263 distinct positions')
    rho = metric_values(data.rows['DMS_score'], data.rows['zeroshot_masked'])['spearman']
    if not np.isclose(rho, metrics['spearman_rho'], atol=1e-12, rtol=0):
        raise ValueError('masked predictions do not reproduce their saved Spearman')
    return data


def metric_values(y, predicted, regression=False):
    y, predicted = finite_values(y, 'target'), finite_values(predicted, 'prediction')
    if y.ndim != 1 or predicted.shape != y.shape or len(y) < 3:
        raise ValueError('require at least three paired scalar observations')
    if np.unique(y).size < 2 or np.unique(predicted).size < 2:
        raise ValueError('Spearman is undefined for constant scores')
    result = {'spearman': float(spearmanr(y, predicted).statistic)}
    if regression:
        result.update(rmse=float(np.sqrt(mean_squared_error(y, predicted))),
                      mae=float(mean_absolute_error(y, predicted)))
    return result


def summarize(fold_reports, y, predicted, regression=False):
    keys = ('spearman', 'rmse', 'mae') if regression else ('spearman',)
    return dict(folds=fold_reports,
                fold_summary={key: dict(mean=float(np.mean([r['metrics'][key] for r in fold_reports])),
                                        std=float(np.std([r['metrics'][key] for r in fold_reports], ddof=1)))
                              for key in keys},
                oof_metrics=metric_values(y, predicted, regression))


def fit_outer_fold(X, y, positions, folds, k, alphas):
    """All scaling and alpha selection take place within outer training positions."""
    test = folds == k
    train = ~test
    if set(positions[train]) & set(positions[test]) or not test.any():
        raise ValueError('outer split leaks positions or has no test rows')
    inner = list(GroupKFold(4).split(X[train], y[train], groups=positions[train]))
    audit = []
    for fit, valid in inner:
        train_pos, val_pos = positions[train][fit], positions[train][valid]
        if set(train_pos) & set(val_pos):
            raise ValueError('inner split leaks positions')
        audit.append(dict(n_train=len(fit), n_validation=len(valid),
                          n_train_positions=len(set(train_pos)), n_validation_positions=len(set(val_pos))))
    search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge(solver='cholesky'))]),
                          {'ridge__alpha': alphas}, scoring='neg_mean_squared_error',
                          cv=inner, n_jobs=1, refit=True, error_score='raise')
    search.fit(X[train], y[train])
    predicted = search.predict(X[test])
    report = dict(fold=k, alpha=float(search.best_params_['ridge__alpha']),
                  selected_inner_mse=float(-search.best_score_),
                  n_train=int(train.sum()), n_test=int(test.sum()),
                  n_train_positions=len(set(positions[train])), n_test_positions=len(set(positions[test])),
                  inner_splits=audit,
                  inner_mse_by_alpha=[dict(alpha=float(a), mean_mse=float(-m))
                                      for a, m in zip(search.cv_results_['param_ridge__alpha'],
                                                      search.cv_results_['mean_test_score'])],
                  metrics=metric_values(y[test], predicted, True))
    return test, predicted, report


def benchmark(data, representations=REPRESENTATIONS, alphas=ALPHA_GRID, threads=2):
    representations = tuple(representations)
    alphas = np.asarray(alphas, dtype=float)
    if (not representations or len(set(representations)) != len(representations)
            or set(representations) - set(REPRESENTATIONS)):
        raise ValueError('choose unique, known representations')
    if threads < 1 or alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or (alphas <= 0).any():
        raise ValueError('require positive threads and finite positive alphas')
    rows = data.rows
    predictions = rows[['mutant', 'position', 'fold', 'DMS_score', *ZERO_SHOT]].copy()
    y = rows['DMS_score'].to_numpy(dtype=float)
    positions, folds = rows['position'].to_numpy(), rows['fold'].to_numpy()
    if set(folds) != set(range(5)) or not rows.groupby('position')['fold'].nunique().eq(1).all():
        raise ValueError('require five fixed position folds')
    reports = {}
    for name in ZERO_SHOT:
        z = rows[name].to_numpy()
        reports[name] = dict(kind='zero_shot', representation=None, n_features=0,
            **summarize([dict(fold=k, n_test=int((folds == k).sum()), alpha=None,
                             metrics=metric_values(y[folds == k], z[folds == k])) for k in range(5)], y, z))
    with threadpool_limits(limits=threads):
        pools = threadpool_info()
        for name in representations:
            start = time.perf_counter()
            X = data.representation(name)
            oof = np.full(len(y), np.nan)
            visits = np.zeros(len(y), dtype=int)
            fold_reports = []
            for k in range(5):
                test, pred, report = fit_outer_fold(X, y, positions, folds, k, alphas)
                oof[test] = pred
                visits[test] += 1
                fold_reports.append(report)
                print(f'{name}: fold {k}, alpha={report["alpha"]:g}, '
                      f'Spearman={report["metrics"]["spearman"]:.6f}', flush=True)
            if not (visits == 1).all() or not np.isfinite(oof).all():
                raise ValueError('every variant must receive exactly one finite OOF prediction')
            predictions['ridge_' + name] = oof
            reports['ridge_' + name] = dict(kind='ridge', representation=name, n_features=X.shape[1],
                seconds=time.perf_counter() - start, **summarize(fold_reports, y, oof, True))
    return predictions, reports, pools


def result_table(result):
    """One table for CLI and notebook; fold means and pooled OOF kept separate."""
    rows = []
    for name, report in result['methods'].items():
        row = dict(method=name, n_features=report['n_features'])
        for metric, values in report['fold_summary'].items():
            row[metric + '_fold_mean'] = values['mean']
            row[metric + '_fold_std'] = values['std']
            row[metric + '_oof'] = report['oof_metrics'][metric]
        rows.append(row)
    return pd.DataFrame(rows).set_index('method')


def load_results(output=DEFAULT_OUTPUT):
    output = Path(output)
    result = json.loads((output / 'metrics.json').read_text())
    if result['status'] != 'complete' or sha256(output / 'predictions.csv') != result['predictions_sha256']:
        raise ValueError('incomplete results or predictions checksum mismatch')
    predictions = pd.read_csv(output / 'predictions.csv', float_precision='round_trip')
    if len(predictions) != result['validation']['n_variants'] or predictions['mutant'].duplicated().any():
        raise ValueError('saved OOF variant count/uniqueness mismatch')
    return predictions, result


def main(argv=None, default_representations=REPRESENTATIONS, default_output=DEFAULT_OUTPUT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=InputPaths.features)
    parser.add_argument('--folds', type=Path, default=InputPaths.folds)
    parser.add_argument('--wt', type=Path, default=InputPaths.wt)
    parser.add_argument('--archive', type=Path, default=InputPaths.archive)
    parser.add_argument('--masked-dir', type=Path, default=InputPaths.masked_dir)
    parser.add_argument('--representations', nargs='+', choices=REPRESENTATIONS, default=default_representations)
    parser.add_argument('--output', type=Path, default=Path(default_output))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a new or empty output directory')
    paths = InputPaths(args.features, args.folds, args.wt, args.archive, args.masked_dir)
    start = time.perf_counter()
    data = load_inputs(paths)
    predictions, methods, pools = benchmark(data, args.representations, threads=args.threads)
    result = dict(status='complete', assay_id=ASSAY_ID, validation=data.validation, methods=methods,
        configuration=dict(representations=args.representations, alpha_grid=ALPHA_GRID.tolist(),
            outer_cv='fixed five position folds from CSV', inner_cv='GroupKFold(4), shuffle=False',
            alpha_selection='minimum unweighted mean validation MSE across inner folds',
            preprocessing='StandardScaler inside Pipeline fitted in every inner split; refit on outer train',
            estimator='Ridge(solver=cholesky)', compute_dtype='float64',
            stored_embedding_dtype='float16', n_jobs=1, threads=args.threads, device='cpu',
            summary='unweighted fold mean and sample standard deviation (ddof=1); pooled OOF separate',
            seed=None, randomness='deterministic GroupKFold and Ridge; no random sampling',
            checkpoint=CHECKPOINT, masked_checkpoint_revision=REVISION,
            embedding_revision='not recorded in original artifacts; input bytes identified by SHA-256',
            zero_shot_methods=dict(zeroshot_wt='unmasked WT-marginals from Parquet',
                                   zeroshot_masked='masked-marginals from saved 150M predictions'),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}),
        input_sha256={str(p): sha256(p) for p in paths.files()},
        source_sha256={str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name('data.py')]},
        environment=dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
            packages={p: importlib.metadata.version(p) for p in
                      ('numpy', 'scipy', 'scikit-learn', 'pandas', 'pyarrow', 'threadpoolctl')},
            threadpools=pools),
        finished_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - start,
        limitations=['One assay and fixed five position folds; no independent protein benchmark.',
                     'Fold SD is descriptive, not a confidence interval; training sets overlap.',
                     'Representation comparisons use the same outer folds; selecting the winner makes these folds exploratory.',
                     'Zero-shot scores are rankings, not calibrated fitness; RMSE/MAE are reported only for regressors.',
                     'Pooled OOF Spearman also compares predictions from different fitted models across folds.'])
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    result['predictions_sha256'] = sha256(args.output / 'predictions.csv')
    write_json(args.output / 'metrics.json', result)
    print(result_table(result).to_string(float_format=lambda value: f'{value:.6f}'))
    print(f'Results: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
