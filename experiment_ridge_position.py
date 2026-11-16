"""Model (b): Ridge on mutation-position embedding features (TEM-1).

Pure tabular ML in sklearn: no ESM-2 inference. Features per variant are the
concatenation of ``emb_mut_pos`` and the mutation delta
``emb_mut_pos - wt_hidden[position-1]`` (1280 dims). Evaluation protocol is
identical to ``experiment_zeroshot_ridge.py``: frozen outer 5-fold by
position, inner GroupKFold(4) alpha selection on the outer training rows
only, same alpha grid, same metrics.

Run from the repository root:

    python experiment_ridge_position.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = "esm_variant_features.parquet"
WT_NPZ = "esm_wt.npz"
FOLDS = "stiffler_single_folds.csv"
N_FOLDS = 5
INNER_FOLDS = 4
ALPHA_GRID = 10 ** np.arange(-2, 6, 0.5)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = pd.read_parquet(FEATURES)
    folds = pd.read_csv(FOLDS)

    if "DMS_score" in features.columns:
        data = features
    else:
        data = features.merge(folds[["mutant", "DMS_score"]], on="mutant", validate="one_to_one")

    for column in ("mutant", "position", "fold", "emb_mut_pos"):
        if column not in data.columns:
            fail(f"{FEATURES} lacks column {column!r}")

    # NaN handling: drop rows with a missing target before anything else.
    n_before = len(data)
    data = data.dropna(subset=["DMS_score"]).reset_index(drop=True)
    n_dropped = n_before - len(data)
    if n_dropped:
        print(f"dropped {n_dropped} rows with NaN DMS_score")

    fold_values = sorted(pd.unique(data["fold"]))
    if fold_values != list(range(N_FOLDS)):
        fail(f"fold must be exactly 0..{N_FOLDS - 1}, found {fold_values}")
    if data["mutant"].duplicated().any():
        fail("duplicate variants in the feature table")
    if not data.groupby("position")["fold"].nunique().eq(1).all():
        fail("a position appears in more than one fold; folds are not group-frozen")
    if data["emb_mut_pos"].isna().any():
        fail("some variants lack emb_mut_pos")

    wt_hidden = np.load(WT_NPZ)["wt_hidden"].astype(np.float32)
    positions = data["position"].to_numpy()
    if positions.min() < 1 or positions.max() > wt_hidden.shape[0]:
        fail(f"positions {positions.min()}..{positions.max()} out of bounds for "
             f"wt_hidden length {wt_hidden.shape[0]}")

    emb = np.stack(data["emb_mut_pos"].to_numpy())
    if emb.ndim != 2 or emb.shape[1] != wt_hidden.shape[1] or not np.isfinite(emb).all():
        fail(f"emb_mut_pos has wrong shape {emb.shape} or non-finite values")
    emb = emb.astype(np.float32)
    delta = emb - wt_hidden[positions - 1]          # X2: mutation delta
    X = np.concatenate([emb, delta], axis=1).astype(np.float32)  # [n, 1280]

    y = data["DMS_score"].to_numpy(dtype=np.float32)
    fold_col = data["fold"].to_numpy()

    print(f"rows {len(data)}, positions {len(set(positions))}, "
          f"X {X.shape} {X.dtype}, y {y.shape} {y.dtype}, wt_hidden {wt_hidden.shape}")
    print("rows per fold:", {k: int((fold_col == k).sum()) for k in range(N_FOLDS)})
    return data, X, y, delta, positions


def correlations(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return float(pearsonr(pred, target)[0]), float(spearmanr(pred, target)[0])


def evaluate_ridge(X: np.ndarray, y: np.ndarray, positions: np.ndarray,
                   fold_col: np.ndarray) -> pd.DataFrame:
    """Identical protocol to Model (a): inner GroupKFold alpha on train only."""
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": ALPHA_GRID},
        cv=GroupKFold(INNER_FOLDS),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rows = []
    for k in range(N_FOLDS):
        test = fold_col == k
        train = ~test
        search.fit(X[train], y[train], groups=positions[train])
        best_alpha = float(search.best_params_["ridge__alpha"])
        pred = search.best_estimator_.predict(X[test])
        pearson, spearman = correlations(pred, y[test])
        rows.append({"fold": k, "alpha": best_alpha, "pearson": pearson, "spearman": spearman})
        print(f"  fold {k}: alpha {best_alpha:g}  pearson {pearson:+.4f}  spearman {spearman:+.4f}")
    return pd.DataFrame(rows)


def main() -> int:
    data, X, y, delta, positions = load_data()
    fold_col = data["fold"].to_numpy()

    print("\nModel (b) - Ridge on [emb_mut_pos | delta] (inner GroupKFold alpha selection)")
    rg = evaluate_ridge(X, y, positions, fold_col)

    p, s = rg["pearson"], rg["spearman"]
    print("\nfinal result (outer 5-fold, frozen folds)")
    print(f"  {'model':<20} | {'pearson (mean +/- std)':<24} | {'spearman (mean +/- std)':<24}")
    print(f"  {'-' * 20}-+-{'-' * 24}-+-{'-' * 24}")
    print(f"  {'ridge position (b)':<20} | {p.mean():.4f} +/- {p.std(ddof=1):.4f}{'':<7} | "
          f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}")

    print("\nper-fold detail")
    print(rg.assign(model="ridge position (b)")[["model", "fold", "alpha", "pearson", "spearman"]]
          .to_string(index=False))

    # Exploratory note only - not part of the evaluation.
    norms = np.linalg.norm(delta.astype(np.float64), axis=1)
    r = pearsonr(norms, np.abs(y.astype(np.float64)))[0]
    print(f"\nexploratory (not part of the evaluation): "
          f"||X2|| mean {norms.mean():.4f} +/- {norms.std(ddof=1):.4f}, "
          f"min {norms.min():.4f}, max {norms.max():.4f}; "
          f"pearson(||X2||, |DMS_score|) = {r:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
