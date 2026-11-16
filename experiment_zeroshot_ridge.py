"""Zero-shot baseline vs Ridge on mean-pooled ESM-2 embeddings (TEM-1).

Pure tabular ML in sklearn: ESM-2 is never loaded or touched. The only valid
evaluation split is the frozen GroupKFold-by-position ``fold`` column (0..4),
used exactly as read. Both parts share the same outer 5-fold loop so the
comparison is fair.

Part 1  zero-shot: ``zeroshot`` used directly as the predicted score.
Part 2  ridge:    Pipeline(StandardScaler, Ridge) on ``mean_pooled`` only,
                  alpha selected by GridSearchCV with an inner GroupKFold(4)
                  on the outer training positions.

Run from the repository root:

    python experiment_zeroshot_ridge.py
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
FOLDS = "stiffler_single_folds.csv"
N_FOLDS = 5
INNER_FOLDS = 4
ALPHA_GRID = 10 ** np.arange(-2, 6, 0.5)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_data() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES)
    folds = pd.read_csv(FOLDS)

    if "DMS_score" in features.columns:
        data = features
    else:
        data = features.merge(folds[["mutant", "DMS_score"]], on="mutant", validate="one_to_one")

    for column in ("mutant", "position", "fold", "zeroshot", "mean_pooled"):
        if column not in data.columns:
            fail(f"{FEATURES} lacks column {column!r}")

    # NaN handling: drop rows with a missing target before anything else.
    n_before = len(data)
    data = data.dropna(subset=["DMS_score"]).reset_index(drop=True)
    n_dropped = n_before - len(data)
    if n_dropped:
        print(f"dropped {n_dropped} rows with NaN DMS_score")

    folds_values = sorted(pd.unique(data["fold"]))
    if folds_values != list(range(N_FOLDS)):
        fail(f"fold must be exactly 0..{N_FOLDS - 1}, found {folds_values}")
    if data["mutant"].duplicated().any():
        fail("duplicate variants in the feature table")
    if not data.groupby("position")["fold"].nunique().eq(1).all():
        fail("a position appears in more than one fold; folds are not group-frozen")
    if data["mean_pooled"].isna().any():
        fail("some variants lack mean_pooled")

    X = np.stack(data["mean_pooled"].to_numpy()).astype(np.float32)
    if X.ndim != 2 or not np.isfinite(X).all():
        fail(f"mean_pooled is not a finite 2D matrix (shape {X.shape})")
    y = data["DMS_score"].to_numpy(dtype=np.float32)
    z = data["zeroshot"].to_numpy(dtype=np.float64)
    if not np.isfinite(z).all():
        fail("non-finite zeroshot values")
    positions = data["position"].to_numpy()
    fold_col = data["fold"].to_numpy()

    print(f"rows {len(data)}, positions {len(set(positions))}, "
          f"X {X.shape} {X.dtype}, y {y.shape} {y.dtype}")
    print("rows per fold:", {k: int((fold_col == k).sum()) for k in range(N_FOLDS)})
    return data, X, y, z, positions, fold_col


def correlations(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return float(pearsonr(pred, target)[0]), float(spearmanr(pred, target)[0])


def evaluate_zeroshot(z: np.ndarray, y: np.ndarray, fold_col: np.ndarray) -> pd.DataFrame:
    """No training: zeroshot is the prediction on each outer test fold."""
    rows = []
    for k in range(N_FOLDS):
        test = fold_col == k
        pearson, spearman = correlations(z[test], y[test])
        rows.append({"fold": k, "alpha": None, "pearson": pearson, "spearman": spearman})
        print(f"  fold {k}: pearson {pearson:+.4f}  spearman {spearman:+.4f}")
    return pd.DataFrame(rows)


def evaluate_ridge(X: np.ndarray, y: np.ndarray, positions: np.ndarray,
                   fold_col: np.ndarray) -> pd.DataFrame:
    """Ridge on mean_pooled; alpha chosen by inner GroupKFold on train only."""
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
    data, X, y, z, positions, fold_col = load_data()

    print("\nPart 1 - zero-shot baseline")
    zs = evaluate_zeroshot(z, y, fold_col)

    print("\nPart 2 - Ridge on mean_pooled (inner GroupKFold alpha selection)")
    rg = evaluate_ridge(X, y, positions, fold_col)

    def summary(frame: pd.DataFrame) -> tuple[str, str]:
        p, s = frame["pearson"], frame["spearman"]
        return f"{p.mean():.4f} +/- {p.std(ddof=1):.4f}", f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}"

    zs_p, zs_s = summary(zs)
    rg_p, rg_s = summary(rg)

    print("\nfinal comparison (outer 5-fold, frozen folds)")
    print(f"  {'model':<20} | {'pearson (mean +/- std)':<24} | {'spearman (mean +/- std)':<24}")
    print(f"  {'-' * 20}-+-{'-' * 24}-+-{'-' * 24}")
    print(f"  {'zero-shot':<20} | {zs_p:<24} | {zs_s:<24}")
    print(f"  {'ridge mean_pooled':<20} | {rg_p:<24} | {rg_s:<24}")

    print("\nper-fold detail")
    print(zs.assign(model="zero-shot")[["model", "fold", "alpha", "pearson", "spearman"]].to_string(index=False))
    print(rg.assign(model="ridge mean_pooled")[["model", "fold", "alpha", "pearson", "spearman"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
