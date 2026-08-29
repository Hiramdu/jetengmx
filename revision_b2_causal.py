"""
Revision experiment for Reviewer B, comment 2 (causal imputation sensitivity).

The submitted pipeline imputed missing sensor values by a forward- then
backward-fill within each (engine, snapshot) group, with a global median for
residual gaps, and it did so on the complete record before the temporal split.
The backward fill and the global median are not available to an online
implementation.  This script re-runs the whole pipeline twice, changing nothing
but the imputation, and reports the effect on prediction accuracy (Table 2 of
the manuscript) and on conformal coverage (Table 5).

  as_submitted : ffill then bfill within (ESN, Snapshot); residual gaps filled
                 with the global median of the column.
  causal       : ffill only within (ESN, Snapshot); residual gaps filled with
                 the median of the TRAINING WINDOW of that column, computed
                 after the temporal cut and never using validation or test
                 rows.  No value is ever imputed from a later observation.

Everything downstream is held fixed so that the imputation is the only
difference: the same feature families, the same feature-selection rule (first
60 % of each engine's timeline), the same temporal split, the same four base
learners with the stored Optuna hyperparameters, the same ridge meta-learner
and the same per-bin bias correction.  Hyperparameters are NOT re-tuned, on
purpose: re-tuning would confound the imputation effect with a search effect.

Output: experiment_results/b2_causal_imputation.csv
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "code"))
from common import (  # noqa: E402  (import after sys.path edit)
    MAINT_COLS, OUTLIER_CAPS, RANDOM_STATE, TARGETS,
    build_all_features, select_features_by_importance, temporal_split,
)

RESULTS = Path(__file__).parent / "experiment_results"
ROOT = Path(__file__).parent
RAW = ROOT / "training_data.csv"
if not RAW.exists():                       # also accept data/training_data.csv
    RAW = ROOT / "data" / "training_data.csv"
SHORT = {"Cycles_to_WW": "WW", "Cycles_to_HPT_SV": "HPT_SV",
         "Cycles_to_HPC_SV": "HPC_SV"}
ALPHA = 0.10
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2


def cap_outliers(df):
    for col, (lo, hi) in OUTLIER_CAPS.items():
        if lo is not None:
            df.loc[df[col] < lo, col] = np.nan
        if hi is not None:
            df.loc[df[col] > hi, col] = np.nan
    return df


def preprocess(path, mode):
    """Load, cap outliers, and impute under one of the two regimes."""
    df = pd.read_csv(path)
    sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
    df = cap_outliers(df)
    df = df.sort_values(["ESN", "Snapshot", "Cycles_Since_New"]).reset_index(drop=True)

    if mode == "as_submitted":
        for col in sensor_cols:
            df[col] = df.groupby(["ESN", "Snapshot"])[col].transform(
                lambda s: s.ffill().bfill())
            df[col] = df[col].fillna(df[col].median())
        return df

    if mode != "causal":
        raise ValueError(mode)

    # Forward fill only: never use a later observation.
    for col in sensor_cols:
        df[col] = df.groupby(["ESN", "Snapshot"])[col].transform(lambda s: s.ffill())

    # Residual gaps (a channel missing from the very first cycles of an engine)
    # are filled with the median of the training window only.  The training cut
    # is defined per engine on the raw cycle axis, matching temporal_split.
    train_mask = np.zeros(len(df), dtype=bool)
    for esn in df.ESN.unique():
        cyc = np.sort(df.loc[df.ESN == esn, "Cycles_Since_New"].unique())
        cut = cyc[int(len(cyc) * TRAIN_FRAC) - 1]
        train_mask |= (df.ESN == esn) & (df.Cycles_Since_New <= cut)
    for col in sensor_cols:
        med = df.loc[train_mask, col].median()
        df[col] = df[col].fillna(med)
        n_left = int(df[col].isna().sum())
        if n_left:                      # column entirely absent in training
            df[col] = df[col].fillna(0.0)
    return df


def asymmetric_objective(y_true, y_pred):
    lam = 0.01
    err = y_pred - y_true
    w = (1.0 + (err >= 0)) / (1.0 + lam * y_true)
    return 2.0 * w * err, 2.0 * w


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(absres, lvl, method="higher"))


def run_pipeline(train_df, val_df, test_df, selected, optuna_params):
    """The submitted Stack+Cal pipeline, unchanged."""
    Xtr, Xva, Xte = (d[selected].to_numpy() for d in (train_df, val_df, test_df))
    scaler = StandardScaler().fit(Xtr)
    out = {}
    for target in TARGETS:
        ytr, yva = train_df[target].to_numpy(), val_df[target].to_numpy()
        base = {
            "Ridge": Ridge(alpha=10.0).fit(scaler.transform(Xtr), ytr),
            "GBR": GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                             learning_rate=0.1, subsample=0.8,
                                             random_state=RANDOM_STATE).fit(Xtr, ytr),
            "RF": RandomForestRegressor(n_estimators=300, max_depth=20,
                                        min_samples_leaf=3, n_jobs=-1,
                                        random_state=RANDOM_STATE).fit(Xtr, ytr),
            "LGBM": lgb.LGBMRegressor(**optuna_params[target],
                                      random_state=RANDOM_STATE, verbose=-1,
                                      n_jobs=-1,
                                      objective=asymmetric_objective).fit(Xtr, ytr),
        }
        ctx = [c for c in ("Cycles_Since_Last_WW", "Cycles_Since_Last_HPC_SV",
                           "Cycles_Since_Last_HPT_SV") if c in selected]
        ci = [selected.index(c) for c in ctx]

        def meta_X(X):
            cols = [np.clip(m.predict(scaler.transform(X) if n == "Ridge" else X),
                            0, None) for n, m in base.items()]
            return np.column_stack(cols + [X[:, ci]])

        meta = Ridge(alpha=1.0).fit(meta_X(Xva), yva)
        raw_va = np.clip(meta.predict(meta_X(Xva)), 0, None)
        raw_te = np.clip(meta.predict(meta_X(Xte)), 0, None)

        n_bins = 20
        edges = np.linspace(0, raw_va.max() * 1.01, n_bins + 1)
        iv = np.clip(np.digitize(raw_va, edges) - 1, 0, n_bins - 1)
        corr = np.zeros(n_bins)
        resid = raw_va - yva
        for b in range(n_bins):
            m_ = iv == b
            if m_.sum() > 5:
                corr[b] = np.median(resid[m_])
        it = np.clip(np.digitize(raw_te, edges) - 1, 0, n_bins - 1)
        out[target] = (np.clip(raw_va - corr[iv], 0, None),
                       np.clip(raw_te - corr[it], 0, None))
    return out


def main():
    with open(RESULTS / "optuna_best_params.json") as f:
        optuna_params = json.load(f)

    rows = []
    for mode in ("as_submitted", "causal"):
        print(f"\n############ imputation = {mode} ############", flush=True)
        df = preprocess(RAW, mode)
        cycle_df = build_all_features(df)
        all_feat = [c for c in cycle_df.columns
                    if c not in ["ESN", "Cycles_Since_New"] + TARGETS]
        selected = select_features_by_importance(cycle_df, all_feat,
                                                 train_frac=TRAIN_FRAC)
        train_df, val_df, test_df = temporal_split(cycle_df, TRAIN_FRAC, VAL_FRAC)
        print(f"  rows {len(train_df)}/{len(val_df)}/{len(test_df)}, "
              f"features {len(selected)}", flush=True)

        preds = run_pipeline(train_df, val_df, test_df, selected, optuna_params)
        for target in TARGETS:
            pv, pt = preds[target]
            yv = val_df[target].to_numpy()
            yt = test_df[target].to_numpy()
            q = conformal_q(np.abs(yv - pv))
            cov = float(((yt >= np.clip(pt - q, 0, None)) & (yt <= pt + q)).mean())
            r = dict(imputation=mode, target=SHORT[target], n_features=len(selected),
                     val_mae=round(mean_absolute_error(yv, pv), 1),
                     val_r2=round(r2_score(yv, pv), 3),
                     test_mae=round(mean_absolute_error(yt, pt), 1),
                     test_r2=round(r2_score(yt, pt), 3),
                     half_width=round(q, 1), test_coverage=round(cov, 3))
            rows.append(r)
            print("   ", r, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "b2_causal_imputation.csv", index=False)
    print("\n=== B2: causal-imputation sensitivity ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
