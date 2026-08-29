"""
Shared utilities for the PHM 2025 jet engine predictive maintenance pipeline.

This module provides:
- Data loading and preprocessing (outlier capping, imputation)
- Feature engineering (cycle aggregation, rolling, tsfresh, cycles-since-last)
- Temporal train/val/test splitting
- Competition scoring function and custom asymmetric loss
- Feature selection via LightGBM importance
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
TARGET_LABELS = ["Water-Wash (WW)", "HPC Shop Visit", "HPT Shop Visit"]
MAINT_COLS = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]
SINCE_LAST_COLS = [
    "Cycles_Since_Last_WW",
    "Cycles_Since_Last_HPC_SV",
    "Cycles_Since_Last_HPT_SV",
]

# Physics-based outlier thresholds
OUTLIER_CAPS = {
    "Sensed_T25": (None, 2000),
    "Sensed_P25": (0, None),
    "Sensed_Core_Speed": (None, 30000),
    "Sensed_T3": (None, 3000),
    "Sensed_T45": (0.5, 3500),
    "Sensed_T5": (0.5, 3500),
    "Sensed_WFuel": (None, 5),
    "Sensed_Fan_Speed": (100, None),
}

# Sensors used for tsfresh automated feature extraction
TSFRESH_SENSORS = [
    "Sensed_TAT", "Sensed_T3", "Sensed_Ps3", "Sensed_T45",
    "Sensed_Fan_Speed", "Sensed_Core_Speed", "Sensed_WFuel",
    "Sensed_Mach", "Sensed_Altitude",
]

# Output directory
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Data loading and preprocessing
# ══════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path: str = "training_data.csv") -> pd.DataFrame:
    """Load raw CSV, apply outlier capping, and forward-fill impute."""
    df = pd.read_csv(path)
    sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]

    # Outlier capping (replace with NaN, then impute)
    for col, (lo, hi) in OUTLIER_CAPS.items():
        if lo is not None:
            df.loc[df[col] < lo, col] = np.nan
        if hi is not None:
            df.loc[df[col] > hi, col] = np.nan

    # Forward/backward fill within (ESN, Snapshot) groups; then global median
    df = df.sort_values(["ESN", "Snapshot", "Cycles_Since_New"]).reset_index(drop=True)
    for col in sensor_cols:
        df[col] = df.groupby(["ESN", "Snapshot"])[col].transform(
            lambda s: s.ffill().bfill()
        )
        df[col] = df[col].fillna(df[col].median())

    return df


# ══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════

def build_cycle_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate snapshot-level rows to cycle-level with mean/std/min/max."""
    sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
    agg_funcs = {col: ["mean", "std", "min", "max"] for col in sensor_cols}
    cycle_agg = df.groupby(["ESN", "Cycles_Since_New"]).agg(agg_funcs)
    cycle_agg.columns = [f"{col}_{stat}" for col, stat in cycle_agg.columns]
    cycle_agg = cycle_agg.reset_index().fillna(0)

    first_per_cycle = (
        df.groupby(["ESN", "Cycles_Since_New"])[MAINT_COLS + TARGETS]
        .first()
        .reset_index()
    )
    cycle_df = cycle_agg.merge(first_per_cycle, on=["ESN", "Cycles_Since_New"])
    return cycle_df.sort_values(["ESN", "Cycles_Since_New"]).reset_index(drop=True)


def add_cycles_since_last(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add Cycles_Since_Last_* features by detecting counter increments.

    These turned out to be the most important features (see SHAP analysis).
    """
    pairs = [
        ("Cumulative_WWs", "Cycles_Since_Last_WW"),
        ("Cumulative_HPC_SVs", "Cycles_Since_Last_HPC_SV"),
        ("Cumulative_HPT_SVs", "Cycles_Since_Last_HPT_SV"),
    ]
    for cum_col, since_col in pairs:
        vals = []
        for esn in sorted(cycle_df["ESN"].unique()):
            mask = cycle_df["ESN"] == esn
            cum = cycle_df.loc[mask, cum_col].values
            cycles = cycle_df.loc[mask, "Cycles_Since_New"].values
            last_event_cycle = cycles[0]
            result = np.zeros(len(cum))
            for j in range(len(cum)):
                if j > 0 and cum[j] > cum[j - 1]:
                    last_event_cycle = cycles[j]
                result[j] = cycles[j] - last_event_cycle
            vals.extend(result)
        cycle_df[since_col] = vals
    return cycle_df


def add_delta_and_rolling(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add delta, multi-scale rolling means, and rolling std features."""
    sensor_cols = [c for c in cycle_df.columns if c.startswith("Sensed_")
                   and c.endswith("_mean")]
    for mc in sensor_cols:
        base = mc.replace("_mean", "")
        # Delta
        cycle_df[f"{base}_delta"] = (
            cycle_df.groupby("ESN")[mc].diff().fillna(0)
        )
        # Rolling means at multiple scales
        for w in [5, 20, 50]:
            cycle_df[f"{base}_roll{w}"] = cycle_df.groupby("ESN")[mc].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
        # Rolling std for degradation volatility
        for w in [10, 20]:
            cycle_df[f"{base}_rollstd{w}"] = (
                cycle_df.groupby("ESN")[mc]
                .transform(lambda s: s.rolling(w, min_periods=2).std())
                .fillna(0)
            )
    return cycle_df


def add_tsfresh_features(
    cycle_df: pd.DataFrame,
    sensors: list = TSFRESH_SENSORS,
    window: int = 10,
) -> pd.DataFrame:
    """Extract tsfresh MinimalFCParameters features via rolling sub-series."""
    all_features = []
    for sensor in sensors:
        mc = f"{sensor}_mean"
        rows = []
        for esn in sorted(cycle_df["ESN"].unique()):
            ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
            cycles = ed["Cycles_Since_New"].values
            values = ed[mc].values
            for i in range(len(cycles)):
                start = max(0, i - window + 1)
                win_vals = values[start : i + 1]
                cid = f"{esn}_{int(cycles[i])}"
                for t, v in enumerate(win_vals):
                    rows.append({"id": cid, "time": t, "value": v})

        ts_df = pd.DataFrame(rows)
        feats = extract_features(
            ts_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
        tsfresh_impute(feats)
        feats.columns = [f"tsf_{sensor}_{c}" for c in feats.columns]
        all_features.append(feats)

    combined = pd.concat(all_features, axis=1)
    combined["_id"] = combined.index
    combined["ESN"] = combined["_id"].str.split("_").str[0].astype(int)
    combined["Cycles_Since_New"] = combined["_id"].str.split("_").str[1].astype(int)
    combined = combined.drop(columns=["_id"])

    return cycle_df.merge(
        combined, on=["ESN", "Cycles_Since_New"], how="left"
    ).fillna(0)


def build_all_features(df: pd.DataFrame, include_tsfresh: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    cycle_df = build_cycle_level_features(df)
    cycle_df = add_cycles_since_last(cycle_df)
    cycle_df = add_delta_and_rolling(cycle_df)
    if include_tsfresh:
        cycle_df = add_tsfresh_features(cycle_df)
    return cycle_df


# ══════════════════════════════════════════════════════════════════════════
# Feature selection
# ══════════════════════════════════════════════════════════════════════════

def select_features_by_importance(
    cycle_df: pd.DataFrame,
    feature_cols: list,
    train_frac: float = 0.6,
    random_state: int = RANDOM_STATE,
) -> list:
    """Drop features with zero LightGBM importance across all three targets.

    Uses only the training portion (first ``train_frac`` of each engine's
    timeline) to avoid leakage.
    """
    train_parts = []
    for esn in sorted(cycle_df["ESN"].unique()):
        ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
        train_parts.append(ed.iloc[: int(len(ed) * train_frac)])
    train_df = pd.concat(train_parts)

    X = train_df[feature_cols].values
    imp_sum = np.zeros(len(feature_cols))
    for target in TARGETS:
        m = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        m.fit(X, train_df[target].values)
        imp_sum += m.feature_importances_

    selected = [f for f, imp in zip(feature_cols, imp_sum) if imp > 0]
    return selected


# ══════════════════════════════════════════════════════════════════════════
# Temporal split
# ══════════════════════════════════════════════════════════════════════════

def temporal_split(
    cycle_df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple:
    """Split each engine's timeline chronologically into train/val/test.

    Respects the "no-peek-into-the-future" constraint by ensuring training
    data always precedes validation data, which precedes test data.
    """
    train_parts, val_parts, test_parts = [], [], []
    for esn in sorted(cycle_df["ESN"].unique()):
        ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
        n = len(ed)
        train_parts.append(ed.iloc[: int(n * train_frac)])
        val_parts.append(ed.iloc[int(n * train_frac) : int(n * (train_frac + val_frac))])
        test_parts.append(ed.iloc[int(n * (train_frac + val_frac)) :])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


# ══════════════════════════════════════════════════════════════════════════
# Scoring functions
# ══════════════════════════════════════════════════════════════════════════

def time_weighted_error(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.01, beta: float = 1.0
) -> np.ndarray:
    """Asymmetric time-weighted squared error per sample (PHM 2025 metric)."""
    error = y_pred - y_true
    weight = np.where(
        error >= 0,
        2.0 / (1.0 + alpha * y_true),
        1.0 / (1.0 + alpha * y_true),
    )
    return weight * (error ** 2) * beta


def compute_target_betas(train_df: pd.DataFrame) -> dict:
    """Per-target beta normalization factors from training data."""
    return {
        "Cycles_to_WW": 1.0 / train_df["Cycles_to_WW"].max(),
        "Cycles_to_HPC_SV": 2.0 / train_df["Cycles_to_HPC_SV"].max(),
        "Cycles_to_HPT_SV": 2.0 / train_df["Cycles_to_HPT_SV"].max(),
    }


def competition_score_single(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float, alpha: float = 0.01
) -> float:
    """Competition score for a single target."""
    return float(np.mean(time_weighted_error(y_true, y_pred, alpha, beta)))


def competition_score_all(
    ref_df: pd.DataFrame, preds: dict, train_df: pd.DataFrame = None
) -> tuple:
    """Average competition score across the three targets."""
    betas = compute_target_betas(train_df if train_df is not None else ref_df)
    scores = {
        t: competition_score_single(ref_df[t].values, preds[t], betas[t])
        for t in TARGETS
    }
    return float(np.mean(list(scores.values()))), scores


# ══════════════════════════════════════════════════════════════════════════
# Custom asymmetric loss for LightGBM
# ══════════════════════════════════════════════════════════════════════════

def asymmetric_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Custom LightGBM objective aligned with the competition scoring function.

    Loss: L = w(y_pred, y_true) * (y_pred - y_true)^2
    where w is the asymmetric time-weighted factor.

    Returns (gradient, hessian) per the LightGBM sklearn API convention.
    """
    alpha = 0.01
    error = y_pred - y_true
    late_mask = (error >= 0).astype(float)
    late_factor = 1.0 + late_mask  # 1 for early, 2 for late
    weight = late_factor / (1.0 + alpha * y_true)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess
