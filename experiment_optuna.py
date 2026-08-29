"""
Experiment: Longer rolling windows + Per-target Optuna hyperparameter tuning
=============================================================================
1. Adds 20-cycle and 50-cycle rolling features on top of existing 5-cycle
2. Runs Optuna per-target to find best LGBM params using val competition score
3. Compares tuned results vs previous baseline
"""

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
TARGET_LABELS = ["Water-Wash (WW)", "HPC Shop Visit", "HPT Shop Visit"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PIPELINE (same as experiment_plan.py)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Data pipeline")
print("=" * 60)

df = pd.read_csv("training_data.csv")
sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
maint_cols = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]

# Outlier capping
caps = {
    "Sensed_T25": (None, 2000), "Sensed_P25": (0, None),
    "Sensed_Core_Speed": (None, 30000), "Sensed_T3": (None, 3000),
    "Sensed_T45": (0.5, 3500), "Sensed_T5": (0.5, 3500),
    "Sensed_WFuel": (None, 5), "Sensed_Fan_Speed": (100, None),
}
for col, (lo, hi) in caps.items():
    if lo is not None: df.loc[df[col] < lo, col] = np.nan
    if hi is not None: df.loc[df[col] > hi, col] = np.nan

df = df.sort_values(["ESN", "Snapshot", "Cycles_Since_New"]).reset_index(drop=True)
for col in sensor_cols:
    df[col] = df.groupby(["ESN", "Snapshot"])[col].transform(lambda s: s.ffill().bfill())
    df[col] = df[col].fillna(df[col].median())

# Cycle-level aggregation
agg_funcs = {col: ["mean", "std", "min", "max"] for col in sensor_cols}
cycle_agg = df.groupby(["ESN", "Cycles_Since_New"]).agg(agg_funcs)
cycle_agg.columns = [f"{col}_{stat}" for col, stat in cycle_agg.columns]
cycle_agg = cycle_agg.reset_index().fillna(0)

first_per_cycle = df.groupby(["ESN", "Cycles_Since_New"])[maint_cols + TARGETS].first().reset_index()
cycle_df = cycle_agg.merge(first_per_cycle, on=["ESN", "Cycles_Since_New"])
cycle_df = cycle_df.sort_values(["ESN", "Cycles_Since_New"]).reset_index(drop=True)

# Cycles-since-last-maintenance
for cum_col, since_col in [("Cumulative_WWs", "Cycles_Since_Last_WW"),
                            ("Cumulative_HPC_SVs", "Cycles_Since_Last_HPC_SV"),
                            ("Cumulative_HPT_SVs", "Cycles_Since_Last_HPT_SV")]:
    vals = []
    for esn in sorted(cycle_df["ESN"].unique()):
        mask = cycle_df["ESN"] == esn
        cum = cycle_df.loc[mask, cum_col].values
        cyc = cycle_df.loc[mask, "Cycles_Since_New"].values
        last = cyc[0]
        res = np.zeros(len(cum))
        for j in range(len(cum)):
            if j > 0 and cum[j] > cum[j-1]: last = cyc[j]
            res[j] = cyc[j] - last
        vals.extend(res)
    cycle_df[since_col] = vals

# Delta + rolling 5
for col in sensor_cols:
    mc = f"{col}_mean"
    if mc in cycle_df.columns:
        cycle_df[f"{col}_delta"] = cycle_df.groupby("ESN")[mc].diff().fillna(0)
        cycle_df[f"{col}_roll5"] = cycle_df.groupby("ESN")[mc].transform(
            lambda s: s.rolling(5, min_periods=1).mean())

print(f"  Base features: {cycle_df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: LONGER ROLLING WINDOWS (20 and 50 cycles)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Adding 20-cycle and 50-cycle rolling features")
print("=" * 60)

for window in [20, 50]:
    for col in sensor_cols:
        mc = f"{col}_mean"
        if mc in cycle_df.columns:
            cycle_df[f"{col}_roll{window}"] = cycle_df.groupby("ESN")[mc].transform(
                lambda s: s.rolling(window, min_periods=1).mean())
    print(f"  Added roll{window} for {len(sensor_cols)} sensors")

# Also add rolling std (captures volatility / instability)
for window in [10, 20]:
    for col in sensor_cols:
        mc = f"{col}_mean"
        if mc in cycle_df.columns:
            cycle_df[f"{col}_rollstd{window}"] = cycle_df.groupby("ESN")[mc].transform(
                lambda s: s.rolling(window, min_periods=2).std()).fillna(0)
    print(f"  Added rollstd{window} for {len(sensor_cols)} sensors")

print(f"  Columns after rolling: {cycle_df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: tsfresh features (same as experiment_plan.py)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: tsfresh features")
print("=" * 60)

TSFRESH_WINDOW = 10
tsfresh_sensors = [
    "Sensed_TAT", "Sensed_T3", "Sensed_Ps3", "Sensed_T45",
    "Sensed_Fan_Speed", "Sensed_Core_Speed", "Sensed_WFuel",
    "Sensed_Mach", "Sensed_Altitude",
]

all_tsf = []
for sensor in tsfresh_sensors:
    mc = f"{sensor}_mean"
    rows = []
    for esn in sorted(cycle_df["ESN"].unique()):
        ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
        cyc = ed["Cycles_Since_New"].values
        vals = ed[mc].values
        for i in range(len(cyc)):
            s = max(0, i - TSFRESH_WINDOW + 1)
            wv = vals[s:i+1]
            cid = f"{esn}_{int(cyc[i])}"
            for t, v in enumerate(wv):
                rows.append({"id": cid, "time": t, "value": v})
    ts_df = pd.DataFrame(rows)
    feats = extract_features(ts_df, column_id="id", column_sort="time",
                             default_fc_parameters=MinimalFCParameters(),
                             disable_progressbar=True, n_jobs=0)
    tsfresh_impute(feats)
    feats.columns = [f"tsf_{sensor}_{c}" for c in feats.columns]
    all_tsf.append(feats)

tsf_combined = pd.concat(all_tsf, axis=1)
tsf_combined["_id"] = tsf_combined.index
tsf_combined["ESN"] = tsf_combined["_id"].str.split("_").str[0].astype(int)
tsf_combined["Cycles_Since_New"] = tsf_combined["_id"].str.split("_").str[1].astype(int)
tsf_combined = tsf_combined.drop(columns=["_id"])
cycle_df = cycle_df.merge(tsf_combined, on=["ESN", "Cycles_Since_New"], how="left").fillna(0)
print(f"  tsfresh: {len([c for c in tsf_combined.columns if c.startswith('tsf_')])} features")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Feature selection
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Feature selection")
print("=" * 60)

all_feat = [c for c in cycle_df.columns if c not in ["ESN", "Cycles_Since_New"] + TARGETS]
print(f"  Candidates: {len(all_feat)}")

# Use train portion for selection
fs_parts = []
for esn in sorted(cycle_df["ESN"].unique()):
    ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
    fs_parts.append(ed.iloc[:int(len(ed) * 0.6)])
fs_df = pd.concat(fs_parts)

imp_sum = np.zeros(len(all_feat))
for target in TARGETS:
    m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m.fit(fs_df[all_feat].values, fs_df[target].values)
    imp_sum += m.feature_importances_

imp_df = pd.DataFrame({"feature": all_feat, "importance": imp_sum}).sort_values("importance", ascending=False)
selected = imp_df[imp_df["importance"] > 0]["feature"].tolist()
print(f"  Selected: {len(selected)} (dropped {len(all_feat) - len(selected)})")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Temporal split
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Temporal split")
print("=" * 60)

train_parts, val_parts, test_parts = [], [], []
for esn in sorted(cycle_df["ESN"].unique()):
    ed = cycle_df[cycle_df["ESN"] == esn].sort_values("Cycles_Since_New")
    n = len(ed)
    train_parts.append(ed.iloc[:int(n*0.6)])
    val_parts.append(ed.iloc[int(n*0.6):int(n*0.8)])
    test_parts.append(ed.iloc[int(n*0.8):])

train_df = pd.concat(train_parts).reset_index(drop=True)
val_df = pd.concat(val_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

X_train = train_df[selected].values
X_val = val_df[selected].values
X_test = test_df[selected].values
print(f"  Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Features={len(selected)}")

# Competition scoring
def time_weighted_error(y_true, y_pred, alpha=0.01, beta=1):
    error = y_pred - y_true
    weight = np.where(error >= 0, 2/(1+alpha*y_true), 1/(1+alpha*y_true))
    return weight * (error**2) * beta

def comp_score_single(y_true, y_pred, alpha, beta):
    return np.mean(time_weighted_error(y_true, y_pred, alpha, beta))

def asymmetric_objective(y_true, y_pred):
    alpha = 0.01
    error = y_pred - y_true
    late_mask = (error >= 0).astype(float)
    late_factor = 1.0 + late_mask
    weight = late_factor / (1.0 + alpha * y_true)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Baseline (default params, custom loss)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Baseline (default params + custom loss)")
print("=" * 60)

key_map = {"Cycles_to_WW": "WW", "Cycles_to_HPC_SV": "HPC", "Cycles_to_HPT_SV": "HPT"}
target_alpha = 0.01
target_betas = {
    "Cycles_to_WW": 1.0 / train_df["Cycles_to_WW"].max(),
    "Cycles_to_HPC_SV": 2.0 / train_df["Cycles_to_HPC_SV"].max(),
    "Cycles_to_HPT_SV": 2.0 / train_df["Cycles_to_HPT_SV"].max(),
}

baseline_scores = {}
for target, label in zip(TARGETS, TARGET_LABELS):
    m = lgb.LGBMRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        objective=asymmetric_objective,
    )
    m.fit(X_train, train_df[target].values)
    vp = np.clip(m.predict(X_val), 0, None)
    tp = np.clip(m.predict(X_test), 0, None)

    val_cs = comp_score_single(val_df[target].values, vp, target_alpha, target_betas[target])
    val_mae = mean_absolute_error(val_df[target].values, vp)
    val_r2 = r2_score(val_df[target].values, vp)
    test_mae = mean_absolute_error(test_df[target].values, tp)
    test_r2 = r2_score(test_df[target].values, tp)

    baseline_scores[target] = {"val_cs": val_cs, "val_mae": val_mae, "val_r2": val_r2,
                                "test_mae": test_mae, "test_r2": test_r2}
    print(f"  {label}: Val MAE={val_mae:.1f}, R²={val_r2:.3f}, CS={val_cs:.4f} | Test MAE={test_mae:.1f}, R²={test_r2:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Per-target Optuna tuning
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Per-target Optuna tuning (50 trials each)")
print("=" * 60)

best_params = {}
tuned_scores = {}

for target, label in zip(TARGETS, TARGET_LABELS):
    print(f"\n  ── Tuning {label} ──")
    y_tr = train_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values
    beta = target_betas[target]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        m = lgb.LGBMRegressor(
            **params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
            objective=asymmetric_objective,
        )
        m.fit(X_train, y_tr)
        vp = np.clip(m.predict(X_val), 0, None)
        return comp_score_single(y_va, vp, target_alpha, beta)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    bp = study.best_params
    best_params[target] = bp
    print(f"    Best params: {bp}")
    print(f"    Best val CS: {study.best_value:.4f} (baseline: {baseline_scores[target]['val_cs']:.4f})")

    # Retrain with best params and evaluate
    m = lgb.LGBMRegressor(
        **bp, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        objective=asymmetric_objective,
    )
    m.fit(X_train, y_tr)
    vp = np.clip(m.predict(X_val), 0, None)
    tp = np.clip(m.predict(X_test), 0, None)

    val_cs = comp_score_single(y_va, vp, target_alpha, beta)
    val_mae = mean_absolute_error(y_va, vp)
    val_r2 = r2_score(y_va, vp)
    test_mae = mean_absolute_error(y_te, tp)
    test_r2 = r2_score(y_te, tp)
    late_pct = 100 * (vp > y_va).mean()

    tuned_scores[target] = {"val_cs": val_cs, "val_mae": val_mae, "val_r2": val_r2,
                             "test_mae": test_mae, "test_r2": test_r2, "late_pct": late_pct}
    print(f"    Tuned: Val MAE={val_mae:.1f}, R²={val_r2:.3f}, CS={val_cs:.4f}, Late%={late_pct:.1f}%")
    print(f"           Test MAE={test_mae:.1f}, R²={test_r2:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 8: Summary comparison
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Summary — Baseline vs Tuned")
print("=" * 60)

print(f"\n  {'Target':<20s} {'Metric':<12s} {'Baseline':>10s} {'Tuned':>10s} {'Change':>10s}")
print(f"  {'-'*62}")

for target, label in zip(TARGETS, TARGET_LABELS):
    b = baseline_scores[target]
    t = tuned_scores[target]
    for metric in ["val_cs", "val_mae", "val_r2", "test_mae", "test_r2"]:
        bv, tv = b[metric], t[metric]
        change = tv - bv
        sign = "+" if change > 0 else ""
        print(f"  {label:<20s} {metric:<12s} {bv:>10.3f} {tv:>10.3f} {sign}{change:>9.3f}")
    print()

# Overall competition score
val_cs_baseline = np.mean([baseline_scores[t]["val_cs"] for t in TARGETS])
val_cs_tuned = np.mean([tuned_scores[t]["val_cs"] for t in TARGETS])
test_cs_baseline_parts = []
test_cs_tuned_parts = []

for target in TARGETS:
    # Recompute test competition scores
    b_model = lgb.LGBMRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        objective=asymmetric_objective,
    )
    b_model.fit(X_train, train_df[target].values)
    b_tp = np.clip(b_model.predict(X_test), 0, None)
    test_cs_baseline_parts.append(comp_score_single(test_df[target].values, b_tp, target_alpha, target_betas[target]))

    t_model = lgb.LGBMRegressor(
        **best_params[target], random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        objective=asymmetric_objective,
    )
    t_model.fit(X_train, train_df[target].values)
    t_tp = np.clip(t_model.predict(X_test), 0, None)
    test_cs_tuned_parts.append(comp_score_single(test_df[target].values, t_tp, target_alpha, target_betas[target]))

test_cs_baseline = np.mean(test_cs_baseline_parts)
test_cs_tuned = np.mean(test_cs_tuned_parts)

print(f"  Overall Val Competition Score:  Baseline={val_cs_baseline:.4f}  Tuned={val_cs_tuned:.4f}  Change={val_cs_tuned-val_cs_baseline:+.4f}")
print(f"  Overall Test Competition Score: Baseline={test_cs_baseline:.4f}  Tuned={test_cs_tuned:.4f}  Change={test_cs_tuned-test_cs_baseline:+.4f}")

# Save best params
import json
with open("experiment_results/optuna_best_params.json", "w") as f:
    json.dump({t: best_params[t] for t in TARGETS}, f, indent=2)
print(f"\n  Best params saved to experiment_results/optuna_best_params.json")
print("\nDone.")
