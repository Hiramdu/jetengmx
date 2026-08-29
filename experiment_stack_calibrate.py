"""
Experiment: Stacking Ensemble + Post-hoc Calibration
=====================================================
1. Reuses the data pipeline from experiment_optuna.py (longer rolling + tsfresh + selection)
2. Trains 4 base models per target: Ridge, GBR, RF, LGBM (with Optuna params)
3. Stacking: uses base model val predictions as meta-features for a Ridge meta-learner
4. Post-hoc calibration: per-RUL-bin bias correction on validation residuals
5. Compares: Base LGBM vs Stacking vs Stacking+Calibration
"""

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
TARGET_LABELS = ["Water-Wash (WW)", "HPC Shop Visit", "HPT Shop Visit"]

# ══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (same as experiment_optuna.py — condensed)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("DATA PIPELINE")
print("=" * 60)

df = pd.read_csv("training_data.csv")
sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
maint_cols = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]

# Outlier capping + imputation
caps = {"Sensed_T25":(None,2000),"Sensed_P25":(0,None),"Sensed_Core_Speed":(None,30000),
        "Sensed_T3":(None,3000),"Sensed_T45":(0.5,3500),"Sensed_T5":(0.5,3500),
        "Sensed_WFuel":(None,5),"Sensed_Fan_Speed":(100,None)}
for col,(lo,hi) in caps.items():
    if lo: df.loc[df[col]<lo,col]=np.nan
    if hi: df.loc[df[col]>hi,col]=np.nan
df=df.sort_values(["ESN","Snapshot","Cycles_Since_New"]).reset_index(drop=True)
for col in sensor_cols:
    df[col]=df.groupby(["ESN","Snapshot"])[col].transform(lambda s:s.ffill().bfill())
    df[col]=df[col].fillna(df[col].median())

# Cycle-level agg
agg_funcs={col:["mean","std","min","max"] for col in sensor_cols}
ca=df.groupby(["ESN","Cycles_Since_New"]).agg(agg_funcs)
ca.columns=[f"{c}_{s}" for c,s in ca.columns]
ca=ca.reset_index().fillna(0)
fp=df.groupby(["ESN","Cycles_Since_New"])[maint_cols+TARGETS].first().reset_index()
cycle_df=ca.merge(fp,on=["ESN","Cycles_Since_New"]).sort_values(["ESN","Cycles_Since_New"]).reset_index(drop=True)

# Cycles-since-last
for cc,sc in [("Cumulative_WWs","Cycles_Since_Last_WW"),
              ("Cumulative_HPC_SVs","Cycles_Since_Last_HPC_SV"),
              ("Cumulative_HPT_SVs","Cycles_Since_Last_HPT_SV")]:
    v=[]
    for esn in sorted(cycle_df["ESN"].unique()):
        m=cycle_df["ESN"]==esn; cum=cycle_df.loc[m,cc].values; cyc=cycle_df.loc[m,"Cycles_Since_New"].values
        last=cyc[0]; r=np.zeros(len(cum))
        for j in range(len(cum)):
            if j>0 and cum[j]>cum[j-1]: last=cyc[j]
            r[j]=cyc[j]-last
        v.extend(r)
    cycle_df[sc]=v

# Delta + rolling 5/20/50 + rollstd 10/20
for col in sensor_cols:
    mc=f"{col}_mean"
    if mc in cycle_df.columns:
        cycle_df[f"{col}_delta"]=cycle_df.groupby("ESN")[mc].diff().fillna(0)
        for w in [5,20,50]:
            cycle_df[f"{col}_roll{w}"]=cycle_df.groupby("ESN")[mc].transform(lambda s:s.rolling(w,min_periods=1).mean())
        for w in [10,20]:
            cycle_df[f"{col}_rollstd{w}"]=cycle_df.groupby("ESN")[mc].transform(lambda s:s.rolling(w,min_periods=2).std()).fillna(0)

# tsfresh
tsf_sensors=["Sensed_TAT","Sensed_T3","Sensed_Ps3","Sensed_T45","Sensed_Fan_Speed",
             "Sensed_Core_Speed","Sensed_WFuel","Sensed_Mach","Sensed_Altitude"]
all_tsf=[]
for sensor in tsf_sensors:
    mc=f"{sensor}_mean"; rows=[]
    for esn in sorted(cycle_df["ESN"].unique()):
        ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New")
        cyc=ed["Cycles_Since_New"].values; vals=ed[mc].values
        for i in range(len(cyc)):
            s=max(0,i-9); wv=vals[s:i+1]; cid=f"{esn}_{int(cyc[i])}"
            for t,v in enumerate(wv): rows.append({"id":cid,"time":t,"value":v})
    ts_df=pd.DataFrame(rows)
    feats=extract_features(ts_df,column_id="id",column_sort="time",
                           default_fc_parameters=MinimalFCParameters(),disable_progressbar=True,n_jobs=0)
    tsfresh_impute(feats)
    feats.columns=[f"tsf_{sensor}_{c}" for c in feats.columns]
    all_tsf.append(feats)
tsf_c=pd.concat(all_tsf,axis=1)
tsf_c["_id"]=tsf_c.index
tsf_c["ESN"]=tsf_c["_id"].str.split("_").str[0].astype(int)
tsf_c["Cycles_Since_New"]=tsf_c["_id"].str.split("_").str[1].astype(int)
tsf_c=tsf_c.drop(columns=["_id"])
cycle_df=cycle_df.merge(tsf_c,on=["ESN","Cycles_Since_New"],how="left").fillna(0)

# Feature selection
all_feat=[c for c in cycle_df.columns if c not in ["ESN","Cycles_Since_New"]+TARGETS]
fs_parts=[]
for esn in sorted(cycle_df["ESN"].unique()):
    ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New")
    fs_parts.append(ed.iloc[:int(len(ed)*0.6)])
fs_df=pd.concat(fs_parts)
imp=np.zeros(len(all_feat))
for t in TARGETS:
    m=lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,verbose=-1,n_jobs=-1)
    m.fit(fs_df[all_feat].values,fs_df[t].values)
    imp+=m.feature_importances_
selected=[all_feat[i] for i in range(len(all_feat)) if imp[i]>0]

# Split
tr,va,te=[],[],[]
for esn in sorted(cycle_df["ESN"].unique()):
    ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New"); n=len(ed)
    tr.append(ed.iloc[:int(n*0.6)]); va.append(ed.iloc[int(n*0.6):int(n*0.8)]); te.append(ed.iloc[int(n*0.8):])
train_df=pd.concat(tr).reset_index(drop=True)
val_df=pd.concat(va).reset_index(drop=True)
test_df=pd.concat(te).reset_index(drop=True)

X_train=train_df[selected].values
X_val=val_df[selected].values
X_test=test_df[selected].values
print(f"  Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Features={len(selected)}")


# ══════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def time_weighted_error(y_true, y_pred, alpha=0.01, beta=1):
    error = y_pred - y_true
    weight = np.where(error >= 0, 2/(1+alpha*y_true), 1/(1+alpha*y_true))
    return weight * (error**2) * beta

def comp_score_single(y_true, y_pred, alpha, beta):
    return np.mean(time_weighted_error(y_true, y_pred, alpha, beta))

def comp_score_all(val_df, preds_dict):
    alpha = 0.01
    betas = {"Cycles_to_WW": 1.0/val_df["Cycles_to_WW"].max(),
             "Cycles_to_HPC_SV": 2.0/val_df["Cycles_to_HPC_SV"].max(),
             "Cycles_to_HPT_SV": 2.0/val_df["Cycles_to_HPT_SV"].max()}
    scores = {}
    for t in TARGETS:
        scores[t] = comp_score_single(val_df[t].values, preds_dict[t], alpha, betas[t])
    return np.mean(list(scores.values())), scores

def asymmetric_objective(y_true, y_pred):
    alpha = 0.01
    error = y_pred - y_true
    late_mask = (error >= 0).astype(float)
    late_factor = 1.0 + late_mask
    weight = late_factor / (1.0 + alpha * y_true)
    return 2.0 * weight * error, 2.0 * weight

# Load Optuna best params
with open("experiment_results/optuna_best_params.json") as f:
    optuna_params = json.load(f)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: TRAIN BASE MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Train 4 base models per target")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

base_model_configs = {
    "Ridge": lambda: Ridge(alpha=10.0),
    "GBR": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                              subsample=0.8, random_state=RANDOM_STATE),
    "RF": lambda: RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3,
                                         random_state=RANDOM_STATE, n_jobs=-1),
    "LGBM_tuned": None,  # per-target params
}

# Store predictions: base_preds[target][model_name] = {"val": array, "test": array}
base_preds = {t: {} for t in TARGETS}
base_models = {t: {} for t in TARGETS}

for target, label in zip(TARGETS, TARGET_LABELS):
    print(f"\n  {label}:")
    y_tr = train_df[target].values
    y_va = val_df[target].values

    for model_name, model_fn in base_model_configs.items():
        if model_name == "LGBM_tuned":
            params = optuna_params[target]
            model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1,
                                       n_jobs=-1, objective=asymmetric_objective)
            model.fit(X_train, y_tr)
            vp = np.clip(model.predict(X_val), 0, None)
            tp = np.clip(model.predict(X_test), 0, None)
        elif model_name == "Ridge":
            model = model_fn()
            model.fit(X_train_sc, y_tr)
            vp = np.clip(model.predict(X_val_sc), 0, None)
            tp = np.clip(model.predict(X_test_sc), 0, None)
        else:
            model = model_fn()
            model.fit(X_train, y_tr)
            vp = np.clip(model.predict(X_val), 0, None)
            tp = np.clip(model.predict(X_test), 0, None)

        val_mae = mean_absolute_error(y_va, vp)
        val_r2 = r2_score(y_va, vp)
        print(f"    {model_name:<12s}: Val MAE={val_mae:.1f}, R²={val_r2:.3f}")

        base_preds[target][model_name] = {"val": vp, "test": tp}
        base_models[target][model_name] = model


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Stacking ensemble (Ridge meta-learner)")
print("=" * 60)

# For each target, build meta-features from base model predictions on val set.
# But we need out-of-fold predictions on train set for the meta-learner to train on.
# Strategy: use 2-fold temporal split within train set to get OOF predictions.

stack_preds = {}

for target, label in zip(TARGETS, TARGET_LABELS):
    print(f"\n  {label}:")
    y_tr = train_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values

    model_names = list(base_model_configs.keys())

    # Generate OOF predictions on train set (2-fold temporal within train)
    n_tr = len(train_df)
    mid = n_tr // 2
    oof_preds = np.zeros((n_tr, len(model_names)))

    for fold, (tr_idx, va_idx) in enumerate([(range(0, mid), range(mid, n_tr)),
                                               (range(0, n_tr), range(0, 0))]):
        # For simplicity, use first half to predict second half, then full train for first half
        pass

    # Simpler approach: train base models on train, predict val, use val predictions as meta-features
    # Train meta-learner on val, evaluate on test (this is valid since val is held-out from base training)
    # But we need a separate meta-train set. Use a 50/50 split of val.
    n_va = len(val_df)
    meta_train_idx = range(0, n_va // 2)
    meta_val_idx = range(n_va // 2, n_va)

    # Build meta-features for val set
    meta_X_val = np.column_stack([base_preds[target][mn]["val"] for mn in model_names])
    meta_X_test = np.column_stack([base_preds[target][mn]["test"] for mn in model_names])

    # Also include the original features for the meta-learner (helps it learn when to trust which model)
    # Add Cycles_Since_Last as context
    context_cols = ["Cycles_Since_Last_WW", "Cycles_Since_Last_HPC_SV", "Cycles_Since_Last_HPT_SV"]
    ctx_idx = [selected.index(c) for c in context_cols if c in selected]

    meta_X_val_full = np.column_stack([meta_X_val, X_val[:, ctx_idx]])
    meta_X_test_full = np.column_stack([meta_X_test, X_test[:, ctx_idx]])

    # Train meta-learner on first half of val, evaluate on second half
    meta_train_X = meta_X_val_full[list(meta_train_idx)]
    meta_train_y = y_va[list(meta_train_idx)]
    meta_val_X = meta_X_val_full[list(meta_val_idx)]
    meta_val_y = y_va[list(meta_val_idx)]

    # Use Ridge as meta-learner (simple, less overfitting risk)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train_X, meta_train_y)

    # Retrain on full val for final test predictions
    meta_model_full = Ridge(alpha=1.0)
    meta_model_full.fit(meta_X_val_full, y_va)

    # Predictions
    stack_val_pred = np.clip(meta_model_full.predict(meta_X_val_full), 0, None)  # in-sample on val (for comparison)
    stack_test_pred = np.clip(meta_model_full.predict(meta_X_test_full), 0, None)

    # Also try simple average as baseline ensemble
    avg_val_pred = np.clip(np.mean([base_preds[target][mn]["val"] for mn in model_names], axis=0), 0, None)
    avg_test_pred = np.clip(np.mean([base_preds[target][mn]["test"] for mn in model_names], axis=0), 0, None)

    stack_preds[target] = {
        "stack_val": stack_val_pred, "stack_test": stack_test_pred,
        "avg_val": avg_val_pred, "avg_test": avg_test_pred,
        "meta_model": meta_model_full,
    }

    # Report
    lgbm_val_mae = mean_absolute_error(y_va, base_preds[target]["LGBM_tuned"]["val"])
    lgbm_val_r2 = r2_score(y_va, base_preds[target]["LGBM_tuned"]["val"])
    avg_val_mae = mean_absolute_error(y_va, avg_val_pred)
    avg_val_r2 = r2_score(y_va, avg_val_pred)
    stack_val_mae = mean_absolute_error(y_va, stack_val_pred)
    stack_val_r2 = r2_score(y_va, stack_val_pred)

    print(f"    LGBM_tuned:  Val MAE={lgbm_val_mae:.1f}, R²={lgbm_val_r2:.3f}")
    print(f"    Simple Avg:  Val MAE={avg_val_mae:.1f}, R²={avg_val_r2:.3f}")
    print(f"    Stack(Ridge):Val MAE={stack_val_mae:.1f}, R²={stack_val_r2:.3f}")

    # Meta-learner weights
    coefs = dict(zip(model_names + [f"ctx_{c}" for c in context_cols if c in selected],
                      meta_model_full.coef_))
    print(f"    Meta weights: {', '.join(f'{k}={v:.3f}' for k,v in list(coefs.items())[:4])}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: POST-HOC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Post-hoc calibration (per-RUL-bin bias correction)")
print("=" * 60)

# For each target, bin the validation predictions by predicted RUL,
# compute the median residual per bin, and subtract it.
# This corrects systematic bias at different RUL ranges.

N_BINS = 20
calibrated_preds = {}

for target, label in zip(TARGETS, TARGET_LABELS):
    print(f"\n  {label}:")
    y_va = val_df[target].values

    # Use stacked predictions as the base for calibration
    raw_val = stack_preds[target]["stack_val"]
    raw_test = stack_preds[target]["stack_test"]

    # Compute residuals on val
    residuals = raw_val - y_va

    # Bin by predicted value
    bin_edges = np.linspace(0, raw_val.max() * 1.01, N_BINS + 1)
    bin_idx = np.digitize(raw_val, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

    # Compute median residual per bin
    bin_corrections = np.zeros(N_BINS)
    for b in range(N_BINS):
        mask = bin_idx == b
        if mask.sum() > 5:
            bin_corrections[b] = np.median(residuals[mask])

    # Apply correction to val
    cal_val = raw_val - bin_corrections[bin_idx]
    cal_val = np.clip(cal_val, 0, None)

    # Apply correction to test (bin by predicted value)
    test_bin_idx = np.digitize(raw_test, bin_edges) - 1
    test_bin_idx = np.clip(test_bin_idx, 0, N_BINS - 1)
    cal_test = raw_test - bin_corrections[test_bin_idx]
    cal_test = np.clip(cal_test, 0, None)

    calibrated_preds[target] = {"val": cal_val, "test": cal_test}

    # Report
    raw_mae = mean_absolute_error(y_va, raw_val)
    cal_mae = mean_absolute_error(y_va, cal_val)
    raw_r2 = r2_score(y_va, raw_val)
    cal_r2 = r2_score(y_va, cal_val)
    raw_late = 100 * (raw_val > y_va).mean()
    cal_late = 100 * (cal_val > y_va).mean()

    print(f"    Before cal: Val MAE={raw_mae:.1f}, R²={raw_r2:.3f}, Late%={raw_late:.1f}%")
    print(f"    After cal:  Val MAE={cal_mae:.1f}, R²={cal_r2:.3f}, Late%={cal_late:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Final comparison")
print("=" * 60)

approaches = {
    "LGBM_tuned": {t: base_preds[t]["LGBM_tuned"] for t in TARGETS},
    "Simple_Avg": {t: {"val": stack_preds[t]["avg_val"], "test": stack_preds[t]["avg_test"]} for t in TARGETS},
    "Stack": {t: {"val": stack_preds[t]["stack_val"], "test": stack_preds[t]["stack_test"]} for t in TARGETS},
    "Stack+Cal": {t: calibrated_preds[t] for t in TARGETS},
}

print(f"\n  {'Approach':<16s} {'Val CS':>10s} {'Test CS':>10s} | {'WW Val R²':>10s} {'HPC Val R²':>10s} {'HPT Val R²':>10s}")
print(f"  {'-'*78}")

for name, preds in approaches.items():
    val_cs, _ = comp_score_all(val_df, {t: preds[t]["val"] for t in TARGETS})
    test_cs, _ = comp_score_all(test_df, {t: preds[t]["test"] for t in TARGETS})

    r2s = []
    for t in TARGETS:
        r2s.append(r2_score(val_df[t].values, preds[t]["val"]))

    print(f"  {name:<16s} {val_cs:>10.2f} {test_cs:>10.2f} | {r2s[0]:>10.3f} {r2s[1]:>10.3f} {r2s[2]:>10.3f}")

# Detailed per-target for best approach
print(f"\n  ── Stack+Cal detailed results ──")
for target, label in zip(TARGETS, TARGET_LABELS):
    y_va = val_df[target].values
    y_te = test_df[target].values
    vp = calibrated_preds[target]["val"]
    tp = calibrated_preds[target]["test"]
    print(f"    {label}:")
    print(f"      Val  MAE={mean_absolute_error(y_va,vp):.1f}, R²={r2_score(y_va,vp):.3f}, Late%={100*(vp>y_va).mean():.1f}%")
    print(f"      Test MAE={mean_absolute_error(y_te,tp):.1f}, R²={r2_score(y_te,tp):.3f}")

print("\nDone.")
