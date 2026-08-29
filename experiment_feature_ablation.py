"""
Feature Category Ablation Study
================================
Tests the isolated contribution of each feature category by dropping
each category one at a time and re-training LightGBM.

Categories:
  - Cycle-level aggregation (mean/std/min/max)
  - Delta features
  - Rolling averages (5/20/50-cycle)
  - Rolling volatility (10/20-cycle std)
  - tsfresh features
  - Maintenance counters + cycles-since-last
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
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

# ══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (condensed)
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv("training_data.csv")
sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
maint_cols = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]

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

agg_funcs={col:["mean","std","min","max"] for col in sensor_cols}
ca=df.groupby(["ESN","Cycles_Since_New"]).agg(agg_funcs)
ca.columns=[f"{c}_{s}" for c,s in ca.columns]; ca=ca.reset_index().fillna(0)
fp=df.groupby(["ESN","Cycles_Since_New"])[maint_cols+TARGETS].first().reset_index()
cycle_df=ca.merge(fp,on=["ESN","Cycles_Since_New"]).sort_values(["ESN","Cycles_Since_New"]).reset_index(drop=True)

for cc,sc in [("Cumulative_WWs","Cycles_Since_Last_WW"),("Cumulative_HPC_SVs","Cycles_Since_Last_HPC_SV"),
              ("Cumulative_HPT_SVs","Cycles_Since_Last_HPT_SV")]:
    v=[]
    for esn in sorted(cycle_df["ESN"].unique()):
        m=cycle_df["ESN"]==esn;cum=cycle_df.loc[m,cc].values;cyc=cycle_df.loc[m,"Cycles_Since_New"].values
        last=cyc[0];r=np.zeros(len(cum))
        for j in range(len(cum)):
            if j>0 and cum[j]>cum[j-1]:last=cyc[j]
            r[j]=cyc[j]-last
        v.extend(r)
    cycle_df[sc]=v

for col in sensor_cols:
    mc=f"{col}_mean"
    if mc in cycle_df.columns:
        cycle_df[f"{col}_delta"]=cycle_df.groupby("ESN")[mc].diff().fillna(0)
        for w in [5,20,50]:
            cycle_df[f"{col}_roll{w}"]=cycle_df.groupby("ESN")[mc].transform(lambda s:s.rolling(w,min_periods=1).mean())
        for w in [10,20]:
            cycle_df[f"{col}_rollstd{w}"]=cycle_df.groupby("ESN")[mc].transform(lambda s:s.rolling(w,min_periods=2).std()).fillna(0)

tsf_sensors=["Sensed_TAT","Sensed_T3","Sensed_Ps3","Sensed_T45","Sensed_Fan_Speed",
             "Sensed_Core_Speed","Sensed_WFuel","Sensed_Mach","Sensed_Altitude"]
all_tsf=[]
for sensor in tsf_sensors:
    mc=f"{sensor}_mean";rows=[]
    for esn in sorted(cycle_df["ESN"].unique()):
        ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New")
        cyc=ed["Cycles_Since_New"].values;vals=ed[mc].values
        for i in range(len(cyc)):
            s=max(0,i-9);wv=vals[s:i+1];cid=f"{esn}_{int(cyc[i])}"
            for t,v in enumerate(wv):rows.append({"id":cid,"time":t,"value":v})
    ts_df=pd.DataFrame(rows)
    feats=extract_features(ts_df,column_id="id",column_sort="time",default_fc_parameters=MinimalFCParameters(),disable_progressbar=True,n_jobs=0)
    tsfresh_impute(feats);feats.columns=[f"tsf_{sensor}_{c}" for c in feats.columns];all_tsf.append(feats)
tsf_c=pd.concat(all_tsf,axis=1);tsf_c["_id"]=tsf_c.index
tsf_c["ESN"]=tsf_c["_id"].str.split("_").str[0].astype(int)
tsf_c["Cycles_Since_New"]=tsf_c["_id"].str.split("_").str[1].astype(int)
tsf_c=tsf_c.drop(columns=["_id"])
cycle_df=cycle_df.merge(tsf_c,on=["ESN","Cycles_Since_New"],how="left").fillna(0)

# Define feature categories
all_feat = [c for c in cycle_df.columns if c not in ["ESN","Cycles_Since_New"]+TARGETS]

def categorize(feat):
    if feat.startswith("tsf_"): return "tsfresh"
    if feat in maint_cols: return "maintenance_counters"
    if feat.startswith("Cycles_Since_Last"): return "cycles_since_last"
    if "_delta" in feat: return "delta"
    if "_rollstd" in feat: return "rolling_std"
    if "_roll" in feat: return "rolling_mean"
    if any(feat.endswith(s) for s in ["_mean","_std","_min","_max"]): return "cycle_agg"
    return "other"

categories = {}
for f in all_feat:
    cat = categorize(f)
    categories.setdefault(cat, []).append(f)
print(f"Feature categories:")
for cat, feats in categories.items():
    print(f"  {cat}: {len(feats)} features")

# Temporal split
tr,va,te=[],[],[]
for esn in sorted(cycle_df["ESN"].unique()):
    ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New");n=len(ed)
    tr.append(ed.iloc[:int(n*0.6)]);va.append(ed.iloc[int(n*0.6):int(n*0.8)]);te.append(ed.iloc[int(n*0.8):])
train_df=pd.concat(tr).reset_index(drop=True);val_df=pd.concat(va).reset_index(drop=True);test_df=pd.concat(te).reset_index(drop=True)

# Load Optuna params
with open("experiment_results/optuna_best_params.json") as f:
    optuna_params = json.load(f)

def time_weighted_error(y_true, y_pred, alpha=0.01, beta=1):
    error = y_pred - y_true
    weight = np.where(error >= 0, 2/(1+alpha*y_true), 1/(1+alpha*y_true))
    return weight * (error**2) * beta

def asymmetric_objective(y_true, y_pred):
    alpha = 0.01; error = y_pred - y_true
    lm = (error >= 0).astype(float); lf = 1.0 + lm
    w = lf / (1.0 + alpha * y_true)
    return 2.0 * w * error, 2.0 * w

target_betas = {
    "Cycles_to_WW": 1.0 / train_df["Cycles_to_WW"].max(),
    "Cycles_to_HPC_SV": 2.0 / train_df["Cycles_to_HPC_SV"].max(),
    "Cycles_to_HPT_SV": 2.0 / train_df["Cycles_to_HPT_SV"].max(),
}

def evaluate_config(features_to_use):
    """Train LGBM per target on given features, return avg val comp score."""
    X_tr = train_df[features_to_use].values
    X_va = val_df[features_to_use].values
    scores = {}
    for target in TARGETS:
        m = lgb.LGBMRegressor(**optuna_params[target], random_state=RANDOM_STATE,
                               verbose=-1, n_jobs=-1, objective=asymmetric_objective)
        m.fit(X_tr, train_df[target].values)
        vp = np.clip(m.predict(X_va), 0, None)
        mae = mean_absolute_error(val_df[target].values, vp)
        r2 = r2_score(val_df[target].values, vp)
        cs = np.mean(time_weighted_error(val_df[target].values, vp, 0.01, target_betas[target]))
        scores[target] = {"mae": mae, "r2": r2, "cs": cs}
    avg_cs = np.mean([scores[t]["cs"] for t in TARGETS])
    return avg_cs, scores

# Full feature set baseline
print("\n" + "=" * 60)
print("FEATURE ABLATION: Drop each category one at a time")
print("=" * 60)

full_cs, full_scores = evaluate_config(all_feat)
print(f"\nFull ({len(all_feat)} features):")
print(f"  Val CS: {full_cs:.4f}")
for t in TARGETS:
    print(f"    {t}: MAE={full_scores[t]['mae']:.1f}, R²={full_scores[t]['r2']:.3f}")

# Drop each category
results = {"Full": {"n_features": len(all_feat), "val_cs": full_cs, **{f"{t}_mae": full_scores[t]["mae"] for t in TARGETS}, **{f"{t}_r2": full_scores[t]["r2"] for t in TARGETS}}}

for drop_cat, drop_feats in categories.items():
    kept = [f for f in all_feat if f not in drop_feats]
    cs, scores = evaluate_config(kept)
    delta = cs - full_cs
    print(f"\n− {drop_cat} ({len(drop_feats)} features dropped, {len(kept)} kept):")
    print(f"  Val CS: {cs:.4f} (Δ={delta:+.4f})")
    for t in TARGETS:
        print(f"    {t}: MAE={scores[t]['mae']:.1f}, R²={scores[t]['r2']:.3f}")
    results[f"w/o {drop_cat}"] = {
        "n_features": len(kept), "val_cs": cs,
        **{f"{t}_mae": scores[t]["mae"] for t in TARGETS},
        **{f"{t}_r2": scores[t]["r2"] for t in TARGETS},
    }

# Save
import pandas as pd
results_df = pd.DataFrame(results).T
results_df.to_csv("experiment_results/feature_ablation_results.csv")
print(f"\n→ Saved: experiment_results/feature_ablation_results.csv")
print("\nDone.")
