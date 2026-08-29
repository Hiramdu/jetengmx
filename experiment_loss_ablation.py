"""
Ablation Study: Training Objective Alignment for Asymmetric Prognostic Metrics
===============================================================================
Compares 3 loss functions:
  1. MSE (default) — symmetric, no time-weighting
  2. Asymmetric-only — late predictions penalized 2x, no time-weighting
  3. Asymmetric + time-weighted — full competition-aligned loss

For each, we measure: MAE, R², competition score, late prediction %, mean bias
"""

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
TARGET_LABELS = ["WW", "HPC", "HPT"]
OUTPUT_DIR = Path("experiment_results")

# ══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (condensed)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("DATA PIPELINE")
print("=" * 60)

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

all_feat=[c for c in cycle_df.columns if c not in ["ESN","Cycles_Since_New"]+TARGETS]
fs_parts=[]
for esn in sorted(cycle_df["ESN"].unique()):
    ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New");fs_parts.append(ed.iloc[:int(len(ed)*0.6)])
fs_df=pd.concat(fs_parts);imp=np.zeros(len(all_feat))
for t in TARGETS:
    m=lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,verbose=-1,n_jobs=-1)
    m.fit(fs_df[all_feat].values,fs_df[t].values);imp+=m.feature_importances_
selected=[all_feat[i] for i in range(len(all_feat)) if imp[i]>0]

tr,va,te=[],[],[]
for esn in sorted(cycle_df["ESN"].unique()):
    ed=cycle_df[cycle_df["ESN"]==esn].sort_values("Cycles_Since_New");n=len(ed)
    tr.append(ed.iloc[:int(n*0.6)]);va.append(ed.iloc[int(n*0.6):int(n*0.8)]);te.append(ed.iloc[int(n*0.8):])
train_df=pd.concat(tr).reset_index(drop=True);val_df=pd.concat(va).reset_index(drop=True);test_df=pd.concat(te).reset_index(drop=True)
X_train=train_df[selected].values;X_val=val_df[selected].values;X_test=test_df[selected].values
print(f"  Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Features={len(selected)}")


# ══════════════════════════════════════════════════════════════════════════
# DEFINE 3 LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def loss_mse(y_true, y_pred):
    """Standard MSE — symmetric, no time-weighting."""
    error = y_pred - y_true
    grad = 2.0 * error
    hess = 2.0 * np.ones_like(error)
    return grad, hess

def loss_asymmetric_only(y_true, y_pred):
    """Asymmetric loss — late predictions penalized 2x, but NO time-weighting."""
    error = y_pred - y_true
    late_mask = (error >= 0).astype(float)
    late_factor = 1.0 + late_mask  # 1 for early, 2 for late
    grad = 2.0 * late_factor * error
    hess = 2.0 * late_factor
    return grad, hess

def loss_full_competition(y_true, y_pred):
    """Full competition-aligned loss — asymmetric + time-weighted."""
    alpha = 0.01
    error = y_pred - y_true
    late_mask = (error >= 0).astype(float)
    late_factor = 1.0 + late_mask
    weight = late_factor / (1.0 + alpha * y_true)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess

# Competition scoring
def time_weighted_error(y_true, y_pred, alpha=0.01, beta=1):
    error = y_pred - y_true
    weight = np.where(error >= 0, 2/(1+alpha*y_true), 1/(1+alpha*y_true))
    return weight * (error**2) * beta

def comp_score_single(y_true, y_pred, alpha, beta):
    return np.mean(time_weighted_error(y_true, y_pred, alpha, beta))

# Load Optuna params
with open("experiment_results/optuna_best_params.json") as f:
    optuna_params = json.load(f)

# ══════════════════════════════════════════════════════════════════════════
# ABLATION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ABLATION: MSE vs Asymmetric-Only vs Full Competition Loss")
print("=" * 60)

loss_configs = {
    "MSE (default)": None,  # use default objective
    "Asymmetric only": loss_asymmetric_only,
    "Asym + time-weighted": loss_full_competition,
}

target_alpha = 0.01
target_betas = {
    "Cycles_to_WW": 1.0 / train_df["Cycles_to_WW"].max(),
    "Cycles_to_HPC_SV": 2.0 / train_df["Cycles_to_HPC_SV"].max(),
    "Cycles_to_HPT_SV": 2.0 / train_df["Cycles_to_HPT_SV"].max(),
}

# Results storage
results = []

for loss_name, loss_fn in loss_configs.items():
    print(f"\n  ── {loss_name} ──")

    total_val_cs_parts = []

    for target, label in zip(TARGETS, TARGET_LABELS):
        params = optuna_params[target].copy()
        if loss_fn is not None:
            model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1,
                                       n_jobs=-1, objective=loss_fn)
        else:
            model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1,
                                       n_jobs=-1)

        model.fit(X_train, train_df[target].values)
        vp = np.clip(model.predict(X_val), 0, None)
        tp = np.clip(model.predict(X_test), 0, None)

        y_va = val_df[target].values
        y_te = test_df[target].values

        val_mae = mean_absolute_error(y_va, vp)
        val_r2 = r2_score(y_va, vp)
        val_cs = comp_score_single(y_va, vp, target_alpha, target_betas[target])
        late_pct = 100 * (vp > y_va).mean()
        mean_bias = (vp - y_va).mean()

        test_mae = mean_absolute_error(y_te, tp)
        test_r2 = r2_score(y_te, tp)

        total_val_cs_parts.append(val_cs)

        results.append({
            "Loss": loss_name,
            "Target": label,
            "Val MAE": val_mae,
            "Val R²": val_r2,
            "Val CS": val_cs,
            "Late %": late_pct,
            "Mean Bias": mean_bias,
            "Test MAE": test_mae,
            "Test R²": test_r2,
        })

        print(f"    {label}: MAE={val_mae:.1f}, R²={val_r2:.3f}, CS={val_cs:.2f}, "
              f"Late%={late_pct:.1f}%, Bias={mean_bias:+.1f}")

    avg_cs = np.mean(total_val_cs_parts)
    print(f"    Overall Val CS: {avg_cs:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "loss_ablation_results.csv", index=False)
print(f"\n  → Saved: {OUTPUT_DIR}/loss_ablation_results.csv")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

sns.set_theme(style="whitegrid", font_scale=1.1)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
loss_names = list(loss_configs.keys())
colors = ["#90CAF9", "#FFE082", "#EF9A9A"]

# Panel 1: Competition Score per target
ax = axes[0]
x = np.arange(3)
width = 0.25
for i, loss_name in enumerate(loss_names):
    vals = [results_df[(results_df["Loss"]==loss_name) & (results_df["Target"]==t)]["Val CS"].values[0]
            for t in TARGET_LABELS]
    ax.bar(x + i*width, vals, width, label=loss_name, color=colors[i], edgecolor="gray")
ax.set_xticks(x + width)
ax.set_xticklabels(TARGET_LABELS)
ax.set_ylabel("Val Competition Score")
ax.set_title("Competition Score\n(lower is better)")
ax.legend(fontsize=8)

# Panel 2: Late prediction %
ax = axes[1]
for i, loss_name in enumerate(loss_names):
    vals = [results_df[(results_df["Loss"]==loss_name) & (results_df["Target"]==t)]["Late %"].values[0]
            for t in TARGET_LABELS]
    ax.bar(x + i*width, vals, width, label=loss_name, color=colors[i], edgecolor="gray")
ax.set_xticks(x + width)
ax.set_xticklabels(TARGET_LABELS)
ax.set_ylabel("Late Prediction %")
ax.set_title("Late Prediction Rate\n(lower is better)")
ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.legend(fontsize=8)

# Panel 3: Mean Bias
ax = axes[2]
for i, loss_name in enumerate(loss_names):
    vals = [results_df[(results_df["Loss"]==loss_name) & (results_df["Target"]==t)]["Mean Bias"].values[0]
            for t in TARGET_LABELS]
    ax.bar(x + i*width, vals, width, label=loss_name, color=colors[i], edgecolor="gray")
ax.set_xticks(x + width)
ax.set_xticklabels(TARGET_LABELS)
ax.set_ylabel("Mean Bias (cycles)")
ax.set_title("Prediction Bias\n(0 = unbiased)")
ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.legend(fontsize=8)

# Panel 4: Val MAE
ax = axes[3]
for i, loss_name in enumerate(loss_names):
    vals = [results_df[(results_df["Loss"]==loss_name) & (results_df["Target"]==t)]["Val MAE"].values[0]
            for t in TARGET_LABELS]
    ax.bar(x + i*width, vals, width, label=loss_name, color=colors[i], edgecolor="gray")
ax.set_xticks(x + width)
ax.set_xticklabels(TARGET_LABELS)
ax.set_ylabel("Val MAE (cycles)")
ax.set_title("MAE\n(lower is better)")
ax.legend(fontsize=8)

plt.suptitle("Loss Function Ablation: Training Objective Alignment", fontsize=15, y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_loss_ablation.png", dpi=200)
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/10_loss_ablation.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

pivot = results_df.pivot_table(index="Loss", columns="Target",
                                values=["Val CS", "Late %", "Mean Bias", "Val MAE"])
print(pivot.to_string())

# Overall competition scores
print("\n  Overall Val Competition Score:")
for loss_name in loss_names:
    sub = results_df[results_df["Loss"] == loss_name]
    avg_cs = sub["Val CS"].mean()
    avg_late = sub["Late %"].mean()
    avg_bias = sub["Mean Bias"].mean()
    print(f"    {loss_name:<25s}: CS={avg_cs:.4f}, Late%={avg_late:.1f}%, Bias={avg_bias:+.1f}")

print("\nDone.")
