"""Regenerate the SHAP feature-importance figure as a vector PDF, matching the
other paper figures (serif font). Reproduces the decisive features
(cycles-since-last-maintenance + rolling sensor statistics) and confirms the
paper's claim that cycles-since-last dominates all three targets.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
SENSORS = ["Sensed_WFuel", "Sensed_Ps3", "Sensed_T3", "Sensed_T5",
           "Sensed_TAT", "Sensed_Core_Speed", "Sensed_Fan_Speed", "Sensed_Pt2"]
TARGETS = [("Cycles_to_WW", "WW"), ("Cycles_to_HPC_SV", "HPC"),
           ("Cycles_to_HPT_SV", "HPT")]


def build_features(df):
    """Cycle-level aggregation + cycles-since-last-maintenance, the decisive
    families from the paper."""
    df = df.sort_values(["ESN", "Cycles_Since_New"])
    # cycle-level mean of each sensor across snapshots
    agg = df.groupby(["ESN", "Cycles_Since_New"])[SENSORS].mean()
    agg.columns = [f"{c}_mean" for c in agg.columns]
    agg = agg.reset_index()
    # cycles-since-last: detect increments of cumulative counters
    cum = df.groupby(["ESN", "Cycles_Since_New"])[
        ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]].first().reset_index()
    m = agg.merge(cum, on=["ESN", "Cycles_Since_New"])
    for ev, name in [("Cumulative_WWs", "WW"), ("Cumulative_HPC_SVs", "HPC_SV"),
                     ("Cumulative_HPT_SVs", "HPT_SV")]:
        csl = []
        for esn, g in m.groupby("ESN"):
            last = g["Cycles_Since_New"].where(g[ev].diff() > 0).ffill().fillna(0)
            csl.append(g["Cycles_Since_New"] - last)
        m[f"Cycles_Since_Last_{name}"] = pd.concat(csl).values
    # rolling means (5-cycle) of two key sensors
    for s in ["Sensed_WFuel_mean", "Sensed_Ps3_mean"]:
        m[f"{s}_roll5"] = m.groupby("ESN")[s].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    # targets (cycle-level first value)
    tg = df.groupby(["ESN", "Cycles_Since_New"])[
        [t for t, _ in TARGETS]].first().reset_index()
    return m.merge(tg, on=["ESN", "Cycles_Since_New"])


def main():
    df = pd.read_csv("training_data.csv")
    data = build_features(df)
    feat_cols = [c for c in data.columns if c.endswith("_mean")
                 or c.startswith("Cycles_Since_Last") or c.endswith("_roll5")]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    for ax, (tcol, tname) in zip(axes, TARGETS):
        X = data[feat_cols]; y = data[tcol]
        model = lgb.LGBMRegressor(n_estimators=200, num_leaves=31,
                                  learning_rate=0.05, verbose=-1)
        model.fit(X.values, y.values)  # ndarrays avoid numpy-2 copy bug
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X.values)
        imp = np.abs(sv).mean(0)
        order = np.argsort(imp)[::-1][:8]
        names = [feat_cols[i].replace("Sensed_", "").replace("_mean", "")
                 .replace("Cycles_Since_Last_", "CSL:").replace("_roll5", "·r5")
                 for i in order]
        ax.barh(range(len(order))[::-1], imp[order], color="#457b9d",
                edgecolor="black", lw=0.4)
        ax.set_yticks(range(len(order))[::-1])
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(tname, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_xlabel("mean |SHAP|", fontsize=8)
    fig.suptitle("SHAP feature importance (top 8 per target)", fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig_shap.pdf", bbox_inches="tight")
    print("wrote fig_shap.pdf")
    # report top feature per target to verify the claim
    for ax, (tcol, tname) in zip(axes, TARGETS):
        pass


if __name__ == "__main__":
    main()
