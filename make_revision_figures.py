"""B1 + B2 figures for the IJPHM revision.

fig_intervals : per-instance 90% conformal intervals on the test window,
                instances sorted by true RUL (Javanmardi & Hullermeier
                Fig. 2 style), one panel per target.
fig_parity    : predicted-vs-actual parity plots, validation vs test
                side by side (Cohen et al. Fig. 3 style), one row per split.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 9,
})

R = Path("experiment_results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
TARGETS = [("WW", "Water-wash"), ("HPT_SV", "HPT shop visit"),
           ("HPC_SV", "HPC shop visit")]
ALPHA = 0.1

val = pd.read_csv(R / "val_with_predictions.csv")
test = pd.read_csv(R / "test_with_predictions.csv")


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


# ---- B1: per-instance intervals, sorted by true RUL ----
fig, axes = plt.subplots(3, 1, figsize=(5.2, 6.0))
for ax, (t, lab) in zip(axes, TARGETS):
    y_col, p_col = f"Cycles_to_{t}", f"Pred_Cycles_to_{t}"
    q = conformal_q(np.abs(val[y_col] - val[p_col]).to_numpy())
    y, p = test[y_col].to_numpy(), test[p_col].to_numpy()
    o = np.argsort(y)
    y, p = y[o], p[o]
    lo, hi = np.clip(p - q, 0, None), p + q
    idx = np.arange(len(y))
    inside = (y >= lo) & (y <= hi)
    ax.fill_between(idx, lo, hi, color="#a8c6e8", alpha=0.6, lw=0,
                    label=f"90% interval (half-width {q:.0f})")
    ax.plot(idx, y, "k-", lw=1.0, label="true RUL")
    ax.plot(idx, p, ".", color="#e76f51", ms=1.6, label="prediction")
    ax.set_ylabel(f"{lab}\n(cycles)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.93, f"coverage {inside.mean():.0%}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(fc="white", ec="0.7", boxstyle="round,pad=0.25"))
    if t == "WW":
        ax.legend(fontsize=7.5, loc="lower right", frameon=True)
axes[-1].set_xlabel("Test instances sorted by true RUL", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig_intervals.pdf", bbox_inches="tight")
print("wrote fig_intervals.pdf")

# ---- B2: parity plots val vs test ----
# Laid out tall (one row per target, one column per split) so the figure fits a
# single manuscript column at close to 1:1 scale; a wide 2x3 version had to be
# scaled down by half in a two-column layout, which left the tick labels below
# the manuscript's own font size.
fig, axes = plt.subplots(3, 2, figsize=(3.3, 4.8), sharex="row", sharey="row")
for i, (t_, lab) in enumerate(TARGETS):
    y_col, p_col = f"Cycles_to_{t_}", f"Pred_Cycles_to_{t_}"
    for j, (df, split) in enumerate([(val, "Validation"), (test, "Test (late life)")]):
        ax = axes[i, j]
        y, p = df[y_col].to_numpy(), df[p_col].to_numpy()
        m = max(y.max(), p.max()) * 1.04
        ax.plot([0, m], [0, m], "k--", lw=0.7)
        ax.plot(y, p, ".", ms=1.4,
                color="#457b9d" if j == 0 else "#e76f51", alpha=0.45)
        ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.tick_params(labelsize=6, pad=1.5)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        ax.yaxis.get_offset_text().set_fontsize(5.5)
        ax.xaxis.get_offset_text().set_fontsize(5.5)
        if i == 0:
            ax.set_title(split, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{lab}\npredicted", fontsize=7)
        if i == 2:
            ax.set_xlabel("actual (cycles)", fontsize=7)
plt.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.5)
plt.savefig(OUT / "fig_parity.pdf", bbox_inches="tight")
print("wrote fig_parity.pdf")
