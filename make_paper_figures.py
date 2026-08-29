"""Generate the two core figures for the reframed IJPHM paper:
  fig_coverage : per-target conformal coverage under shift (Direction B)
  fig_decision : policy cost & failures on reliable targets (Direction A)
Saved into figures/ for inclusion in the manuscript.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Match the IJPHM body font (Times / serif) for a consistent look.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
})

R = Path("experiment_results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ---- Figure 1: reliability under shift ----
conf = pd.read_csv(R / "conformal_results.csv")
c90 = conf[conf.nominal_coverage == 0.9].set_index("target")
targets = ["WW", "HPT_SV", "HPC_SV"]
labels = ["Water-wash", "HPT shop visit", "HPC shop visit"]
ach = [c90.loc[t, "empirical_coverage"] for t in targets]

fig, ax = plt.subplots(figsize=(5.0, 3.2))
colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
bars = ax.bar(labels, ach, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.90, ls="--", c="black", lw=1)
ax.text(2.05, 0.91, "nominal 90%", ha="right", fontsize=8)
ax.set_ylabel("Achieved test coverage")
ax.set_ylim(0, 1.0)
for b, v in zip(bars, ach):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
ax.set_title("Conformal interval coverage under late-life shift", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig_coverage.pdf", bbox_inches="tight")
print("wrote fig_coverage.pdf")

# ---- Figure 2: decision policies on reliable targets (WW+HPT), C_fail=500 ----
dr = pd.read_csv(R / "decision_robust_results.csv")
dr = dr[dr.target.isin(["WW", "HPT_SV"])]
heur = dr.groupby("policy").agg(cost=("total_cost", "sum"), fails=("failures", "sum"))
md = pd.read_csv(R / "mdp_results.csv")
mdp_cost = md[md.target.isin(["WW", "HPT_SV"])].total_cost.sum()
mdp_fail = int(md[md.target.isin(["WW", "HPT_SV"])].failures.sum())

policies = ["naive", "fixed_buffer", "uncertainty_aware", "MDP"]
plabels = ["Naive", "Fixed\nbuffer", "Conformal\nbuffer", "MDP\n(opt. stop)"]
cost = [heur.loc["naive", "cost"], heur.loc["fixed_buffer", "cost"],
        heur.loc["uncertainty_aware", "cost"], mdp_cost]
fails = [heur.loc["naive", "fails"], heur.loc["fixed_buffer", "fails"],
         heur.loc["uncertainty_aware", "fails"], mdp_fail]

# ---- Figure 0: degradation sawtooth (regenerated as vector, clean proportions) ----
tr = pd.read_csv("training_data.csv")
figd, axd = plt.subplots(3, 1, figsize=(5.0, 4.8), sharex=True)
tg = [("Cycles_to_WW", "Water-wash\n(cycles)"),
      ("Cycles_to_HPC_SV", "HPC shop visit\n(cycles)"),
      ("Cycles_to_HPT_SV", "HPT shop visit\n(cycles)")]
colors = plt.cm.tab10(np.linspace(0, 1, 4))
for ax, (col, lab) in zip(axd, tg):
    for c, (esn, g) in zip(colors, tr.groupby("ESN")):
        g = g.sort_values("Cycles_Since_New")
        ax.plot(g["Cycles_Since_New"], g[col], lw=0.7, color=c,
                label=f"ESN {esn}")
    ax.set_ylabel(lab, fontsize=9)
    ax.tick_params(labelsize=9)
axd[-1].set_xlabel("Cycles since new", fontsize=10)
axd[0].legend(fontsize=8, ncol=4, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, 1.45))
figd.suptitle("Remaining-cycle degradation profiles", fontsize=10, y=0.99)
plt.tight_layout()
plt.savefig(OUT / "fig_degradation.pdf", bbox_inches="tight")
print("wrote fig_degradation.pdf")

# ---- Figure 3: cost vs failure-penalty sensitivity (crossover) ----
sens = pd.read_csv(R / "decision_sensitivity.csv")
figs, axs = plt.subplots(figsize=(5.0, 3.2))
cf = sens["C_fail"].to_numpy()
axs.plot(cf, sens["naive"], "o-", color="#888888", label="Naive")
axs.plot(cf, sens["fixed_buffer"], "s-", color="#457b9d", label="Fixed buffer")
axs.plot(cf, sens["uncertainty_aware"], "^-", color="#e76f51",
         label="Conformal buffer")
axs.set_xscale("log")
axs.set_xlabel("Failure penalty $C_{\\mathrm{fail}}$ (log scale)")
axs.set_ylabel("Total cost (all targets)")
axs.axvspan(100, 250, color="#cccccc", alpha=0.35)
axs.text(155, axs.get_ylim()[1]*0.6, "naive\nbest", ha="center", fontsize=7.5)
axs.annotate("crossover\n$C_{\\mathrm{fail}}\\approx250$", xy=(250, 6450),
             xytext=(420, 17000), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axs.set_title("When does uncertainty-awareness pay off?", fontsize=10)
axs.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
print("wrote fig_sensitivity.pdf")

fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
x = np.arange(len(policies))
b1 = ax1.bar(x - 0.2, cost, 0.4, label="Total cost", color="#264653", edgecolor="black", lw=0.5)
ax1.set_ylabel("Total cost (relative units)")
ax1.set_xticks(x); ax1.set_xticklabels(plabels, fontsize=8)
ax2 = ax1.twinx()
b2 = ax2.bar(x + 0.2, fails, 0.4, label="Failures", color="#e76f51", edgecolor="black", lw=0.5)
ax2.set_ylabel("Unplanned failures (of 31)")
ax1.set_title("Decision policies on reliable targets ($C_{\\mathrm{fail}}=500$)", fontsize=10)
lines = [b1, b2]
ax1.legend(lines, ["Total cost", "Failures"], loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_decision.pdf", bbox_inches="tight")
print("wrote fig_decision.pdf")
