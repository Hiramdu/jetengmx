"""
Revision experiment for Reviewer B, comment 5 (decision outcomes by engine).

The reviewer asks whether the improvement of the conformal buffer over the tuned
fixed buffer is consistent across the four available units.  It is not: the
aggregate 14 % is carried by ESN 102 and ESN 103, while ESN 101 and ESN 104 are
marginally worse.  This script produces Table 10 of the revised manuscript.

Water-wash and HPT only; HPC's six instances are excluded from the aggregate for
the reason given in Section 6.3 (no constant-buffer policy triggers at all).

Output: experiment_results/decision_per_engine.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV"]
C_FAIL, C_EARLY, C_LATE = 500.0, 1.0, 2.0
ALPHA = 0.10


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(absres, lvl, method="higher"))


def segment_cycles(df, t):
    """Split each engine's timeline into degrade-to-event cycles."""
    segs = []
    for esn, g in df.groupby("ESN"):
        g = g.sort_values("Cycles_Since_New")
        y = g[f"Cycles_to_{t}"].to_numpy()
        for seg in np.split(np.arange(len(y)), np.where(np.diff(y) > 0)[0] + 1):
            if len(seg) > 3 and y[seg].min() <= 10:
                segs.append(g.iloc[seg])
    return segs


def policy_cost(seg, t, buffer):
    p = seg[f"Pred_Cycles_to_{t}"].to_numpy()
    y = seg[f"Cycles_to_{t}"].to_numpy()
    for k in range(len(p)):
        if p[k] <= buffer:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (C_FAIL + C_LATE * abs(r), 1)
    return C_FAIL, 1


def main():
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")

    conf = {t: conformal_q(np.abs(val[f"Cycles_to_{t}"]
                                  - val[f"Pred_Cycles_to_{t}"]).to_numpy())
            for t in TARGETS}
    fixed = {}
    for t in TARGETS:
        segs = segment_cycles(val, t)
        fixed[t] = float(min(((b, sum(policy_cost(s, t, b)[0] for s in segs))
                              for b in np.linspace(0, 600, 61)),
                             key=lambda x: x[1])[0])
    print("conformal half-widths", {t: round(v, 1) for t, v in conf.items()},
          "| validation-tuned fixed buffers", fixed)

    rows = []
    for esn in sorted(test.ESN.unique()):
        sub = test[test.ESN == esn]
        for t in TARGETS:
            segs = segment_cycles(sub, t)
            r = dict(ESN=esn, target=t, n_cycles=len(segs))
            for nm, b in (("naive", 0.0), ("fixed", fixed[t]), ("conf", conf[t])):
                c = f = 0
                for s in segs:
                    ci, fi = policy_cost(s, t, b)
                    c += ci
                    f += fi
                r[f"{nm}_cost"], r[f"{nm}_fail"] = c, f
            rows.append(r)
    d = pd.DataFrame(rows)
    d["conf_vs_fixed_pct"] = np.where(d.fixed_cost > 0,
                                      (1 - d.conf_cost / d.fixed_cost) * 100,
                                      np.nan).round(1)
    d.to_csv(RESULTS / "decision_per_engine.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== per-engine, per-target (C_fail=500) ===")
    print(d.to_string(index=False))

    cols = ["naive_cost", "fixed_cost", "conf_cost",
            "naive_fail", "fixed_fail", "conf_fail"]
    g = d.groupby("ESN")[cols].sum()
    g["conf_vs_fixed_pct"] = ((1 - g.conf_cost / g.fixed_cost) * 100).round(1)
    g["conf_vs_naive_pct"] = ((1 - g.conf_cost / g.naive_cost) * 100).round(1)
    print("\n=== Table 10 of the manuscript: per engine, WW+HPT combined ===")
    print(g.to_string())
    tot = d[cols].sum()
    print("\nAll engines: naive %.0f/%d  fixed %.0f/%d  conformal %.0f/%d "
          "(conf vs fixed %+.1f%%)"
          % (tot.naive_cost, tot.naive_fail, tot.fixed_cost, tot.fixed_fail,
             tot.conf_cost, tot.conf_fail,
             (1 - tot.conf_cost / tot.fixed_cost) * 100))


if __name__ == "__main__":
    main()
