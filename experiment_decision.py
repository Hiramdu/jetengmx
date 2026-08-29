"""
Direction A: From RUL prediction to a cost-optimal maintenance decision,
with an uncertainty-aware policy that uses the conformal intervals from
experiment_conformal.py.

Decision setting (per target, per engine timeline)
--------------------------------------------------
Walking forward in cycles, at each observation we hold a predicted RUL.
A maintenance action is triggered the first time the *decision statistic*
drops to <= 0. The decision statistic is RUL_pred minus a safety buffer b:

    trigger when  (RUL_pred - b) <= 0     i.e.  RUL_pred <= b

We compare three buffer policies:
  1. Naive            : b = 0 (act when point prediction hits 0)
  2. Fixed-buffer     : b = b* , a single constant tuned on VAL
  3. Uncertainty-aware: b = conformal half-width (per target), so the buffer
                         automatically widens for unreliable targets (HPC)
                         and stays tight for reliable ones (WW).

Cost model (built from the challenge's safety-critical asymmetry)
-----------------------------------------------------------------
At the triggered cycle, let r = true RUL at that cycle.
  - If r > 0  : we serviced early. Cost = c_early * r   (wasted useful life)
  - If r <= 0 : we serviced late -> the event already occurred = unplanned
                failure. Cost = C_fail  (large fixed penalty) + c_late*|r|

C_fail >> c_early to encode that an unplanned in-service failure is far
worse than early servicing -- exactly the 2x-late asymmetry of the metric,
taken to its operational conclusion.

This uses only test_with_intervals.csv (no model retraining). Per-engine
timelines are reconstructed from (ESN, Cycles_Since_New).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPC_SV", "HPT_SV"]

# Cost constants (relative units). C_fail dominates: one unplanned failure
# costs as much as servicing ~500 cycles early.
C_EARLY = 1.0
C_LATE = 2.0      # late also accrues per-cycle, 2x early (challenge asymmetry)
C_FAIL = 500.0    # fixed unplanned-failure penalty


def first_trigger_cost(df_engine, target, buffer_fn):
    """Walk one engine's timeline in cycle order; trigger maintenance the
    first time RUL_pred <= buffer. Return (cost, triggered, late_flag, lead).
    buffer_fn(row) -> b for that observation."""
    yhat = df_engine[f"Pred_Cycles_to_{target}"].to_numpy()
    ytrue = df_engine[f"Cycles_to_{target}"].to_numpy()
    b = buffer_fn(df_engine)
    trig = np.where(yhat <= b)[0]
    if len(trig) == 0:
        # never triggered within the window -> treat as a missed event if the
        # true event actually occurred (true RUL hit ~0 somewhere)
        if (ytrue <= 0).any():
            return C_FAIL, False, True, np.nan
        return 0.0, False, False, np.nan
    i = trig[0]
    r = ytrue[i]
    if r > 0:
        return C_EARLY * r, True, False, r
    else:
        return C_FAIL + C_LATE * abs(r), True, True, r


def evaluate(df, buffer_policies):
    """Return per-(policy,target) aggregated cost/late-rate over engines."""
    rows = []
    for pol_name, buffer_fns in buffer_policies.items():
        for t in TARGETS:
            costs, lates, leads = [], [], []
            for esn, g in df.groupby("ESN"):
                g = g.sort_values("Cycles_Since_New")
                c, trig, late, lead = first_trigger_cost(g, t, buffer_fns[t])
                costs.append(c); lates.append(late)
                if not np.isnan(lead):
                    leads.append(lead)
            rows.append({
                "policy": pol_name,
                "target": t,
                "total_cost": round(float(np.sum(costs)), 1),
                "mean_cost": round(float(np.mean(costs)), 1),
                "late_failures": int(np.sum(lates)),
                "n_engines": len(costs),
                "mean_lead_cycles": round(float(np.mean(leads)), 1) if leads else np.nan,
            })
    return pd.DataFrame(rows)


def tune_fixed_buffer(val):
    """Pick one global buffer per target that minimizes total cost on VAL."""
    best = {}
    for t in TARGETS:
        grid = np.arange(0, 600, 10)
        best_b, best_c = 0, np.inf
        for b in grid:
            tot = 0.0
            for esn, g in val.groupby("ESN"):
                g = g.sort_values("Cycles_Since_New")
                c, *_ = first_trigger_cost(g, t, lambda gg, bb=b: bb)
                tot += c
            if tot < best_c:
                best_c, best_b = tot, b
        best[t] = best_b
    return best


def main():
    test = pd.read_csv(RESULTS / "test_with_intervals.csv")
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")

    fixed_b = tune_fixed_buffer(val)
    print(f"Fixed-buffer tuned on VAL: {fixed_b}\n")

    # Conformal half-widths (per target) were written as Width90_<t> on test.
    conf_w = {t: float(test[f"Width90_{t}"].iloc[0]) for t in TARGETS}
    print(f"Uncertainty-aware buffers (conformal 90% half-width): "
          f"{ {t: round(w,1) for t,w in conf_w.items()} }\n")

    policies = {
        "naive": {t: (lambda g: 0.0) for t in TARGETS},
        "fixed_buffer": {t: (lambda g, b=fixed_b[t]: b) for t in TARGETS},
        "uncertainty_aware": {t: (lambda g, w=conf_w[t]: w) for t in TARGETS},
    }

    res = evaluate(test, policies)
    res.to_csv(RESULTS / "decision_results.csv", index=False)
    pd.set_option("display.width", 130)
    print("=== Maintenance-decision policies on TEST (4 engines) ===\n")
    print(res.to_string(index=False))

    print("\n=== Summary: total cost & unplanned failures by policy ===")
    agg = res.groupby("policy").agg(total_cost=("total_cost", "sum"),
                                    late_failures=("late_failures", "sum"))
    agg = agg.reindex(["naive", "fixed_buffer", "uncertainty_aware"])
    print(agg.to_string())
    base = agg.loc["naive", "total_cost"]
    print("\nCost reduction vs naive:")
    for p in ["fixed_buffer", "uncertainty_aware"]:
        print(f"  {p:18s}: {(1 - agg.loc[p,'total_cost']/base)*100:+.1f}%  "
              f"(failures {int(agg.loc[p,'late_failures'])} vs {int(agg.loc['naive','late_failures'])})")


if __name__ == "__main__":
    main()
