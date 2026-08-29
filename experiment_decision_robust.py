"""
Direction A, hardened: per-maintenance-cycle decision evaluation
(leave-one-engine-out style expansion of decision instances) plus a
sensitivity analysis over the failure/early cost ratio.

Two methodological gaps from experiment_decision.py are addressed:

1. Sample size. Instead of one decision per engine (12 instances), we
   segment each engine's TEST timeline into individual maintenance cycles
   using Cycles_Since_Last_<target> resets, and make one decision per cycle.
   This yields ~26 decision instances (WW ~17, HPT ~7, HPC ~2). HPC remains
   sparse and is reported as a difficult case, not a headline claim.

2. Cost-constant sensitivity. The "uncertainty-aware wins" conclusion must
   not be an artifact of a single C_FAIL value. We sweep C_FAIL over
   {100, 250, 500, 1000, 2000} and report the policy ranking at each.

Within each maintenance cycle we walk forward and trigger when
RUL_pred <= buffer; cost is early-waste, or a failure penalty if the true
event was already reached at trigger time.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPC_SV", "HPT_SV"]
C_EARLY, C_LATE = 1.0, 2.0


def segment_cycles(g, target):
    """Split one engine timeline into maintenance cycles via csl resets.
    Returns list of sub-DataFrames, each a single degrade->event cycle."""
    g = g.sort_values("Cycles_Since_New").reset_index(drop=True)
    csl = g[f"Cycles_Since_Last_{target}"].to_numpy()
    boundaries = [0] + list(np.where(np.diff(csl) < 0)[0] + 1) + [len(g)]
    segs = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = g.iloc[a:b]
        # only keep cycles that actually contain the event (true RUL reaches 0)
        if (seg[f"Cycles_to_{target}"] <= 0).any() and len(seg) >= 3:
            segs.append(seg)
    return segs


def cycle_cost(seg, target, buffer, c_fail):
    yhat = seg[f"Pred_Cycles_to_{target}"].to_numpy()
    ytrue = seg[f"Cycles_to_{target}"].to_numpy()
    trig = np.where(yhat <= buffer)[0]
    if len(trig) == 0:
        return c_fail, True  # never acted but event occurred -> failure
    i = trig[0]
    r = ytrue[i]
    if r > 0:
        return C_EARLY * r, False
    return c_fail + C_LATE * abs(r), True


def buffers_for(test, val):
    """Return the three policies' per-target buffers."""
    # fixed buffer tuned on VAL per target (minimize total cost at C_FAIL=500)
    fixed = {}
    for t in TARGETS:
        best_b, best_c = 0, np.inf
        for b in np.arange(0, 600, 10):
            tot = 0.0
            for _, g in val.groupby("ESN"):
                for seg in segment_cycles(g, t):
                    c, _ = cycle_cost(seg, t, b, 500.0)
                    tot += c
            if tot < best_c:
                best_c, best_b = tot, b
        fixed[t] = float(best_b)
    conf = {t: float(test[f"Width90_{t}"].iloc[0]) for t in TARGETS}
    return fixed, conf


def run(test, policies, c_fail):
    rows = []
    for pol, bufs in policies.items():
        for t in TARGETS:
            costs, fails, n = [], 0, 0
            for _, g in test.groupby("ESN"):
                for seg in segment_cycles(g, t):
                    c, f = cycle_cost(seg, t, bufs[t], c_fail)
                    costs.append(c); fails += int(f); n += 1
            rows.append({"policy": pol, "target": t, "C_fail": c_fail,
                         "n_cycles": n, "total_cost": round(sum(costs), 1),
                         "failures": fails})
    return pd.DataFrame(rows)


def main():
    test = pd.read_csv(RESULTS / "test_with_intervals.csv")
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    fixed, conf = buffers_for(test, val)
    print(f"Fixed buffers (VAL-tuned): {fixed}")
    print(f"Conformal buffers:         { {t: round(v,1) for t,v in conf.items()} }\n")

    policies = {
        "naive": {t: 0.0 for t in TARGETS},
        "fixed_buffer": fixed,
        "uncertainty_aware": conf,
    }

    # report instance counts
    nseg = {t: sum(len(segment_cycles(g, t)) for _, g in test.groupby("ESN")) for t in TARGETS}
    print(f"Decision instances on TEST (maintenance cycles): {nseg} "
          f"(total {sum(nseg.values())})\n")

    # main table at C_FAIL=500
    main_res = run(test, policies, 500.0)
    main_res.to_csv(RESULTS / "decision_robust_results.csv", index=False)
    print("=== Per-target decision cost (C_fail=500) ===")
    print(main_res.to_string(index=False))

    print("\n=== Sensitivity sweep: total cost by policy across C_fail ===")
    sens_rows = []
    for cf in [100, 250, 500, 1000, 2000]:
        r = run(test, policies, cf)
        agg = r.groupby("policy").total_cost.sum()
        sens_rows.append({"C_fail": cf,
                          "naive": agg["naive"],
                          "fixed_buffer": agg["fixed_buffer"],
                          "uncertainty_aware": agg["uncertainty_aware"],
                          "best": agg.idxmin()})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(RESULTS / "decision_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    print("\n=== Headline (C_fail=500), WW+HPT only (reliable targets) ===")
    rel = main_res[main_res.target.isin(["WW", "HPT_SV"])]
    agg = rel.groupby("policy").agg(total_cost=("total_cost", "sum"),
                                    failures=("failures", "sum")).reindex(
                                        ["naive", "fixed_buffer", "uncertainty_aware"])
    base = agg.loc["naive", "total_cost"]
    for p in agg.index:
        red = (1 - agg.loc[p, "total_cost"] / base) * 100 if base else 0
        print(f"  {p:18s} cost={agg.loc[p,'total_cost']:7.0f} ({red:+5.1f}% vs naive)  "
              f"failures={int(agg.loc[p,'failures'])}")


if __name__ == "__main__":
    main()
