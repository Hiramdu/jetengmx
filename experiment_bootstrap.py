"""
#3 Bootstrap confidence intervals for the decision-layer headline numbers.

The 36% cost reduction and the failure counts rest on a modest number of
maintenance-cycle decision instances. We quantify their sampling uncertainty
by resampling the decision instances (with replacement) and recomputing the
policy comparison on each resample.

Resampling unit = one maintenance cycle (the independent decision instance).
We resample WW and HPT cycles (the reliable targets used in the headline),
recompute total cost and failures per policy, and report percentile CIs for:
  - total cost per policy
  - cost reduction of uncertainty-aware vs naive (the headline 36%)
  - failures per policy

Uses the per-cycle costs already produced by the decision logic, recomputed
here from test_with_intervals.csv so each bootstrap draw is self-consistent.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
RELIABLE = ["WW", "HPT_SV"]
C_EARLY, C_LATE, C_FAIL = 1.0, 2.0, 500.0
N_BOOT = 5000
SEED = 0


def maintenance_cost(r):
    return C_EARLY * r if r > 0 else C_FAIL + C_LATE * abs(r)


def segment_cycles(g, target):
    g = g.sort_values("Cycles_Since_New").reset_index(drop=True)
    csl = g[f"Cycles_Since_Last_{target}"].to_numpy()
    bnds = [0] + list(np.where(np.diff(csl) < 0)[0] + 1) + [len(g)]
    out = []
    for a, b in zip(bnds[:-1], bnds[1:]):
        seg = g.iloc[a:b]
        if (seg[f"Cycles_to_{target}"] <= 0).any() and len(seg) >= 3:
            out.append(seg)
    return out


def cycle_outcome(seg, target, buffer):
    """Return (cost, failure) for one maintenance cycle under a buffer."""
    yhat = seg[f"Pred_Cycles_to_{target}"].to_numpy()
    ytrue = seg[f"Cycles_to_{target}"].to_numpy()
    trig = np.where(yhat <= buffer)[0]
    if len(trig) == 0:
        return C_FAIL, 1
    r = ytrue[trig[0]]
    return maintenance_cost(r), int(r <= 0)


def build_instances(test, val):
    """Pre-compute, for every reliable maintenance cycle, the (cost, fail)
    under each of the three policies. Returns a DataFrame of instances."""
    # buffers: naive=0; fixed (VAL-tuned, from decision_robust); conformal width
    fixed = {"WW": 150.0, "HPT_SV": 180.0}  # matches experiment_decision_robust
    conf = {t: float(test[f"Width90_{t}"].iloc[0]) for t in RELIABLE}
    rows = []
    for t in RELIABLE:
        for _, g in test.groupby("ESN"):
            for seg in segment_cycles(g, t):
                cn, fn = cycle_outcome(seg, t, 0.0)
                cx, fx = cycle_outcome(seg, t, fixed[t])
                cu, fu = cycle_outcome(seg, t, conf[t])
                rows.append({"target": t,
                             "naive_cost": cn, "naive_fail": fn,
                             "fixed_cost": cx, "fixed_fail": fx,
                             "unc_cost": cu, "unc_fail": fu})
    return pd.DataFrame(rows)


def pct_ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def main():
    test = pd.read_csv(RESULTS / "test_with_intervals.csv")
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    inst = build_instances(test, val)
    n = len(inst)
    print(f"Reliable-target decision instances (WW+HPT): {n}\n")

    rng = np.random.default_rng(SEED)
    naive_c, fixed_c, unc_c = [], [], []
    naive_f, fixed_f, unc_f = [], [], []
    red_fixed, red_unc = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        s = inst.iloc[idx]
        nc, fc, uc = s.naive_cost.sum(), s.fixed_cost.sum(), s.unc_cost.sum()
        naive_c.append(nc); fixed_c.append(fc); unc_c.append(uc)
        naive_f.append(s.naive_fail.sum()); fixed_f.append(s.fixed_fail.sum())
        unc_f.append(s.unc_fail.sum())
        red_fixed.append((1 - fc / nc) * 100 if nc else 0)
        red_unc.append((1 - uc / nc) * 100 if nc else 0)

    def report(name, arr, fmt="{:.0f}"):
        lo, hi = pct_ci(arr)
        print(f"  {name:28s} {fmt.format(np.mean(arr))}  "
              f"95% CI [{fmt.format(lo)}, {fmt.format(hi)}]")

    print("=== Bootstrap (5000 resamples) on reliable targets, C_fail=500 ===\n")
    print("Total cost:")
    report("naive", naive_c); report("fixed_buffer", fixed_c)
    report("uncertainty_aware", unc_c)
    print("\nCost reduction vs naive (%):")
    report("fixed_buffer", red_fixed, "{:.1f}")
    report("uncertainty_aware", red_unc, "{:.1f}")
    print("\nUnplanned failures (out of %d cycles):" % n)
    report("naive", naive_f); report("fixed_buffer", fixed_f)
    report("uncertainty_aware", unc_f)

    # probability uncertainty-aware beats naive on cost
    p_better = np.mean(np.array(unc_c) < np.array(naive_c))
    p_beat_fixed = np.mean(np.array(unc_c) < np.array(fixed_c))
    print(f"\nP(uncertainty-aware cost < naive cost)        = {p_better:.3f}")
    print(f"P(uncertainty-aware cost < fixed-buffer cost) = {p_beat_fixed:.3f}")

    pd.DataFrame({
        "metric": ["red_unc_mean", "red_unc_lo", "red_unc_hi",
                   "p_unc_lt_naive", "p_unc_lt_fixed"],
        "value": [np.mean(red_unc), *pct_ci(red_unc), p_better, p_beat_fixed],
    }).to_csv(RESULTS / "bootstrap_results.csv", index=False)
    print("\nWrote bootstrap_results.csv")


if __name__ == "__main__":
    main()
