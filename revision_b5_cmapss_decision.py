"""
Extension of the answer to Reviewer B, comment 5.

B5 observes that the bootstrap interval for the conformal buffer's advantage over
a tuned fixed buffer includes zero, and asks whether the improvement is
consistent across the four available units.  It is not (Table 10), and under a
nested calibration protocol the sign is not even stable (Table 15).  Four
engines simply cannot order the two policies.

This script gives that specific comparison the statistical power the challenge
fleet cannot, by running the same decision experiment on the public NASA C-MAPSS
benchmark, where each subset provides 100-260 independent run-to-failure units
instead of four engines.  Nothing else about the paper's argument depends on it;
the purpose is only to say whether the comparison B5 questions resolves once the
sample is large enough.

Protocol.  A first attempt reused the within-unit chronological split of
experiment_cmapss_replication.py (fit on the first 60 % of each unit's life).
That is degenerate for a *decision* experiment: with RUL rectified at 125, the
first 60 % of a unit's life is almost entirely at the cap, so the fitted model
never learns the run-down, its predictions never descend to any buffer, and every
policy fails on every unit.  We therefore use the cross-unit protocol instead --
the same mature-fleet setting as the cross-unit column of Table 6 -- which is
also the setting in which a scheduling buffer is meaningful at all:

  * units are partitioned 50/25/25: half are used in full for fitting, a quarter
    supply the calibration window, a quarter are deployed;
  * calibration and deployment both use the final 20 % of a unit's life, so the
    conformal quantile and the decision are measured in the same late-life
    regime;
  * LightGBM on the standard 14 informative sensors plus operating settings,
    rectified RUL capped at 125;
  * split-conformal 90 % half-width from the calibration units.

Decision setting: one instance per unit.  Walking the deployment window forward,
maintenance is triggered the first time the predicted RUL falls to or below the
buffer.  Cost follows Eq. 6 of the paper in forfeited-cycle units: c_early * r
if the true RUL r at the trigger is positive, otherwise C_fail + c_late * |r|;
a unit that is never triggered before its final cycle counts as an unplanned
failure.  The fixed buffer is tuned on decision instances built the same way
from the calibration window, never on the deployment window.

Uncertainty is a unit-level bootstrap (units are the independent sampling unit
here, exactly as engines were in the paper).

Output: experiment_results/b5_cmapss_decision.csv
"""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("CMAPSS_DIR", "/persist/cmapss/data"))
OUT = Path(__file__).parent / "experiment_results"
ALPHA = 0.10
RUL_MAX = 125
C_EARLY, C_LATE, C_FAIL = 1.0, 2.0, 500.0
SEED = 42
N_BOOT = 5000
N_JOBS = 4            # keep LightGBM from thrashing when several runs share the box
DROP_SENSORS = [1, 5, 6, 10, 16, 18, 19]          # near-constant, Li et al. 2018
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
FEATS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)
                                 if i not in DROP_SENSORS]


def load(subset):
    df = pd.read_csv(DATA / f"train_{subset}.txt", sep=r"\s+", header=None,
                     names=COLS)
    fail = df.groupby("unit")["cycle"].max().rename("fail_cycle")
    df = df.join(fail, on="unit")
    df["RUL"] = np.minimum(df["fail_cycle"] - df["cycle"], RUL_MAX)
    for c in FEATS:
        df[f"{c}_rm"] = df.groupby("unit")[c].transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    df["life_frac"] = df["cycle"] / df["fail_cycle"]
    return df


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(absres, lvl, method="higher"))


def unit_cost(g, buffer, pred_col="pred"):
    """One decision instance: walk a unit's window forward, trigger at buffer."""
    g = g.sort_values("cycle")
    p = g[pred_col].to_numpy()
    y = g["RUL"].to_numpy()
    for k in range(len(p)):
        if p[k] <= buffer:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (C_FAIL + C_LATE * abs(r), 1)
    return C_FAIL, 1                       # never triggered -> unplanned failure


def unit_cost_cf(g, buffer, c_fail, pred_col="pred"):
    """unit_cost with an explicit failure penalty, for the sweep."""
    g = g.sort_values("cycle")
    p = g[pred_col].to_numpy()
    y = g["RUL"].to_numpy()
    for k in range(len(p)):
        if p[k] <= buffer:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (c_fail + C_LATE * abs(r), 1)
    return c_fail, 1


def tune_fixed(cal, grid):
    best = None
    for b in grid:
        tot = sum(unit_cost(g, b)[0] for _, g in cal.groupby("unit"))
        if best is None or tot < best[1]:
            best = (b, tot)
    return float(best[0])


def main():
    feat_cols = FEATS + [f"{c}_rm" for c in FEATS] + ["cycle"]
    rows, per_unit_all, fitted = [], {}, {}

    for subset in ("FD001", "FD002", "FD003", "FD004"):
        df = load(subset)
        units = np.sort(df.unit.unique())
        rs = np.random.RandomState(SEED)
        perm = rs.permutation(units)
        n = len(perm)
        fit_u = perm[: n // 2]
        cal_u = perm[n // 2: n // 2 + n // 4]
        dep_u = perm[n // 2 + n // 4:]
        fit = df[df.unit.isin(fit_u)]                       # full trajectories
        cal = df[df.unit.isin(cal_u) & (df.life_frac > 0.80)].copy()
        dep = df[df.unit.isin(dep_u) & (df.life_frac > 0.80)].copy()

        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1, n_jobs=N_JOBS)
        m.fit(fit[feat_cols], fit["RUL"])
        cal["pred"] = np.clip(m.predict(cal[feat_cols]), 0, None)
        dep["pred"] = np.clip(m.predict(dep[feat_cols]), 0, None)

        q = conformal_q(np.abs(cal["RUL"].to_numpy() - cal["pred"].to_numpy()))
        cov = float(((dep.RUL >= np.clip(dep.pred - q, 0, None)) &
                     (dep.RUL <= dep.pred + q)).mean())
        fb = tune_fixed(cal, np.linspace(0, 120, 61))

        policies = {"naive": 0.0, "fixed": fb, "conformal": q}
        per_unit = {}
        for u, g in dep.groupby("unit"):
            per_unit[u] = {}
            for nm, b in policies.items():
                c, f = unit_cost(g, b)
                per_unit[u][nm + "_cost"] = c
                per_unit[u][nm + "_fail"] = f
        pu = pd.DataFrame(per_unit).T.astype(float)
        fitted[subset] = (dep, q, fb)
        per_unit_all[subset] = pu

        # unit-level bootstrap of the conformal-vs-fixed cost reduction,
        # vectorized over replicates (the per-unit table is small)
        rng = np.random.RandomState(0)
        cc = pu["conformal_cost"].to_numpy()
        fc_ = pu["fixed_cost"].to_numpy()
        idx = rng.randint(0, len(cc), size=(N_BOOT, len(cc)))
        red = 1 - cc[idx].sum(axis=1) / fc_[idx].sum(axis=1)
        tot = pu.sum()
        rows.append(dict(
            subset=subset, units=len(pu), dep_rows=len(dep),
            half_width=round(q, 1), fixed_buffer=round(fb, 1),
            coverage_dep=round(cov, 3),
            naive_cost=int(tot.naive_cost), naive_fail=int(tot.naive_fail),
            fixed_cost=int(tot.fixed_cost), fixed_fail=int(tot.fixed_fail),
            conf_cost=int(tot.conformal_cost), conf_fail=int(tot.conformal_fail),
            conf_vs_naive=round(1 - tot.conformal_cost / tot.naive_cost, 3),
            conf_vs_fixed=round(1 - tot.conformal_cost / tot.fixed_cost, 3),
            boot_lo=round(float(np.percentile(red, 2.5)), 3),
            boot_hi=round(float(np.percentile(red, 97.5)), 3)))
        print(f"{subset}: {len(pu)} units, q={q:.1f}, fixed={fb:.1f}, "
              f"cov(dep)={cov:.3f} | conf vs fixed "
              f"{rows[-1]['conf_vs_fixed']:+.1%} "
              f"[{rows[-1]['boot_lo']:+.1%}, {rows[-1]['boot_hi']:+.1%}] | "
              f"fails {int(tot.naive_fail)}/{int(tot.fixed_fail)}/"
              f"{int(tot.conformal_fail)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "b5_cmapss_decision.csv", index=False)

    # --- failure-penalty sweep -------------------------------------------
    # At C_fail = 500 the tuned fixed buffer is cheaper although it leaves
    # unplanned failures that the conformal buffer avoids, so the ordering must
    # depend on the penalty.  We locate the crossover on the pooled units.
    sweep = []
    for cf in (250, 500, 1000, 1500, 2000, 4000):
        tot = {k: [0.0, 0] for k in ("naive", "fixed", "conformal")}
        for subset, meta in fitted.items():
            dep, q, fb = meta
            for nm, b in (("naive", 0.0), ("fixed", fb), ("conformal", q)):
                for _, g in dep.groupby("unit"):
                    c, f = unit_cost_cf(g, b, cf)
                    tot[nm][0] += c
                    tot[nm][1] += f
        sweep.append(dict(c_fail=cf,
                          naive_cost=int(tot["naive"][0]), naive_fail=tot["naive"][1],
                          fixed_cost=int(tot["fixed"][0]), fixed_fail=tot["fixed"][1],
                          conf_cost=int(tot["conformal"][0]), conf_fail=tot["conformal"][1],
                          conf_vs_fixed=round(1 - tot["conformal"][0] / tot["fixed"][0], 3),
                          best=min(tot, key=lambda k: tot[k][0])))
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "b5_cmapss_sweep.csv", index=False)
    print("\n=== pooled failure-penalty sweep (178 units) ===")
    print(sw.to_string(index=False))

    # pooled across subsets, still bootstrapping over units
    pooled = pd.concat(per_unit_all.values()).astype(float)
    rng = np.random.RandomState(1)
    cc = pooled["conformal_cost"].to_numpy()
    fc_ = pooled["fixed_cost"].to_numpy()
    idx = rng.randint(0, len(cc), size=(N_BOOT, len(cc)))
    red = 1 - cc[idx].sum(axis=1) / fc_[idx].sum(axis=1)
    tot = pooled.sum()
    print("\n=== pooled over all four subsets ===")
    print(f"units={len(pooled)}  naive {tot.naive_cost:.0f}/{tot.naive_fail:.0f}"
          f"  fixed {tot.fixed_cost:.0f}/{tot.fixed_fail:.0f}"
          f"  conformal {tot.conformal_cost:.0f}/{tot.conformal_fail:.0f}")
    print(f"conf vs fixed {1 - tot.conformal_cost / tot.fixed_cost:+.1%} "
          f"95% CI [{np.percentile(red, 2.5):+.1%}, {np.percentile(red, 97.5):+.1%}]")
    print(f"conf vs naive {1 - tot.conformal_cost / tot.naive_cost:+.1%}")
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
