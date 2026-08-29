"""
Experiments answering the simulated-review panel (R3, R4, R5):

R3  Engine-cluster bootstrap for the decision-layer cost reduction and
    cluster-aware coverage uncertainty (replaces i.i.d. Wilson/bootstrap as
    the primary uncertainty statement).
R4  Scheduled-maintenance fallback policy for HPC: trigger at a fixed cycle
    count tuned on validation mean inter-event interval (prediction-free).
R5  Alternative-explanation ablation: split-conformal coverage with the
    cycles-since-last feature family REMOVED, and with the ridge learner
    (which extrapolates linearly) alone.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV", "HPC_SV"]
ALPHA = 0.1
C_FAIL, C_EARLY, C_LATE = 500, 1, 2


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


def segment_cycles(df, t):
    """Split each engine's test timeline into degrade-to-event cycles using
    resets of the true remaining-cycles counter."""
    y_col = f"Cycles_to_{t}"
    out = []
    for esn, g in df.groupby("ESN"):
        g = g.sort_values("Cycles_Since_New")
        y = g[y_col].to_numpy()
        resets = np.where(np.diff(y) > 0)[0] + 1
        for seg in np.split(np.arange(len(y)), resets):
            if len(seg) > 3 and y[seg].min() <= 10:  # complete cycle
                out.append(g.iloc[seg])
    return out


def policy_cost(seg, t, buffer):
    """Walk forward; trigger when prediction <= buffer. Returns (cost, fail)."""
    p = seg[f"Pred_Cycles_to_{t}"].to_numpy()
    y = seg[f"Cycles_to_{t}"].to_numpy()
    for k in range(len(p)):
        if p[k] <= buffer:
            r = y[k]
            if r > 0:
                return C_EARLY * r, 0
            return C_FAIL + C_LATE * abs(r), 1
    return C_FAIL, 1


def scheduled_cost(seg, t, interval):
    """Prediction-free policy: service when cycles-since-last reaches
    `interval` (age-based trigger)."""
    y = seg[f"Cycles_to_{t}"].to_numpy()
    age = seg[f"Cycles_Since_Last_{t if t != 'WW' else 'WW'}"].to_numpy() \
        if f"Cycles_Since_Last_{t}" in seg.columns else None
    # age within the cycle = position from cycle start (10-cycle steps)
    n = len(y)
    steps = np.arange(n) * 10
    for k in range(n):
        if steps[k] >= interval:
            r = y[k]
            if r > 0:
                return C_EARLY * r, 0
            return C_FAIL + C_LATE * abs(r), 1
    return C_FAIL, 1


def main():
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")
    train = pd.read_csv(RESULTS / "train_with_predictions.csv")

    # ------------------------------------------------------------------
    # R4: scheduled fallback for HPC (and all targets, for completeness)
    # ------------------------------------------------------------------
    print("=== R4: scheduled (age-based) policy, C_fail=500 ===")
    conf_widths = {t: conformal_q(np.abs(val[f"Cycles_to_{t}"] -
                                         val[f"Pred_Cycles_to_{t}"]).to_numpy())
                   for t in TARGETS}
    for t in TARGETS:
        segs = segment_cycles(test, t)
        # tune interval on VALIDATION mean inter-event gap minus a margin grid
        val_segs = segment_cycles(val, t)
        mean_len = np.mean([len(s) * 10 for s in val_segs]) if val_segs else 500
        best = None
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            iv = mean_len * frac
            c = sum(scheduled_cost(s, t, iv)[0] for s in val_segs) if val_segs else np.inf
            if best is None or c < best[1]:
                best = (iv, c)
        iv = best[0]
        costs, fails = zip(*[scheduled_cost(s, t, iv) for s in segs])
        # conformal policy for reference
        ccosts, cfails = zip(*[policy_cost(s, t, conf_widths[t]) for s in segs])
        print(f"{t:7s} n={len(segs):2d} scheduled(interval={iv:.0f}): "
              f"cost={sum(costs):6.0f} fails={sum(fails)} | "
              f"conformal: cost={sum(ccosts):6.0f} fails={sum(cfails)}")

    # ------------------------------------------------------------------
    # R3: engine-cluster bootstrap for the fixed-vs-conformal comparison
    # ------------------------------------------------------------------
    print("\n=== R3: engine-cluster bootstrap (reliable targets WW+HPT) ===")
    # per-engine, per-policy totals
    per_engine = {}
    # tuned fixed buffers (validation-tuned per target, as in the paper)
    fixed_buffers = {}
    for t in ["WW", "HPT_SV"]:
        val_segs = segment_cycles(val, t)
        best = None
        for b in np.linspace(0, 600, 61):
            c = sum(policy_cost(s, t, b)[0] for s in val_segs)
            if best is None or c < best[1]:
                best = (b, c)
        fixed_buffers[t] = best[0]
    for esn in sorted(test.ESN.unique()):
        sub = test[test.ESN == esn]
        row = {}
        for t in ["WW", "HPT_SV"]:
            segs = segment_cycles(sub, t)
            for name, buf in [("naive", 0.0),
                              ("fixed", fixed_buffers[t]),
                              ("conf", conf_widths[t])]:
                c, f = (0, 0)
                for s in segs:
                    ci, fi = policy_cost(s, t, buf)
                    c += ci; f += fi
                row[f"{t}_{name}_cost"] = c
                row[f"{t}_{name}_fail"] = f
        per_engine[esn] = row
    pe = pd.DataFrame(per_engine).T
    tot = pe.sum()
    for name in ["naive", "fixed", "conf"]:
        c = tot[f"WW_{name}_cost"] + tot[f"HPT_SV_{name}_cost"]
        f = tot[f"WW_{name}_fail"] + tot[f"HPT_SV_{name}_fail"]
        print(f"{name:6s} total cost={c:7.0f} fails={f:.0f}")
    # cluster bootstrap over engines
    rng = np.random.RandomState(0)
    red_fixed, red_naive = [], []
    engines = list(per_engine)
    for _ in range(5000):
        pick = rng.choice(engines, size=len(engines), replace=True)
        cs = {n: sum(per_engine[e][f"{t}_{n}_cost"]
                     for e in pick for t in ["WW", "HPT_SV"]) for n in
              ["naive", "fixed", "conf"]}
        if cs["fixed"] > 0:
            red_fixed.append(1 - cs["conf"] / cs["fixed"])
        if cs["naive"] > 0:
            red_naive.append(1 - cs["conf"] / cs["naive"])
    for lbl, arr in [("conf vs naive", red_naive), ("conf vs fixed", red_fixed)]:
        a = np.array(arr)
        print(f"{lbl}: median {np.median(a):.1%} "
              f"95% cluster-CI [{np.percentile(a,2.5):.1%}, {np.percentile(a,97.5):.1%}]")

    # ------------------------------------------------------------------
    # R5: coverage without the cycles-since-last family / ridge-only
    # ------------------------------------------------------------------
    print("\n=== R5: coverage ablations (nominal 90%) ===")
    drop = ["ESN"] + [f"Cycles_to_{t}" for t in TARGETS] + \
        [f"Pred_Cycles_to_{t}" for t in TARGETS]
    feat_all = [c for c in train.columns if c not in drop]
    csl = [c for c in feat_all if "Cycles_Since_Last" in c or
           "Cumulative" in c or "Cycles_Since_New" in c]
    feat_nocsl = [c for c in feat_all if c not in csl]
    print(f"removed {len(csl)} counter features, kept {len(feat_nocsl)}")

    configs = {
        "lgbm_nocounter": (lgb.LGBMRegressor(n_estimators=300, max_depth=6,
                                             learning_rate=0.05, random_state=42,
                                             verbose=-1, n_jobs=-1), feat_nocsl),
        "ridge_full": (make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler(), Ridge(alpha=10.0)), feat_all),
        "ridge_nocounter": (make_pipeline(SimpleImputer(strategy="median"),
                                          StandardScaler(), Ridge(alpha=10.0)),
                            feat_nocsl),
    }
    rows = []
    for name, (model, fc) in configs.items():
        for t in TARGETS:
            y_col = f"Cycles_to_{t}"
            import copy
            m = copy.deepcopy(model)
            m.fit(train[fc], train[y_col])
            pv = np.clip(m.predict(val[fc]), 0, None)
            pt = np.clip(m.predict(test[fc]), 0, None)
            q = conformal_q(np.abs(val[y_col].to_numpy() - pv))
            y = test[y_col].to_numpy()
            cov = float(((y >= np.clip(pt - q, 0, None)) & (y <= pt + q)).mean())
            rows.append([name, t, round(cov, 3)])
    df = pd.DataFrame(rows, columns=["config", "target", "coverage"])
    print(df.pivot(index="target", columns="config", values="coverage").to_string())
    df.to_csv(RESULTS / "review_fix_ablations.csv", index=False)


if __name__ == "__main__":
    main()
