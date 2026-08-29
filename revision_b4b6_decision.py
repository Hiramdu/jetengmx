"""
Revision experiments for Reviewer B, comments 4 and 6.

B4  The submitted uncertainty-aware policy uses one symmetric conformal
    half-width per target, so the buffer is constant across operating points
    and RUL regions, while the decision risk is directional (over-predicting
    remaining life is what delays maintenance).  We therefore add:
      * one-sided conformal buffer from SIGNED over-prediction residuals,
        q = Quantile({yhat_i - y_i}; ceil((n+1)(1-alpha))/n), which gives the
        one-sided guarantee P(y >= yhat - q) >= 1-alpha.  Note the residuals
        are NOT clipped at zero: clipping would mix a point mass at zero into
        the quantile and deflate the buffer.
      * normalized (adaptive) conformal, width q * sigma(x) with
        sigma(x) = a + b*yhat fit on the calibration window, so the buffer
        varies within a component.
      * conformalized quantile regression (CQR), whose lower quantile model
        supplies a directly instance-dependent trigger.
    The comment asks for at least one of the adaptive constructions of Section
    5.3 to be propagated into the decision experiment; for completeness we
    propagate ALL five constructions of that section, adding the parametric
    Gaussian interval and the non-exchangeable weighted conformal method
    (nex-SCP) to the three above.

B6  The submitted analysis applies one C_fail to all three targets, although a
    water wash is an on-wing action and a shop visit requires engine removal.
    We sweep target-specific failure penalties and report whether the
    crossover conclusion survives.

Everything runs from the stored prediction/feature files; only the CQR quantile
models are refit (on the training window only).

Outputs (experiment_results/):
    b4_buffers_coverage.csv    buffer width and coverage per construction
    b4_decision.csv            decision outcomes per policy, target, engine
    b4_bootstrap.csv           engine-level cluster bootstrap per comparison
    b6_target_costs.csv        target-specific C_fail sweep
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV", "HPC_SV"]
ALPHA = 0.10
C_EARLY, C_LATE = 1.0, 2.0
C_FAIL_DEFAULT = 500.0
SEED = 42


def q_level(n, alpha=ALPHA):
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


def conformal_two_sided(res):
    """Symmetric radius from absolute residuals (the submitted construction)."""
    return float(np.quantile(np.abs(res), q_level(len(res)), method="higher"))


def conformal_one_sided(over, alpha=ALPHA):
    """Radius from SIGNED over-prediction residuals yhat - y, unclipped."""
    return float(np.quantile(over, q_level(len(over), alpha), method="higher"))


def segment_cycles(df, t):
    segs = []
    for esn, g in df.groupby("ESN"):
        g = g.sort_values("Cycles_Since_New")
        y = g[f"Cycles_to_{t}"].to_numpy()
        for seg in np.split(np.arange(len(y)), np.where(np.diff(y) > 0)[0] + 1):
            if len(seg) > 3 and y[seg].min() <= 10:
                segs.append(g.iloc[seg])
    return segs


def run_policy(seg, t, buffer, c_fail, trigger_col=None):
    """Walk the cycle forward; trigger when the decision statistic <= buffer.

    `buffer` may be a scalar or a per-row array (instance-dependent buffers).
    `trigger_col` allows a policy to trigger on a column other than the point
    prediction (used by CQR, which triggers on its lower quantile).
    """
    p = seg[trigger_col or f"Pred_Cycles_to_{t}"].to_numpy()
    y = seg[f"Cycles_to_{t}"].to_numpy()
    b = np.full(len(p), buffer, dtype=float) if np.isscalar(buffer) \
        else np.asarray(buffer, dtype=float)
    for k in range(len(p)):
        if p[k] <= b[k]:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (c_fail + C_LATE * abs(r), 1)
    return c_fail, 1


def tune_fixed(val, t, c_fail):
    segs = segment_cycles(val, t)
    if not segs:
        return 0.0
    return float(min(((b, sum(run_policy(s, t, b, c_fail)[0] for s in segs))
                      for b in np.linspace(0, 600, 61)), key=lambda x: x[1])[0])


def main():
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")
    train = pd.read_csv(RESULTS / "train_with_predictions.csv")
    drop = ["ESN", "Cycles_Since_New"] + [f"Cycles_to_{t}" for t in TARGETS] \
        + [f"Pred_Cycles_to_{t}" for t in TARGETS]
    feats = [c for c in train.columns if c not in drop]

    cov_rows, buffers = [], {}
    for t in TARGETS:
        yv = val[f"Cycles_to_{t}"].to_numpy()
        pv = val[f"Pred_Cycles_to_{t}"].to_numpy()
        yt = test[f"Cycles_to_{t}"].to_numpy()
        pt = test[f"Pred_Cycles_to_{t}"].to_numpy()
        res_v = yv - pv                     # y - yhat
        over_v = pv - yv                    # yhat - y  (positive = over-predict)

        # --- symmetric split-conformal (submitted) ---
        q_sym = conformal_two_sided(res_v)
        cov_rows.append(dict(construction="symmetric", target=t,
                             buffer_mean=round(q_sym, 1),
                             cov_two_sided=round(float(((yt >= pt - q_sym) &
                                                        (yt <= pt + q_sym)).mean()), 3),
                             cov_lower_only=round(float((yt >= pt - q_sym).mean()), 3)))

        # --- one-sided conformal on over-prediction residuals ---
        # At the SAME nominal level alpha the one-sided quantile is narrower
        # than the two-sided one, so a like-for-like comparison with the
        # symmetric 90 % interval must match the LOWER bound's confidence
        # level: a symmetric 1-alpha interval has a one-sided lower bound at
        # 1-alpha/2.  We report both readings.
        q_one = conformal_one_sided(over_v)                       # alpha = 0.10
        cov_rows.append(dict(construction="one_sided", target=t,
                             buffer_mean=round(q_one, 1), cov_two_sided=np.nan,
                             cov_lower_only=round(float((yt >= pt - q_one).mean()), 3)))
        q_one_m = conformal_one_sided(over_v, alpha=ALPHA / 2)    # alpha = 0.05
        cov_rows.append(dict(construction="one_sided_matched", target=t,
                             buffer_mean=round(q_one_m, 1), cov_two_sided=np.nan,
                             cov_lower_only=round(float((yt >= pt - q_one_m).mean()), 3)))

        # --- normalized (adaptive) conformal: sigma(x) = a + b*yhat ---
        a, b = np.polyfit(pv, np.abs(res_v), 1)[::-1]
        sig_v = np.clip(a + b * pv, 1e-6, None)
        sig_t = np.clip(a + b * pt, 1e-6, None)
        q_norm = float(np.quantile(np.abs(res_v) / sig_v, q_level(len(res_v)),
                                  method="higher"))
        w_t = q_norm * sig_t
        cov_rows.append(dict(construction="normalized", target=t,
                             buffer_mean=round(float(w_t.mean()), 1),
                             cov_two_sided=round(float(((yt >= pt - w_t) &
                                                        (yt <= pt + w_t)).mean()), 3),
                             cov_lower_only=round(float((yt >= pt - w_t).mean()), 3)))

        # --- parametric Gaussian interval (Table 5, row 2) propagated ---
        # Homoscedastic Gaussian residuals: half-width z_{1-alpha/2} * sigma.
        z = 1.6448536269514722          # Phi^{-1}(0.95) for the 90 % interval
        q_gauss = float(z * res_v.std(ddof=1))
        cov_rows.append(dict(construction="parametric_gaussian", target=t,
                             buffer_mean=round(q_gauss, 1),
                             cov_two_sided=round(float(((yt >= pt - q_gauss) &
                                                        (yt <= pt + q_gauss)).mean()), 3),
                             cov_lower_only=round(float((yt >= pt - q_gauss).mean()), 3)))

        # --- nex-SCP (Barber et al., 2023) propagated: weighted quantile of
        # the absolute residuals with fixed time-decay weights 0.99^{delta t},
        # so that stale calibration residuals count for less.
        dt = val["Cycles_Since_New"].max() - val["Cycles_Since_New"].to_numpy()
        w = 0.99 ** (dt / 10.0)         # cycles are sampled every 10 cycles
        order = np.argsort(np.abs(res_v))
        aw = np.abs(res_v)[order]
        cw = np.cumsum(w[order]) / w.sum()
        idx = int(np.searchsorted(cw, 1 - ALPHA))
        q_nex = float(aw[min(idx, len(aw) - 1)])
        cov_rows.append(dict(construction="nex_scp", target=t,
                             buffer_mean=round(q_nex, 1),
                             cov_two_sided=round(float(((yt >= pt - q_nex) &
                                                        (yt <= pt + q_nex)).mean()), 3),
                             cov_lower_only=round(float((yt >= pt - q_nex).mean()), 3)))

        # --- CQR: quantile models refit on the training window only ---
        ycol = f"Cycles_to_{t}"
        qlo = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2,
                                n_estimators=300, learning_rate=0.05, max_depth=6,
                                random_state=SEED, verbose=-1, n_jobs=-1
                                ).fit(train[feats], train[ycol])
        qhi = lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2,
                                n_estimators=300, learning_rate=0.05, max_depth=6,
                                random_state=SEED, verbose=-1, n_jobs=-1
                                ).fit(train[feats], train[ycol])
        lo_v, hi_v = qlo.predict(val[feats]), qhi.predict(val[feats])
        lo_t, hi_t = qlo.predict(test[feats]), qhi.predict(test[feats])
        E_v = np.maximum(lo_v - yv, yv - hi_v)
        q_cqr = float(np.quantile(E_v, q_level(len(E_v)), method="higher"))
        cqr_lo_t = np.clip(lo_t - q_cqr, 0, None)
        cov_rows.append(dict(construction="cqr", target=t,
                             buffer_mean=round(float(np.mean(pt - cqr_lo_t)), 1),
                             cov_two_sided=round(float(((yt >= cqr_lo_t) &
                                                        (yt <= hi_t + q_cqr)).mean()), 3),
                             cov_lower_only=round(float((yt >= cqr_lo_t).mean()), 3)))

        # --- normalized conformal with a COVARIATE-dependent difficulty model.
        # The reviewer notes that a buffer scaled by yhat alone still cannot
        # vary with the operating point.  We therefore fit sigma(x) on the
        # training window as a LightGBM regression of the absolute residual on
        # the full feature vector, which does depend on flight condition.
        res_tr = np.abs(train[ycol].to_numpy() - train[f"Pred_Cycles_to_{t}"].to_numpy())
        sig_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=6, random_state=SEED,
                                      verbose=-1, n_jobs=-1).fit(train[feats], res_tr)
        sg_v = np.clip(sig_model.predict(val[feats]), 1.0, None)
        sg_t = np.clip(sig_model.predict(test[feats]), 1.0, None)
        q_cov = float(np.quantile(np.abs(res_v) / sg_v, q_level(len(res_v)),
                                 method="higher"))
        w_cov_t = q_cov * sg_t
        cov_rows.append(dict(construction="normalized_covariate", target=t,
                             buffer_mean=round(float(w_cov_t.mean()), 1),
                             cov_two_sided=round(float(((yt >= pt - w_cov_t) &
                                                        (yt <= pt + w_cov_t)).mean()), 3),
                             cov_lower_only=round(float((yt >= pt - w_cov_t).mean()), 3)))

        buffers[t] = dict(symmetric=q_sym, one_sided=q_one,
                          one_sided_matched=q_one_m, normalized=w_t,
                          normalized_covariate=w_cov_t, cqr_trigger=cqr_lo_t,
                          parametric_gaussian=q_gauss, nex_scp=q_nex)
        test[f"CQRlo_{t}"] = cqr_lo_t

    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RESULTS / "b4_buffers_coverage.csv", index=False)
    print("=== B4: buffer constructions, widths and coverage on test ===")
    print(cov.to_string(index=False))

    # ------------------------------------------------------------------
    # B4: decision experiment with every buffer construction
    # ------------------------------------------------------------------
    rows = []
    for t in TARGETS:
        fb = tune_fixed(val, t, C_FAIL_DEFAULT)
        for esn in sorted(test.ESN.unique()):
            segs = segment_cycles(test[test.ESN == esn], t)
            r = dict(ESN=esn, target=t, n_cycles=len(segs),
                     fixed_buffer=round(fb, 1))
            policies = {
                "naive": lambda s: run_policy(s, t, 0.0, C_FAIL_DEFAULT),
                "fixed": lambda s: run_policy(s, t, fb, C_FAIL_DEFAULT),
                "symmetric": lambda s: run_policy(s, t, buffers[t]["symmetric"],
                                                  C_FAIL_DEFAULT),
                "one_sided": lambda s: run_policy(s, t, buffers[t]["one_sided"],
                                                  C_FAIL_DEFAULT),
                "one_sided_matched": lambda s: run_policy(
                    s, t, buffers[t]["one_sided_matched"], C_FAIL_DEFAULT),
                "normalized_cov": lambda s: run_policy(
                    s, t, buffers[t]["normalized_covariate"][s.index.to_numpy()],
                    C_FAIL_DEFAULT),
                "normalized": lambda s: run_policy(
                    s, t, buffers[t]["normalized"][s.index.to_numpy()], C_FAIL_DEFAULT),
                "parametric_gaussian": lambda s: run_policy(
                    s, t, buffers[t]["parametric_gaussian"], C_FAIL_DEFAULT),
                "nex_scp": lambda s: run_policy(s, t, buffers[t]["nex_scp"],
                                                C_FAIL_DEFAULT),
                "cqr": lambda s: run_policy(s, t, 0.0, C_FAIL_DEFAULT,
                                            trigger_col=f"CQRlo_{t}"),
            }
            for nm, fn in policies.items():
                c = f = 0
                for s in segs:
                    ci, fi = fn(s)
                    c += ci
                    f += fi
                r[f"{nm}_cost"], r[f"{nm}_fail"] = c, f
            rows.append(r)
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS / "b4_decision.csv", index=False)
    pol_names = ["naive", "fixed", "symmetric", "one_sided",
                 "one_sided_matched", "normalized", "normalized_cov",
                 "parametric_gaussian", "nex_scp", "cqr"]
    print("\n=== B4: decision outcomes by target (C_fail=500) ===")
    print(dec.groupby("target")[[f"{p}_{k}" for p in pol_names
                                 for k in ("cost", "fail")]].sum().to_string())

    # cluster bootstrap over engines, WW+HPT (the targets with usable signal)
    rng = np.random.RandomState(0)
    sub = dec[dec.target.isin(["WW", "HPT_SV"])]
    per_engine = {e: g[[f"{p}_cost" for p in pol_names]].sum()
                  for e, g in sub.groupby("ESN")}
    engines = list(per_engine)
    boot = {p: [] for p in pol_names}
    for _ in range(5000):
        pick = rng.choice(engines, size=len(engines), replace=True)
        tot = {p: sum(per_engine[e][f"{p}_cost"] for e in pick) for p in pol_names}
        for p in pol_names:
            boot[p].append(tot[p])
    brows = []
    for p in pol_names:
        arr = np.array(boot[p])
        ref = np.array(boot["fixed"])
        red = 1 - arr / ref
        brows.append(dict(policy=p, median_cost=float(np.median(arr)),
                          vs_fixed_median=round(float(np.median(red)), 3),
                          vs_fixed_lo=round(float(np.percentile(red, 2.5)), 3),
                          vs_fixed_hi=round(float(np.percentile(red, 97.5)), 3)))
    bs = pd.DataFrame(brows)
    bs.to_csv(RESULTS / "b4_bootstrap.csv", index=False)
    print("\n=== B4: engine-level cluster bootstrap, WW+HPT, relative to fixed ===")
    print(bs.to_string(index=False))

    # ------------------------------------------------------------------
    # B6: target-specific failure penalties
    # ------------------------------------------------------------------
    grids = {"WW": [25, 50, 100, 250, 500],
             "HPT_SV": [250, 500, 1000, 2000, 4000],
             "HPC_SV": [250, 500, 1000, 2000, 4000]}
    trows = []
    for t in TARGETS:
        for cf in grids[t]:
            fb = tune_fixed(val, t, cf)
            segs = segment_cycles(test, t)
            out = {}
            for nm, buf in (("naive", 0.0), ("fixed", fb),
                            ("symmetric", buffers[t]["symmetric"]),
                            ("one_sided", buffers[t]["one_sided"])):
                c = f = 0
                for s in segs:
                    ci, fi = run_policy(s, t, buf, cf)
                    c += ci
                    f += fi
                out[nm] = (c, f)
            best = min(out, key=lambda k: out[k][0])
            trows.append(dict(target=t, c_fail=cf, fixed_buffer=round(fb, 1),
                              **{f"{k}_cost": v[0] for k, v in out.items()},
                              **{f"{k}_fail": v[1] for k, v in out.items()},
                              best=best))
    tc = pd.DataFrame(trows)
    tc.to_csv(RESULTS / "b6_target_costs.csv", index=False)
    print("\n=== B6: target-specific failure penalties ===")
    print(tc.to_string(index=False))


if __name__ == "__main__":
    main()
