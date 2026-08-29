"""
Revision experiment for Reviewer B, comments 1 and 3.

B1: the submitted analysis calibrated split-conformal intervals on the
    validation window, but that window was used (a) to select LightGBM
    hyperparameters with Optuna, (b) to FIT the ridge meta-learner, and
    (c) to FIT the per-bin bias correction.  The calibration residuals were
    therefore in-sample for the final Stack+Cal predictor.

B3: the reliability gate was demonstrated retrospectively (achieved coverage
    on the evaluation window identified HPC as unreliable, and the fallback
    was then evaluated on that same window).

Both are addressed by one strictly nested temporal design.  Per engine, the
timeline is cut into four disjoint, time-ordered windows:

    A  0-60 %   fit the four base learners.  Deliberately the same window, and
                the same 4,800 rows, as the training split of the main analysis,
                so that nothing here is confounded with a smaller training set.
    B  60-70 %  model selection: Optuna for LightGBM (same search space and
                budget as experiment_optuna.py), ridge meta-learner, per-bin
                bias correction.
    C  70-75 %  split-conformal calibration  (untouched by fitting/tuning)
    D  75-80 %  GATE measurement: achieved coverage -> which targets may be
                driven by predictions.  Available strictly BEFORE deployment.
    E  80-100 % deployment: reported coverage and the decision experiment.
                Identical to the test window of the submitted paper (n=1,604).

Windows B, C and D together are exactly the 60-80 % range that the main analysis
used as one undifferentiated validation window for tuning, for fitting the
meta-learner, for fitting the bias correction AND for conformal calibration.

Everything the gate uses (A-D) precedes E, so the gate-and-policy workflow is
prospective.  Feature construction is inherited from the stored feature matrix
(the causal-imputation question is Reviewer B2 and handled separately in
revision_b2_causal.py).

Outputs (experiment_results/):
    nested_accuracy.csv     honest out-of-sample accuracy per window
    nested_coverage.csv     coverage on C (in-window), D (gate), E (deploy)
    nested_gate.csv         gate decision per target + resulting policy routing
    nested_decision.csv     decision outcomes on E, gated vs ungated, by engine
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV", "HPC_SV"]
SEED = 42
ALPHA = 0.10          # nominal miscoverage -> 90 % intervals
LAMBDA = 0.01         # time-weight of Eq. 1 as PRINTED in the manuscript
N_TRIALS = 50         # matches experiment_optuna.py, which produced the
                      # hyperparameters used in the main analysis
GATE_TOL = 0.10       # a target is trusted if achieved coverage >= 1-alpha-tol
C_EARLY, C_LATE, C_FAIL = 1.0, 2.0, 500.0

# window boundaries as fractions of each engine's own timeline
# The fit window is held at the first 60 % of each timeline -- exactly the
# training window of the main analysis -- so that the comparison with
# Tables 3-4 isolates the calibration protocol and is not confounded with a
# smaller training set.  Model selection, calibration and the gate are then
# carved out of the 60-80 % range that the main analysis used as one
# undifferentiated validation window.
WINDOWS = {"A": (0.00, 0.60), "B": (0.60, 0.70), "C": (0.70, 0.75),
           "D": (0.75, 0.80), "E": (0.80, 1.00)}


# ----------------------------------------------------------------- data ----
def load_panel():
    """Rebuild the full cycle-level panel from the stored split files."""
    parts = [pd.read_csv(RESULTS / f"{s}_with_predictions.csv")
             for s in ("train", "val", "test")]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["ESN", "Cycles_Since_New"]).reset_index(drop=True)
    drop = ["ESN", "Cycles_Since_New"] + [f"Cycles_to_{t}" for t in TARGETS] \
        + [f"Pred_Cycles_to_{t}" for t in TARGETS]
    feats = [c for c in df.columns if c not in drop]
    return df, feats


def cut_windows(df):
    """Split each engine's timeline into the five disjoint windows."""
    out = {k: [] for k in WINDOWS}
    for esn in sorted(df.ESN.unique()):
        ed = df[df.ESN == esn].sort_values("Cycles_Since_New")
        n = len(ed)
        for k, (lo, hi) in WINDOWS.items():
            out[k].append(ed.iloc[int(n * lo):int(n * hi)])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in out.items()}


# --------------------------------------------------------------- scoring ---
def comp_score(y, p, beta, lam=LAMBDA):
    err = p - y
    w = np.where(err >= 0, 2.0 / (1 + lam * y), 1.0 / (1 + lam * y))
    return float(np.mean(w * err ** 2 * beta))


def asymmetric_objective(y_true, y_pred):
    err = y_pred - y_true
    w = (1.0 + (err >= 0)) / (1.0 + LAMBDA * y_true)
    return 2.0 * w * err, 2.0 * w


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(absres, lvl, method="higher"))


def coverage_trend(W, preds, target, q, n_blocks=4):
    """Slope of achieved coverage across equal blocks of the pre-deployment
    windows C and D, with the half-width q fixed from window C.

    Motivation: the level-based gate of Section 5.6 asks whether coverage is
    currently adequate, but distribution shift is progressive, so a component
    can still be inside tolerance while deteriorating.  The slope uses only
    windows that precede deployment, so it is available at the same time as the
    level.  The decision rule -- flag a target when the slope is significantly
    negative -- is fixed here rather than tuned on the deployment outcome, and
    the significance test is a two-sided t-test on the OLS slope of coverage
    against block index (n_blocks - 2 degrees of freedom).
    """
    ycol = f"Cycles_to_{target}"
    frames, ps = [], []
    for k in ("C", "D"):
        frames.append(W[k])
        ps.append(preds[k])
    df = pd.concat(frames, ignore_index=True)
    pr = np.concatenate(ps)
    order = np.argsort(df["Cycles_Since_New"].to_numpy(), kind="stable")
    y = df[ycol].to_numpy()[order]
    pr = pr[order]
    hit = ((y >= np.clip(pr - q, 0, None)) & (y <= pr + q)).astype(float)
    blocks = np.array_split(hit, n_blocks)
    cov = np.array([b.mean() for b in blocks])
    x = np.arange(1, n_blocks + 1, dtype=float)
    slope, intercept = np.polyfit(x, cov, 1)
    resid = cov - (slope * x + intercept)
    dof = n_blocks - 2
    se = np.sqrt((resid ** 2).sum() / dof / ((x - x.mean()) ** 2).sum()) \
        if dof > 0 and (resid ** 2).sum() > 0 else 0.0
    tstat = slope / se if se > 0 else (-np.inf if slope < 0 else np.inf)
    # two-sided p from Student t with dof degrees of freedom
    from scipy import stats
    pval = float(2 * (1 - stats.t.cdf(abs(tstat), dof))) if se > 0 else 0.0
    return dict(block_coverage=[round(float(c), 3) for c in cov],
                slope=round(float(slope), 4), t=round(float(tstat), 2),
                p=round(pval, 4))


def coverage(y, p, q):
    return float(((y >= np.clip(p - q, 0, None)) & (y <= p + q)).mean())


# ------------------------------------------------------- decision policy ---
def segment_cycles(df, t):
    segs = []
    for esn, g in df.groupby("ESN"):
        g = g.sort_values("Cycles_Since_New")
        y = g[f"Cycles_to_{t}"].to_numpy()
        for seg in np.split(np.arange(len(y)), np.where(np.diff(y) > 0)[0] + 1):
            if len(seg) > 3 and y[seg].min() <= 10:
                segs.append(g.iloc[seg])
    return segs


def policy_cost(seg, t, buffer, pred_col=None):
    p = seg[pred_col or f"Pred_Cycles_to_{t}"].to_numpy()
    y = seg[f"Cycles_to_{t}"].to_numpy()
    for k in range(len(p)):
        if p[k] <= buffer:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (C_FAIL + C_LATE * abs(r), 1)
    return C_FAIL, 1


def scheduled_cost(seg, t, interval):
    """Prediction-free age-based fallback (10-cycle sampling as in the paper)."""
    y = seg[f"Cycles_to_{t}"].to_numpy()
    for k in range(len(y)):
        if k * 10 >= interval:
            r = y[k]
            return (C_EARLY * r, 0) if r > 0 else (C_FAIL + C_LATE * abs(r), 1)
    return C_FAIL, 1


# ------------------------------------------------------------- pipeline ----
def fit_pipeline(W, feats, target):
    """Fit the paper's stacking pipeline with a strictly nested protocol.

    Base learners are fit on window A only.  Optuna, the ridge meta-learner
    and the per-bin bias correction all use window B only.  Windows C, D, E
    are never seen during fitting or tuning.
    """
    ycol = f"Cycles_to_{target}"
    XA, yA = W["A"][feats].to_numpy(), W["A"][ycol].to_numpy()
    XB, yB = W["B"][feats].to_numpy(), W["B"][ycol].to_numpy()
    beta = (1.0 if target == "WW" else 2.0) / W["A"][ycol].max()

    scaler = StandardScaler().fit(XA)

    # --- Optuna for LightGBM, scored on window B (model-selection window) ---
    def objective(trial):
        params = dict(
            # Search space identical to experiment_optuna.py, so that the only
            # difference from the main analysis is the temporal protocol.
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
        )
        m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1, n_jobs=-1,
                              objective=asymmetric_objective)
        m.fit(XA, yA)
        return comp_score(yB, np.clip(m.predict(XB), 0, None), beta)

    study = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params

    # --- base learners, all fit on window A only ---
    models = {
        "Ridge": (Ridge(alpha=10.0).fit(scaler.transform(XA), yA), True),
        "GBR": (GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, subsample=0.8,
                                          random_state=SEED).fit(XA, yA), False),
        "RF": (RandomForestRegressor(n_estimators=300, max_depth=20,
                                     min_samples_leaf=3, random_state=SEED,
                                     n_jobs=-1).fit(XA, yA), False),
        "LGBM": (lgb.LGBMRegressor(**best, random_state=SEED, verbose=-1,
                                   n_jobs=-1,
                                   objective=asymmetric_objective).fit(XA, yA), False),
    }

    ctx = [c for c in ("Cycles_Since_Last_WW", "Cycles_Since_Last_HPC_SV",
                       "Cycles_Since_Last_HPT_SV") if c in feats]
    ctx_idx = [feats.index(c) for c in ctx]

    def base_matrix(X):
        cols = [np.clip(m.predict(scaler.transform(X) if sc else X), 0, None)
                for m, sc in models.values()]
        return np.column_stack(cols + [X[:, ctx_idx]])

    # --- meta-learner on window B only ---
    meta = Ridge(alpha=1.0).fit(base_matrix(XB), yB)

    # --- per-bin bias correction, also on window B only ---
    raw_B = np.clip(meta.predict(base_matrix(XB)), 0, None)
    n_bins, resid = 20, raw_B - yB
    edges = np.linspace(0, raw_B.max() * 1.01, n_bins + 1)
    idx = np.clip(np.digitize(raw_B, edges) - 1, 0, n_bins - 1)
    corr = np.zeros(n_bins)
    for b in range(n_bins):
        m_ = idx == b
        if m_.sum() > 5:
            corr[b] = np.median(resid[m_])

    def predict(X):
        raw = np.clip(meta.predict(base_matrix(X)), 0, None)
        i = np.clip(np.digitize(raw, edges) - 1, 0, n_bins - 1)
        return np.clip(raw - corr[i], 0, None)

    return predict, beta, best


def main():
    df, feats = load_panel()
    W = cut_windows(df)
    print("window sizes:", {k: len(v) for k, v in W.items()}, "features:", len(feats))

    acc, cov, gate, best_params = [], [], [], {}
    preds = {}
    for t in TARGETS:
        print(f"\n=== {t} ===", flush=True)
        predict, beta, best = fit_pipeline(W, feats, t)
        best_params[t] = best
        ycol = f"Cycles_to_{t}"
        p = {k: predict(W[k][feats].to_numpy()) for k in WINDOWS}
        preds[t] = p

        for k in ("B", "C", "D", "E"):
            y = W[k][ycol].to_numpy()
            acc.append(dict(target=t, window=k, n=len(y),
                            mae=round(mean_absolute_error(y, p[k]), 1),
                            r2=round(r2_score(y, p[k]), 3),
                            comp_score=round(comp_score(y, p[k], beta), 3)))

        # conformal calibration on window C only
        q = conformal_q(np.abs(W["C"][ycol].to_numpy() - p["C"]))
        for k in ("C", "D", "E"):
            cov.append(dict(target=t, window=k, nominal=1 - ALPHA,
                            half_width=round(q, 1),
                            achieved=round(coverage(W[k][ycol].to_numpy(), p[k], q), 3)))

        cov_D = coverage(W["D"][ycol].to_numpy(), p["D"], q)
        cov_E = coverage(W["E"][ycol].to_numpy(), p["E"], q)
        trusted = cov_D >= (1 - ALPHA - GATE_TOL)
        tr = coverage_trend(W, p, t, q)
        # trend gate: flag when coverage is deteriorating across the
        # pre-deployment blocks, in addition to the level rule
        trend_flag = (tr["slope"] < 0) and (tr["p"] < 0.10)
        trusted_trend = trusted and not trend_flag
        gate.append(dict(target=t, half_width=round(q, 1),
                         gate_coverage_D=round(cov_D, 3),
                         gate_threshold=round(1 - ALPHA - GATE_TOL, 2),
                         decision="prediction-driven" if trusted else "fallback",
                         block_coverage=tr["block_coverage"], slope=tr["slope"],
                         slope_t=tr["t"], slope_p=tr["p"],
                         decision_trend=("prediction-driven" if trusted_trend
                                         else "fallback"),
                         deployed_coverage_E=round(cov_E, 3)))
        print(f"  q={q:.1f}  cov(C)={coverage(W['C'][ycol].to_numpy(),p['C'],q):.3f}"
              f"  gate cov(D)={cov_D:.3f} -> "
              f"{'TRUST' if trusted else 'FALLBACK'}   cov(E)={cov_E:.3f}\n"
              f"      blocks={tr['block_coverage']} slope={tr['slope']:+.4f} "
              f"p={tr['p']:.3f} -> trend gate: "
              f"{'TRUST' if trusted_trend else 'FALLBACK'}", flush=True)

    pd.DataFrame(acc).to_csv(RESULTS / "nested_accuracy.csv", index=False)
    pd.DataFrame(cov).to_csv(RESULTS / "nested_coverage.csv", index=False)
    gate_df = pd.DataFrame(gate)
    gate_df.to_csv(RESULTS / "nested_gate.csv", index=False)
    with open(RESULTS / "nested_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # ------------------------------------------------------------------
    # prospective gate-and-policy workflow on window E
    # ------------------------------------------------------------------
    E = W["E"].copy()
    for t in TARGETS:
        E[f"Pred_Cycles_to_{t}"] = preds[t]["E"]
    C_ = W["C"].copy()
    for t in TARGETS:
        C_[f"Pred_Cycles_to_{t}"] = preds[t]["C"]

    rows = []
    for t in TARGETS:
        q = float(gate_df.loc[gate_df.target == t, "half_width"].iloc[0])
        trusted = gate_df.loc[gate_df.target == t, "decision"].iloc[0] == "prediction-driven"
        # fixed buffer tuned on the calibration window (never on E)
        cal_segs = segment_cycles(C_, t)
        fixed_b = 0.0
        if cal_segs:
            fixed_b = min(((b, sum(policy_cost(s, t, b)[0] for s in cal_segs))
                           for b in np.linspace(0, 600, 61)), key=lambda x: x[1])[0]
        mean_len = np.mean([len(s) * 10 for s in cal_segs]) if cal_segs else 500.0
        sched_iv = min(((iv, sum(scheduled_cost(s, t, iv)[0] for s in cal_segs))
                        for iv in mean_len * np.array([0.5, 0.6, 0.7, 0.8, 0.9])),
                       key=lambda x: x[1])[0] if cal_segs else mean_len
        for esn in sorted(E.ESN.unique()):
            segs = segment_cycles(E[E.ESN == esn], t)
            r = dict(ESN=esn, target=t, n_cycles=len(segs),
                     gate="trust" if trusted else "fallback")
            for nm, fn in (("naive", lambda s: policy_cost(s, t, 0.0)),
                           ("fixed", lambda s: policy_cost(s, t, fixed_b)),
                           ("conformal", lambda s: policy_cost(s, t, q)),
                           ("scheduled", lambda s: scheduled_cost(s, t, sched_iv)),
                           ("gated", lambda s: policy_cost(s, t, q) if trusted
                            else scheduled_cost(s, t, sched_iv))):
                c = f = 0
                for s in segs:
                    ci, fi = fn(s)
                    c += ci
                    f += fi
                r[f"{nm}_cost"], r[f"{nm}_fail"] = c, f
            rows.append(r)
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS / "nested_decision.csv", index=False)

    print("\n=== nested coverage ===")
    print(pd.DataFrame(cov).to_string(index=False))
    print("\n=== prospective gate ===")
    print(gate_df.to_string(index=False))
    print("\n=== decision on window E, summed over engines ===")
    g = dec.groupby("target")[[c for c in dec.columns if c.endswith(("_cost", "_fail"))]].sum()
    print(g.to_string())
    print("\n=== decision on window E, per engine (all targets) ===")
    print(dec.groupby("ESN")[[c for c in dec.columns
                              if c.endswith(("_cost", "_fail"))]].sum().to_string())


if __name__ == "__main__":
    main()
