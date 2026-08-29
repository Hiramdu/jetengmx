"""
Extended UQ comparison for the IJPHM revision. Two questions a reviewer of
Javanmardi & Hullermeier (IJPHM 2023) will ask:

(A) METHOD ROBUSTNESS -- add the two missing members of the standard 5-way
    conformal lineup:
    4. nex-SCP (Barber et al. 2023): weighted split conformal with fixed
       time-decay weights w_i = rho^{(t_test - t_i)/step}. Designed exactly
       for the non-exchangeable regime we diagnose; requires NO
       post-deployment labels (unlike Gibbs & Candes online adaptation).
    5. CQR (Romano et al. 2019): conformalized quantile regression on
       LightGBM quantile models (5%/95%), calibrated on VAL.

(B) BASE-MODEL ROBUSTNESS -- is the coverage collapse a property of the
    stacking ensemble, or of the shift? Re-run split-conformal on three
    heterogeneous single-point predictors trained from scratch on TRAIN:
    boosted trees (LightGBM), linear ridge, and a feed-forward neural net.

All methods calibrated on VAL, evaluated on the late-life TEST window,
nominal 90% intervals, mirroring experiment_uq_compare.py.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPC_SV", "HPT_SV"]
ALPHA = 0.1
RHO = 0.99          # nex-SCP decay per calibration step (Javanmardi/Barber convention)
CYCLE_STEP = 10     # snapshots are every ~10 cycles


def conformal_q(absres, alpha):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


def weighted_conformal_q(scores, weights, alpha):
    """Barber et al. (2023) weighted quantile with +infinity point mass."""
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    wn = w / (w.sum() + 1.0)  # +1 = weight of the test point's infinity mass
    cw = np.cumsum(wn)
    idx = np.searchsorted(cw, 1 - alpha)
    if idx >= len(s):
        return np.inf
    return s[idx]


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def main():
    train = pd.read_csv(RESULTS / "train_with_predictions.csv")
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")

    drop = ["ESN"] + [f"Cycles_to_{t}" for t in TARGETS] + \
        [f"Pred_Cycles_to_{t}" for t in TARGETS]
    feat_cols = [c for c in train.columns if c not in drop]
    Xtr, Xva, Xte = train[feat_cols], val[feat_cols], test[feat_cols]

    rows = []

    # ------------------------------------------------------------------
    # (A) nex-SCP and CQR on the paper's stacking predictions
    # ------------------------------------------------------------------
    for t in TARGETS:
        y_col, p_col = f"Cycles_to_{t}", f"Pred_Cycles_to_{t}"
        yv, pv = val[y_col].to_numpy(), val[p_col].to_numpy()
        yt, pt = test[y_col].to_numpy(), test[p_col].to_numpy()
        scores = np.abs(yv - pv)

        # --- nex-SCP: per-test-point weighted quantile, time-decay weights
        t_cal = val["Cycles_Since_New"].to_numpy()
        t_test = test["Cycles_Since_New"].to_numpy()
        covered, widths, n_inf = [], [], 0
        for i in range(len(yt)):
            w = RHO ** ((t_test[i] - t_cal) / CYCLE_STEP)
            q = weighted_conformal_q(scores, w, ALPHA)
            lo, hi = max(pt[i] - q, 0.0), pt[i] + q
            covered.append(lo <= yt[i] <= hi)
            if np.isinf(hi):
                n_inf += 1
            else:
                widths.append(hi - lo)
        # An infinite interval trivially covers; report the finite-interval
        # mean width plus the fraction of vacuous (infinite) intervals.
        rows.append([t, "nex_scp", float(np.mean(covered)),
                     round(float(np.mean(widths)), 1) if widths else np.nan,
                     f"stacking (inf: {100*n_inf/len(yt):.0f}%)"])

        # --- CQR on LightGBM quantile models
        q_lo_m = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2,
                                   n_estimators=300, random_state=42, verbose=-1)
        q_hi_m = lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2,
                                   n_estimators=300, random_state=42, verbose=-1)
        ytr = train[y_col].to_numpy()
        q_lo_m.fit(Xtr, ytr)
        q_hi_m.fit(Xtr, ytr)
        lo_v, hi_v = q_lo_m.predict(Xva), q_hi_m.predict(Xva)
        cqr_scores = np.maximum(lo_v - yv, yv - hi_v)
        c = conformal_q(cqr_scores, ALPHA)
        lo_t = np.clip(q_lo_m.predict(Xte) - c, 0, None)
        hi_t = q_hi_m.predict(Xte) + c
        rows.append([t, "cqr", coverage(yt, lo_t, hi_t),
                     round(float(np.mean(hi_t - lo_t)), 1), "stacking/quantile-LGBM"])

    # ------------------------------------------------------------------
    # (B) split-conformal coverage under three alternative base models
    # ------------------------------------------------------------------
    base_models = {
        "lgbm_single": lambda: lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1),
        "ridge": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "mlp": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         early_stopping=True, random_state=42)),
    }
    for name, mk in base_models.items():
        for t in TARGETS:
            y_col = f"Cycles_to_{t}"
            m = mk()
            m.fit(Xtr, train[y_col])
            pv = np.clip(m.predict(Xva), 0, None)
            pt = np.clip(m.predict(Xte), 0, None)
            yv, yt = val[y_col].to_numpy(), test[y_col].to_numpy()
            q = conformal_q(np.abs(yv - pv), ALPHA)
            lo, hi = np.clip(pt - q, 0, None), pt + q
            rows.append([t, "split_conformal", coverage(yt, lo, hi),
                         round(float(np.mean(hi - lo)), 1), name])

    df = pd.DataFrame(rows, columns=["target", "method", "test_coverage",
                                     "mean_width", "base_model"])
    df.to_csv(RESULTS / "uq_extended_results.csv", index=False)
    pd.set_option("display.width", 140)

    print("=== (A) nex-SCP / CQR on stacking predictions (nominal 90%) ===")
    print(df[df.method.isin(["nex_scp", "cqr"])].to_string(index=False))
    print("\n=== (B) split-conformal under alternative base models ===")
    piv = df[df.method == "split_conformal"].pivot(
        index="target", columns="base_model", values="test_coverage")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
