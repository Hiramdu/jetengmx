"""
External replication of the late-life coverage-collapse phenomenon on the
public NASA C-MAPSS benchmark (all four subsets FD001-FD004).

Protocol mirrors the paper's Section 5 exactly:
  - per-unit chronological split: first 60% of each unit's cycles -> train,
    next 20% -> calibration (validation), final 20% -> test (late-life).
  - base model: gradient-boosted trees (LightGBM) on the standard 14
    informative sensors (7 near-constant sensors removed, per Li et al. 2018 /
    Javanmardi & Hullermeier 2023), plus operating settings.
  - rectified RUL labels capped at RUL_max = 125 (standard convention).
  - split-conformal 90% intervals calibrated on the calibration window;
    achieved coverage measured on the late-life test window.

The paper's hypothesis: late-life coverage loss grows with how far the test
regime departs from what the calibration window represents. In C-MAPSS every
unit runs to failure exactly once, so the late-life window is ALWAYS the
approach to failure -- analogous to the rare-event (HPC-like) regime. We
therefore also report an in-distribution control: conformal calibrated and
evaluated on a random (unit-stratified) split of the same mid-life window,
which should retain nominal coverage if the collapse is due to the temporal
shift rather than the method.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
import lightgbm as lgb

DATA = Path(os.environ.get("CMAPSS_DIR", "cmapss"))
OUT = Path(__file__).parent / "experiment_results"
ALPHA = 0.1
RUL_MAX = 125
DROP_SENSORS = [1, 5, 6, 10, 16, 18, 19]  # near-constant (Li et al. 2018)
COLS = (["unit", "cycle", "op1", "op2", "op3"] +
        [f"s{i}" for i in range(1, 22)])
FEATS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)
                                 if i not in DROP_SENSORS]


def load(subset):
    df = pd.read_csv(DATA / f"train_{subset}.txt", sep=r"\s+", header=None,
                     names=COLS)
    fail = df.groupby("unit")["cycle"].max().rename("fail_cycle")
    df = df.join(fail, on="unit")
    df["RUL"] = np.minimum(df["fail_cycle"] - df["cycle"], RUL_MAX)
    # rolling-mean smoothing per unit (5-cycle) as a light feature aid
    for c in FEATS:
        df[f"{c}_rm"] = df.groupby("unit")[c].transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    df["life_frac"] = df["cycle"] / df["fail_cycle"]
    return df


def conformal_q(absres, alpha):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


def fit_predict(tr, other_frames):
    fcols = FEATS + [f"{c}_rm" for c in FEATS]
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          random_state=42, verbose=-1, n_jobs=-1)
    m.fit(tr[fcols], tr["RUL"])
    return [np.clip(m.predict(f[fcols]), 0, None) for f in other_frames]


def main():
    rows = []
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        df = load(subset)

        # --- late-life protocol (mirrors the paper) ---
        tr = df[df.life_frac <= 0.60]
        cal = df[(df.life_frac > 0.60) & (df.life_frac <= 0.80)]
        te = df[df.life_frac > 0.80]
        (pc, pt) = fit_predict(tr, [cal, te])
        q = conformal_q(np.abs(cal["RUL"].to_numpy() - pc), ALPHA)
        y = te["RUL"].to_numpy()
        cov_late = float(((y >= np.clip(pt - q, 0, None)) & (y <= pt + q)).mean())

        # --- in-distribution control: random unit-stratified split of the
        #     SAME calibration window (60-80% life) ---
        units = cal["unit"].unique()
        rng = np.random.RandomState(0)
        rng.shuffle(units)
        half = len(units) // 2
        cal_a = cal[cal.unit.isin(units[:half])]
        cal_b = cal[cal.unit.isin(units[half:])]
        (pa, pb) = fit_predict(tr, [cal_a, cal_b])
        q2 = conformal_q(np.abs(cal_a["RUL"].to_numpy() - pa), ALPHA)
        yb = cal_b["RUL"].to_numpy()
        cov_ctrl = float(((yb >= np.clip(pb - q2, 0, None)) & (yb <= pb + q2)).mean())

        rows.append({"subset": subset, "n_units": df.unit.nunique(),
                     "coverage_latelife": round(cov_late, 3),
                     "coverage_control": round(cov_ctrl, 3),
                     "halfwidth_latelife": round(float(q), 1),
                     "n_test": len(te)})
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cmapss_replication.csv", index=False)
    print("\nnominal coverage = 0.90 everywhere")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
