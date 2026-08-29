"""
Cross-unit late-life calibration on C-MAPSS (the reviewer-suggested
"rescue" experiment).

Question: is the late-life coverage collapse a property of run-to-failure
data per se, or of the WITHIN-UNIT CHRONOLOGICAL calibration protocol that a
young fleet (no completed failure histories) is forced into?

Protocol per subset:
  - model: same LGBM as experiment_cmapss_replication.py, trained on the
    first 60% of every unit's life.
  - within-unit protocol (paper's Table): calibrate on the 60-80% window of
    ALL units, evaluate on the final 20% (late-life) -- no late-life
    residuals available at calibration time.
  - cross-unit protocol (mature fleet): split units 50/50; calibrate on the
    LATE-LIFE (>80%) residuals of half the fleet ("historical engines that
    already ran to failure"), evaluate on the late-life of the other half.
    Both folds are run and averaged.

If cross-unit coverage recovers to ~nominal, the collapse is a protocol
property -- fixable with fleet history -- which converts the paper's
Limitations speculation into an evidence-backed statement.
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
DROP_SENSORS = [1, 5, 6, 10, 16, 18, 19]
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
    for c in FEATS:
        df[f"{c}_rm"] = df.groupby("unit")[c].transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    df["life_frac"] = df["cycle"] / df["fail_cycle"]
    return df


def conformal_q(absres, alpha=ALPHA):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


def main():
    fcols = FEATS + [f"{c}_rm" for c in FEATS]
    rows = []
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        df = load(subset)
        tr = df[df.life_frac <= 0.60]
        late = df[df.life_frac > 0.80]

        m = lgb.LGBMRegressor(n_estimators=300, max_depth=6,
                              learning_rate=0.05, random_state=42,
                              verbose=-1, n_jobs=-1)
        m.fit(tr[fcols], tr["RUL"])
        late = late.copy()
        late["pred"] = np.clip(m.predict(late[fcols]), 0, None)
        late["absres"] = np.abs(late["RUL"] - late["pred"])

        # within-unit chronological (paper protocol, for reference)
        cal_mid = df[(df.life_frac > 0.60) & (df.life_frac <= 0.80)]
        p_mid = np.clip(m.predict(cal_mid[fcols]), 0, None)
        q_within = conformal_q(np.abs(cal_mid["RUL"].to_numpy() - p_mid))
        cov_within = float(((late["RUL"] >= np.clip(late["pred"] - q_within, 0, None)) &
                            (late["RUL"] <= late["pred"] + q_within)).mean())

        # cross-unit late-life calibration, 2 folds
        units = np.array(sorted(df.unit.unique()))
        rng = np.random.RandomState(0)
        rng.shuffle(units)
        half = len(units) // 2
        folds = [(units[:half], units[half:]), (units[half:], units[:half])]
        covs, widths = [], []
        for cal_u, test_u in folds:
            cal = late[late.unit.isin(cal_u)]
            te = late[late.unit.isin(test_u)]
            q = conformal_q(cal["absres"].to_numpy())
            cov = float(((te["RUL"] >= np.clip(te["pred"] - q, 0, None)) &
                         (te["RUL"] <= te["pred"] + q)).mean())
            covs.append(cov)
            widths.append(q)
        rows.append({"subset": subset,
                     "within_unit": round(cov_within, 3),
                     "cross_unit": round(float(np.mean(covs)), 3),
                     "cross_unit_folds": f"{covs[0]:.3f}/{covs[1]:.3f}",
                     "halfwidth_within": round(float(q_within), 1),
                     "halfwidth_cross": round(float(np.mean(widths)), 1)})
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cmapss_crossunit.csv", index=False)
    print("\nnominal coverage = 0.90")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
