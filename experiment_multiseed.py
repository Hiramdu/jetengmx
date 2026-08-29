"""
Multi-seed repetition of the prediction pipeline for the IJPHM revision.

Reruns the seed-dependent portion of experiment_stack_calibrate.py (base
models -> ridge stacking -> per-bin calibration) across 5 random seeds,
reusing the engineered features already stored in the prediction CSVs.
Reports mean +/- std for the headline numbers: validation competition
score, and per-target validation/test MAE and R^2 (Table 2 of the paper).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
SEEDS = [0, 1, 2, 3, 42]
N_BINS = 20


def asymmetric_objective(y_true, y_pred):
    alpha = 0.01
    error = y_pred - y_true
    late_factor = 1.0 + (error >= 0).astype(float)
    weight = late_factor / (1.0 + alpha * y_true)
    return 2.0 * weight * error, 2.0 * weight


def comp_score_all(df, preds):
    alpha = 0.01
    betas = {t: (1.0 if t.endswith("WW") else 2.0) / df[t].max() for t in TARGETS}
    scores = []
    for t in TARGETS:
        y, p = df[t].values, preds[t]
        e = p - y
        w = np.where(e >= 0, 2 / (1 + alpha * y), 1 / (1 + alpha * y))
        scores.append(np.mean(w * e ** 2 * betas[t]))
    return float(np.mean(scores))


def run_seed(seed, train, val, test, feat_cols, optuna_params):
    Xtr, Xva, Xte = train[feat_cols].values, val[feat_cols].values, test[feat_cols].values
    scaler = StandardScaler().fit(np.nan_to_num(Xtr))
    ctx = [feat_cols.index(c) for c in
           ["Cycles_Since_Last_WW", "Cycles_Since_Last_HPC_SV", "Cycles_Since_Last_HPT_SV"]
           if c in feat_cols]

    val_preds, test_preds = {}, {}
    for t in TARGETS:
        ytr, yva = train[t].values, val[t].values
        base_val, base_test = [], []
        # Ridge (deterministic) on scaled features
        m = Ridge(alpha=10.0).fit(scaler.transform(np.nan_to_num(Xtr)), ytr)
        base_val.append(np.clip(m.predict(scaler.transform(np.nan_to_num(Xva))), 0, None))
        base_test.append(np.clip(m.predict(scaler.transform(np.nan_to_num(Xte))), 0, None))
        # GBR
        m = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                      subsample=0.8, random_state=seed).fit(np.nan_to_num(Xtr), ytr)
        base_val.append(np.clip(m.predict(np.nan_to_num(Xva)), 0, None))
        base_test.append(np.clip(m.predict(np.nan_to_num(Xte)), 0, None))
        # RF
        m = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3,
                                  random_state=seed, n_jobs=-1).fit(np.nan_to_num(Xtr), ytr)
        base_val.append(np.clip(m.predict(np.nan_to_num(Xva)), 0, None))
        base_test.append(np.clip(m.predict(np.nan_to_num(Xte)), 0, None))
        # LGBM tuned + asymmetric objective
        m = lgb.LGBMRegressor(**optuna_params[t], random_state=seed, verbose=-1,
                              n_jobs=-1, objective=asymmetric_objective).fit(Xtr, ytr)
        base_val.append(np.clip(m.predict(Xva), 0, None))
        base_test.append(np.clip(m.predict(Xte), 0, None))

        # Ridge meta-learner on val (same protocol as the paper pipeline)
        mv = np.column_stack(base_val + [Xva[:, c] for c in ctx])
        mt = np.column_stack(base_test + [Xte[:, c] for c in ctx])
        meta = Ridge(alpha=1.0).fit(np.nan_to_num(mv), yva)
        sv = np.clip(meta.predict(np.nan_to_num(mv)), 0, None)
        st = np.clip(meta.predict(np.nan_to_num(mt)), 0, None)

        # per-bin median-residual calibration (fit on val)
        edges = np.linspace(0, sv.max() * 1.01, N_BINS + 1)
        bi = np.clip(np.digitize(sv, edges) - 1, 0, N_BINS - 1)
        corr = np.zeros(N_BINS)
        for b in range(N_BINS):
            m_ = bi == b
            if m_.sum() > 5:
                corr[b] = np.median((sv - yva)[m_])
        cal_v = np.clip(sv - corr[bi], 0, None)
        ti = np.clip(np.digitize(st, edges) - 1, 0, N_BINS - 1)
        cal_t = np.clip(st - corr[ti], 0, None)
        val_preds[t], test_preds[t] = cal_v, cal_t

    row = {"seed": seed, "val_cs": comp_score_all(val, val_preds)}
    for t in TARGETS:
        k = t.replace("Cycles_to_", "")
        row[f"{k}_val_mae"] = mean_absolute_error(val[t], val_preds[t])
        row[f"{k}_val_r2"] = r2_score(val[t], val_preds[t])
        row[f"{k}_test_mae"] = mean_absolute_error(test[t], test_preds[t])
        row[f"{k}_test_r2"] = r2_score(test[t], test_preds[t])
    return row


def main():
    train = pd.read_csv(RESULTS / "train_with_predictions.csv")
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")
    with open(RESULTS / "optuna_best_params.json") as f:
        optuna_params = json.load(f)

    drop = ["ESN"] + TARGETS + [f"Pred_{t}" for t in TARGETS]
    feat_cols = [c for c in train.columns if c not in drop]

    rows = [run_seed(s, train, val, test, feat_cols, optuna_params) for s in SEEDS]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multiseed_results.csv", index=False)

    print("=== Per-seed results ===")
    print(df.round(3).to_string(index=False))
    print("\n=== mean +/- std over", len(SEEDS), "seeds ===")
    for c in df.columns:
        if c == "seed":
            continue
        print(f"{c:16s} {df[c].mean():10.3f} +/- {df[c].std():.3f}")


if __name__ == "__main__":
    main()
