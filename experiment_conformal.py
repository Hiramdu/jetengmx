"""
Direction B (reframed): Uncertainty-aware RUL via split-conformal prediction
intervals, and an honest characterization of how the late-life temporal
distribution shift degrades interval coverage.

We do NOT claim to "fix" the shift. We quantify it: calibrate conformal
intervals on the validation window and measure empirical coverage on the
later test window. The gap between nominal and achieved coverage is itself
the reported result, and it is what motivates the conservative decision
policy in experiment_decision.py.

Uses only the existing prediction CSVs (no model retraining).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPC_SV", "HPT_SV"]
ALPHAS = [0.1, 0.2]  # nominal miscoverage -> 90% and 80% intervals


def load(split):
    return pd.read_csv(RESULTS / f"{split}_with_predictions.csv")


def split_conformal(cal_resid, alpha):
    """Symmetric split-conformal quantile of |residual| at level 1-alpha.
    Uses the finite-sample-corrected rank ceil((n+1)(1-alpha))/n."""
    n = len(cal_resid)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(np.abs(cal_resid), q_level, method="higher")


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def main():
    train, val, test = load("train"), load("val"), load("test")

    # IMPORTANT: calibrate on VAL residuals, not TRAIN. The base/stacking
    # models were fit using TRAIN, so TRAIN residuals are optimistically
    # small and would yield meaningless (too-narrow) intervals. VAL is the
    # first split the model did not fit, so it is the valid calibration set.
    # We then evaluate coverage on TEST (the later, shifted window). The
    # nominal-vs-achieved gap on TEST is the honest distribution-shift result.
    rows = []
    for t in TARGETS:
        yhat_col, y_col = f"Pred_Cycles_to_{t}", f"Cycles_to_{t}"
        cal_resid = (val[y_col] - val[yhat_col]).to_numpy()
        for alpha in ALPHAS:
            width = split_conformal(cal_resid, alpha)  # one-sided half-width
            nominal = 1 - alpha
            for split_name, df in [("test", test)]:
                yhat = df[yhat_col].to_numpy()
                y = df[y_col].to_numpy()
                lo = np.clip(yhat - width, 0, None)
                hi = yhat + width
                cov = coverage(y, lo, hi)
                # RUL can't be negative; report effective (clipped) mean width
                mean_width = float((hi - lo).mean())
                rows.append({
                    "target": t,
                    "nominal_coverage": nominal,
                    "split": split_name,
                    "empirical_coverage": round(cov, 3),
                    "coverage_gap": round(nominal - cov, 3),
                    "interval_halfwidth": round(float(width), 1),
                    "mean_width": round(mean_width, 1),
                })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "conformal_results.csv", index=False)
    pd.set_option("display.width", 120)
    print("=== Split-conformal coverage (calibrated on TRAIN) ===\n")
    print(out.to_string(index=False))

    print("\n=== Key finding: per-target reliability under late-life shift ===")
    print("    (90% intervals calibrated on VAL, coverage measured on TEST)")
    for t in TARGETS:
        sub = out[(out.target == t) & (out.nominal_coverage == 0.9)]
        ct = sub.empirical_coverage.iloc[0]
        flag = "reliable" if ct >= 0.85 else ("degraded" if ct >= 0.4 else "UNRELIABLE")
        print(f"  {t:7s} nominal 90% -> achieved {ct:.0%} on test   [{flag}]")

    # Export per-row 90% intervals (VAL-calibrated) for the decision experiment.
    for t in TARGETS:
        yhat_col, y_col = f"Pred_Cycles_to_{t}", f"Cycles_to_{t}"
        width = split_conformal((val[y_col] - val[yhat_col]).to_numpy(), 0.1)
        test[f"Lo90_{t}"] = np.clip(test[yhat_col] - width, 0, None)
        test[f"Hi90_{t}"] = test[yhat_col] + width
        test[f"Width90_{t}"] = width
    test.to_csv(RESULTS / "test_with_intervals.csv", index=False)
    print("\nWrote conformal_results.csv, test_with_intervals.csv")


if __name__ == "__main__":
    main()
