"""
B4 + B5 for the IJPHM revision:
  B4 -- Wilson 95% confidence intervals for the achieved test coverages in
        Table 3 (point estimates alone invite a sample-size objection).
  B5 -- leave-one-engine-out (LOEO) robustness: recompute split-conformal
        90% coverage four times, each time calibrating on 3 engines'
        validation residuals and evaluating on the held-out engine's test
        window. No model retraining (uses stored stacking predictions),
        exactly like the paper's main analysis.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPT_SV", "HPC_SV"]
ALPHA = 0.1


def conformal_q(absres, alpha):
    n = len(absres)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(absres, lvl, method="higher")


def wilson(p, n, z=1.96):
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return center - half, center + half


def main():
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_predictions.csv")

    print("=== B4: Wilson 95% CIs for Table 3 (nominal 90%) ===")
    for t in TARGETS:
        y_col, p_col = f"Cycles_to_{t}", f"Pred_Cycles_to_{t}"
        q = conformal_q(np.abs(val[y_col] - val[p_col]).to_numpy(), ALPHA)
        y, p = test[y_col].to_numpy(), test[p_col].to_numpy()
        cov = ((y >= np.clip(p - q, 0, None)) & (y <= p + q))
        lo, hi = wilson(cov.mean(), len(cov))
        print(f"{t:7s} coverage {cov.mean():.3f}  n={len(cov)}  "
              f"Wilson95 [{lo:.3f}, {hi:.3f}]")

    print("\n=== B5: leave-one-engine-out coverage (nominal 90%) ===")
    rows = []
    for t in TARGETS:
        y_col, p_col = f"Cycles_to_{t}", f"Pred_Cycles_to_{t}"
        for held in sorted(test.ESN.unique()):
            cal = val[val.ESN != held]
            ev = test[test.ESN == held]
            q = conformal_q(np.abs(cal[y_col] - cal[p_col]).to_numpy(), ALPHA)
            y, p = ev[y_col].to_numpy(), ev[p_col].to_numpy()
            cov = float(((y >= np.clip(p - q, 0, None)) & (y <= p + q)).mean())
            rows.append({"target": t, "held_out_engine": held,
                         "coverage": round(cov, 3), "n": len(ev)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "loeo_coverage.csv", index=False)
    piv = df.pivot(index="target", columns="held_out_engine", values="coverage")
    print(piv.to_string())
    print("\nrange per target:")
    for t in TARGETS:
        s = df[df.target == t].coverage
        print(f"  {t:7s} {s.min():.2f} - {s.max():.2f}")


if __name__ == "__main__":
    main()
