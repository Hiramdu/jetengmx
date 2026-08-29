"""
Direction A, formalized: maintenance scheduling as an optimal-stopping MDP,
solved by value iteration, replacing the threshold-buffer heuristic.

State    : discretized predicted RUL (hat-y) bin.
Actions  : MAINTAIN (terminal) or WAIT.
Immediate maintenance cost m(b): expected cost of servicing now given the
            predicted-RUL bin b, estimated from VALIDATION (pred, true) pairs.
            cost(true=r) = c_early*r           if r > 0  (early service)
                           C_fail + c_late|r|  if r <= 0 (unplanned failure)
            Because m(b) is an EXPECTATION over the true-RUL spread within a
            predicted bin, it automatically encodes each component's predictive
            reliability: a tight pred->true mapping (water-wash) yields a clean,
            low-variance m(b); a diffuse mapping (HPC) puts failure mass in
            many bins, raising m(b) everywhere.
Dynamics : one-step transition P(b'|b) estimated from VAL engine timelines
            (how the predicted-RUL bin evolves one cycle-step forward).
Solve    : discounted value iteration to a fixed point; STOP iff
            m(b) <= E[V(b') | b].  The resulting policy is a STATE-DEPENDENT
            stopping rule, not a constant buffer.

Train (estimate m, P; solve V/policy) on VAL only; evaluate on TEST segments
with the same per-maintenance-cycle accounting as experiment_decision_robust.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiment_results"
TARGETS = ["WW", "HPC_SV", "HPT_SV"]
C_EARLY, C_LATE = 1.0, 2.0
GAMMA = 0.99
N_BINS = 40


def maintenance_cost(r, c_fail):
    return C_EARLY * r if r > 0 else c_fail + C_LATE * abs(r)


def build_bins(pred_vals):
    # bins span observed predicted-RUL range; a dedicated bin 0 absorbs <=0
    hi = np.quantile(pred_vals, 0.99)
    edges = np.linspace(0, max(hi, 1.0), N_BINS)
    edges = np.concatenate([[-1e9], edges, [1e9]])
    return edges


def to_bin(pred, edges):
    return np.clip(np.digitize(pred, edges) - 1, 0, len(edges) - 2)


def estimate_m(val, target, edges, c_fail):
    """Expected immediate maintenance cost per predicted-RUL bin (from VAL)."""
    pred = val[f"Pred_Cycles_to_{target}"].to_numpy()
    true = val[f"Cycles_to_{target}"].to_numpy()
    b = to_bin(pred, edges)
    nb = len(edges) - 1
    m = np.full(nb, np.nan)
    for j in range(nb):
        idx = b == j
        if idx.sum() >= 3:
            m[j] = np.mean([maintenance_cost(r, c_fail) for r in true[idx]])
    # fill empty bins by nearest-filled (monotone-ish): forward then backward
    s = pd.Series(m).ffill().bfill()
    return s.to_numpy()


def estimate_P(val, target, edges):
    """Transition model over predicted-RUL bins WITH a forced failure
    absorbing state.

    The key modeling fix: waiting is not free. From each bin b there is an
    empirically estimated probability p_event(b) that the true maintenance
    event occurs at the next step (true RUL crosses <= 0). If it does and we
    have not maintained, that is an unplanned failure (absorbing, cost
    C_fail). With probability 1 - p_event(b) the system transitions among
    the predicted-RUL bins as observed in VAL.

    p_event(b) is estimated as: among VAL observations in bin b, the fraction
    whose true RUL is within one cycle-step (10 cycles) of the event. This
    gives imminent-event bins (small predicted RUL) a high forced-failure
    probability, so the optimal policy must stop there.

    Returns (P, p_event) where P is the (nb x nb) among-bins transition
    (row-stochastic over the non-failure mass) and p_event is length nb.
    """
    nb = len(edges) - 1
    P = np.zeros((nb, nb))
    for _, g in val.groupby("ESN"):
        g = g.sort_values("Cycles_Since_New")
        b = to_bin(g[f"Pred_Cycles_to_{target}"].to_numpy(), edges)
        for cur, nxt in zip(b[:-1], b[1:]):
            P[cur, nxt] += 1
    row = P.sum(1, keepdims=True)
    for j in range(nb):
        if row[j, 0] == 0:
            P[j, j] = 1.0
    P = P / P.sum(1, keepdims=True)

    # forced-event probability per bin, from VAL true RUL within one step
    pred = val[f"Pred_Cycles_to_{target}"].to_numpy()
    true = val[f"Cycles_to_{target}"].to_numpy()
    b = to_bin(pred, edges)
    p_event = np.zeros(nb)
    for j in range(nb):
        idx = b == j
        if idx.sum() >= 3:
            # event imminent if true RUL <= one cycle step (10 cycles)
            p_event[j] = float(np.mean(true[idx] <= 10))
    return P, p_event


def value_iteration(m, P, p_event, c_fail, gamma=GAMMA, iters=5000, tol=1e-6):
    """Optimal-stopping value iteration with a forced-failure absorbing state.

    WAIT cost-to-go from bin b:
        p_event(b) * C_fail                      (event strikes -> failure)
      + (1 - p_event(b)) * gamma * E[V(b') | b]  (survive -> continue)

    Because p_event(b) > 0 near the event, waiting is no longer free and the
    optimal policy stops once immediate maintenance cost m(b) is cheaper than
    this risk-adjusted continuation. STOP iff m(b) <= wait(b).
    """
    V = m.copy()
    for _ in range(iters):
        wait = p_event * c_fail + (1 - p_event) * gamma * (P @ V)
        Vn = np.minimum(m, wait)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    wait = p_event * c_fail + (1 - p_event) * gamma * (P @ V)
    stop = m <= wait
    return V, stop


def segment_cycles(g, target):
    g = g.sort_values("Cycles_Since_New").reset_index(drop=True)
    csl = g[f"Cycles_Since_Last_{target}"].to_numpy()
    bnds = [0] + list(np.where(np.diff(csl) < 0)[0] + 1) + [len(g)]
    segs = []
    for a, b in zip(bnds[:-1], bnds[1:]):
        seg = g.iloc[a:b]
        if (seg[f"Cycles_to_{target}"] <= 0).any() and len(seg) >= 3:
            segs.append(seg)
    return segs


def eval_policy(test, target, edges, stop, c_fail):
    """Apply the MDP stopping rule on test maintenance cycles."""
    costs, fails = [], 0
    for _, g in test.groupby("ESN"):
        for seg in segment_cycles(g, target):
            pred = seg[f"Pred_Cycles_to_{target}"].to_numpy()
            true = seg[f"Cycles_to_{target}"].to_numpy()
            b = to_bin(pred, edges)
            trig = np.where(stop[b])[0]
            if len(trig) == 0:
                costs.append(c_fail); fails += 1; continue
            i = trig[0]; r = true[i]
            costs.append(maintenance_cost(r, c_fail)); fails += int(r <= 0)
    return float(np.sum(costs)), fails, len(costs)


def main():
    val = pd.read_csv(RESULTS / "val_with_predictions.csv")
    test = pd.read_csv(RESULTS / "test_with_intervals.csv")

    print("=== Optimal-stopping MDP policy (trained on VAL, eval on TEST) ===\n")
    rows = []
    for cf in [250, 500, 1000]:
        per_t = {}
        for t in TARGETS:
            edges = build_bins(val[f"Pred_Cycles_to_{t}"].to_numpy())
            m = estimate_m(val, t, edges, cf)
            P, p_event = estimate_P(val, t, edges)
            V, stop = value_iteration(m, P, p_event, cf)
            cost, fails, n = eval_policy(test, t, edges, stop, cf)
            per_t[t] = (cost, fails, n)
            if cf == 500:
                rows.append({"target": t, "C_fail": cf, "total_cost": round(cost, 1),
                             "failures": fails, "n_cycles": n})
        tot = sum(v[0] for v in per_t.values())
        rel = per_t["WW"][0] + per_t["HPT_SV"][0]
        print(f"C_fail={cf:5d}  MDP total cost(all)={tot:8.0f}  "
              f"reliable(WW+HPT)={rel:7.0f}  "
              f"failures(all)={sum(v[1] for v in per_t.values())}")

    md = pd.DataFrame(rows)
    md.to_csv(RESULTS / "mdp_results.csv", index=False)
    print("\n=== MDP per-target (C_fail=500) ===")
    print(md.to_string(index=False))

    # Head-to-head vs heuristics on reliable targets at C_fail=500
    print("\n=== Reliable targets (WW+HPT), C_fail=500: MDP vs heuristics ===")
    dr = pd.read_csv(RESULTS / "decision_robust_results.csv")
    dr = dr[dr.target.isin(["WW", "HPT_SV"])]
    heur = dr.groupby("policy").agg(cost=("total_cost", "sum"),
                                    fails=("failures", "sum"))
    mdp_cost = md[md.target.isin(["WW", "HPT_SV"])].total_cost.sum()
    mdp_fail = int(md[md.target.isin(["WW", "HPT_SV"])].failures.sum())
    naive_cost = heur.loc["naive", "cost"]
    print(f"  naive             cost={heur.loc['naive','cost']:7.0f}  "
          f"fails={int(heur.loc['naive','fails'])}")
    print(f"  fixed_buffer      cost={heur.loc['fixed_buffer','cost']:7.0f}  "
          f"fails={int(heur.loc['fixed_buffer','fails'])}  "
          f"({(1-heur.loc['fixed_buffer','cost']/naive_cost)*100:+.1f}% vs naive)")
    print(f"  conformal_buffer  cost={heur.loc['uncertainty_aware','cost']:7.0f}  "
          f"fails={int(heur.loc['uncertainty_aware','fails'])}  "
          f"({(1-heur.loc['uncertainty_aware','cost']/naive_cost)*100:+.1f}% vs naive)")
    print(f"  MDP (optimal stop) cost={mdp_cost:7.0f}  fails={mdp_fail}  "
          f"({(1-mdp_cost/naive_cost)*100:+.1f}% vs naive)")


if __name__ == "__main__":
    main()
