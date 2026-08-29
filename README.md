# From Prediction to Decision: Uncertainty-Aware Maintenance Scheduling for Commercial Turbofan Engines

Experiment code for the IJPHM paper by Gaoyuan Du, Ark Ifeanyi and Anahita
Khojandi.

The paper studies the full prognostics-and-health-management loop on the
[PHM North America 2025 Data Challenge](https://data.phmsociety.org/phm-north-america-2025-conference-data-challenge/):
a prediction pipeline, a per-component reliability diagnosis via split-conformal
prediction intervals, and an uncertainty-aware maintenance-scheduling policy
whose safety buffer is the conformal half-width.

Table and section numbers below refer to the **revised** manuscript.

## Contents

```
code/common.py            shared preprocessing, features, splitting, scoring
experiment_*.py           the pipeline and analyses of the original submission
revision_*.py             the revision round, one script per reviewer comment
make_*.py                 figures (written to figures/)
experiment_results/       locally generated outputs (not tracked)
data/                     where to put the input data (not redistributed)
```

## Setup

Python 3.11, CPU only. Pinned versions are in `requirements.txt`, which also
documents the one non-obvious install (tsfresh on systems with an old GCC).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Then obtain the two datasets as described in [`data/README.md`](data/README.md).
Neither is redistributed here.

## Running

Two scripts must run first, because they produce the prediction files that
everything else consumes. Generated outputs are written to the local,
untracked `experiment_results/` directory:

```bash
python experiment_stack_calibrate.py     # -> {train,val,test}_with_predictions.csv
python experiment_conformal.py           # -> {val,test}_with_intervals.csv
```

After that every script below runs standalone and writes to
`experiment_results/`.

### Original submission

| Stage | Script | Paper artifact |
|---|---|---|
| Hyperparameter search (50 Optuna trials per target) | `experiment_optuna.py` | Table 18 |
| Prediction pipeline (features → stacking → bias correction) | `experiment_stack_calibrate.py` | §4, Table 3 |
| Loss ablation | `experiment_loss_ablation.py` | Table 2 |
| Feature-family ablation | `experiment_feature_ablation.py` | §4.2 |
| Multi-seed repetition | `experiment_multiseed.py` | §4.3 (2.0 ± 0.2) |
| Conformal reliability diagnosis | `experiment_conformal.py` | §5, Table 4, Fig. 5 |
| UQ comparison (Gaussian, normalized) | `experiment_uq_compare.py` | Table 5, rows 1–3 |
| UQ extensions (nex-SCP, CQR, alternative base models) | `experiment_uq_extended.py` | Table 5 rows 4–5, §5.3 |
| Wilson intervals, leave-one-engine-out | `experiment_loeo_wilson.py` | §5.5 |
| C-MAPSS replication | `experiment_cmapss_replication.py` | Table 6, within-unit and control |
| C-MAPSS cross-unit calibration | `experiment_cmapss_crossunit.py` | Table 6, cross-unit column |
| Decision policies | `experiment_decision.py`, `experiment_decision_robust.py` | §6, Table 11 |
| Cost-sensitivity sweep | `experiment_bootstrap.py` | Table 12, Fig. 8 |
| Optimal-stopping MDP baseline | `experiment_mdp.py` | §6.7 |
| First-round analyses (cluster bootstrap, scheduled fallback, counter ablation) | `experiment_review_fixes.py` | §5.3, §6.3 |
| Figures | `make_paper_figures.py`, `make_revision_figures.py`, `make_shap_figure.py` | Figs. 2–8 |

### Revision round (one script per reviewer comment)

| Comment | Script | Paper artifact |
|---|---|---|
| **A3** — what if the uncertainty is a predictive density (Bayesian) rather than a population interval? | `revision_a3_pdf_uq.py` | §5.7, Table 8 — MC-dropout MLP and Gaussian-NLL deep ensemble; credible coverage, CRPS, PIT |
| **B1** — the calibration window is not independent of model selection<br>**B3** — the reliability gate is retrospective | `revision_b1b3_nested.py` | §5.6, Table 7 and §6.9, Table 16 — five disjoint temporal windows, prospective gate, decision experiment under out-of-sample calibration |
| **B2** — backward filling is not causal | `revision_b2_causal.py` | §5.8, Table 9 — forward-fill-only variant with a training-window median |
| **B4** — the buffer is symmetric and constant<br>**B6** — one cost applies to all three targets | `revision_b4b6_decision.py` | §6.6, Table 14 (one-sided at two levels, normalized in $\hat y$ and in $x$, CQR) and §6.5, Table 13 (target-specific failure penalties) |
| **B5** — report the decision outcomes by engine | `revision_b5_per_engine.py` | §6.3, Table 10 |
| **B5** — the same comparison with enough units to resolve it | `revision_b5_cmapss_decision.py` | §6.9, Table 15 — C-MAPSS mature-fleet protocol, 178 deployment units, failure-penalty sweep |

`revision_b5_cmapss_decision.py` needs the C-MAPSS files (see
[`data/README.md`](data/README.md)) and finishes in under a minute.
`revision_b1b3_nested.py` runs 50 Optuna trials per target and takes roughly
half an hour; its `WINDOWS` dictionary selects between the two apportionment
variants reported in Table 7. `revision_b2_causal.py` rebuilds all features
twice from `training_data.csv` and is the slowest script here. The rest finish
in seconds to minutes.

## Notes on reproducibility

- Seeds: `{0, 1, 2, 3, 42}` for the multi-seed study, `42` everywhere else
  (`RANDOM_STATE`/`SEED` in each script).
- The competition score uses the time weight λ = 0.01 of Eq. 1 throughout; every
  call site passes it explicitly.
- `revision_b2_causal.py` reruns feature extraction under whatever tsfresh
  release is installed. With tsfresh 0.21.2 it selects the same 229 features and
  reproduces validation R² to within 0.01, but the achieved test coverages move
  relative to Table 4 (HPT 0.568, HPC 0.284 against 0.47 and 0.17). Section 5.8
  of the paper discusses this; what is stable is the separation between the
  calibrated and the uncalibrated targets, not the individual coverage values.
- `make_clean_release.sh` in the parent directory regenerates this repository
  from the authors' working tree and rebuilds the archives.

## License

MIT for the code (see `LICENSE`). The challenge dataset and C-MAPSS are
distributed by the PHM Society and NASA under their own terms and are not
redistributed here.
