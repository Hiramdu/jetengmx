# Results directory

The small CSV and JSON files committed here are the exact outputs behind the
tables of the paper, so the numbers can be checked without rerunning anything.

Five large files are **not** committed, because they are intermediate artifacts
that any run regenerates and they total roughly 50 MB:

| File | Produced by | Consumed by |
|---|---|---|
| `train_with_predictions.csv` | `experiment_stack_calibrate.py` | ablations, UQ extensions, nested design |
| `val_with_predictions.csv` | `experiment_stack_calibrate.py` | conformal calibration, all decision scripts |
| `test_with_predictions.csv` | `experiment_stack_calibrate.py` | coverage and decision evaluation |
| `val_with_intervals.csv`, `test_with_intervals.csv` | `experiment_conformal.py` | `experiment_decision.py` |

So the order is: obtain `training_data.csv` (see `../data/README.md`), run
`experiment_stack_calibrate.py`, then `experiment_conformal.py`, after which
every other script runs standalone. The two strictly nested allocations are
generated with:

```bash
NESTED_VARIANT=I  python revision_b1b3_nested.py
NESTED_VARIANT=II python revision_b1b3_nested.py
```

Variant II writes the `_fit40` files committed in this directory.
