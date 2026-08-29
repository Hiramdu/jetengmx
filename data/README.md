# Input data (not redistributed)

Neither dataset is included here; both are freely available from their
publishers under their own terms.

## PHM North America 2025 Data Challenge (the paper's main dataset)

Download `training_data.csv` from
<https://data.phmsociety.org/phm-north-america-2025-conference-data-challenge/>
and place it either at the repository root or in this directory. It contains
59,702 snapshot rows for four engines (ESN 101-104).

## NASA C-MAPSS (external replication, Section 5.5, Table 6)

Download the Turbofan Engine Degradation Simulation Data Set from the NASA
Prognostics Data Repository
(<https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data>) and
unpack `train_FD001.txt` ... `train_FD004.txt` into a directory of your choice,
then point the two C-MAPSS scripts at it:

```bash
export CMAPSS_DIR=/path/to/cmapss
python experiment_cmapss_replication.py
python experiment_cmapss_crossunit.py
```

The default is `./cmapss`. Avoid `/tmp` on systems that sweep it by age.
