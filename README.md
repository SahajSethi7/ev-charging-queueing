# EV Charging Queueing and Operations Research

This repository contains a reproducible research pipeline for analyzing EV-charging demand, queueing performance, charger faults, flexibility, scheduling, and demand-responsive charger activation. It uses the public Jiaxing charging-transactions dataset as the primary source and ACN-Data for limited cross-validation.

The repository intentionally excludes raw and session-level datasets. It includes source code, tests, aggregate results, figures, and the lookup data used by the React simulator.

## Headline results

- 441,077 Jiaxing charging sessions across 13 stations; overall observed fault rate 18.58%.
- Poisson arrivals rejected at all 13 stations, with residual within-hour overdispersion.
- XGBoost holdout performance: MAE 1.518, RMSE 2.256, and R² 0.508.
- LSTM horizon-1 MAE 1.533; weather ablation changed MAE by only -0.35%.
- 11.5% of analyzed sessions classified as likely flexible and 25.5% as likely or possibly flexible.
- Cost-first LP scheduling increased mean wait by 19.47 minutes versus the zero-wait FCFS replay baseline.
- Demand-responsive activation reduced active charger-hours by 21.8–32.7% while meeting the service target at three of four representative stations; the fourth required a one-charger-per-hour tuned margin.
- The frontend's 10 exact-fleet validation configurations passed independent-seed SimPy checks.

## Repository layout

```text
Code/                 Weekly analysis and simulation scripts
Results/              Aggregate tables, figures, and validation outputs
tests/                Regression tests for simulation and scheduling
my-appcd/              React/Vite lookup-based simulator
requirements.txt       Bounded Python dependencies
DATA_ACCESS.md         Dataset sources and redistribution guidance
NOTICE.md              Third-party and licensing notices
```

## Data access

Download source data from the original providers; do not expect a `Data/` directory in this repository.

1. Jiaxing EV charging transactions: [Figshare dataset, version 3](https://doi.org/10.6084/m9.figshare.28182251.v3). The associated article is [A high-resolution electric vehicle charging transaction dataset with multidimensional features in China](https://doi.org/10.1038/s41597-025-04982-1).
2. ACN-Data: use the [official Caltech portal](https://ev.caltech.edu/dataset.html). Registration requires agreement to educational/research use.

Place authorized downloads in `Data/` using the filenames expected by the scripts. See [DATA_ACCESS.md](DATA_ACCESS.md) for details.

## Python setup

Python 3.11–3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Week 5 defaults to CUDA for LSTM training. Use `--device auto` or `--device cpu` when CUDA is unavailable. Install the appropriate PyTorch build for your hardware separately if required.

## Pipeline

Run scripts from the repository root. Their defaults use `Data/` and `Results/weekN_results/`.

```powershell
python Code/ingest_jiaxing.py
python Code/week1_wrapup.py
python Code/week2_eda.py
python Code/week3_analysis.py
python Code/week4_analysis.py
python Code/week5_analysis.py --device auto
python Code/week6_analysis.py
python Code/week7_analysis.py
python Code/week8_analysis.py
python Code/week9_analysis.py
python Code/week10_validation.py
python Code/export_simulator_data.py
```

Some stages are computationally expensive. The checked-in aggregate outputs allow inspection of the reported results without rerunning the complete pipeline.

## Tests

```powershell
python -m pytest
```

The test suite covers fleet-independent exogenous arrivals, customer identity across fault retries, and Week 8 arrival/scheduling constraints.

## Frontend

```powershell
cd my-appcd
npm ci
npm run dev
```

The simulator reads `my-appcd/public/simulator-data.json` and exposes only exact precomputed fleet sizes; it does not interpolate unsupported configurations.

## Privacy and reproducibility

Only aggregate outputs are published here. Files containing per-session identifiers, exact session records, payment fields, user inputs, or machine-specific absolute paths are intentionally excluded. Provenance hashes in the private research workspace were used for verification, while public documentation uses repository-relative paths.

## Licence

The software source is released under the MIT License. Dataset rights remain with their respective providers, and the software licence does not grant rights to redistribute either source dataset. See [NOTICE.md](NOTICE.md).
