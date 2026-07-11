# Data access

The datasets used by this project are available from their original providers and are not duplicated in this repository.

## Jiaxing EV charging transactions

Download version 3 of the dataset from its official Figshare record:

- [Dataset on Figshare](https://doi.org/10.6084/m9.figshare.28182251.v3)
- [Associated Scientific Data article](https://doi.org/10.1038/s41597-025-04982-1)

The Figshare release provides the three source files required by the pipeline:

```text
Charging_Data.csv
Weather_Data.csv
Time-of-use_Price.csv
```

Create a `Data/` directory at the repository root and place the files there without renaming them:

```text
ev-charging-queueing/
├── Code/
├── Data/
│   ├── Charging_Data.csv
│   ├── Weather_Data.csv
│   └── Time-of-use_Price.csv
└── Results/
```

The records are described by the source authors as desensitized and anonymized, including randomly mapped user identifiers. Refer to the Figshare landing page for the dataset's current licence and citation information.

Running the ingestion and Week 1 scripts creates the canonical local inputs used by later stages:

```text
Data/jiaxing_clean.parquet
Data/jiaxing_hourly.parquet
Data/jiaxing_daily.parquet
Data/jiaxing_iat.parquet
```

```powershell
python Code/ingest_jiaxing.py
python Code/week1_wrapup.py
```

## ACN-Data cross-validation

ACN-Data is used only for the external cross-validation analysis.

- [Official ACN-Data portal](https://ev.caltech.edu/dataset.html)
- [Registration and access](https://ev.caltech.edu/register)

Download the JPL session export from the provider and save it as either:

```text
Data/acndata_sessions.json
```

or:

```text
Data/acn_jpl.json
```

Access requires registration and agreement to the provider's educational/research-use terms.

## Included results

The checked-in [`Results/`](Results/) directory contains the aggregate tables and figures needed to inspect the reported findings without downloading the session-level data or rerunning the complete pipeline.
