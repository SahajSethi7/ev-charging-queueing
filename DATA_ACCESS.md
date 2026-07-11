# Data access and redistribution

Raw, cleaned, and session-level data are not distributed in this repository.

## Jiaxing dataset

- Dataset DOI: <https://doi.org/10.6084/m9.figshare.28182251.v3>
- Article DOI: <https://doi.org/10.1038/s41597-025-04982-1>

The source describes the records as desensitized and anonymized, including randomly mapped user IDs. Nevertheless, the records retain detailed timestamps, charging locations/categories, payments, and longitudinal pseudonymous identifiers. Obtain the dataset from Figshare and follow the licence displayed on the dataset's own landing page.

A Google Drive mirror is not included. Rehosting a cleaned derivative on Google Drive is still redistribution; add such a link only after confirming that the Figshare item licence permits redistribution of the derivative and that the shared file contains no fields beyond the intended research release.

Expected primary files under `Data/`:

```text
Charging_Data.csv
Weather_Data.csv
Time-of-use_Price.csv
```

The pipeline produces `jiaxing_clean.parquet`, `jiaxing_hourly.parquet`, `jiaxing_daily.parquet`, and `jiaxing_iat.parquet` locally.

## ACN-Data

- Official portal: <https://ev.caltech.edu/dataset.html>
- Registration: <https://ev.caltech.edu/register>

ACN-Data access requires registration and agreement to educational/research use. This repository does not redistribute its session export. Download it directly from the provider when reproducing the cross-validation stage.

## Public result boundary

Published result files contain station-, configuration-, day-, or replication-level aggregates. The following private-workspace artifacts are deliberately excluded:

- raw and cleaned session tables;
- ACN JSON/XLSX exports;
- `flexibility_analysis.csv`;
- `acn_flexibility.csv`;
- `representative_days.csv`;
- trained PyTorch pickle/checkpoint files;
- provenance JSON containing absolute workstation paths.
