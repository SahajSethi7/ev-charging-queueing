# Phase 2 Conclusions

*Quantitative claims assembled from Weeks 4–6 outputs.*


## Claim 1: M/M/s and M/G/s Recommendations Are Close, but a Direct Simulation Gap Is Not Evaluated

- **Expressway A (DC Fast)**: Week 4 recommends s*=5 under M/M/s and s*=5 under the M/G/s approximation (gap=0). Week 6 does not compute a direct analytical-versus-simulation gap because the available metrics differ (peak-hour P(W>0) versus full-day P(W>15min)).

- **Technology Park (Mixed)**: Week 4 recommends s*=6 under M/M/s and s*=6 under the M/G/s approximation (gap=0). Week 6 does not compute a direct analytical-versus-simulation gap because the available metrics differ (peak-hour P(W>0) versus full-day P(W>15min)).

- **Gov Agency (L2)**: Week 4 recommends s*=8 under M/M/s and s*=8 under the M/G/s approximation (gap=0). Week 6 does not compute a direct analytical-versus-simulation gap because the available metrics differ (peak-hour P(W>0) versus full-day P(W>15min)).

- **Bus Station (L2, High-Vol)**: Week 4 recommends s*=10 under M/M/s and s*=11 under the M/G/s approximation (gap=1). Week 6 does not compute a direct analytical-versus-simulation gap because the available metrics differ (peak-hour P(W>0) versus full-day P(W>15min)).


The Week 4 M/G/s approximation changes the historical-average recommendation by only 0-1 chargers across these stations. A direct comparison against the Week 6 simulation is intentionally not reported because their probability metrics are not the same.


## Claim 2: Fault Capacity Penalty

- **Expressway A (DC Fast)**: s*(faults ON)=4, s*(faults OFF)=4, fault tax=+0 chargers. Empirical fault fraction: 16.9%.

- **Technology Park (Mixed)**: s*(faults ON)=6, s*(faults OFF)=6, fault tax=+0 chargers. Empirical fault fraction: 15.6%.

- **Gov Agency (L2)**: s*(faults ON)=8, s*(faults OFF)=6, fault tax=+2 chargers. Empirical fault fraction: 16.6%.

- **Bus Station (L2, High-Vol)**: s*(faults ON)=10, s*(faults OFF)=10, fault tax=+0 chargers. Empirical fault fraction: 17.2%.


Sensitivity (Gov Agency, L2-heavy): s* steps from 6 to 8 at ~15% fault rate. Below 10% fault rate, the fault tax disappears at step-2 grid resolution.


## Claim 3: Demand Growth Impact on Fleet Sizing

LSTM vs XGBoost (horizon-1, Sep-Dec 2021): XGBoost MAE=1.518, LSTM MAE=1.533 (-1.0%). LSTM does not outperform XGBoost. XGBoost is the adopted forecasting model.


### Peak lambda comparison: historical average vs recent quarter

The recent-quarter scenario uses NHPP rates from the final quarter of the dataset, reflecting observed demand growth. If s* changes between scenarios, that is a demand-trend effect; no fleet-sizing effect is inferred from forecasting-model MAE alone.


- **Expressway A (DC Fast)**: historical peak lambda=1.72 (s*=5), recent-quarter peak=3.52 (s*=7). Fleet size CHANGES.

- **Technology Park (Mixed)**: historical peak lambda=2.78 (s*=6), recent-quarter peak=2.78 (s*=6). Fleet size unchanged.

- **Gov Agency (L2)**: historical peak lambda=3.70 (s*=8), recent-quarter peak=5.43 (s*=10). Fleet size CHANGES.

- **Bus Station (L2, High-Vol)**: historical peak lambda=5.58 (s*=10), recent-quarter peak=7.90 (s*=13). Fleet size CHANGES.


Accounting for recent demand growth changes the fleet recommendation at some stations. This is a demand trend effect, not a benefit specific to ML over growth-adjusted NHPP.
