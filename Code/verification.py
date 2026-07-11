"""Fast structural verification checks for regenerated project artifacts."""

import json

import pandas as pd

from project_paths import DATA_DIR, RESULTS_DIR


def main() -> None:
    df = pd.read_parquet(DATA_DIR / 'jiaxing_clean.parquet')
    assert 'user_stop_proxy' in df.columns
    assert 'flexibility_tier' not in df.columns

    hourly = pd.read_parquet(DATA_DIR / 'jiaxing_hourly.parquet')
    station_col = ('station_name' if 'station_name' in hourly.columns
                   else 'station_id')
    for station in hourly[station_col].unique():
        rows = hourly[hourly[station_col] == station]
        assert len(rows) == rows['date_dt'].nunique() * 24, \
            f'{station}: incomplete hourly grid'

    assert not (hourly['lag_1'].isna()
                & hourly['rolling_7d_mean'].notna()).any()
    for station in list(hourly[station_col].unique())[:3]:
        rows = hourly[hourly[station_col] == station].sort_values(
            'datetime').reset_index(drop=True)
        for index in range(24, min(len(rows), 100)):
            if pd.notna(rows.loc[index, 'lag_24']):
                assert rows.loc[index, 'lag_24'] == rows.loc[
                    index - 24, 'arrivals']

    week4_dir = RESULTS_DIR / 'week4_results'
    with open(week4_dir / 'parameter_summary.json', encoding='utf-8') as f:
        parameters = json.load(f)
    sized_stations = parameters['fleet_sizing_scope'][
        'representative_stations']
    assert len(sized_stations) == 4

    mgs = pd.read_csv(week4_dir / 'mgs_comparison_4rep.csv')
    assert set(mgs['sizing_scenario'].unique()) == {
        'historical_avg', 'recent_quarter', 'stress_case'}
    if 'probability_metric' in mgs.columns:
        assert mgs['probability_metric'].eq('p_wait_gt_15min').all()
    else:
        print('Note: existing M/G/s output predates the probability-metric schema; '
              'Week 4 will regenerate it.')
    print('All structural verification checks passed.')


if __name__ == '__main__':
    main()
