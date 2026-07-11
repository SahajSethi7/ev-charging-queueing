"""Export regenerated Week 8/9 results for the web simulator.

Run after week9_analysis.py. The frontend intentionally consumes this JSON at
runtime so simulation output is not duplicated inside the JavaScript bundle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from project_paths import PROJECT_ROOT, RESULTS_DIR, to_builtin


def main() -> None:
    week9 = RESULTS_DIR / 'week9_results'
    week8 = RESULTS_DIR / 'week8_results'
    pareto = pd.read_csv(week9 / 'pareto_activation_results.csv')
    schedules = pd.read_csv(week9 / 'activation_schedules.csv')
    comparison_path = week8 / 'lp_vs_fcfs_comparison.csv'
    comparison = (pd.read_csv(comparison_path)
                  if comparison_path.exists() else pd.DataFrame())
    metadata_path = week9 / 'week9_metadata.json'
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    simulation_days = int(metadata['simulation_days'])

    payload = {
        'schema_version': 2,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'simulation_days': simulation_days,
        'pareto': pareto.to_dict(orient='records'),
        'schedules': schedules.to_dict(orient='records'),
        'scheduling_comparison': comparison.to_dict(orient='records'),
    }
    target = PROJECT_ROOT / 'my-appcd' / 'public' / 'simulator-data.json'
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(to_builtin(payload), handle, separators=(',', ':'),
                  allow_nan=False)
    print(f'Exported simulator data to {target}')


if __name__ == '__main__':
    main()
