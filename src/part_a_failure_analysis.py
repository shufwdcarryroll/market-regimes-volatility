"""Reproduce market-only Ridge test forecasts and extract three large forecast errors.
Run from any directory: python part_a_failure_analysis.py --project-root /path/to/SIADS696_market-regimes-volatility
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main(root: Path):
    sys.path.insert(0, str(root / 'src'))
    from forecasting import MARKET_FEATURES
    data = pd.read_csv(root / 'data/processed/feature_dataset.csv', index_col=0, parse_dates=True).sort_index()
    output = root / 'results' / 'part_a'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in (5, 10, 20):
        target = f'target_rv_{horizon}'
        usable = data.loc[:, MARKET_FEATURES + [target]].dropna()
        train = usable.loc[usable.index <= '2018-12-31']
        test = usable.loc[(usable.index > '2021-12-31') & (usable.index <= '2026-09-30')]
        model = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0, random_state=42))])
        model.fit(train[MARKET_FEATURES], train[target])
        predicted = model.predict(test[MARKET_FEATURES])
        out = pd.DataFrame({'Date': test.index, 'Horizon': f'{horizon}d', 'Actual_RV':test[target].to_numpy(), 'Predicted_RV': predicted})
        out['Error'] = out['Predicted_RV'] - out['Actual_RV']
        out['Absolute_Error'] = out['Error'].abs()
        rows.append(out)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(output / 'ridge_test_predictions_all_horizons.csv', index=False)
    top = pd.concat([group.nlargest(1, 'Absolute_Error') for _, group in predictions.groupby('Horizon')]).sort_values('Horizon')
    top.to_csv(output / 'ridge_representative_failures.csv', index=False)
    predictions.nlargest(30, 'Absolute_Error').to_csv(output / 'ridge_largest_errors.csv', index=False)
    print('Largest error per horizon:')
    print(top.round(5).to_string(index=False))
    print('Saved under:', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent / 'siads_work' / 'SIADS696_market-regimes-volatility copy')
    main(parser.parse_args().project_root)
