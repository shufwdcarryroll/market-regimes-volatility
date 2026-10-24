"""SIADS 696 Section 5.2.2.2 — 10-day Ridge alpha sensitivity.

Copy this script into src/part_a_sensitivity.py in your project and run:
    python src/part_a_sensitivity.py

Uses the existing feature dataset, splits, market feature list and preprocessing.
Does not modify the existing model results, notebooks, or Part B.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

CONFIG = {
    "data_file": "data/processed/feature_dataset.csv",
    "results_dir": "results/part_a",
    "figures_dir": "figures/part_a",
    "train_end": "2018-12-31",
    "validation_end": "2021-12-31",
    "horizon": 10,
    "alphas": (0.1, 1.0, 10.0, 100.0),
    "random_seed": 42,
    "output_stem": "ridge_sensitivity_10d",
    "figure_title": "Ridge regularization sensitivity — 10-day volatility",
    "figsize": (7.5, 4.5),
    "dpi": 300,
}


def project_root():
    here = Path(__file__).resolve().parent
    for folder in (here, here.parent, Path.cwd()):
        if (folder / CONFIG["data_file"]).exists():
            return folder
    raise FileNotFoundError("Cannot locate feature dataset. Run inside your project, or copy script to src/.")


def main():
    root = project_root()
    # Import the source project's exact market predictor list.
    sys.path.insert(0, str(root / 'src'))
    from forecasting import MARKET_FEATURES

    df = pd.read_csv(root / CONFIG['data_file'], index_col=0, parse_dates=True).sort_index()
    cols = list(MARKET_FEATURES)
    target = f"target_rv_{CONFIG['horizon']}"
    clean = df[cols + [target]].dropna()
    train = clean.loc[clean.index <= pd.Timestamp(CONFIG['train_end'])]
    validation = clean.loc[(clean.index > pd.Timestamp(CONFIG['train_end'])) &
                           (clean.index <= pd.Timestamp(CONFIG['validation_end']))]
    test = clean.loc[clean.index > pd.Timestamp(CONFIG['validation_end'])]
    if train.empty or validation.empty or test.empty:
        raise ValueError('Empty split: check project data and dates')

    rows = []
    for alpha in CONFIG['alphas']:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=alpha, random_state=CONFIG['random_seed']))
        ])
        model.fit(train[cols], train[target])
        pred_val = model.predict(validation[cols])
        pred_test = model.predict(test[cols])
        rows.append({
            'alpha': alpha,
            'val_rmse': float(np.sqrt(mean_squared_error(validation[target], pred_val))),
            'val_mae': float(mean_absolute_error(validation[target], pred_val)),
            'test_rmse': float(np.sqrt(mean_squared_error(test[target], pred_test))),
            'test_mae': float(mean_absolute_error(test[target], pred_test)),
            'train_n': len(train), 'validation_n': len(validation), 'test_n': len(test),
        })

    results = pd.DataFrame(rows)
    results_dir = root / CONFIG['results_dir']
    figures_dir = root / CONFIG['figures_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{CONFIG['output_stem']}.csv"
    results.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=CONFIG['figsize'], layout='constrained')
    ax.plot(results['alpha'], results['test_rmse'], 'o-', linewidth=2, label='Test RMSE')
    ax.set_xscale('log')
    ax.set_xticks(results['alpha'])
    ax.set_xticklabels([f'{v:g}' for v in results['alpha']])
    ax.set_xlabel('Ridge regularization strength (alpha; log scale)')
    ax.set_ylabel('10-day test RMSE')
    ax.set_title(CONFIG['figure_title'])
    ax.grid(alpha=0.25)
    for ext in ('png', 'pdf'):
        fig.savefig(figures_dir / f"{CONFIG['output_stem']}.{ext}", dpi=CONFIG['dpi'])
    plt.close(fig)

    print('10-day Ridge regularization sensitivity:')
    print(results[['alpha','test_rmse','test_mae','val_rmse','val_mae']].round(6).to_string(index=False))
    print(f'Observations: train {len(train)}, validation {len(validation)}, test {len(test)}')
    print('Saved:', out_csv)
    print('Saved:', figures_dir / f"{CONFIG['output_stem']}.png")
    print('Saved:', figures_dir / f"{CONFIG['output_stem']}.pdf")


if __name__ == '__main__':
    main()
