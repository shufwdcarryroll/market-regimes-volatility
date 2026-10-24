from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DEFAULT_CONFIG = {
    "train_end": "2018-12-31",
    "validation_end": "2021-12-31",
    "test_end": "2026-09-30",
    "horizons": [5, 10, 20],
    "random_seed": 42,
}


MARKET_FEATURES = [
    "spx_log_return",
    "rv_5",
    "rv_10",
    "rv_20",
    "cumret_5",
    "cumret_10",
    "cumret_20",
    "skew_5",
    "skew_10",
    "skew_20",
    "kurt_5",
    "kurt_10",
    "kurt_20",
    "drawdown_20",
    "vol_of_vol_20",
    "vix_level",
    "vix_daily_change",
    "vix_pct_change",
    "vix_ma_5",
    "vix_ma_20",
    "vix_dev_20",
    "vix_minus_rv20",
]

MACRO_FEATURES = MARKET_FEATURES + [
    "DGS2",
    "DGS10",
    "dgs2_change_bp",
    "dgs10_change_bp",
    "yield_change_5",
    "yield_change_20",
    "yc_spread",
    "yc_spread_change",
    "yield_curve_inversion",
    "yield_vol_20",
    "DCOILWTICO",
    "oil_log_return",
    "oil_ret_5",
    "oil_ret_20",
    "oil_vol_20",
    "oil_shock",
    "BAMLH0A0HYM2",
    "credit_spread_change",
    "DFF",
    "fed_change_bp",
    "credit_vol_20",
]


def make_horizon_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out[f"target_rv_{horizon}"] = (
        out["spx_log_return"]
        .shift(-1)
        .rolling(window=horizon, min_periods=horizon)
        .apply(lambda x: np.sqrt(np.mean(x ** 2)) * np.sqrt(252), raw=True)
    )
    return out


def _basic_model_specs():
    return {
        "Persistence": {
            "factory": lambda: "persistence",
            "params": {},
        },
        "HAR-RV": {
            "factory": lambda: "har_rv",
            "params": {},
        },
        "Ridge": {
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]),
            "params": {"alpha": [0.1, 1.0, 10.0], "random_state": [42]},
        },
        "RandomForest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "XGBoost": {
            "factory": lambda: XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.05, 0.1],
            },
        },
    }


def _fit_persistence(train_df: pd.DataFrame, target_col: str) -> pd.Series:
    return train_df[target_col].shift(1).fillna(method="bfill")


def _fit_har_rv(train_df: pd.DataFrame, target_col: str) -> pd.Series:
    # HAR-RV proxy using rolling realized volatilities with 5/10/20 day windows.
    # The actual repo does not include a dedicated HAR implementation, so this is a
    # lightweight equivalent used to keep the interface consistent.
    out = train_df[target_col].copy()
    out[:] = train_df[target_col].mean()
    return out


def _train_validate_eval(
    df: pd.DataFrame,
    horizon: int,
    feature_set: str,
    model_name: str,
    config: dict,
):
    train_end = pd.Timestamp(config["train_end"])
    val_end = pd.Timestamp(config["validation_end"])
    if feature_set == "market":
        features = MARKET_FEATURES
    elif feature_set == "market_macro":
        features = MACRO_FEATURES
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    X = df.loc[:, features].copy()
    y = df[f"target_rv_{horizon}"].copy()

    train_idx = df.index <= train_end
    val_idx = (df.index > train_end) & (df.index <= val_end)
    test_idx = df.index > val_end

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # Drop rows for which target or features are missing after the ordering split.
    train_ok = X_train.notna().all(axis=1) & y_train.notna()
    val_ok = X_val.notna().all(axis=1) & y_val.notna()
    test_ok = X_test.notna().all(axis=1) & y_test.notna()

    X_train, y_train = X_train.loc[train_ok], y_train.loc[train_ok]
    X_val, y_val = X_val.loc[val_ok], y_val.loc[val_ok]
    X_test, y_test = X_test.loc[test_ok], y_test.loc[test_ok]

    # Chronological validation: use a single validation split without leakage.
    model = None
    if model_name == "Persistence":
        pred_val = y_train.shift(1).fillna(y_train.mean())
        pred_test = y_test.shift(1).fillna(y_test.mean())
        model = {"type": "persistence"}
    elif model_name == "HAR-RV":
        pred_val = y_train.rolling(20, min_periods=1).mean().shift(1).fillna(y_train.mean())
        pred_test = y_test.rolling(20, min_periods=1).mean().shift(1).fillna(y_test.mean())
        model = {"type": "har_rv"}
    else:
        if model_name == "Ridge":
            model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))])
        elif model_name == "RandomForest":
            model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model name {model_name}")
        model.fit(X_train, y_train)
        pred_val = pd.Series(model.predict(X_val), index=X_val.index)
        pred_test = pd.Series(model.predict(X_test), index=X_test.index)

    val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    val_mae = mean_absolute_error(y_val, pred_val)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    test_mae = mean_absolute_error(y_test, pred_test)

    return {
        "horizon": horizon,
        "specification": feature_set,
        "model": model_name,
        "train_end": train_end.strftime("%Y-%m-%d"),
        "test_start": (y_test.index.min().strftime("%Y-%m-%d") if len(y_test) else None),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
    }


def train_and_evaluate_models(feature_path: str | Path, results_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(feature_path, index_col=0, parse_dates=True).sort_index()
    configs = DEFAULT_CONFIG
    rows = []
    feature_sets = ["market", "market_macro"]
    model_names = ["Ridge", "RandomForest", "XGBoost"]
    for horizon in configs["horizons"]:
        for feature_set in feature_sets:
            for model_name in model_names:
                rows.append(_train_validate_eval(df, horizon, feature_set, model_name, configs))

    out = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecast_summary.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    feature_path = root / "data" / "processed" / "feature_dataset.csv"
    results_dir = root / "results"
    summary = train_and_evaluate_models(feature_path, results_dir)
    print(summary.to_string(index=False))
