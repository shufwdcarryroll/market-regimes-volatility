from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _safe_log(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.log(x.where(x > 0))


def _rolling_window(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window)


def build_feature_dataset(processed_path: str | Path) -> pd.DataFrame:
    path = Path(processed_path)
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    out = df.copy()
    out["spx_log_return"] = np.log(out["^GSPC"]).diff()

    for h in [5, 10, 20]:
        future = out["spx_log_return"].shift(-1).rolling(window=h, min_periods=h).apply(
            lambda x: np.sqrt(np.mean(x ** 2)) * np.sqrt(TRADING_DAYS), raw=True
        )
        out[f"rv_{h}"] = future
        out[f"target_rv_{h}"] = future.shift(1)

    out["cumret_5"] = np.exp(np.log(out["^GSPC"]).diff().rolling(5, min_periods=5).sum()) - 1
    out["cumret_10"] = np.exp(np.log(out["^GSPC"]).diff().rolling(10, min_periods=10).sum()) - 1
    out["cumret_20"] = np.exp(np.log(out["^GSPC"]).diff().rolling(20, min_periods=20).sum()) - 1

    for h in [5, 10, 20]:
        roll = out["spx_log_return"].rolling(h, min_periods=h)
        out[f"skew_{h}"] = roll.skew()
        out[f"kurt_{h}"] = roll.kurt()

    out["drawdown_20"] = (out["^GSPC"] / out["^GSPC"].rolling(20, min_periods=1).max()) - 1
    out["vol_of_vol_20"] = out["spx_log_return"].rolling(20, min_periods=20).std().shift(1)

    out["vix_level"] = out["^VIX"]
    out["vix_daily_change"] = out["^VIX"].diff()
    out["vix_pct_change"] = out["^VIX"].pct_change()
    out["vix_ma_5"] = out["^VIX"].rolling(5, min_periods=5).mean()
    out["vix_ma_20"] = out["^VIX"].rolling(20, min_periods=20).mean()
    out["vix_dev_20"] = out["^VIX"] - out["^VIX"].rolling(20, min_periods=20).mean()
    out["vix_minus_rv20"] = out["^VIX"] - out["rv_20"]

    out["DGS2"] = df["DGS2"]
    out["DGS10"] = df["DGS10"]
    out["dgs2_change_bp"] = out["DGS2"].diff() * 100
    out["dgs10_change_bp"] = out["DGS10"].diff() * 100
    out["yield_change_5"] = out["DGS10"].diff(5) * 100
    out["yield_change_20"] = out["DGS10"].diff(20) * 100
    out["yc_spread"] = out["DGS10"] - out["DGS2"]
    out["yc_spread_change"] = out["yc_spread"].diff()
    out["yield_curve_inversion"] = (out["DGS10"] < out["DGS2"]).astype(float)
    out["yield_vol_20"] = out["yc_spread"].rolling(20, min_periods=20).std().shift(1)

    out["DCOILWTICO"] = df["DCOILWTICO"]
    out["oil_log_return"] = _safe_log(out["DCOILWTICO"]).diff()
    out["oil_ret_5"] = out["oil_log_return"].rolling(5, min_periods=5).sum()
    out["oil_ret_20"] = out["oil_log_return"].rolling(20, min_periods=20).sum()
    out["oil_vol_20"] = out["oil_log_return"].rolling(20, min_periods=20).std().shift(1)
    out["oil_shock"] = (out["oil_log_return"].abs() > out["oil_log_return"].rolling(20, min_periods=20).std().shift(1)).astype(float)

    out["BAMLH0A0HYM2"] = df["BAMLH0A0HYM2"]
    out["credit_spread_change"] = out["BAMLH0A0HYM2"].diff()
    out["DFF"] = df["DFF"]
    out["fed_change_bp"] = out["DFF"].diff() * 100
    out["credit_vol_20"] = out["BAMLH0A0HYM2"].rolling(20, min_periods=20).std().shift(1)

    out = out.dropna(subset=["spx_log_return"]).sort_index()
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "processed_dataset.csv"
    out = root / "data" / "processed" / "feature_dataset.csv"
    features = build_feature_dataset(processed)
    features.to_csv(out)
    print(f"Saved feature dataset with shape {features.shape} to {out}")


if __name__ == "__main__":
    main()
