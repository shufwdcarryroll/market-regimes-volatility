from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_market_and_macro(start: str | None = None, end: str | None = None, raw_dir: str | Path | None = None):
    """Load the repository's existing raw market and macro data.

    The project stores raw CSV exports in data/raw. This helper keeps the original
    notebook contract working even when the historic download step is not rerun.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    market_path = raw_dir / "market_data_raw.csv"
    macro_path = raw_dir / "macro_data_raw.csv"
    if not market_path.exists() or not macro_path.exists():
        raise FileNotFoundError(f"Missing raw data files under {raw_dir}")

    market_df = pd.read_csv(market_path, index_col=0, parse_dates=True).sort_index()
    macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True).sort_index()

    if start is not None:
        market_df = market_df.loc[market_df.index >= pd.Timestamp(start)]
        macro_df = macro_df.loc[macro_df.index >= pd.Timestamp(start)]
    if end is not None:
        market_df = market_df.loc[market_df.index <= pd.Timestamp(end)]
        macro_df = macro_df.loc[macro_df.index <= pd.Timestamp(end)]

    combined = market_df.join(macro_df, how="left").sort_index()
    return market_df, macro_df, combined
