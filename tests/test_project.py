from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists():
    processed = ROOT / "data" / "processed" / "processed_dataset.csv"
    assert processed.exists(), "Processed data must be created."


def test_feature_file_exists():
    feature_path = ROOT / "data" / "processed" / "feature_dataset.csv"
    assert feature_path.exists(), "Feature dataset must be created."


def test_feature_columns():
    df = pd.read_csv(ROOT / "data" / "processed" / "feature_dataset.csv", index_col=0, parse_dates=True)
    assert "target_rv_5" in df.columns
    assert "target_rv_10" in df.columns
    assert "target_rv_20" in df.columns
    assert not df.empty
