"""SIADS 696, Section 5.2.2.1: Ridge coefficients and saved-results ablation.

Copy this file to src/part_a_feature_analysis.py and run from the project root:
    python src/part_a_feature_analysis.py

Uses saved forecast_summary.csv; only fits one Ridge model for the coefficient plot.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# All user-editable settings are here.
CONFIG = {
    "data_file": "data/processed/feature_dataset.csv",
    "summary_file": "results/forecast_summary.csv",
    "results_dir": "results/part_a",
    "figures_dir": "figures/part_a",
    "train_end": "2018-12-31",
    "horizon": 5,
    "ridge_alpha": 1.0,
    "top_n": 12,
    "figsize": (9, 5.6),
    "dpi": 300,
    "title": "Ridge feature importance: standardized coefficients",
    "coefficient_stem": "ridge_feature_importance",
    "ablation_filename": "feature_ablation_5d.csv",
    "positive_color": "#4B739B",
    "negative_color": "#BA755C",
}

# Use the EXACT market-only feature list from the existing forecasting.py.
MARKET_FEATURES = [
    "spx_log_return", "rv_5", "rv_10", "rv_20",
    "cumret_5", "cumret_10", "cumret_20",
    "skew_5", "skew_10", "skew_20",
    "kurt_5", "kurt_10", "kurt_20",
    "drawdown_20", "vol_of_vol_20",
    "vix_level", "vix_daily_change", "vix_pct_change",
    "vix_ma_5", "vix_ma_20", "vix_dev_20", "vix_minus_rv20",
]


def find_project_root() -> Path:
    # Supports running a copied script from project_root/src or project_root.
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, Path.cwd()]
    for folder in candidates:
        if (folder / CONFIG["data_file"]).exists():
            return folder
    raise FileNotFoundError(
        f"Cannot find {CONFIG['data_file']}. Put script in src/ of the extracted project "
        "and run it from that project."
    )


def write_ablation(root: Path, out_dir: Path) -> None:
    path = root / CONFIG["summary_file"]
    saved = pd.read_csv(path)
    h = CONFIG["horizon"]
    subset = saved[(saved["horizon"].astype(int) == h) &
                   (saved["model"].str.lower() == "ridge")]
    rows = []
    for specification, label in [
        ("market", "Market only"),
        ("market_macro", "Market + all macro"),
    ]:
        matched = subset[subset["specification"] == specification]
        if len(matched) != 1:
            raise ValueError(f"Expected exactly one {h}-day Ridge '{specification}' row; found {len(matched)}")
        rec = matched.iloc[0]
        rows.append({"Feature specification": label,
                     "RMSE": float(rec["test_rmse"]),
                     "MAE": float(rec["test_mae"])})
    base = rows[0]["RMSE"]
    for row in rows:
        row["Delta RMSE vs baseline"] = row["RMSE"] - base
    table = pd.DataFrame(rows)
    out_path = out_dir / CONFIG["ablation_filename"]
    table.to_csv(out_path, index=False)
    print("\n5-day Ridge ablation (saved results):\n", table.round(5).to_string(index=False))
    print("Saved:", out_path)


def create_coefficient_chart(root: Path, out_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(root / CONFIG["data_file"], index_col=0, parse_dates=True)
    target = f"target_rv_{CONFIG['horizon']}"
    missing = sorted(set(MARKET_FEATURES + [target]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in existing dataset: {missing}")
    training = data.loc[data.index <= pd.Timestamp(CONFIG["train_end"]), MARKET_FEATURES + [target]].dropna()
    if training.empty:
        raise ValueError("No valid training observations.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=CONFIG["ridge_alpha"], random_state=42)),
    ])
    model.fit(training[MARKET_FEATURES], training[target])
    # Coefficients are on the standardized-feature scale because the scaler is fitted in the pipeline.
    importance = pd.DataFrame({
        "feature": MARKET_FEATURES,
        "standardized_coefficient": model.named_steps["reg"].coef_,
    })
    importance["abs_coefficient"] = importance["standardized_coefficient"].abs()
    importance.sort_values("abs_coefficient", ascending=False, inplace=True)
    csv_path = out_dir / "ridge_feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    top = importance.head(CONFIG["top_n"]).iloc[::-1]
    colors = [CONFIG["positive_color"] if c >= 0 else CONFIG["negative_color"]
              for c in top["standardized_coefficient"]]
    fig, ax = plt.subplots(figsize=CONFIG["figsize"], layout="constrained")
    ax.barh(top["feature"], top["standardized_coefficient"], color=colors)
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.set_xlabel("Ridge coefficient (per 1-SD increase in feature)")
    ax.set_title(CONFIG["title"])
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for extension in ["png", "pdf"]:
        destination = figure_dir / f"{CONFIG['coefficient_stem']}.{extension}"
        fig.savefig(destination, dpi=CONFIG["dpi"], bbox_inches="tight")
        print("Saved:", destination)
    plt.close(fig)
    print("Saved:", csv_path)
    print("\nTop five predictors by absolute Ridge coefficient:\n",
          importance[["feature", "standardized_coefficient"]].head(5).to_string(index=False))


def main() -> None:
    root = find_project_root()
    out_dir = root / CONFIG["results_dir"]
    figure_dir = root / CONFIG["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_ablation(root, out_dir)
    create_coefficient_chart(root, out_dir, figure_dir)


if __name__ == "__main__":
    main()
