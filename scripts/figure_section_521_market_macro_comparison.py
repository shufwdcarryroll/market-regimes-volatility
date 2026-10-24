from __future__ import annotations

from pathlib import Path

from src.plotting import plot_forecast_rmse_dumbbell_heatmap

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "results" / "part_a" / "forecast_summary.csv"
    output_dir = ROOT / "figures" / "part_a" / "final"
    plot_forecast_rmse_dumbbell_heatmap(
        summary_path=summary_path,
        out_dir=output_dir,
        semantic_name="forecast_rmse_comparison_v2",
    )


if __name__ == "__main__":
    main()
