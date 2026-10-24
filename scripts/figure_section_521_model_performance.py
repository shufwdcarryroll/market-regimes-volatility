from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

FIGURE_CONFIG = {
    "figure_title": "Forecasting performance by horizon",
    "semantic_figure_name": "supervised_model_performance",
    "output_dir": ROOT / "figures" / "part_a" / "final",
    "figsize": (11.0, 4.7),
    "dpi": 300,
    "colors": {
        "Ridge": "#4C78A8",
        "Random Forest": "#F58518",
        "XGBoost": "#54A24B",
    },
    "font_sizes": {
        "title": 14,
        "panel_title": 12,
        "label": 11,
        "tick": 10,
        "legend": 9,
    },
    "legend": {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, -0.02),
        "ncol": 3,
        "frameon": False,
        "handlelength": 1.4,
    },
}

DATA = {
    "5d": {
        "Ridge": {"Test_RMSE": 0.0647, "Test_MAE": 0.0435},
        "Random Forest": {"Test_RMSE": 0.0659, "Test_MAE": 0.0434},
        "XGBoost": {"Test_RMSE": 0.0656, "Test_MAE": 0.0442},
    },
    "10d": {
        "Ridge": {"Test_RMSE": 0.0604, "Test_MAE": 0.0388},
        "Random Forest": {"Test_RMSE": 0.0615, "Test_MAE": 0.0390},
        "XGBoost": {"Test_RMSE": 0.0633, "Test_MAE": 0.0403},
    },
    "20d": {
        "Ridge": {"Test_RMSE": 0.0559, "Test_MAE": 0.0363},
        "Random Forest": {"Test_RMSE": 0.0561, "Test_MAE": 0.0365},
        "XGBoost": {"Test_RMSE": 0.0567, "Test_MAE": 0.0367},
    },
}

MODELS = ["Ridge", "Random Forest", "XGBoost"]
HORIZONS = ["5d", "10d", "20d"]


def build_figure() -> plt.Figure:
    x = np.arange(len(HORIZONS))
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_CONFIG["figsize"], constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4a4a4a")
        ax.spines["bottom"].set_color("#4a4a4a")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=FIGURE_CONFIG["font_sizes"]["tick"], colors="#2f2f2f")

    rmse_ax, mae_ax = axes
    handles = []

    for idx, model in enumerate(MODELS):
        offset = (idx - 1) * width
        rmse_vals = [DATA[horizon][model]["Test_RMSE"] for horizon in HORIZONS]
        mae_vals = [DATA[horizon][model]["Test_MAE"] for horizon in HORIZONS]
        rmse_ax.bar(x + offset, rmse_vals, width=width, label=model, color=FIGURE_CONFIG["colors"][model], edgecolor="white")
        mae_ax.bar(x + offset, mae_vals, width=width, label=model, color=FIGURE_CONFIG["colors"][model], edgecolor="white")
        handles.append(Patch(facecolor=FIGURE_CONFIG["colors"][model], edgecolor="none", label=model))

    rmse_ax.set_title("A. Test RMSE", fontsize=FIGURE_CONFIG["font_sizes"]["panel_title"], pad=8)
    rmse_ax.set_xlabel("Forecast horizon", fontsize=FIGURE_CONFIG["font_sizes"]["label"])
    rmse_ax.set_ylabel("RMSE", fontsize=FIGURE_CONFIG["font_sizes"]["label"])
    rmse_ax.set_xticks(x)
    rmse_ax.set_xticklabels(HORIZONS)
    rmse_ax.margins(x=0.08)
    rmse_ax.set_ylim(0, 0.07)
    rmse_ax.set_yticks(np.arange(0.0, 0.071, 0.01))

    mae_ax.set_title("B. Test MAE", fontsize=FIGURE_CONFIG["font_sizes"]["panel_title"], pad=8)
    mae_ax.set_xlabel("Forecast horizon", fontsize=FIGURE_CONFIG["font_sizes"]["label"])
    mae_ax.set_ylabel("MAE", fontsize=FIGURE_CONFIG["font_sizes"]["label"])
    mae_ax.set_xticks(x)
    mae_ax.set_xticklabels(HORIZONS)
    mae_ax.margins(x=0.08)
    mae_ax.set_ylim(0, 0.05)
    mae_ax.set_yticks(np.arange(0.0, 0.051, 0.01))

    fig.suptitle(FIGURE_CONFIG["figure_title"], fontsize=FIGURE_CONFIG["font_sizes"]["title"], y=0.98)
    fig.legend(
        handles=handles,
        loc=FIGURE_CONFIG["legend"]["loc"],
        bbox_to_anchor=FIGURE_CONFIG["legend"]["bbox_to_anchor"],
        ncol=FIGURE_CONFIG["legend"]["ncol"],
        frameon=FIGURE_CONFIG["legend"]["frameon"],
        fontsize=FIGURE_CONFIG["font_sizes"]["legend"],
        handlelength=FIGURE_CONFIG["legend"]["handlelength"],
        columnspacing=1.4,
    )
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, semantic_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{semantic_name}.png"
    pdf_path = output_dir / f"{semantic_name}.pdf"
    fig.savefig(png_path, dpi=FIGURE_CONFIG["dpi"], facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    return png_path, pdf_path


def main() -> None:
    fig = build_figure()
    png_path, pdf_path = save_figure(fig, FIGURE_CONFIG["output_dir"], FIGURE_CONFIG["semantic_figure_name"])
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
