from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULTS_PATH = ROOT / "results" / "part_b"
FIGNES_PATH = ROOT / "figures" / "part_b" / "final"

FIGURE_CONFIG = {
    "hmm_model_selection": {
        "title": "HMM model selection",
        "output_stem": "fig_hmm_model_selection",
        "figure_size": (8.0, 5.3),
        "dpi": 300,
        "line_colors": {"AIC": "#4C78A8", "BIC": "#F58518"},
        "markers": {"AIC": "o", "BIC": "s"},
        "highlight_state": 4,
    },
    "hmm_macro_sensitivity": {
        "title": "Market-only vs. market+macro HMM",
        "output_stem": "fig_hmm_macro_sensitivity",
        "figure_size": (14.5, 7.2),
        "dpi": 300,
        "panel_titles": {
            "rv": "20-day realized volatility",
            "market_only": "Market-only HMM",
            "macro": "Market + macro HMM",
        },
        "state_palette": {
            0: "#4C78A8",
            1: "#F58518",
            2: "#54A24B",
            3: "#E45756",
            4: "#B279A2",
            5: "#72B7B2",
        },
        "legend_title_market": "Market-only HMM",
        "legend_title_macro": "Market + macro HMM",
    },
}

MARKET_ONLY_FEATURES = [
    "spx_log_return",
    "rv_20",
    "cumret_20",
    "vix_level",
    "vix_daily_change",
    "vix_dev_20",
]
MACRO_FEATURES = MARKET_ONLY_FEATURES + [
    "DGS2",
    "DGS10",
    "yc_spread",
    "DCOILWTICO",
    "BAMLH0A0HYM2",
    "DFF",
]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()


def fit_hmm(df: pd.DataFrame, feature_list: list[str], n_states: int = 6, seed: int = 42) -> tuple[pd.Index, np.ndarray, GaussianHMM, np.ndarray]:
    subset = df.loc[:, feature_list].dropna()
    X = StandardScaler().fit_transform(subset)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=seed,
        init_params="stmc",
    )
    model.fit(X)
    labels = model.predict(X)
    return subset.index, X, model, labels


def plot_hmm_model_selection() -> None:
    model_df = pd.read_csv(RESULTS_PATH / "hmm_model_selection.csv")
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG["hmm_model_selection"]["figure_size"])
    x = model_df["n_states"].astype(int).to_numpy()
    ax.plot(x, model_df["aic"], color=FIGURE_CONFIG["hmm_model_selection"]["line_colors"]["AIC"], marker=FIGURE_CONFIG["hmm_model_selection"]["markers"]["AIC"], linewidth=2.2, markersize=6, label="AIC")
    ax.plot(x, model_df["bic"], color=FIGURE_CONFIG["hmm_model_selection"]["line_colors"]["BIC"], marker=FIGURE_CONFIG["hmm_model_selection"]["markers"]["BIC"], linewidth=2.2, markersize=6, label="BIC")

    low_idx = model_df[["aic", "bic"]].idxmin().min()
    low_state = int(model_df.loc[low_idx, "n_states"])
    low_aic = float(model_df.loc[low_idx, "aic"])
    low_bic = float(model_df.loc[low_idx, "bic"])
    ax.scatter([low_state], [low_aic], color=FIGURE_CONFIG["hmm_model_selection"]["line_colors"]["AIC"], s=60, zorder=5)
    ax.scatter([low_state], [low_bic], color=FIGURE_CONFIG["hmm_model_selection"]["line_colors"]["BIC"], s=60, zorder=5)
    ax.annotate(f"Lowest AIC/BIC\n({low_state} states)", xy=(low_state, low_bic), xytext=(12, 12), textcoords="offset points", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))

    ax.set_xticks(x)
    ax.set_xlabel("Number of states")
    ax.set_ylabel("Information criterion")
    ax.set_title(FIGURE_CONFIG["hmm_model_selection"]["title"])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(False)
    fig.tight_layout()
    out_png = FIGNES_PATH / f"{FIGURE_CONFIG['hmm_model_selection']['output_stem']}.png"
    out_pdf = FIGNES_PATH / f"{FIGURE_CONFIG['hmm_model_selection']['output_stem']}.pdf"
    fig.savefig(out_png, dpi=FIGURE_CONFIG["hmm_model_selection"]["dpi"], bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def draw_state_strip(ax, dates: pd.DatetimeIndex, labels: np.ndarray, color_map: dict[int, str], label: str) -> None: 
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel(label)
    if len(labels) == 0:
        return
    prev_label = int(labels[0])
    prev_time = dates[0]
    for i in range(1, len(labels)):
        cur_label = int(labels[i])
        cur_time = dates[i]
        if cur_label == prev_label:
            continue
        ax.hlines(0, prev_time, cur_time, color=color_map[prev_label], linewidth=12, alpha=0.95)
        prev_time = cur_time
        prev_label = cur_label
    ax.hlines(0, prev_time, dates[-1], color=color_map[prev_label], linewidth=12, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))


def plot_hmm_macro_sensitivity() -> None:
    df = load_dataset()
    market_dates, market_X, market_model, market_labels = fit_hmm(df, MARKET_ONLY_FEATURES, n_states=6, seed=42)
    macro_dates, macro_X, macro_model, macro_labels = fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_CONFIG["hmm_macro_sensitivity"]["figure_size"], sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})
    ax_rv, ax_market, ax_macro = axes

    rv_series = df.loc[market_dates, "rv_20"] * 100
    ax_rv.plot(rv_series.index, rv_series.values, color="#2F4F4F", linewidth=1.5)
    ax_rv.set_ylabel("RV (%)")
    ax_rv.set_title(FIGURE_CONFIG["hmm_macro_sensitivity"]["panel_titles"]["rv"])
    ax_rv.grid(False)
    for side in ["top", "right", "left", "bottom"]:
        ax_rv.spines[side].set_visible(False)

    draw_state_strip(ax_market, pd.DatetimeIndex(market_dates), np.asarray(market_labels), FIGURE_CONFIG["hmm_macro_sensitivity"]["state_palette"], "Market-only")
    ax_market.set_title(FIGURE_CONFIG["hmm_macro_sensitivity"]["panel_titles"]["market_only"])

    draw_state_strip(ax_macro, pd.DatetimeIndex(macro_dates), np.asarray(macro_labels), FIGURE_CONFIG["hmm_macro_sensitivity"]["state_palette"], "Market + macro")
    ax_macro.set_title(FIGURE_CONFIG["hmm_macro_sensitivity"]["panel_titles"]["macro"])
    ax_macro.set_xlabel("Date")

    # separate legends to avoid implying direct state comparability across independently fitted models
    market_handles = [Line2D([0], [0], linestyle="None", marker="s", markersize=8, markerfacecolor=FIGURE_CONFIG["hmm_macro_sensitivity"]["state_palette"][i], markeredgecolor="none") for i in range(6)]
    macro_handles = [Line2D([0], [0], linestyle="None", marker="s", markersize=8, markerfacecolor=FIGURE_CONFIG["hmm_macro_sensitivity"]["state_palette"][i], markeredgecolor="none") for i in range(6)]
    market_legend = fig.legend(market_handles, [f"State {i + 1}" for i in range(6)], title=FIGURE_CONFIG["hmm_macro_sensitivity"]["legend_title_market"], loc="center left", bbox_to_anchor=(0.78, 0.66), frameon=False, handletextpad=0.6, borderaxespad=0.0)
    macro_legend = fig.legend(macro_handles, [f"State {i + 1}" for i in range(6)], title=FIGURE_CONFIG["hmm_macro_sensitivity"]["legend_title_macro"], loc="center left", bbox_to_anchor=(0.78, 0.35), frameon=False, handletextpad=0.6, borderaxespad=0.0)
    for text in market_legend.get_texts():
        text.set_fontsize(8)
    for text in macro_legend.get_texts():
        text.set_fontsize(8)

    fig.suptitle(FIGURE_CONFIG["hmm_macro_sensitivity"]["title"], y=0.98)
    fig.subplots_adjust(left=0.08, right=0.75, bottom=0.1, top=0.92, hspace=0.18)
    out_png = FIGNES_PATH / f"{FIGURE_CONFIG['hmm_macro_sensitivity']['output_stem']}.png"
    out_pdf = FIGNES_PATH / f"{FIGURE_CONFIG['hmm_macro_sensitivity']['output_stem']}.pdf"
    fig.savefig(out_png, dpi=FIGURE_CONFIG["hmm_macro_sensitivity"]["dpi"], bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGNES_PATH.mkdir(parents=True, exist_ok=True)
    plot_hmm_model_selection()
    plot_hmm_macro_sensitivity()
    print(f"Saved model selection to {FIGNES_PATH / 'fig_hmm_model_selection.png'} and {FIGNES_PATH / 'fig_hmm_model_selection.pdf'}")
    print(f"Saved macro sensitivity to {FIGNES_PATH / 'fig_hmm_macro_sensitivity.png'} and {FIGNES_PATH / 'fig_hmm_macro_sensitivity.pdf'}")
