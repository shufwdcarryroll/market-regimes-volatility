from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULTS_PATH = ROOT / "results" / "part_b"
FIGURES_PATH = ROOT / "figures" / "part_b" / "final"

FIGURE_CONFIG = {
    "figure_label": "Figure",
    "figure_number": "research",
    "figure_title": "K-means regime timeline and HMM comparison",
    "panel_a_title": "20-day realized volatility",
    "panel_b_title": "HMM regime timeline",
    "panel_c_title": "K-means cluster timeline",
    "output_stem": "fig_kmeans_hmm_regime_timeline",
    "output_dir": FIGURES_PATH,
    "dpi": 300,
    "figure_size": (14.5, 8.0),
    "font_sizes": {"title": 13, "panel": 10.0, "tick": 8.5, "summary": 8.5},
    "colors": {
        "hmm": {0: "#4C78A8", 1: "#4E9D99", 2: "#5AAE61", 3: "#B8B05A", 4: "#E59A3E", 5: "#C85C5C"},
        "kmeans": {0: "#4C78A8", 1: "#F58518"},
    },
    "summary_table": {
        "metrics_csv": "kmeans_hmm_switching_metrics.csv",
        "annual_csv": "kmeans_hmm_annual_switching.csv",
        "aligned_csv": "kmeans_hmm_regime_timeline.csv",
        "hmm_pair_breakdown_csv": "hmm_state_1_4_transition_breakdown.csv",
    },
    "subplots": {"hspace": 0.06, "left": 0.08, "right": 0.72, "top": 0.90, "bottom": 0.22},
    "year_locator": 5,
    "summary_position": [0.46, 0.04, 0.28, 0.15],
}

FEATURES = [
    "spx_log_return",
    "rv_20",
    "cumret_20",
    "vix_level",
    "vix_daily_change",
    "vix_dev_20",
    "DGS2",
    "DGS10",
    "yc_spread",
    "DCOILWTICO",
    "BAMLH0A0HYM2",
    "DFF",
]


def form_figure_label(config: dict) -> str:
    return f"{config['figure_label']} {config['figure_number']}" if config.get("figure_number") else config["figure_label"]


def _fit_kmeans_labels(df: pd.DataFrame) -> pd.Series:
    X = df[FEATURES].dropna().copy()
    Z = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=2, random_state=42, n_init=25)
    labels = model.fit_predict(Z)
    return pd.Series(labels, index=X.index, name="kmeans_cluster")


def _compute_switching_stats(series: pd.Series) -> dict:
    s = series.astype(int)
    n_obs = len(s)
    n_switches = int((s != s.shift(1)).fillna(False).sum())
    switching_rate = float(n_switches / max(n_obs - 1, 1))
    run_lengths = []
    current = 1
    for i in range(1, len(s)):
        if s.iloc[i] == s.iloc[i - 1]:
            current += 1
        else:
            run_lengths.append(current)
            current = 1
    run_lengths.append(current)
    run_lengths = np.asarray(run_lengths, dtype=float)
    mean_dur = float(run_lengths.mean())
    median_dur = float(np.median(run_lengths))
    start_date = s.index[0]
    end_date = s.index[-1]
    return {
        "n_observations": n_obs,
        "n_switches": n_switches,
        "switching_rate": switching_rate,
        "mean_run_duration": mean_dur,
        "median_run_duration": median_dur,
        "date_start": start_date,
        "date_end": end_date,
    }


def _annual_switching_stats(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"label": series.astype(int), "year": series.index.year})
    rows = []
    for year, group in df.groupby("year"):
        s = group["label"].reset_index(drop=True)
        n_obs = len(s)
        n_switches = int((s != s.shift(1)).fillna(False).sum()) if n_obs > 1 else 0
        switching_rate = float(n_switches / max(n_obs - 1, 1)) if n_obs > 1 else 0.0
        run_lengths = []
        current = 1
        for i in range(1, len(s)):
            if s.iloc[i] == s.iloc[i - 1]:
                current += 1
            else:
                run_lengths.append(current)
                current = 1
        run_lengths.append(current)
        if not run_lengths:
            mean_dur = median_dur = 0.0
        else:
            arr = np.asarray(run_lengths, dtype=float)
            mean_dur = float(arr.mean())
            median_dur = float(np.median(arr))
        rows.append({
            "year": int(year),
            "n_observations": int(n_obs),
            "n_switches": int(n_switches),
            "switching_rate": switching_rate,
            "mean_run_duration": mean_dur,
            "median_run_duration": median_dur,
        })
    return pd.DataFrame(rows)


def _draw_regime_strip(ax, dates: pd.DatetimeIndex, labels: pd.Series, color_map: dict[int, str], label_name: str) -> None:
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel(label_name)
    ax.set_axisbelow(False)
    prev_label = labels.iloc[0]
    prev_date = dates[0]
    for idx in range(1, len(labels)):
        current_label = labels.iloc[idx]
        current_date = dates[idx]
        if current_label == prev_label:
            continue
        ax.hlines(0, prev_date, current_date, color=color_map[int(prev_label)], linewidth=12, alpha=0.95)
        prev_date = current_date
        prev_label = current_label
    ax.hlines(0, prev_date, dates[-1], color=color_map[int(prev_label)], linewidth=12, alpha=0.95)
    ax.set_yticks([])
    # Keep the strip aligned to the shared date axis; the bottom panel will handle visible calendar tick labels.
    for spine in ax.spines.values():
        spine.set_visible(False)


def _state_transition_breakdown(series: pd.Series) -> pd.DataFrame:
    pairs = [(int(series.iloc[i]), int(series.iloc[i + 1])) for i in range(len(series) - 1)]
    from collections import Counter
    counts = Counter(pairs)
    rows = []
    for (src, dst), count in sorted(counts.items()):
        rows.append({"from_state": src, "to_state": dst, "count": count, "share_of_all_transitions": count / len(pairs)})
    return pd.DataFrame(rows)


def _plot_regime_comparison() -> None:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    hmm = pd.read_csv(RESULTS_PATH / "hmm_6state_assignments.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    hmm = hmm.rename(columns={"state": "hmm_state"})
    kmeans = _fit_kmeans_labels(df)
    common_index = df.index.intersection(hmm.index).intersection(kmeans.index)
    df = df.loc[common_index]
    hmm = hmm.loc[common_index]
    kmeans = kmeans.loc[common_index]
    aligned = pd.DataFrame({
        "Date": common_index,
        "hmm_state": hmm["hmm_state"].astype(int),
        "kmeans_cluster": kmeans.astype(int),
    }).set_index("Date")
    aligned.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["aligned_csv"], index=True)

    hmm_stats = _compute_switching_stats(aligned["hmm_state"])
    kmeans_stats = _compute_switching_stats(aligned["kmeans_cluster"])
    stats_df = pd.DataFrame([
        {"model": "HMM", **hmm_stats},
        {"model": "K-means", **kmeans_stats},
    ])
    stats_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["metrics_csv"], index=False)

    annual_stats = pd.concat([
        _annual_switching_stats(aligned["hmm_state"]).assign(model="HMM"),
        _annual_switching_stats(aligned["kmeans_cluster"]).assign(model="K-means"),
    ], ignore_index=True)
    annual_stats.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["annual_csv"], index=False)

    hmm_breakdown = _state_transition_breakdown(aligned["hmm_state"])
    hmm_breakdown.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["hmm_pair_breakdown_csv"], index=False)

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_CONFIG["figure_size"], gridspec_kw={"height_ratios": [1.5, 1.0, 1.0]}, sharex=True)
    ax_market, ax_hmm, ax_kmeans = axes
    ax_summary = fig.add_axes(FIGURE_CONFIG["summary_position"])

    market = df["rv_20"].copy()
    ax_market.plot(market.index, market.values * 100, color="#2F4F4F", linewidth=1.6)
    ax_market.set_ylabel("RV (%)")
    ax_market.set_title(FIGURE_CONFIG["panel_a_title"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"], pad=6)
    ax_market.grid(False)
    ax_market.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=FIGURE_CONFIG["year_locator"]))
    ax_market.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    for spine in ax_market.spines.values():
        spine.set_visible(False)

    _draw_regime_strip(ax_hmm, aligned.index, aligned["hmm_state"], FIGURE_CONFIG["colors"]["hmm"], "HMM")
    ax_hmm.set_title(FIGURE_CONFIG["panel_b_title"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"], pad=6)
    ax_hmm.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=FIGURE_CONFIG["year_locator"]))
    ax_hmm.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax_hmm.tick_params(axis='x', labelbottom=True, labelsize=8)

    _draw_regime_strip(ax_kmeans, aligned.index, aligned["kmeans_cluster"], FIGURE_CONFIG["colors"]["kmeans"], "K-means")
    ax_kmeans.set_title(FIGURE_CONFIG["panel_c_title"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"], pad=6)
    ax_kmeans.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=FIGURE_CONFIG["year_locator"]))
    ax_kmeans.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax_kmeans.tick_params(axis='x', labelbottom=True, labelsize=8)

    ax_kmeans.set_xlabel("Date")
    ax_market.set_xlim(aligned.index.min(), aligned.index.max())

    kmeans_legend = [
        plt.Line2D([0], [0], linestyle='None', marker='s', markersize=8, markerfacecolor=FIGURE_CONFIG["colors"]["kmeans"][k], markeredgecolor='none', label=f"Cluster {k}")
        for k in [0, 1]
    ]
    fig.legend(handles=kmeans_legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.38, 0.05), fontsize=9)

    summary_rows = [
        ["Regime changes", f"{hmm_stats['n_switches']}", f"{kmeans_stats['n_switches']}"],
        ["Switching rate", f"{hmm_stats['switching_rate']:.3f}", f"{kmeans_stats['switching_rate']:.3f}"],
        ["Mean duration", f"{hmm_stats['mean_run_duration']:.1f} d", f"{kmeans_stats['mean_run_duration']:.1f} d"],
        ["Median duration", f"{hmm_stats['median_run_duration']:.1f} d", f"{kmeans_stats['median_run_duration']:.1f} d"],
    ]
    ax_summary.axis("off")
    table = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Metric", "HMM", "K-means"],
        cellLoc="center",
        loc="center",
        colWidths=[0.42, 0.28, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FIGURE_CONFIG["font_sizes"]["summary"])
    table.scale(1.0, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DADADA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")

    fig.suptitle(FIGURE_CONFIG["figure_title"], fontsize=FIGURE_CONFIG["font_sizes"]["title"], y=0.98)
    fig.subplots_adjust(**FIGURE_CONFIG["subplots"])
    FIGURE_CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_CONFIG["output_dir"] / f"{FIGURE_CONFIG['output_stem']}.png", dpi=FIGURE_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(FIGURE_CONFIG["output_dir"] / f"{FIGURE_CONFIG['output_stem']}.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved:", FIGURE_CONFIG["output_dir"] / f"{FIGURE_CONFIG['output_stem']}.png")
    print("Saved:", FIGURE_CONFIG["output_dir"] / f"{FIGURE_CONFIG['output_stem']}.pdf")
    print("HMM stats:", hmm_stats)
    print("K-means stats:", kmeans_stats)
    print("HMM State 1↔4 transition breakdown:")
    mask = hmm_breakdown["from_state"].isin([0, 3]) & hmm_breakdown["to_state"].isin([0, 3])
    print(hmm_breakdown.loc[mask].to_string(index=False))


if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURE_CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    _plot_regime_comparison()
