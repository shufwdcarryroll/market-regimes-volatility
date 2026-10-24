from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIGURE_STYLE = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "sans-serif",
}


def save_figure(fig, path: str | Path, dpi: int = 300, tight: bool = True) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def regime_colors(n: int = 6):
    palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2",
        "#EECA3B", "#BCA0DC", "#FF9DA6", "#9D755D", "#BAB0AC", "#79706E",
    ]
    return palette[:n]


def plot_hmm_market_regimes(df: pd.DataFrame, regimes: pd.Series, out_dir: str | Path) -> None:
    d = df.copy().sort_index()
    d_plot = d.loc[regimes.index]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(d_plot.index, d_plot["^GSPC"], color="#2F4F4F", linewidth=1.5, label="S&P 500")
    ax.set_ylabel("Index level")
    ax.set_title("Figure B1: HMM regime assignments across S&P 500 history")
    ax2 = axes[1]
    colors = regime_colors(int(regimes.max()) if len(regimes) else 1)
    for state in sorted(regimes.unique()):
        mask = regimes == state
        ax2.scatter(d_plot.index[mask], np.full(mask.sum(), state), c=colors[int(state) - 1], s=20, alpha=0.8)
    ax2.set_yticks(sorted(regimes.unique()))
    ax2.set_yticklabels([f"State {s}" for s in sorted(regimes.unique())])
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")
    save_figure(fig, Path(out_dir) / "fig_b01_hmm_market_regimes.png")
    fig.savefig(Path(out_dir) / "fig_b01_hmm_market_regimes.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b01_hmm_market_regimes.svg", bbox_inches="tight")


def plot_hmm_transition_matrix(tm: pd.DataFrame, out_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tm.columns)))
    ax.set_yticks(range(len(tm.index)))
    ax.set_xticklabels(tm.columns)
    ax.set_yticklabels(tm.index)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            ax.text(j, i, f"{tm.iloc[i, j]:.2f}", ha="center", va="center", color="black" if tm.iloc[i, j] < 0.7 else "white")
    ax.set_title("Figure B2: HMM transition matrix")
    fig.colorbar(im, ax=ax, label="Probability")
    save_figure(fig, Path(out_dir) / "fig_b02_hmm_transition_matrix.png")
    fig.savefig(Path(out_dir) / "fig_b02_hmm_transition_matrix.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b02_hmm_transition_matrix.svg", bbox_inches="tight")


def plot_kmeans_cluster_scatter(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, out_dir: str | Path) -> None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    c2 = pca.transform(centers)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in np.unique(labels):
        mask = labels == cluster
        ax.scatter(X2[mask, 0], X2[mask, 1], s=18, alpha=0.65, label=f"Cluster {cluster + 1}")
    ax.scatter(c2[:, 0], c2[:, 1], c="black", s=80, marker="X", label="Centroids")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Figure B3: K-means clusters in PCA space")
    ax.legend(frameon=False)
    save_figure(fig, Path(out_dir) / "fig_b03_kmeans_pca_clusters.png")
    fig.savefig(Path(out_dir) / "fig_b03_kmeans_pca_clusters.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b03_kmeans_pca_clusters.svg", bbox_inches="tight")


def plot_kmeans_timeline(df: pd.DataFrame, labels: np.ndarray, out_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.scatter(df.index, labels + 1, c=labels, cmap="viridis", s=10)
    ax.set_yticks(np.unique(labels) + 1)
    ax.set_yticklabels([f"Cluster {k}" for k in np.unique(labels) + 1])
    ax.set_ylabel("Cluster")
    ax.set_title("Figure B4: K-means cluster timeline")
    save_figure(fig, Path(out_dir) / "fig_b04_kmeans_cluster_timeline.png")
    fig.savefig(Path(out_dir) / "fig_b04_kmeans_cluster_timeline.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b04_kmeans_cluster_timeline.svg", bbox_inches="tight")


def plot_state_profiles(profile_df: pd.DataFrame, out_dir: str | Path) -> None:
    profile = profile_df.copy()
    if "state" in profile.columns:
        profile = profile.set_index("state")
    elif profile.index.name == "state":
        profile = profile.copy()
    else:
        profile = profile.set_index(profile.columns[0])
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(profile.values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(profile.columns)))
    ax.set_xticklabels(profile.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(profile.index)))
    ax.set_yticklabels([f"State {s}" for s in profile.index])
    ax.set_title("Figure B5: HMM state profiles")
    fig.colorbar(im, ax=ax, label="Standardized value")
    save_figure(fig, Path(out_dir) / "fig_b05_hmm_state_profiles.png")
    fig.savefig(Path(out_dir) / "fig_b05_hmm_state_profiles.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b05_hmm_state_profiles.svg", bbox_inches="tight")


def plot_model_selection(df: pd.DataFrame, out_dir: str | Path, prefix: str = "fig_b07") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    x = df["n_states"].astype(int)
    axes[0].plot(x, df["log_likelihood"], marker="o")
    axes[0].set_title("Model selection: log-likelihood")
    axes[1].plot(x, df["aic"], marker="o")
    axes[1].set_title("AIC")
    axes[2].plot(x, df["bic"], marker="o")
    axes[2].set_title("BIC")
    for ax in axes:
        ax.set_xlabel("States")
        ax.set_xticks(x)
    save_figure(fig, Path(out_dir) / f"{prefix}_hmm_model_selection.png")
    fig.savefig(Path(out_dir) / f"{prefix}_hmm_model_selection.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / f"{prefix}_hmm_model_selection.svg", bbox_inches="tight")


def plot_kmeans_selection(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    x = df["n_clusters"].astype(int)
    axes[0].plot(x, df["silhouette"], marker="o")
    axes[0].set_title("Silhouette score")
    axes[1].plot(x, df["db_index"], marker="o")
    axes[1].set_title("Davies-Bouldin index")
    axes[2].plot(x, df["switching_rate"], marker="o")
    axes[2].set_title("Switching rate")
    for ax in axes:
        ax.set_xlabel("Clusters")
        ax.set_xticks(x)
    save_figure(fig, Path(out_dir) / "fig_b08_kmeans_model_selection.png")
    fig.savefig(Path(out_dir) / "fig_b08_kmeans_model_selection.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b08_kmeans_model_selection.svg", bbox_inches="tight")


def plot_probability_series(prob_df: pd.DataFrame, out_dir: str | Path) -> None:
    cols = [c for c in prob_df.columns if c.endswith("_prob")]
    fig, ax = plt.subplots(figsize=(14, 5))
    for c in cols:
        ax.plot(prob_df["Date"], prob_df[c], linewidth=1.0, alpha=0.7)
    ax.set_title("Figure B13: HMM filtered regime probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    save_figure(fig, Path(out_dir) / "fig_b13_hmm_probability_time_series.png")
    fig.savefig(Path(out_dir) / "fig_b13_hmm_probability_time_series.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b13_hmm_probability_time_series.svg", bbox_inches="tight")


def plot_stress_periods(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["^GSPC"], color="#2b3d55")
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30"), color="#d0d4d8", alpha=0.6)
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-31"), color="#c7d1d8", alpha=0.6)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="#e7e2d5", alpha=0.6)
    ax.set_title("Figure B12: Historical stress periods")
    ax.set_ylabel("S&P 500")
    save_figure(fig, Path(out_dir) / "fig_b12_stress_periods.png")
    fig.savefig(Path(out_dir) / "fig_b12_stress_periods.pdf", bbox_inches="tight")
    fig.savefig(Path(out_dir) / "fig_b12_stress_periods.svg", bbox_inches="tight")


def plot_forecast_rmse_dumbbell_heatmap(summary_path: str | Path, out_dir: str | Path, semantic_name: str = "forecast_rmse_comparison_v2") -> tuple[Path, Path]:
    """Create an alternative market-vs-macro RMSE comparison figure for Section 5.2.1."""
    df = pd.read_csv(summary_path)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.labelcolor": "#222222",
        "savefig.dpi": 300,
    })

    models = ["Ridge", "RandomForest", "XGBoost"]
    model_lbl = {"Ridge": "Ridge", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
    colors = {"Ridge": "#1f4e79", "RandomForest": "#2a9d8f", "XGBoost": "#c8553d"}
    horizons = [5, 10, 20]

    piv = df.pivot_table(index=["horizon", "model"], columns="specification", values="test_rmse").reset_index()
    piv["pct"] = (piv["market_macro"] / piv["market"] - 1) * 100

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1])

    y = 0
    yticks, ylabels, band_centers = [], [], []
    for h in horizons:
        start = y
        for m in models:
            r = piv[(piv.horizon == h) & (piv.model == m)].iloc[0]
            c = colors[m]
            ax.plot([r.market, r.market_macro], [y, y], color=c, lw=2, alpha=0.35, solid_capstyle="round", zorder=1)
            ax.scatter(r.market, y, s=60, color=c, zorder=3, edgecolor="white", lw=0.8)
            ax.scatter(r.market_macro, y, s=60, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
            yticks.append(y)
            ylabels.append(model_lbl[m])
            y += 1
        band_centers.append((start + y - 1) / 2)
        best = piv[piv.horizon == h].sort_values("market").iloc[0]
        by = start + models.index(best.model)
        ax.scatter(best.market, by - 0.38, marker="v", s=28, color="#222222", zorder=4)
        y += 0.9

    for i, h in enumerate(horizons):
        lo = band_centers[i] - 1.45
        hi = band_centers[i] + 1.45
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#f2f2f2", zorder=0, lw=0)
        ax.text(0.99, band_centers[i] - 1.15, f"{h}-day horizon", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5, fontweight="bold", color="#333333")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Out-of-sample RMSE (annualized realized volatility)")
    ax.xaxis.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("A.  Test RMSE: market-only vs. market + macro", loc="left", fontsize=11, fontweight="bold", pad=10)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white", markersize=8, label="Market only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.6, markersize=8, label="Market + macro"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#222222", markeredgecolor="#222222", markersize=6, label="Lowest market-only RMSE"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)

    mat = np.array([[piv[(piv.horizon == h) & (piv.model == m)].pct.iloc[0] for h in horizons] for m in models])
    vmax = np.ceil(mat.max() / 5) * 5
    im = axh.imshow(mat, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(models)):
        for j in range(len(horizons)):
            v = mat[i, j]
            axh.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if v > vmax * 0.55 else "#222222")
    axh.set_xticks(range(len(horizons)))
    axh.set_xticklabels([f"{h}-day" for h in horizons])
    axh.set_yticks(range(len(models)))
    axh.set_yticklabels([model_lbl[m] for m in models])
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_xticks(np.arange(-0.5, len(horizons)), minor=True)
    axh.set_yticks(np.arange(-0.5, len(models)), minor=True)
    axh.grid(which="minor", color="white", lw=2.5)
    axh.tick_params(which="minor", length=0)
    axh.set_title("B.  Change in test RMSE from adding macro", loc="left", fontsize=11, fontweight="bold", pad=10)
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    cb.set_label("% increase in RMSE vs. market-only", fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Macro features do not improve S&P 500 realized-volatility forecasts", x=0.075, ha="left", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.075, 0.955,
             "Training through Dec 2018, validation 2019–2021, test Jan 2022 onward. Adding macro raises test RMSE for every model and horizon.",
             ha="left", fontsize=9.5, color="#444444")
    fig.text(0.075, -0.10,
             "Note: RMSE on forward realized volatility over 5, 10 and 20 trading days. Macro set adds Treasury yields, 10Y–2Y spread, WTI oil, HY credit spread and fed funds.",
             ha="left", fontsize=8, color="#666666")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{semantic_name}.png"
    pdf_path = out_dir / f"{semantic_name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path
