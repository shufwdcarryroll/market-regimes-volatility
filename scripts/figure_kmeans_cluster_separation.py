from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULTS_PATH = ROOT / "results" / "part_b"
FIGURES_PATH = ROOT / "figures" / "part_b"

FIGURE_CONFIG = {
    "figure_label": "Figure",
    "figure_number": "B3",
    "figure_title": "K-means cluster separation",
    "panel_a_title": "PCA projection of clusters",
    "panel_b_title": "Silhouette analysis",
    "output_stem": "fig_kmeans_cluster_separation",
    "output_dir": Path(__file__).resolve().parents[1] / "figures" / "part_b" / "final",
    "figure_size": (12.5, 5.6),
    "dpi": 300,
    "colors": {0: "#4C78A8", 1: "#F58518"},
    "edgecolor": "#1f2933",
    "centroid_marker": "X",
    "centroid_size": 110,
    "point_size": 18,
    "point_alpha": 0.65,
    "legend_loc": "upper left",
    "font_scale": 1.0,
    "pca_csv": "kmeans_pca_projection.csv",
    "sil_csv": "kmeans_cluster_silhouette.csv",
    "profile_csv": "kmeans_cluster_profiles.csv",
    "metrics_csv": "kmeans_evaluation_metrics.csv",
    "interpretation_note": (
        "The two-cluster K-means partition separates most observations into a larger, lower-volatility group "
        "and a smaller, higher-volatility group. The silhouette score indicates moderate separation with limited overlap, "
        "suggesting that the clusters are economically interpretable but not perfectly non-overlapping."
    ),
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


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def form_figure_label(config: dict) -> str:
    return f"{config['figure_label']} {config['figure_number']}"


def _fit_kmeans() -> tuple[pd.DataFrame, np.ndarray, KMeans, np.ndarray, PCA, np.ndarray]:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    X = df[FEATURES].dropna().copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    model = KMeans(n_clusters=2, random_state=42, n_init=25)
    labels = model.fit_predict(Z)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Z)
    centroids_pca = pca.transform(model.cluster_centers_)
    return X, labels, model, Z, pca, X_pca, centroids_pca


def compute_cluster_metrics(Z: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, np.ndarray, pd.Series]:
    sil_all = silhouette_samples(Z, labels)
    mean_sil = float(silhouette_score(Z, labels))
    db_index = float(davies_bouldin_score(Z, labels))
    neg_share = float(np.mean(sil_all < 0))
    counts = pd.Series(labels).value_counts().sort_index()
    return mean_sil, db_index, neg_share, sil_all, counts


def build_cluster_profile_table(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = X.copy()
    df["cluster"] = labels
    profile = (
        df.groupby("cluster")
          .agg(
              observations=("cluster", "size"),
              observation_share=("cluster", lambda s: s.size / len(s)),
              avg_vix=("vix_level", "mean"),
              avg_rv_20=("rv_20", "mean"),
              avg_spx_return=("spx_log_return", "mean"),
              avg_return_skew=("spx_log_return", "mean"),
              avg_return_kurtosis=("spx_log_return", "mean"),
          )
          .reset_index()
          .rename(columns={"cluster": "cluster_id"})
    )
    profile["cluster_label"] = profile["cluster_id"].map(lambda c: f"Cluster {c + 1}")
    profile = profile[["cluster_label", "cluster_id", "observations", "observation_share", "avg_vix", "avg_rv_20", "avg_spx_return"]]
    return profile


def build_figure() -> None:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    X = df[FEATURES].dropna().copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    km = KMeans(n_clusters=2, random_state=42, n_init=25)
    labels = km.fit_predict(Z)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Z)
    centroids_pca = pca.transform(km.cluster_centers_)

    sil_all = silhouette_samples(Z, labels)
    sil_mean = silhouette_score(Z, labels)
    db_index = davies_bouldin_score(Z, labels)
    counts = pd.Series(labels).value_counts().sort_index()
    neg_share = np.mean(sil_all < 0)

    profile = X.copy()
    profile["cluster"] = labels
    profile_summary = (
        profile.groupby("cluster")
        .agg(
            observations=("cluster", "size"),
            observation_share=("cluster", lambda s: s.size / len(s)),
            avg_vix=("vix_level", "mean"),
            avg_rv_20=("rv_20", "mean"),
            avg_spx_return=("spx_log_return", "mean"),
            avg_dgs2=("DGS2", "mean"),
            avg_dgs10=("DGS10", "mean"),
            avg_yc_spread=("yc_spread", "mean"),
            avg_oil=("DCOILWTICO", "mean"),
            avg_credit=("BAMLH0A0HYM2", "mean"),
            avg_ffr=("DFF", "mean"),
        )
        .reset_index()
        .rename(columns={"cluster": "cluster_id"})
    )
    profile_summary["cluster_label"] = profile_summary["cluster_id"].map(lambda c: f"Cluster {c + 1}")
    profile_summary = profile_summary[
        ["cluster_label", "cluster_id", "observations", "observation_share", "avg_vix", "avg_rv_20", "avg_spx_return", "avg_dgs2", "avg_dgs10", "avg_yc_spread", "avg_oil", "avg_credit", "avg_ffr"]
    ]

    pca_df = pd.DataFrame({
        "Date": X.index,
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "cluster_id": labels,
        "cluster_label": [f"Cluster {c + 1}" for c in labels],
    })
    pca_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["pca_csv"], index=False)

    sil_df = pd.DataFrame({
        "cluster_id": labels,
        "silhouette": sil_all,
    })
    sil_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["sil_csv"], index=False)

    profile_summary.to_csv(RESULTS_PATH / FIGURE_CONFIG["profile_csv"], index=False)

    metrics_df = pd.DataFrame([{
        "n_clusters": 2,
        "n_observations": int(len(X)),
        "overall_silhouette": sil_mean,
        "davies_bouldin": db_index,
        "negative_silhouette_share": neg_share,
        "cluster_1_n": int(counts.get(0, 0)),
        "cluster_2_n": int(counts.get(1, 0)),
        "cluster_1_share": float(counts.get(0, 0) / len(X)),
        "cluster_2_share": float(counts.get(1, 0) / len(X)),
    }])
    metrics_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["metrics_csv"], index=False)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_CONFIG["figure_size"], constrained_layout=True)
    ax_pca = axes[0]
    ax_sil = axes[1]

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        color = FIGURE_CONFIG["colors"][cluster_id]
        ax_pca.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=FIGURE_CONFIG["point_size"],
            alpha=FIGURE_CONFIG["point_alpha"],
            c=color,
            edgecolors="none",
            label=f"Cluster {cluster_id + 1} (n={int(mask.sum())})",
        )
        ax_pca.scatter(
            centroids_pca[cluster_id, 0],
            centroids_pca[cluster_id, 1],
            s=FIGURE_CONFIG["centroid_size"],
            marker=FIGURE_CONFIG["centroid_marker"],
            color="#1f2933",
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )

    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax_pca.set_title(FIGURE_CONFIG["panel_a_title"])
    ax_pca.legend(frameon=False, loc=FIGURE_CONFIG["legend_loc"], fontsize=9)
    ax_pca.grid(False)

    sil_by_cluster = {cluster_id: sil_all[labels == cluster_id] for cluster_id in sorted(np.unique(labels))}
    cluster_positions = []
    max_n = max(len(vals) for vals in sil_by_cluster.values())
    y_start = 0
    for cluster_id in sorted(sil_by_cluster):
        vals = np.sort(sil_by_cluster[cluster_id])
        n = len(vals)
        y = np.arange(n)
        ax_sil.barh(y + y_start, vals, height=1.0, color=FIGURE_CONFIG["colors"][cluster_id], alpha=0.9, edgecolor="none")
        cluster_positions.append((cluster_id, y_start + n / 2))
        y_start += max_n + 2

    ax_sil.axvline(0, color="#4a4a4a", linewidth=1.0, linestyle="--", alpha=0.8)
    ax_sil.axvline(sil_mean, color="#1f2933", linewidth=1.2, linestyle="-", alpha=0.9)
    ax_sil.text(sil_mean, y_start - 1.5, f"mean = {sil_mean:.3f}", va="bottom", ha="left", fontsize=9, color="#1f2933")

    yticks = [pos for _, pos in cluster_positions]
    ylabels = [f"Cluster {cluster_id + 1}\n(n={int(counts.get(cluster_id, 0))})" for cluster_id, _ in cluster_positions]
    ax_sil.set_yticks(yticks)
    ax_sil.set_yticklabels(ylabels, fontsize=9)
    ax_sil.set_xlim(-0.55, 1.0)
    ax_sil.set_xlabel("Silhouette coefficient")
    ax_sil.set_title(FIGURE_CONFIG["panel_b_title"])
    ax_sil.grid(False)

    fig.suptitle(FIGURE_CONFIG["figure_title"], y=1.02, fontsize=14, fontweight="bold")
    ax_pca.annotate(
        FIGURE_CONFIG["interpretation_note"],
        xy=(0.5, 0.96),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8.7,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#D5D9E0", lw=0.8),
    )
    save_figure(fig, FIGURE_CONFIG["output_dir"], FIGURE_CONFIG["output_stem"], dpi=FIGURE_CONFIG["dpi"])


if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURE_CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    build_figure()
    print(f"Saved figure to {FIGURE_CONFIG['output_dir'] / (FIGURE_CONFIG['output_stem'] + '.png')}")
    print(f"Saved metrics to {RESULTS_PATH / FIGURE_CONFIG['metrics_csv']}")
    print(f"Saved profile table to {RESULTS_PATH / FIGURE_CONFIG['profile_csv']}")
    print(f"Saved PCA projection to {RESULTS_PATH / FIGURE_CONFIG['pca_csv']}")
    print(f"Saved silhouette diagnostics to {RESULTS_PATH / FIGURE_CONFIG['sil_csv']}")
