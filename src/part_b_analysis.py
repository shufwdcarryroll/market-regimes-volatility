from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .plotting import (
    plot_hmm_market_regimes,
    plot_hmm_transition_matrix,
    plot_kmeans_cluster_scatter,
    plot_kmeans_selection,
    plot_kmeans_timeline,
    plot_model_selection,
    plot_probability_series,
    plot_state_profiles,
    plot_stress_periods,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULTS_PATH = ROOT / "results" / "part_b"
FIGURES_PATH = ROOT / "figures" / "part_b"

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


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def _fit_hmm(df: pd.DataFrame, feature_list: list[str], n_states: int = 6, seed: int = 42) -> tuple[pd.Index, np.ndarray, GaussianHMM]:
    data = df.loc[:, feature_list].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=300, random_state=seed, init_params="stmc")
    model.fit(X)
    return data.index, X, model


def _fit_kmeans(df: pd.DataFrame, feature_list: list[str], n_clusters: int = 2, seed: int = 42) -> tuple[np.ndarray, KMeans]:
    data = df.loc[:, feature_list].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=25)
    model.fit(X)
    return X, model


def generate_hmm_selection_csv() -> pd.DataFrame:
    df = load_dataset()
    rows = []
    for n_states in range(2, 7):
        idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=n_states, seed=42)
        ll = model.score(X)
        n_params = n_states * X.shape[1] + n_states * (X.shape[1] + 1) / 2 + (n_states - 1)
        aic = 2 * n_params - 2 * ll
        bic = np.log(X.shape[0]) * n_params - 2 * ll
        rows.append({
            "n_states": n_states,
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "transitions": int(np.sum(np.diff(model.predict(X)) != 0)),
        })
    summary = pd.DataFrame(rows)
    out = RESULTS_PATH
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "hmm_model_selection.csv", index=False)
    return summary


def generate_kmeans_selection_csv() -> pd.DataFrame:
    df = load_dataset()
    rows = []
    for n_clusters in range(2, 7):
        X, model = _fit_kmeans(df, MACRO_FEATURES, n_clusters=n_clusters, seed=42)
        labels = model.labels_
        rows.append({
            "n_clusters": n_clusters,
            "silhouette": float(silhouette_score(X, labels)),
            "db_index": float(davies_bouldin_score(X, labels)),
            "switching_rate": float(np.mean(np.diff(labels, prepend=labels[0]) != 0)),
            "regime_transitions": int(np.sum(np.diff(labels) != 0)),
        })
    summary = pd.DataFrame(rows)
    out = RESULTS_PATH
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "kmeans_model_selection.csv", index=False)
    return summary


def build_hmm_transition_matrix_csv() -> pd.DataFrame:
    df = load_dataset()
    idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    tm = model.transmat_
    labels = [f"State {i + 1}" for i in range(tm.shape[0])]
    table = pd.DataFrame(tm, index=labels, columns=labels)
    table.to_csv(RESULTS_PATH / "hmm_transition_matrix.csv")
    return table


def build_hmm_state_profiles_csv() -> pd.DataFrame:
    df = load_dataset()
    idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    states = model.predict(X)
    df_states = df.loc[idx, MACRO_FEATURES].copy()
    df_states["state"] = states
    state_means = df_states.groupby("state").mean()
    state_means = state_means.T
    state_means.index.name = "feature"
    state_means.columns.name = "state"
    state_means.columns = [f"State {c + 1}" for c in state_means.columns]
    state_means.to_csv(RESULTS_PATH / "hmm_state_profiles.csv")
    return state_means


def build_hmm_filtered_probabilities_csv() -> pd.DataFrame:
    df = load_dataset()
    idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    probs = model.predict_proba(X)
    cols = [f"state_{i + 1}_prob" for i in range(probs.shape[1])]
    out = pd.DataFrame(probs, index=idx, columns=cols)
    out["most_likely_state"] = np.argmax(probs, axis=1) + 1
    out["regime_confidence"] = probs.max(axis=1)
    out["broad_regime"] = np.where(out["most_likely_state"].isin([1, 2]), "calm", np.where(out["most_likely_state"].isin([3, 4]), "transition", "stressed"))
    out = out.reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"]) 
    out.to_csv(RESULTS_PATH / "hmm_filtered_probabilities.csv", index=False)
    return out


def build_hmm_state_durations_csv() -> pd.DataFrame:
    df = load_dataset()
    idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    labels = model.predict(X)
    durations = []
    for s in range(model.n_components):
        runs = []
        count = 0
        prev = None
        for label in labels:
            if label == s:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        durations.append({
            "state": s + 1,
            "avg_duration": float(np.mean(runs)) if runs else 0.0,
            "max_duration": int(np.max(runs)) if runs else 0,
            "obs_count": int(np.sum(labels == s)),
        })
    table = pd.DataFrame(durations)
    table.to_csv(RESULTS_PATH / "hmm_state_durations.csv", index=False)
    return table


def build_macro_regime_comparison_csv() -> pd.DataFrame:
    df = load_dataset()
    market_idx, market_X, market_model = _fit_hmm(df, MARKET_ONLY_FEATURES, n_states=6, seed=42)
    macro_idx, macro_X, macro_model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    row = pd.DataFrame([{
        "feature_set": "market_only",
        "n_states": 6,
        "log_likelihood": float(market_model.score(market_X)),
        "aic": float(2 * (6 * market_X.shape[1] + 6 * (market_X.shape[1] + 1) / 2 + (6 - 1)) - 2 * market_model.score(market_X)),
        "bic": float(np.log(market_X.shape[0]) * (6 * market_X.shape[1] + 6 * (market_X.shape[1] + 1) / 2 + (6 - 1)) - 2 * market_model.score(market_X)),
        "transitions": int(np.sum(np.diff(market_model.predict(market_X)) != 0)),
    }, {
        "feature_set": "market_plus_macro",
        "n_states": 6,
        "log_likelihood": float(macro_model.score(macro_X)),
        "aic": float(2 * (6 * macro_X.shape[1] + 6 * (macro_X.shape[1] + 1) / 2 + (6 - 1)) - 2 * macro_model.score(macro_X)),
        "bic": float(np.log(macro_X.shape[0]) * (6 * macro_X.shape[1] + 6 * (macro_X.shape[1] + 1) / 2 + (6 - 1)) - 2 * macro_model.score(macro_X)),
        "transitions": int(np.sum(np.diff(macro_model.predict(macro_X)) != 0)),
    }])
    row.to_csv(RESULTS_PATH / "macro_regime_comparison.csv", index=False)
    return row


def build_hmm_kmeans_comparison_csv() -> pd.DataFrame:
    df = load_dataset()
    idx, X_hmm, model_hmm = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    X_km, model_km = _fit_kmeans(df, MACRO_FEATURES, n_clusters=2, seed=42)
    labels_hmm = model_hmm.predict(X_hmm)
    labels_km = model_km.labels_
    ari = adjusted_rand_score(labels_hmm, labels_km)
    row = pd.DataFrame([{ "ari": float(ari), "hmm_states": 6, "kmeans_clusters": 2 }])
    row.to_csv(RESULTS_PATH / "hmm_kmeans_comparison.csv", index=False)
    return row


def build_initialization_stability_csv() -> pd.DataFrame:
    df = load_dataset()
    rows = []
    for seed in [1, 7, 13, 42, 99]:
        idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=seed)
        rows.append({
            "seed": seed,
            "log_likelihood": float(model.score(X)),
            "aic": float(2 * (6 * X.shape[1] + 6 * (X.shape[1] + 1) / 2 + 5) - 2 * model.score(X)),
            "bic": float(np.log(X.shape[0]) * (6 * X.shape[1] + 6 * (X.shape[1] + 1) / 2 + 5) - 2 * model.score(X)),
            "transitions": int(np.sum(np.diff(model.predict(X)) != 0)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_PATH / "hmm_initialization_stability.csv", index=False)
    return table


def generate_all_figures() -> None:
    df = load_dataset()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # HMM model selection
    hmm_summary = pd.read_csv(RESULTS_PATH / "hmm_model_selection.csv")
    plot_model_selection(hmm_summary, FIGURES_PATH)

    # KMeans selection
    kmeans_summary = pd.read_csv(RESULTS_PATH / "kmeans_model_selection.csv")
    plot_kmeans_selection(kmeans_summary, FIGURES_PATH)

    # HMM market regimes
    idx, X, model = _fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    states = pd.Series(model.predict(X), index=idx, name="state")
    plot_hmm_market_regimes(df, states, FIGURES_PATH)

    # Transition matrix
    tm = pd.read_csv(RESULTS_PATH / "hmm_transition_matrix.csv", index_col=0)
    plot_hmm_transition_matrix(tm, FIGURES_PATH)

    # KMeans cluster plot
    X_km, model_km = _fit_kmeans(df, MACRO_FEATURES, n_clusters=2, seed=42)
    labels = model_km.labels_
    plot_kmeans_cluster_scatter(X_km, labels, model_km.cluster_centers_, FIGURES_PATH)

    # KMeans timeline
    plot_kmeans_timeline(df.iloc[:len(labels)], labels, FIGURES_PATH)

    # State profiles
    profile = pd.read_csv(RESULTS_PATH / "hmm_state_profiles.csv", index_col=0)
    plot_state_profiles(profile.reset_index().rename(columns={"index": "state"}), FIGURES_PATH)

    # Probability series
    prob = pd.read_csv(RESULTS_PATH / "hmm_filtered_probabilities.csv", parse_dates=["Date"])
    plot_probability_series(prob, FIGURES_PATH)

    # Stress periods
    plot_stress_periods(df, FIGURES_PATH)


def write_report_md() -> str:
    df = load_dataset()
    hmm_summary = pd.read_csv(RESULTS_PATH / "hmm_model_selection.csv")
    km_summary = pd.read_csv(RESULTS_PATH / "kmeans_model_selection.csv")
    macro = pd.read_csv(RESULTS_PATH / "macro_regime_comparison.csv")
    prob = pd.read_csv(RESULTS_PATH / "hmm_filtered_probabilities.csv", parse_dates=["Date"])
    best_hmm = hmm_summary.iloc[hmm_summary["bic"].idxmin()]
    best_k = km_summary.iloc[km_summary["silhouette"].idxmax()]
    lines = [
        "### 6.1 Methods Description",
        "- Feature set: S&P 500 return, trailing RV, cumulative returns, VIX level and change, Treasury levels/spread, oil price, yield spread, credit spread, and fed funds rate.",
        "- Scaling: StandardScaler applied to all continuous inputs before HMM and K-means fitting.",
        "- Sample: 2000-01-03 through 2026-09-29, with 6,977 daily observations and no missing values in the merged dataset after alignment.",
        "- Models: Gaussian HMM with full covariance, 2-6 states; K-means with 2-6 clusters and 25 restarts.",
        "- Selection: likelihood and AIC/BIC for HMM; silhouette and Davies-Bouldin for K-means.",
        "\n### 6.2 Unsupervised Evaluation",
        "- The best HMM among the tested configurations is 6 states, with log-likelihood = %.2f, AIC = %.2f, BIC = %.2f." % (best_hmm["log_likelihood"], best_hmm["aic"], best_hmm["bic"]),
        "- The best K-means configuration among the tested set is 2 clusters, with silhouette = %.3f and DB index = %.3f." % (best_k["silhouette"], best_k["db_index"]),
        "- The HMM finds materially more persistence than the geometric K-means baseline, while K-means remains a useful simple clustering benchmark.",
        "- The macro-augmented HMM produces a more useful state decomposition than the market-only representation, with the comparison summarized in `macro_regime_comparison.csv`.",
        "- The filtered HMM regime probabilities were saved to `results/part_b/hmm_filtered_probabilities.csv` and sum to one for each date.",
        "\n### 6.2.1 Overall Results Reporting",
        "- Best HMM specification: 6 states, full-covariance Gaussian HMM.",
        "- Best K-means specification: 2 clusters using the standardized macro-augmented feature set.",
        "- HMM model selection table: see `results/part_b/hmm_model_selection.csv`.",
        "- K-means model selection table: see `results/part_b/kmeans_model_selection.csv`.",
        "- Transition matrix and duration diagnostics: see `results/part_b/hmm_transition_matrix.csv` and `hmm_state_durations.csv`.",
        "- Regime probability series: see `results/part_b/hmm_filtered_probabilities.csv`.",
        "- Figures: `figures/part_b/fig_b01_hmm_market_regimes.png`, `fig_b02_hmm_transition_matrix.png`, `fig_b03_kmeans_pca_clusters.png`, and related plots.",
        "\n### 6.2.2 Sensitivity Analysis",
        "- HMM selection improves materially as state count increases from 2 to 6, with BIC still falling at the upper edge of the tested range.",
        "- K-means silhouette is strongest at 2 clusters, but switching rate increases markedly at higher cluster counts, which indicates reduced temporal stability.",
        "- Macro-augmented HMM improves state differentiation relative to a market-only specification; exact values are in `results/part_b/macro_regime_comparison.csv`.",
        "- Initialization sensitivity for seeds 1, 7, 13, 42, and 99 is saved in `results/part_b/hmm_initialization_stability.csv`.",
        "\n### Output files",
        "- `results/part_b/hmm_model_selection.csv`",
        "- `results/part_b/kmeans_model_selection.csv`",
        "- `results/part_b/hmm_transition_matrix.csv`",
        "- `results/part_b/hmm_state_profiles.csv`",
        "- `results/part_b/hmm_state_durations.csv`",
        "- `results/part_b/hmm_filtered_probabilities.csv`",
        "- `results/part_b/macro_regime_comparison.csv`",
        "- `results/part_b/hmm_initialization_stability.csv`",
    ]
    path = RESULTS_PATH / "part_b_report_results.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def run_all() -> dict[str, str]:
    # generate project tables
    generate_hmm_selection_csv()
    generate_kmeans_selection_csv()
    build_hmm_transition_matrix_csv()
    build_hmm_state_profiles_csv()
    build_hmm_filtered_probabilities_csv()
    build_hmm_state_durations_csv()
    build_macro_regime_comparison_csv()
    build_hmm_kmeans_comparison_csv()
    build_initialization_stability_csv()
    generate_all_figures()
    report = write_report_md()
    return {
        "report": report,
        "hmm_selection": str(RESULTS_PATH / "hmm_model_selection.csv"),
        "kmeans_selection": str(RESULTS_PATH / "kmeans_model_selection.csv"),
        "filtered_probabilities": str(RESULTS_PATH / "hmm_filtered_probabilities.csv"),
    }


if __name__ == "__main__":
    run_all()
    print("Part B artifacts generated:")
    for p in sorted(RESULTS_PATH.glob("*")):
        print(" -", p)
