from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_STATE_RANGE = range(2, 7)
DEFAULT_K_RANGE = range(2, 7)


def load_feature_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def _prepare_matrix(df: pd.DataFrame, feature_list: list[str]) -> tuple[pd.Index, np.ndarray]:
    subset = df.loc[:, feature_list].dropna()
    if subset.empty:
        raise ValueError("No rows available after dropping missing values.")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(subset)
    return subset.index, matrix


def fit_hmm_model(df: pd.DataFrame, feature_list: list[str], n_states: int, random_state: int = 42) -> GaussianHMM:
    idx, X = _prepare_matrix(df, feature_list)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=random_state,
        init_params="stmc",
    )
    model.fit(X)
    return model


def run_hmm_comparison(data_path: str | Path, results_dir: str | Path) -> pd.DataFrame:
    df = load_feature_dataset(data_path)
    feature_list = [
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
    results = []
    for n_states in DEFAULT_STATE_RANGE:
        idx, X = _prepare_matrix(df, feature_list)
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=300, random_state=42, init_params="stmc")
        model.fit(X)
        ll = model.score(X)
        n_obs = X.shape[0]
        n_params = n_states * len(feature_list) + (n_states ** 2) + (n_states - 1)
        aic = 2 * n_params - 2 * ll
        bic = np.log(n_obs) * n_params - 2 * ll
        results.append({
            "n_states": n_states,
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "transitions": int(np.sum(np.diff(model.predict(X)) != 0)),
        })
    summary = pd.DataFrame(results)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "hmm_model_selection.csv", index=False)
    return summary


def run_kmeans_comparison(data_path: str | Path, results_dir: str | Path) -> pd.DataFrame:
    df = load_feature_dataset(data_path)
    feature_list = [
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
    idx, X = _prepare_matrix(df, feature_list)
    records = []
    for n_clusters in DEFAULT_K_RANGE:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        switching_rate = float(np.mean(np.diff(labels, prepend=labels[0]) != 0))
        records.append({
            "n_clusters": n_clusters,
            "silhouette": float(sil),
            "db_index": float(db),
            "switching_rate": switching_rate,
            "regime_transitions": int(np.sum(np.diff(labels) != 0)),
        })
    summary = pd.DataFrame(records)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "kmeans_model_selection.csv", index=False)
    return summary


def build_filtered_probabilities(data_path: str | Path, output_path: str | Path, n_states: int = 6, random_state: int = 42) -> pd.DataFrame:
    df = load_feature_dataset(data_path)
    feature_list = [
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
    idx, X = _prepare_matrix(df, feature_list)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=300, random_state=random_state, init_params="stmc")
    model.fit(X)
    prob = model.predict_proba(X)
    state_cols = [f"state_{i + 1}_prob" for i in range(n_states)]
    df_out = pd.DataFrame(prob, index=idx, columns=state_cols)
    df_out["most_likely_state"] = np.argmax(prob, axis=1) + 1
    df_out["regime_confidence"] = prob.max(axis=1)
    df_out["broad_regime"] = np.where(df_out["most_likely_state"].isin([1, 2]), "calm", np.where(df_out["most_likely_state"].isin([3, 4]), "transition", "stressed"))
    df_out = df_out.reset_index().rename(columns={"index": "Date"})
    df_out["Date"] = pd.to_datetime(df_out["Date"])
    df_out.to_csv(output_path, index=False)
    return df_out
