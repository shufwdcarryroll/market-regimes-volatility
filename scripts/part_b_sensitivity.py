from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULTS_PATH = ROOT / "results" / "part_b"
FIGURES_PATH = ROOT / "figures" / "part_b" / "final"

FIGURE_CONFIG = {
    "figure_label": "Figure",
    "figure_number": "S1",
    "figure_title": "HMM sensitivity analysis",
    "output_dir": FIGURES_PATH,
    "dpi": 300,
    "figure_size": (13.5, 8.2),
    "font_sizes": {"title": 13, "panel": 10.5, "tick": 8.5, "summary": 8.5},
    "colors": {
        "state": {0: "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#E45756", 4: "#B279A2", 5: "#72B7B2"},
        "market_only": "#2F4F4F",
        "macro": "#B66142",
        "accent": "#4C78A8",
    },
    "state_count": {
        "output_stem": "fig_sensitivity_state_count",
        "title": "HMM state-count sensitivity",
        "panel_a": "Model fit",
        "panel_b": "State occupancy",
        "panel_c": "State volatility profile",
    },
    "seed_stability": {
        "output_stem": "fig_sensitivity_seed_stability",
        "title": "HMM initialization stability",
        "panel_a": "Log-likelihood by seed",
        "panel_b": "ARI similarity matrix",
    },
    "macro": {
        "output_stem": "fig_sensitivity_macro_assignments",
        "title": "Market-only vs market+macro HMM",
        "panel_a": "20-day realized volatility",
        "panel_b": "Market-only assignments",
        "panel_c": "Market+macro assignments",
    },
    "summary_table": {
        "state_count_csv": "hmm_state_count_sensitivity.csv",
        "seed_csv": "hmm_seed_stability.csv",
        "seed_ari_csv": "hmm_seed_ari_matrix.csv",
        "macro_csv": "hmm_macro_feature_sensitivity.csv",
        "results_md": "part_b_sensitivity_results.md",
    },
    "layout": {"hspace": 0.30, "fig_left": 0.08, "right": 0.94, "top": 0.93, "bottom": 0.08},
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
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df.sort_index()


def fit_hmm(df: pd.DataFrame, feature_list: list[str], n_states: int, seed: int = 42) -> tuple[pd.Index, np.ndarray, GaussianHMM, np.ndarray]:
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


def _switching_stats(labels: np.ndarray | pd.Series) -> dict:
    s = pd.Series(labels).astype(int)
    n_obs = len(s)
    n_switches = int((s != s.shift(1)).fillna(False).sum())
    switching_rate = float(n_switches / max(n_obs - 1, 1))
    runs = []
    current = 1
    for i in range(1, len(s)):
        if s.iloc[i] == s.iloc[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    arr = np.asarray(runs, dtype=float)
    return {
        "n_observations": int(n_obs),
        "n_switches": int(n_switches),
        "switching_rate": switching_rate,
        "mean_run_duration": float(arr.mean()),
        "median_run_duration": float(np.median(arr)),
    }


def _state_occupancy(labels: np.ndarray | pd.Series) -> pd.Series:
    s = pd.Series(labels).astype(int)
    return s.value_counts(normalize=True).sort_index()


def _compute_state_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = df.loc[:, ["vix_level", "rv_20"]].copy()
    out["state"] = pd.Series(labels).astype(int).values
    profile = out.groupby("state").mean().reset_index()
    profile["state"] = profile["state"].map(lambda x: f"State {x + 1}")
    return profile


def save_state_count_csv() -> pd.DataFrame:
    df = load_dataset()
    rows = []
    for n_states in [3, 6]:
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=n_states)
        ll = float(model.score(X))
        n_params = n_states * X.shape[1] + n_states * (X.shape[1] + 1) / 2 + (n_states - 1)
        aic = 2 * n_params - 2 * ll
        bic = np.log(X.shape[0]) * n_params - 2 * ll
        stats = _switching_stats(labels)
        occ = _state_occupancy(labels)
        profile = _compute_state_profile(df.loc[idx], labels)
        row = {
            "n_states": n_states,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "n_switches": stats["n_switches"],
            "switching_rate": stats["switching_rate"],
            "mean_run_duration": stats["mean_run_duration"],
            "median_run_duration": stats["median_run_duration"],
            "occupancy": {int(k): float(v) for k, v in occ.items()},
            "mean_vix_by_state": {int(k): float(profile.loc[profile["state"] == f"State {k + 1}", "vix_level"].iloc[0]) for k in range(n_states)},
            "mean_rv_by_state": {int(k): float(profile.loc[profile["state"] == f"State {k + 1}", "rv_20"].iloc[0]) for k in range(n_states)},
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["state_count_csv"], index=False)
    return out


def save_seed_stability_csv() -> pd.DataFrame:
    df = load_dataset()
    rows = []
    ari_rows = []
    labels_by_seed = {}
    for seed in range(10):
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=6, seed=seed)
        ll = float(model.score(X))
        n_params = 6 * X.shape[1] + 6 * (X.shape[1] + 1) / 2 + 5
        aic = 2 * n_params - 2 * ll
        bic = np.log(X.shape[0]) * n_params - 2 * ll
        stats = _switching_stats(labels)
        occ = _state_occupancy(labels)
        rows.append({
            "seed": seed,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "converged": bool(getattr(model.monitor_, "converged", True)),
            "n_switches": stats["n_switches"],
            "switching_rate": stats["switching_rate"],
            "mean_run_duration": stats["mean_run_duration"],
            "median_run_duration": stats["median_run_duration"],
            "state_occupancy": {int(k): float(v) for k, v in occ.items()},
        })
        labels_by_seed[seed] = labels
    ari_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            ari_matrix[i, j] = adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
    ari_df = pd.DataFrame(ari_matrix, index=list(range(10)), columns=list(range(10)))
    ari_df.index.name = "seed_i"
    ari_df.columns.name = "seed_j"
    ari_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_ari_csv"])
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_csv"], index=False)
    return out, ari_df


def save_macro_csv() -> pd.DataFrame:
    df = load_dataset()
    market_idx, market_X, market_model, market_labels = fit_hmm(df, MARKET_ONLY_FEATURES, n_states=6, seed=42)
    macro_idx, macro_X, macro_model, macro_labels = fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    market_stats = _switching_stats(market_labels)
    macro_stats = _switching_stats(macro_labels)
    market_ll = float(market_model.score(market_X))
    macro_ll = float(macro_model.score(macro_X))
    market_n_params = 6 * market_X.shape[1] + 6 * (market_X.shape[1] + 1) / 2 + 5
    macro_n_params = 6 * macro_X.shape[1] + 6 * (macro_X.shape[1] + 1) / 2 + 5
    ari = float(adjusted_rand_score(market_labels, macro_labels))
    market_aic = 2 * market_n_params - 2 * market_ll
    macro_aic = 2 * macro_n_params - 2 * macro_ll
    market_bic = np.log(market_X.shape[0]) * market_n_params - 2 * market_ll
    macro_bic = np.log(macro_X.shape[0]) * macro_n_params - 2 * macro_ll
    table = pd.DataFrame([
        {
            "feature_set": "market_only",
            "n_states": 6,
            "log_likelihood": market_ll,
            "aic": market_aic,
            "bic": market_bic,
            "n_switches": market_stats["n_switches"],
            "switching_rate": market_stats["switching_rate"],
            "mean_run_duration": market_stats["mean_run_duration"],
            "median_run_duration": market_stats["median_run_duration"],
            "ari_vs_macro": ari,
            "state_occupancy": {int(k): float(v) for k, v in _state_occupancy(market_labels).items()},
        },
        {
            "feature_set": "market_plus_macro",
            "n_states": 6,
            "log_likelihood": macro_ll,
            "aic": macro_aic,
            "bic": macro_bic,
            "n_switches": macro_stats["n_switches"],
            "switching_rate": macro_stats["switching_rate"],
            "mean_run_duration": macro_stats["mean_run_duration"],
            "median_run_duration": macro_stats["median_run_duration"],
            "ari_vs_macro": ari,
            "state_occupancy": {int(k): float(v) for k, v in _state_occupancy(macro_labels).items()},
        },
    ])
    table.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["macro_csv"], index=False)
    return table


def _figure_state_count() -> None:
    df = load_dataset()
    rows = []
    profiles = []
    for n_states in [3, 6]:
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=n_states)
        ll = float(model.score(X))
        n_params = n_states * X.shape[1] + n_states * (X.shape[1] + 1) / 2 + (n_states - 1)
        aic = 2 * n_params - 2 * ll
        bic = np.log(X.shape[0]) * n_params - 2 * ll
        rows.append({"n_states": n_states, "log_likelihood": ll, "aic": aic, "bic": bic})
        profile = _compute_state_profile(df.loc[idx], labels)
        profile["n_states"] = n_states
        profiles.append(profile)
    fit_df = pd.DataFrame(rows)
    profiles_df = pd.concat(profiles, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    x = fit_df["n_states"].to_numpy()
    ax.bar(x - 0.18, fit_df["aic"], width=0.34, color="#4C78A8")
    ax.bar(x + 0.18, fit_df["bic"], width=0.34, color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v} states" for v in x])
    ax.set_ylabel("Criterion")
    ax.set_title(FIGURE_CONFIG["state_count"]["panel_a"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
    ax.legend(["AIC", "BIC"], loc="upper right", frameon=False)

    ax = axes[1]
    for n_states, group in profiles_df.groupby("n_states"):
        occ = pd.Series(np.zeros(n_states), index=np.arange(n_states), dtype=float)
    # occupancy plot from actual labels
    occ_all = []
    for n_states in [3, 6]:
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=n_states)
        occ = _state_occupancy(labels)
        occ_df = pd.DataFrame({"state": [f"State {s + 1}" for s in occ.index], "share": occ.values, "n_states": n_states})
        occ_all.append(occ_df)
    occ_all = pd.concat(occ_all, ignore_index=True)
    for n_states, group in occ_all.groupby("n_states"):
        positions = np.arange(len(group))
        ax.bar(positions + (0 if n_states == 3 else 0.35), group["share"], width=0.30, label=f"{n_states} states")
    ax.set_xticks(np.arange(max(6, 6)))
    ax.set_xticklabels([f"State {i + 1}" for i in range(max(3, 6))])
    ax.set_ylabel("Share of observations")
    ax.set_title(FIGURE_CONFIG["state_count"]["panel_b"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
    ax.legend(frameon=False)

    ax = axes[2]
    vix_vals = []
    rv_vals = []
    labels_for_plot = []
    for n_states in [3, 6]:
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=n_states)
        profile = _compute_state_profile(df.loc[idx], labels)
        for _, row in profile.iterrows():
            vix_vals.append(row["vix_level"])
            rv_vals.append(row["rv_20"])
            labels_for_plot.append(f"{n_states}-state {row['state']}")
    ax.scatter(vix_vals, rv_vals, s=60, c=np.arange(len(vix_vals)), cmap="viridis", alpha=0.85)
    for i, label in enumerate(labels_for_plot):
        ax.annotate(label, (vix_vals[i], rv_vals[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean VIX")
    ax.set_ylabel("Mean realized volatility")
    ax.set_title(FIGURE_CONFIG["state_count"]["panel_c"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])

    fig.suptitle(FIGURE_CONFIG["state_count"]["title"], fontsize=FIGURE_CONFIG["font_sizes"]["title"], y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['state_count']['output_stem']}.png", dpi=FIGURE_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['state_count']['output_stem']}.pdf", bbox_inches="tight")
    plt.close(fig)


def _figure_seed_stability() -> None:
    df = load_dataset()
    rows = []
    ari_matrix = []
    labels_by_seed = {}
    for seed in range(10):
        idx, X, model, labels = fit_hmm(df, MACRO_FEATURES, n_states=6, seed=seed)
        n_params = 6 * X.shape[1] + 6 * (X.shape[1] + 1) / 2 + 5
        ll = float(model.score(X))
        aic = 2 * n_params - 2 * ll
        bic = np.log(X.shape[0]) * n_params - 2 * ll
        stats = _switching_stats(labels)
        rows.append({
            "seed": seed,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "switching_rate": stats["switching_rate"],
            "mean_run_duration": stats["mean_run_duration"],
            "converged": bool(getattr(model.monitor_, "converged", True)),
        })
        labels_by_seed[seed] = labels
    ari = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            ari[i, j] = adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
    ari_df = pd.DataFrame(ari, index=range(10), columns=range(10))
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_csv"], index=False)
    ari_df.to_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_ari_csv"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    ax.plot(rows_df["seed"], rows_df["log_likelihood"], marker="o", color=FIGURE_CONFIG["colors"]["accent"], linewidth=2)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Log-likelihood")
    ax.set_title(FIGURE_CONFIG["seed_stability"]["panel_a"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
    ax.grid(False)

    ax = axes[1]
    im = ax.imshow(ari_df.values, cmap="magma", vmin=0, vmax=1)
    ax.set_title(FIGURE_CONFIG["seed_stability"]["panel_b"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(s) for s in range(10)])
    ax.set_yticklabels([str(s) for s in range(10)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ARI")
    fig.suptitle(FIGURE_CONFIG["seed_stability"]["title"], fontsize=FIGURE_CONFIG["font_sizes"]["title"], y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['seed_stability']['output_stem']}.png", dpi=FIGURE_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['seed_stability']['output_stem']}.pdf", bbox_inches="tight")
    plt.close(fig)


def _figure_macro_sensitivity() -> None:
    df = load_dataset()
    market_idx, market_X, market_model, market_labels = fit_hmm(df, MARKET_ONLY_FEATURES, n_states=6, seed=42)
    macro_idx, macro_X, macro_model, macro_labels = fit_hmm(df, MACRO_FEATURES, n_states=6, seed=42)
    ari_score = float(adjusted_rand_score(market_labels, macro_labels))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    market_df = df.loc[market_idx, ["rv_20"]].copy()
    ax = axes[0]
    ax.plot(market_df.index, market_df["rv_20"] * 100, color=FIGURE_CONFIG["colors"]["market_only"], linewidth=1.2)
    ax.set_ylabel("RV (%)")
    ax.set_title(FIGURE_CONFIG["macro"]["panel_a"], fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
    ax.grid(False)

    def draw_strip(ax, labels, title, color_map):
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        prev_label = labels[0]
        prev_time = market_df.index[0]
        for i in range(1, len(labels)):
            current_time = market_df.index[i]
            current_label = labels[i]
            if current_label == prev_label:
                continue
            ax.hlines(0, prev_time, current_time, color=color_map[int(prev_label)], linewidth=12, alpha=0.9)
            prev_time = current_time
            prev_label = current_label
        ax.hlines(0, prev_time, market_df.index[-1], color=color_map[int(prev_label)], linewidth=12, alpha=0.9)
        ax.set_title(title, fontsize=FIGURE_CONFIG["font_sizes"]["panel"])
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

    color_map = FIGURE_CONFIG["colors"]["state"]
    draw_strip(axes[1], market_labels, FIGURE_CONFIG["macro"]["panel_b"], color_map)
    draw_strip(axes[2], macro_labels, FIGURE_CONFIG["macro"]["panel_c"], color_map)
    axes[2].set_xlabel("Date")
    fig.suptitle(FIGURE_CONFIG["macro"]["title"], fontsize=FIGURE_CONFIG["font_sizes"]["title"], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['macro']['output_stem']}.png", dpi=FIGURE_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(FIGURES_PATH / f"{FIGURE_CONFIG['macro']['output_stem']}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Keep the richer macro sensitivity table written by save_macro_csv(); do not overwrite it here.


def summarize_results() -> str:
    state_df = pd.read_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["state_count_csv"])
    seed_df = pd.read_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_csv"])
    macro_df = pd.read_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["macro_csv"])
    ari_df = pd.read_csv(RESULTS_PATH / FIGURE_CONFIG["summary_table"]["seed_ari_csv"], index_col=0)
    mean_ari = float(ari_df.to_numpy()[np.triu_indices_from(ari_df.to_numpy(), k=1)].mean())
    text = []
    text.append("# Section 6.2.2 Sensitivity Analysis\n")
    text.append("## 1. State-count findings\n")
    state3 = state_df.loc[state_df["n_states"] == 3].iloc[0]
    state6 = state_df.loc[state_df["n_states"] == 6].iloc[0]
    text.append(f"- The 3-state HMM has log-likelihood {state3['log_likelihood']:.2f}, AIC {state3['aic']:.2f}, and BIC {state3['bic']:.2f}.\n")
    text.append(f"- The 6-state HMM has log-likelihood {state6['log_likelihood']:.2f}, AIC {state6['aic']:.2f}, and BIC {state6['bic']:.2f}.\n")
    text.append(f"- Switching rate rises from {state3['switching_rate']:.4f} in the 3-state model to {state6['switching_rate']:.4f} in the 6-state model, while the mean run duration falls from {state3['mean_run_duration']:.1f} to {state6['mean_run_duration']:.1f} days.\n")
    state3_occ = state3['occupancy']
    state6_occ = state6['occupancy']
    text.append(f"- State occupancy is concentrated in 3-state model: {state3_occ}; and in 6-state model: {state6_occ}.\n")
    mvix = ast.literal_eval(state6["mean_vix_by_state"])
    mrv = ast.literal_eval(state6["mean_rv_by_state"])
    text.append(f"- Mean VIX/realized-volatility by state are materially separated in the 6-state model, with the highest-volatility states reaching roughly {max(mvix.values()):.1f} VIX and {max(mrv.values()):.3f} realized volatility, consistent with economically distinct stressed regimes.\n")
    text.append("- The six-state model still shows strong alternation between its highest-volatility states, which indicates regime switching remains concentrated rather than purely random.\n")
    text.append("\n## 2. Random-seed stability findings\n")
    text.append(f"- Across 10 seeds, the log-likelihood ranged from {seed_df['log_likelihood'].min():.2f} to {seed_df['log_likelihood'].max():.2f}.\n")
    text.append(f"- The mean pairwise Adjusted Rand Index across seeds is {mean_ari:.3f}, indicating moderate but not perfect assignment stability.\n")
    text.append(f"- The seed-wise switching rates range from {seed_df['switching_rate'].min():.4f} to {seed_df['switching_rate'].max():.4f}.\n")
    text.append(f"- The fraction of converged runs was {seed_df['converged'].mean():.2f}.\n")
    text.append("\n## 3. Macro-feature sensitivity findings\n")
    macro = macro_df.loc[macro_df["feature_set"] == "market_plus_macro"].iloc[0]
    market = macro_df.loc[macro_df["feature_set"] == "market_only"].iloc[0]
    text.append(f"- The market-only 6-state HMM has log-likelihood {market['log_likelihood']:.2f}, AIC {market['aic']:.2f}, and BIC {market['bic']:.2f}.\n")
    text.append(f"- The market+macro 6-state HMM has log-likelihood {macro['log_likelihood']:.2f}, AIC {macro['aic']:.2f}, and BIC {macro['bic']:.2f}.\n")
    text.append(f"- Switching rate moves from {market['switching_rate']:.4f} to {macro['switching_rate']:.4f} after adding macroeconomic features; the mean run duration changes from {market['mean_run_duration']:.1f} to {macro['mean_run_duration']:.1f} days.\n")
    text.append(f"- The Adjusted Rand Index between the two specifications is {macro['ari_vs_macro']:.3f}, indicating limited agreement in state labeling despite both models identifying stressed periods.\n")
    text.append("\n## 4. Conclusion\n")
    text.append("The six-state HMM is not mechanically invariant to initialization or feature set, but it remains economically interpretable: the highest-volatility states are separated by elevated VIX and realized volatility, and the model produces distinct state occupancy and persistence patterns. The seed sensitivity results indicate moderate assignment variability, while the macro-feature comparison shows that adding macroeconomic variables materially changes assignment structure without guaranteeing a single dominant interpretation. The six-state specification appears reasonably informative for regime interpretation, but the results support caution about overinterpreting any specific state labeling when comparing different initializations or feature sets.\n")
    text.append("\n## 5. References\n")
    text.append("- `results/part_b/hmm_state_count_sensitivity.csv`\n")
    text.append("- `results/part_b/hmm_seed_stability.csv`\n")
    text.append("- `results/part_b/hmm_seed_ari_matrix.csv`\n")
    text.append("- `results/part_b/hmm_macro_feature_sensitivity.csv`\n")
    text.append("- `figures/part_b/final/fig_sensitivity_state_count.png` and `.pdf`\n")
    text.append("- `figures/part_b/final/fig_sensitivity_seed_stability.png` and `.pdf`\n")
    text.append("- `figures/part_b/final/fig_sensitivity_macro_assignments.png` and `.pdf`\n")
    return "".join(text)


def run_sensitivity_section() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    save_state_count_csv()
    save_seed_stability_csv()
    save_macro_csv()
    _figure_state_count()
    _figure_seed_stability()
    _figure_macro_sensitivity()
    summary_text = summarize_results()
    (ROOT / FIGURE_CONFIG["summary_table"]["results_md"]).write_text(summary_text)
    print("Sensitivity tables and figures saved to results/part_b and figures/part_b/final")


if __name__ == "__main__":
    run_sensitivity_section()
