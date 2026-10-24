### 6.1 Methods Description
- Feature set: S&P 500 return, trailing RV, cumulative returns, VIX level and change, Treasury levels/spread, oil price, yield spread, credit spread, and fed funds rate.
- Scaling: StandardScaler applied to all continuous inputs before HMM and K-means fitting.
- Sample: 2000-01-03 through 2026-09-29, with 6,977 daily observations and no missing values in the merged dataset after alignment.
- Models: Gaussian HMM with full covariance, 2-6 states; K-means with 2-6 clusters and 25 restarts.
- Selection: likelihood and AIC/BIC for HMM; silhouette and Davies-Bouldin for K-means.

### 6.2 Unsupervised Evaluation
- The best HMM among the tested configurations is 6 states, with log-likelihood = 81970.81, AIC = -163787.62, BIC = -163260.36.
- The best K-means configuration among the tested set is 2 clusters, with silhouette = 0.282 and DB index = 1.512.
- The HMM finds materially more persistence than the geometric K-means baseline, while K-means remains a useful simple clustering benchmark.
- The macro-augmented HMM produces a more useful state decomposition than the market-only representation, with the comparison summarized in `macro_regime_comparison.csv`.
- The filtered HMM regime probabilities were saved to `results/part_b/hmm_filtered_probabilities.csv` and sum to one for each date.

### 6.2.1 Overall Results Reporting
- Best HMM specification: 6 states, full-covariance Gaussian HMM.
- Best K-means specification: 2 clusters using the standardized macro-augmented feature set.
- HMM model selection table: see `results/part_b/hmm_model_selection.csv`.
- K-means model selection table: see `results/part_b/kmeans_model_selection.csv`.
- Transition matrix and duration diagnostics: see `results/part_b/hmm_transition_matrix.csv` and `hmm_state_durations.csv`.
- Regime probability series: see `results/part_b/hmm_filtered_probabilities.csv`.
- Figures: `figures/part_b/fig_b01_hmm_market_regimes.png`, `fig_b02_hmm_transition_matrix.png`, `fig_b03_kmeans_pca_clusters.png`, and related plots.

### 6.2.2 Sensitivity Analysis
- HMM selection improves materially as state count increases from 2 to 6, with BIC still falling at the upper edge of the tested range.
- K-means silhouette is strongest at 2 clusters, but switching rate increases markedly at higher cluster counts, which indicates reduced temporal stability.
- Macro-augmented HMM improves state differentiation relative to a market-only specification; exact values are in `results/part_b/macro_regime_comparison.csv`.
- Initialization sensitivity for seeds 1, 7, 13, 42, and 99 is saved in `results/part_b/hmm_initialization_stability.csv`.

### Output files
- `results/part_b/hmm_model_selection.csv`
- `results/part_b/kmeans_model_selection.csv`
- `results/part_b/hmm_transition_matrix.csv`
- `results/part_b/hmm_state_profiles.csv`
- `results/part_b/hmm_state_durations.csv`
- `results/part_b/hmm_filtered_probabilities.csv`
- `results/part_b/macro_regime_comparison.csv`
- `results/part_b/hmm_initialization_stability.csv`
