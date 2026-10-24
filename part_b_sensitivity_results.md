# Section 6.2.2 Sensitivity Analysis
## 1. State-count findings
- The 3-state HMM has log-likelihood 66659.73, AIC -133204.47, and BIC -132810.73.
- The 6-state HMM has log-likelihood 19557.52, AIC -38883.04, and BIC -38088.73.
- Switching rate rises from 0.0022 in the 3-state model to 0.3583 in the 6-state model, while the mean run duration falls from 463.8 to 2.8 days.
- State occupancy is concentrated in 3-state model: {0: 0.0773321834123904, 1: 0.032772746873652434, 2: 0.8898950697139572}; and in 6-state model: {0: 0.17306310191174357, 1: 0.43970102055483684, 2: 0.10679890757510421, 3: 0.19879258300991806, 4: 0.06166451056489866, 5: 0.019979876383498633}.
- Mean VIX/realized-volatility by state are materially separated in the 6-state model, with the highest-volatility states reaching roughly 41.4 VIX and 0.414 realized volatility, consistent with economically distinct stressed regimes.
- The six-state model still shows strong alternation between its highest-volatility states, which indicates regime switching remains concentrated rather than purely random.

## 2. Random-seed stability findings
- Across 10 seeds, the log-likelihood ranged from 21023.71 to 74871.31.
- The mean pairwise Adjusted Rand Index across seeds is 0.310, indicating moderate but not perfect assignment stability.
- The seed-wise switching rates range from 0.0082 to 0.3373.
- The fraction of converged runs was 1.00.

## 3. Macro-feature sensitivity findings
- The market-only 6-state HMM has log-likelihood -23477.24, AIC 47078.48, and BIC 47503.02.
- The market+macro 6-state HMM has log-likelihood 67765.21, AIC -135298.43, and BIC -134504.12.
- Switching rate moves from 0.0381 to 0.0331 after adding macroeconomic features; the mean run duration changes from 26.3 to 30.2 days.
- The Adjusted Rand Index between the two specifications is 0.052, indicating limited agreement in state labeling despite both models identifying stressed periods.

## 4. Conclusion
The six-state HMM is not mechanically invariant to initialization or feature set, but it remains economically interpretable: the highest-volatility states are separated by elevated VIX and realized volatility, and the model produces distinct state occupancy and persistence patterns. The seed sensitivity results indicate moderate assignment variability, while the macro-feature comparison shows that adding macroeconomic variables materially changes assignment structure without guaranteeing a single dominant interpretation. The six-state specification appears reasonably informative for regime interpretation, but the results support caution about overinterpreting any specific state labeling when comparing different initializations or feature sets.

## 5. References
- `results/part_b/hmm_state_count_sensitivity.csv`
- `results/part_b/hmm_seed_stability.csv`
- `results/part_b/hmm_seed_ari_matrix.csv`
- `results/part_b/hmm_macro_feature_sensitivity.csv`
- `figures/part_b/final/fig_sensitivity_state_count.png` and `.pdf`
- `figures/part_b/final/fig_sensitivity_seed_stability.png` and `.pdf`
- `figures/part_b/final/fig_sensitivity_macro_assignments.png` and `.pdf`
