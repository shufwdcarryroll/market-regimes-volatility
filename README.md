# Market Regimes and US Equity Volatility

**Identifying Market States and Forecasting Future Volatility Using Financial and Macroeconomic Indicators**

**Shubhankit Mohan**  
University of Michigan | Master of Applied Data Science  
SIADS 696 – Milestone II | Fall 2026

## Project Overview

This project explores how financial market conditions and macroeconomic indicators influence US equity market volatility. The main objective is to understand whether incorporating macroeconomic information improves volatility forecasts and helps identify different market regimes.

The analysis focuses on two questions:

1. How accurately can we forecast S&P 500 realized volatility over the next 5, 10, and 20 trading days?
2. Can we identify meaningful market regimes using machine learning, and does adding macroeconomic information improve those classifications?

We use historical S&P 500 prices, VIX, Treasury yields, oil prices, credit spreads, and other market indicators from 2000 through 2026.

The project combines supervised learning for volatility forecasting with unsupervised learning for market regime identification.

## Methodology

### Part A: Volatility Forecasting

We compare three supervised learning models:

| Model | Purpose |
|---|---|
| Ridge Regression | Regularized linear forecasting |
| Random Forest | Capturing nonlinear relationships and interactions |
| XGBoost | Gradient-boosted tree forecasting |

We evaluate forecasts over 5-, 10-, and 20-day horizons using RMSE and MAE.

The analysis also includes chronological cross-validation, feature importance, market-only versus macroeconomic feature comparisons, regularization sensitivity, and an examination of forecasting failures during major market events.

### Part B: Market Regime Identification

We use two unsupervised learning approaches:

| Model | Purpose |
|---|---|
| Hidden Markov Model (HMM) | Identifying market states and transitions over time |
| K-means | Grouping similar market conditions without explicitly modeling transitions |

The HMM is evaluated across different numbers of states using AIC, BIC, and transition probabilities. K-means is assessed using clustering metrics such as silhouette scores.

We also examine how incorporating macroeconomic variables changes the identified regimes.

## Key Findings

Our analysis highlights several findings:

- Historical realized volatility and VIX-related indicators are important predictors of future volatility.
- Ridge Regression shows relatively stable cross-validation performance, while more complex tree-based models remain competitive on the test set.
- Adding macroeconomic variables did not improve 5-day Ridge forecasting accuracy in our comparison.
- Ridge performance was relatively insensitive to changes in regularization strength.
- Forecasting errors were particularly pronounced during sudden market shocks, including the April 2025 tariff announcements.
- For regime identification, the six-state HMM achieved the lowest AIC and BIC among the tested specifications, while K-means favored a two-cluster solution.

One interesting takeaway is that short-term volatility appears to be driven largely by recent market conditions. Macroeconomic variables may provide more context for longer-term market developments, although our results do not establish consistent forecasting improvements from adding them.

## Data

Market data comes from Yahoo Finance, while macroeconomic series are obtained from the Federal Reserve Economic Data (FRED) database.

Features include daily returns, historical realized volatility, VIX changes, skewness, kurtosis, drawdowns, Treasury yields, yield-curve spreads, crude oil prices, credit spreads, and the federal funds rate.

The data is aligned to US trading days, with separate historical periods for training, validation, and testing.

## Repository Structure

```text
├── src/              # Python modules and analysis scripts
├── notebooks/        # Five analysis notebooks
├── data/
│   ├── raw/          # Original market and macro data
│   └── processed/    # Cleaned datasets and features
├── figures/          # Model results and visualizations
├── results/          # Forecast evaluations and statistical outputs
├── report/           # Research report
├── tests/            # Project tests
├── config.yaml       # Project configuration
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/shufwdcarryroll/market-regimes-volatility.git
cd market-regimes-volatility
python -m pip install -r requirements.txt
```

## Running the Project

The main analysis is organized into five notebooks, covering data collection, feature engineering, market regime identification, volatility forecasting, and macroeconomic robustness.

The Python modules can be run from the project root:

```bash
python src/download_data.py
python src/preprocessing.py
python src/features.py
python src/hmm_models.py
python src/forecasting.py
python src/evaluation.py
python src/plotting.py
```

Additional scripts used for supervised learning evaluation include:

```bash
python src/part_a_feature_analysis.py
python src/part_a_sensitivity.py
python src/part_a_failure_analysis.py
python src/part_a_cv_statistics.py
```

These produce supporting results for the feature importance, ablation, sensitivity, cross-validation, and failure analyses.

Tests can be run with:

```bash
python -m pytest tests -q
```

## Limitations and Future Work

Financial market relationships change over time, and models trained on historical observations may struggle when unexpected events occur.

Our results also show that increasing model complexity does not always lead to better forecasts. Similarly, while macroeconomic indicators help explain financial conditions, their additional predictive value varies.

Future work could explore rolling-window estimation, regime-dependent volatility forecasting, and more detailed analysis of how macroeconomic conditions affect forecasting performance during periods of market stress.

## Author and Contributions

**Shubhankit Mohan**  
University of Michigan, Master of Applied Data Science

I developed this project as part of SIADS 696, covering research design, data collection, feature engineering, model implementation, evaluation, financial interpretation, and report preparation.

Generative AI tools were used to assist with coding, debugging, visualizations, and editing. The project findings are based on the empirical analysis and documented results.

## References

Key references include:

- Engle and Patton (2001), *What Good Is a Volatility Model?*
- Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility.*
- Ang and Bekaert (2002), *International Asset Allocation with Regime Shifts.*
- Rabiner (1989), *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.*

The full bibliography is available in the research report.

---

**University of Michigan | SIADS 696 | Fall 2026**

**Repository:** https://github.com/shufwdcarryroll/market-regimes-volatility
