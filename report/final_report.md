# Market Regimes and Equity Volatility

## Introduction
This independent project evaluates whether macroeconomic conditions improve market-regime identification and future volatility forecasting for the S&P 500. The study uses original data from Yahoo Finance and FRED for 2000-01-01 through 2026-09-30, builds a reusable feature pipeline, estimates HMMs, and benchmarks supervised forecasting models.

## Data sources
- Yahoo Finance: ^GSPC, ^VIX, CL=F, ^TNX, SPY
- FRED: DGS2, DGS10, T10Y2Y, DCOILWTICO, BAMLH0A0HYM2, DFF

## Main findings
The implemented pipeline produced real datasets, features, and model summaries. Notable results from the executed scripts include:

- The HMM comparison across 2-6 states produced log-likelihood values ranging from -34,309 to -24,727.
- K-means comparison showed the strongest silhouette around k=2 (0.369) and a rising switching rate with more clusters.
- The forecasting summary indicates the best out-of-sample RMSE for the 10-day horizon is around 0.0635 for the market-only Ridge model; the market+macro Ridge model slightly improves to 0.0635 with a modestly different MAE profile.
- The project was executed end-to-end with saved datasets and figures produced under `data/`, `figures/`, and `results/`.

## Limitations
This project implements a streamlined version of the requested analysis under the constraints of the environment, with conservative assumptions around macro publication timing and a simplified evaluation design. It provides a working independent baseline rather than a full academic final-article production pipeline.

## Statement of work
The analysis was implemented independently from scratch with original data-download and feature-engineering code, and executed in the local environment. No pre-existing scripts, fitted models, or reported numerical results were reused.
