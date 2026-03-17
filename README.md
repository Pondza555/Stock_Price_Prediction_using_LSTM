# Stock Price Prediction using LSTM

## 🔹 Introduction

This project builds and evaluates LSTM (Long Short-Term Memory) deep learning models to forecast the close price of four major financial assets: **Gold**, **Bitcoin**, **Nasdaq**, and **SET** (Thailand's stock index). The goal is to leverage sequential patterns in historical price data to generate short-term forecasts useful for market assessment and risk management.

**Main notebook:** `lstm.ipynb`

## 🔹 Objectives

- Forecast close prices of Gold, Bitcoin, Nasdaq, and SET using LSTM.
- Explore price trends, volatility patterns, and cross-asset correlations.
- Evaluate model performance using MAE and RMSE.
- Generate 3-day ahead price forecasts for each asset.
     
## 🔹 Dataset & Setup
     
- **Source:** Yahoo Finance via `yfinance`
- **Assets:** Gold (GC=F), Bitcoin (BTC-USD), Nasdaq (^IXIC), SET (^SET.BK)
- **Time Period:** January 1, 2014 – May 30, 2025 (daily close prices)
- **Features:** Close price, Volume, Daily Return
- Data after May 2025 was excluded due to elevated geopolitical volatility (Iran–Israel conflict) that could distort model learning.

## 🔹 EDA Highlights

- **Gold:** Stable daily returns (±2–4%), with volume peaks during 2018–2020 (pre-COVID period).
- **Bitcoin:** High volatility (±20%), volume peaked in 2021 during COVID.
- **Nasdaq:** Stable returns (±2–5%), with volume and momentum rising in 2024–2025 (AI adoption surge).
- **SET:** Stable returns (±2.5–5%), volume peaked in 2014–2015 (economic stimulus by Thai government).
- **Correlation:** Only SET shows a negative correlation with other assets. Gold and Bitcoin are generally uncorrelated or slightly negatively correlated.
      
## 🔹 Methodology

- **Model Architecture**
  - LSTM with 3 layers, hidden size = 100
  - Sequence length: 30 days of historical data
  - Forecasting horizon: next-day prediction (rolling to 3 days ahead)
  - Loss function: MSE; Optimizer: Adam (lr=0.001)
  - Train/Test split: 80% / 20%
  - Epochs: 30, Batch size: 32
- **Forecasting Approach**
  - A separate LSTM model is trained for each asset independently.
  - Prices normalized with MinMaxScaler before training.
  - 3-day ahead forecasts generated via recursive (autoregressive) prediction.
          
## 🔹 Results
 
  | Asset   | MAE     | RMSE    |
  |---------|---------|---------|
  | Gold    | ~145    | ~162    |
  | Bitcoin | ~2,940  | ~3,800  |       
  | Nasdaq  | ~580    | ~620    |
  | SET     | ~8.9    | ~11.5   |

  ✅ **Best model performance:** SET index — lowest RMSE relative to price scale.
  ⚠️ **Weakest performance:** Bitcoin — high volatility makes price prediction challenging with price-only features.

  **3-Day Forecast (as of late May 2025):**
  - SET: predicted to increase slightly (t+1: ~1163.8, t+2: ~1164.6, t+3: ~1166.3)
  - Gold, Bitcoin, Nasdaq: predicted to decrease in the near term.
 
## 🔹 Conclusion
 
An LSTM model was trained independently for each of the four assets (Gold, Bitcoin, Nasdaq, SET) using 30-day sequences of historical close prices. The model performed best on SET and worst on Bitcoin, due to Bitcoin's high inherent volatility. Future improvements could include adding more features (volume, macroeconomic indicators), applying attention mechanisms, or using multivariate LSTM to capture cross-asset dynamics.
 
## 🔹 Executive Summary
 
This project applies LSTM deep learning to forecast 3-day close prices for Gold, Bitcoin, Nasdaq, and SET using daily data from 2014–2025. EDA reveals distinct volatility profiles and low cross-asset correlation (except during COVID). Separate LSTM models are trained per asset, with SET yielding the best forecast accuracy and Bitcoin the lowest due to high volatility. The 3-day ahead forecast (late May 2025) indicates a slight SET increase while other assets are projected to decline, providing actionable short-term market signals.
