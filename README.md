# Quant Trading Model Suite

A research collection of machine learning and reinforcement learning models for financial time-series forecasting and algorithmic trading. This project explores multiple approaches—from LSTM forecasting to PPO-based agents—to evaluate predictive edge and trade execution logic.

---

## Overview

This repository contains experimental models for:

- Supervised price forecasting using deep learning  
- Hybrid CNN + BiLSTM architectures  
- Market data ingestion and preprocessing  
- Reinforcement learning trading agent (PPO)

The focus is on **directional accuracy, risk-aware evaluation, and reproducible research pipelines.**

---

## Repository Contents

| Notebook | Purpose |
|--------|---------|
| `AMZN Time Series Model.ipynb` | LSTM forecasting pipeline for Amazon |
| `MSFT Time Series Model.ipynb` | Microsoft sequence model |
| `NVDA Time Series Model.ipynb` | NVIDIA time-series predictor |
| `CNN BiLSTM Testing.ipynb` | Hybrid convolution + BiLSTM experiment |
| `Kudan Test Data Pulls.ipynb` | Data acquisition & preprocessing |
| `Trading PPO Initial.ipynb` | PPO reinforcement learning trader |

---

## Modeling Approaches

### 1. Supervised Forecasting
- Sliding window sequence generation  
- Normalization and feature engineering  
- Multi-step horizon prediction  
- Evaluation using:
  - RMSE / MAE  
  - Directional accuracy  
  - Simple trading overlay

### 2. CNN + BiLSTM Hybrid
- CNN layers capture local microstructure patterns  
- BiLSTM learns long-range temporal dependencies  
- Designed for noisy, regime-shifting markets

### 3. Reinforcement Learning (PPO)
- Custom trading environment  
- Actions: Long / Flat / Short  
- Reward shaping using:
  - PnL  
  - Drawdown penalties  
  - Transaction cost estimates

---

## Installation

### Requirements

- Python 3.9+
- Key libraries:

```
numpy
pandas
scikit-learn
matplotlib
tensorflow or torch
stable-baselines3
yfinance (or data provider)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Pipeline

Run:

```
Kudan Test Data Pulls.ipynb
```

Outputs:
- Clean OHLCV dataset  
- Engineered features  
- Train/test splits

---

### 2. Train Forecast Models

Open any asset notebook:

```
AMZN Time Series Model.ipynb
MSFT Time Series Model.ipynb
NVDA Time Series Model.ipynb
```

Workflow:

1. Load processed data  
2. Generate sequences  
3. Train LSTM  
4. Evaluate predictions  
5. Backtest style overlay

---

### 3. Hybrid Model

```
CNN BiLSTM Testing.ipynb
```

- Compare to baseline LSTM  
- Tune convolution filters  
- Multi-feature experiments

---

### 4. RL Trader

```
Trading PPO Initial.ipynb
```

- Custom gym-style environment  
- PPO training loop  
- Performance visualization

---

## Evaluation Philosophy

Models are assessed on:

- Directional edge  
- Out-of-sample robustness  
- Cost-aware returns  
- Drawdown behavior  
- Stability across assets

---

## Roadmap

- Ensemble blending  
- Regime detection  
- Probabilistic forecasts  
- Options-aware rewards  
- Live paper-trading harness

---

## Disclaimer

This is research code only.  
Not financial advice. Use at your own risk.

---

## License

For educational and research use.
