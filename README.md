# 📊 Market Analysis ML - No Look-ahead Bias

A realistic machine learning system for stock and forex market prediction that rigorously prevents look-ahead bias in feature engineering.

## Key Features

- ** No Look-ahead Bias**: All features use only data available before the prediction day
- ** Dual Market Support**: Stocks (any Yahoo Finance ticker) and Forex (EURUSD, GBPUSD, JPY, AUDUSD, USDCAD)
- ** Multiple Models**: XGBoost, Random Forest, and Ensemble Voting
- ** Realistic Validation**: Time series cross-validation that respects chronological order
- ** Interactive Dashboard**: Built with Streamlit for real-time analysis
- ** Cross-Market Comparison**: Compare model performance across different asset classes

## Live Demo

[Application](https://market-analysis-ml-2clfbrpodgbbhiuqhavar2.streamlit.app/) - Demo for Stock and Forex market analysis

## Example Results

| Asset | Typical Accuracy | Interpretation |
|-------|-----------------|----------------|
| Stocks (AAPL) | 53-58% | Good predictive power |
| Forex (EURUSD) | 48-52% | Typical for forex markets |

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Tom223-hub/market-analysis-ml.git
cd market-analysis-ml
