# 🧠 Stock Price Prediction using RNNs

This project leverages Recurrent Neural Networks (RNNs), specifically **Simple RNN** and **LSTM**, to predict the closing prices of multiple stocks simultaneously. It is designed for time series forecasting in the financial domain, using data from tech giants: **Amazon (AMZN), Google (GOOGL), IBM, and Microsoft (MSFT)**.

---

## 📊 Overview

The goal is to build and compare two types of RNN models:
- A **Simple RNN**, which handles short-term dependencies
- An **LSTM (Long Short-Term Memory)** model, which is better suited for capturing long-term trends in sequential data

The models predict **multiple target variables** (closing prices of four companies) simultaneously using past stock market data.

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📁 Project Structure

├── RNN_Assg_Stock_Price_Prediction_Starter_Completed.ipynb
├── RNN_Stocks_Data/
│ ├── AMZN_stocks_data.csv
│ ├── GOOGL_stocks_data.csv
│ ├── IBM_stocks_data.csv
│ └── MSFT_stocks_data.csv
├── README.md



---

## 📈 Features

- Multivariate, multi-target time series forecasting
- Sequence generation using sliding window
- Data normalization with `MinMaxScaler`
- Hyperparameter tuning for both models (units, learning rate)
- Visual comparisons of actual vs predicted prices
- Evaluation using MSE, MAE, and R² Score
- Training vs validation loss monitoring
- Stock-wise performance breakdown
- Dropout regularization to avoid overfitting

---

## ✅ Results

| Model        | MSE     | MAE     | R² Score |
|--------------|---------|---------|----------|
| Simple RNN   | 0.0110  | 0.0805  | 0.2689   |
| LSTM         | 0.0026  | 0.0405  | 0.7703   |

LSTM significantly outperformed Simple RNN, providing smoother and more accurate forecasts across all four stocks.

---

## 📌 Key Takeaways

- LSTM’s ability to remember long sequences gives it a clear edge in stock forecasting.
- Including features like `Open`, `High`, `Low`, and `Volume` greatly improves prediction quality.
- Multi-target models are effective when stock trends are interdependent.

---

## 📦 Future Work

- Add technical indicators (e.g., RSI, MACD)
- Incorporate news sentiment analysis
- Deploy model with a web-based dashboard (e.g., using Flask or Streamlit)
- Try Transformer or Attention-based models

---

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-rnn.git
   cd stock-price-prediction-rnn

---

## 📈 Features

- Multivariate, multi-target time series forecasting
- Sequence generation using sliding window
- Data normalization with `MinMaxScaler`
- Hyperparameter tuning for both models (units, learning rate)
- Visual comparisons of actual vs predicted prices
- Evaluation using MSE, MAE, and R² Score
- Training vs validation loss monitoring
- Stock-wise performance breakdown
- Dropout regularization to avoid overfitting

---

## ✅ Results

| Model        | MSE     | MAE     | R² Score |
|--------------|---------|---------|----------|
| Simple RNN   | 0.0110  | 0.0805  | 0.2689   |
| LSTM         | 0.0026  | 0.0405  | 0.7703   |

LSTM significantly outperformed Simple RNN, providing smoother and more accurate forecasts across all four stocks.

---

## 📌 Key Takeaways

- LSTM’s ability to remember long sequences gives it a clear edge in stock forecasting.
- Including features like `Open`, `High`, `Low`, and `Volume` greatly improves prediction quality.
- Multi-target models are effective when stock trends are interdependent.

---

## 📦 Future Work

- Add technical indicators (e.g., RSI, MACD)
- Incorporate news sentiment analysis
- Deploy model with a web-based dashboard (e.g., using Flask or Streamlit)
- Try Transformer or Attention-based models

---
