import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features
from datetime import date

st.set_page_config(layout="wide")
st.title("Stock Price Prediction with XGBoost")

# Sidebar input options
st.sidebar.header("Select Stock and Time Range")
ticker = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=date.today())

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df.dropna()

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def add_features(df):
    # Ensure required columns exist and are numeric
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Add lag features
    for i in range(1, 11):
        df[f"Close_Lag_{i}"] = df["Close"].shift(i)

    df.dropna(inplace=True)
    return df
    
if st.sidebar.button("Train Model"):
    df = fetch_data(ticker, start_date, end_date)
    df = add_features(df)

    feature_cols = [
        'momentum_rsi', 'momentum_roc', 'trend_macd', 'trend_macd_signal',
        'volatility_bbh', 'volatility_bbl', 'trend_sma_fast', 'trend_ema_fast',
        'volume_adi'] + [f'Close_Lag_{i}' for i in range(1, 11)]

    X = df[feature_cols]
    y = df[['Close']]

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train.ravel())

    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_y.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    st.write(f"### RMSE: {rmse:.2f}")

    test_dates = df.iloc[-len(y_test):].index
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates, y_test_unscaled, label='Actual', color='blue')
    ax.plot(test_dates, y_pred, label='Predicted', linestyle='--', color='red')
    ax.set_title(f"Actual vs. Predicted Close Prices for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
