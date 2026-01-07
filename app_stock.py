# streamlit_app_weekends_only_final.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from pandas.tseries.offsets import BDay  # only business days (skip weekends)

st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
st.title("Stock Price Prediction for Next N Business Days")  # removed Apple emoji

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your stock CSV file", type=["csv"])
forecast_days = st.sidebar.slider("Select Forecast Days (Business Days)", min_value=1, max_value=90, value=30)

# -------------------------
# Main
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns!")
        st.stop()
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Features
    df['MA_21'] = df['Close'].rolling(21).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_21'] = df['Daily_Return'].rolling(21).std()
    earnings_months = [1,4,7,10]
    df['Earnings_Month'] = df.index.month.isin(earnings_months).astype(int)
    
    df = df.dropna()
    
    X = df.drop(columns=['Close'])
    y = df['Close']
    
    # Train-test split
    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Load or train model
    model_file = "xgb_stock_model.pkl"
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Loaded saved model")
    except:
        st.info("⚡ Training XGBoost model. Please wait...")
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist"
        )
        model.fit(X_train, y_train)
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        st.success("✅ Model trained and saved")
    
    # -------------------------
    # Historical metrics
    # -------------------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    direction_actual = np.sign(y_test.diff().fillna(0))
    direction_pred = np.sign(pd.Series(y_pred, index=y_test.index).diff().fillna(0))
    direction_accuracy = (direction_actual == direction_pred).mean() * 100
    
    # -------------------------
    # Forecast next N business days (skip weekends only)
    # -------------------------
    st.subheader(f"{forecast_days}-Business Day Forecast")
    
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + BDay(1), periods=forecast_days, freq=BDay())

    forecast_df = pd.DataFrame(index=forecast_dates, columns=df.columns)
    last_row = df.iloc[-1:].copy()
    close_history = list(df['Close'][-50:])  # last 50 closes for rolling MA
    
    for forecast_date in forecast_dates:
        X_next = last_row.drop(columns=['Close'])
        y_next = model.predict(X_next)[0]
        
        close_history.append(y_next)
        
        # Rolling MAs
        MA_21 = np.mean(close_history[-21:]) if len(close_history) >= 21 else np.mean(close_history)
        MA_50 = np.mean(close_history[-50:]) if len(close_history) >= 50 else np.mean(close_history)
        
        # Forecast row
        new_row = last_row.copy()
        new_row.index = [forecast_date]
        new_row['Close'] = y_next
        new_row['MA_21'] = MA_21
        new_row['MA_50'] = MA_50
        new_row['Daily_Return'] = (y_next - close_history[-2]) / close_history[-2]
        new_row['Volatility_21'] = np.std(np.diff(close_history[-21:])) if len(close_history) >= 21 else 0
        new_row['Earnings_Month'] = int(forecast_date.month in earnings_months)
        
        forecast_df.loc[forecast_date] = new_row.iloc[0]
        last_row = new_row
    
    forecast_df = forecast_df[['Close']].rename(columns={'Close':'Forecast_Close'})
    
    # -------------------------
    # Forecast Plot
    # -------------------------
    plt.figure(figsize=(12,5))
    plt.plot(forecast_df.index, forecast_df['Forecast_Close'], label="Forecasted Close", marker='o')
    plt.title(f"Stock Forecast for Next {forecast_days} Business Days")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # -------------------------
    # Forecast Table
    # -------------------------
    st.subheader("Forecasted Closing Prices")
    st.dataframe(forecast_df)
    
    # -------------------------
    # Last Predicted Closing Price
    # -------------------------
    last_price = forecast_df['Forecast_Close'].iloc[-1]
    st.success(f"💰 Last predicted closing price after {forecast_days} business days: **{last_price:.2f}**")
