import streamlit as st

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📊 Stock Trend Prediction & Trading Intelligence System")

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/trend_model.pkl")

model = load_model()

# --------------------------------
# USER INPUT
# --------------------------------
ticker = st.selectbox(
    "Select Stock",
    ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
)

start_date = "2018-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# --------------------------------
# DOWNLOAD DATA
# --------------------------------
df = yf.download(ticker, start=start_date, end=end_date)

st.subheader("📈 Latest Stock Data")
st.dataframe(df.tail())

# --------------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# --------------------------------
df["Return"] = df["Close"].pct_change()

df["SMA_20"] = df["Close"].rolling(20).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# --------------------------------
# FEATURES (EXACT SAME ORDER AS TRAINING)
# --------------------------------
features = [
    "Close",
    "Return",
    "SMA_20",
    "EMA_20",
    "EMA_50",
    "RSI",
    "Volume"
]

X = df[features]

# --------------------------------
# PREDICTION
# --------------------------------
df["Signal"] = model.predict(X)

# --------------------------------
# LATEST SIGNAL (SAFE SCALAR)
# --------------------------------
latest_signal = int(df["Signal"].iloc[-1])

st.subheader("📌 Latest Trading Signal")

if latest_signal == 1:
    st.success("📈 BUY Signal")
else:
    st.error("📉 SELL Signal")

# --------------------------------
# VISUALIZATION
# --------------------------------
st.subheader("📊 Price & Indicators")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, df["SMA_20"], label="SMA 20")
ax.plot(df.index, df["EMA_20"], label="EMA 20")
ax.plot(df.index, df["EMA_50"], label="EMA 50")
ax.legend()
st.pyplot(fig)

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.markdown("**Capstone Project – Stock Trend Prediction | Amol Purkar**")
