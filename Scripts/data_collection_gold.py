# data_collection_gold.py — Signal generation for XAUUSDm aligned with live trading

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import (
    MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER,
    REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT,
    FRED_API_KEY, ALPHA_VANTAGE_API_KEY, BINANCE_API_KEY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SYMBOL = "XAUUSDm"
NUM_BARS = 20000
SL_MULTIPLIER = 2.0
TP_MULTIPLIER = 2.8

# --- External Data (Live-compatible placeholders) ---
def get_reddit_sentiment_score(symbol):
    logging.info(f"📡 Fetching Reddit sentiment for {symbol}")
    return pd.Series(0.5, index=pd.date_range(end=datetime.now(), periods=NUM_BARS, freq='h'))

def get_macro_data():
    logging.info(f"📡 Fetching macro data")
    return pd.Series(0.5, index=pd.date_range(end=datetime.now(), periods=NUM_BARS, freq='h'))

def get_binance_volatility(symbol):
    logging.info(f"📡 Fetching Binance volatility for {symbol}")
    return pd.Series(0.5, index=pd.date_range(end=datetime.now(), periods=NUM_BARS, freq='h'))

# --- MT5 Setup ---
def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    if not mt5.login(MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER):
        raise RuntimeError("MT5 login failed")
    logging.info("✅ MT5 initialized and logged in")

# --- Indicator Calculations ---
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --- Fetch MT5 Data + External Context ---
def fetch_mt5_data(symbol):
    logging.info(f"Fetching data for {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, NUM_BARS)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['ATR'] = compute_atr(df, 14)
    df['MACD'], df['Signal'] = compute_macd(df['close'])

    df['ATR'] = df['ATR'].bfill()

    logging.info(f"📡 Fetching external data for {symbol}")
    df['sentiment_score'] = get_reddit_sentiment_score(symbol).reindex(df.index, method='nearest')
    df['macro_event_score'] = get_macro_data().reindex(df.index, method='nearest')
    df['volatility'] = get_binance_volatility(symbol).reindex(df.index, method='nearest')
    logging.info(f"✅ External data fetched for {symbol}")

    return df

# --- Signal Generation + TP-before-SL Labeling ---
def generate_signals(df):
    logging.info(f"⚙️ Generating signals for {SYMBOL}")
    entry_condition = (
        (df['EMA_20'] > df['EMA_50']) &
        (df['RSI'] > 50) &
        (df['MACD'] > df['Signal']) &
        (df['sentiment_score'] > 0.4)
    )

    entry_df = df[entry_condition].copy()
    entry_df['signal'] = 0

    for i in range(len(entry_df) - 10):
        entry_price = entry_df.iloc[i]['close']
        atr = entry_df.iloc[i]['ATR']
        tp = entry_price + TP_MULTIPLIER * atr
        sl = entry_price - SL_MULTIPLIER * atr

        for j in range(i + 1, min(i + 10, len(entry_df))):
            price = entry_df.iloc[j]['close']
            if price >= tp:
                entry_df.iloc[i, entry_df.columns.get_loc('signal')] = 1
                break
            elif price <= sl:
                entry_df.iloc[i, entry_df.columns.get_loc('signal')] = 0
                break

    logging.info(f"📊 Signal distribution for {SYMBOL}:\n{entry_df['signal'].value_counts()}")
    return entry_df

# --- Main Pipeline ---
def main():
    initialize_mt5()
    df = fetch_mt5_data(SYMBOL)
    labeled_df = generate_signals(df)
    labeled_df.to_csv("data/gold_data.csv")
    logging.info("✅ Gold data saved to data/gold_data.csv")

if __name__ == "__main__":
    main()
