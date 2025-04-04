# predictive_test.py — Forward prediction using latest models and features

import pandas as pd
import joblib
import MetaTrader5 as mt5
from datetime import datetime
import logging

# === CONFIG ===
START_DATE = "2025-04-01"
END_DATE = "2025-04-02"
SYMBOLS = ["BTCUSDm", "ETHUSDm", "XAUUSDm"]

from config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, MODEL_PATH

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(message)s")

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    if not mt5.login(login=MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError("MT5 login failed")
    logging.info("✅ MT5 initialized and logged in")

def fetch_historical_data(symbol):
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    utc_from = datetime(start.year, start.month, start.day)
    utc_to = datetime(end.year, end.month, end.day)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
    if rates is None:
        raise ValueError(f"No data for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def load_features(symbol):
    path = f"{MODEL_PATH}{symbol}_features.txt"
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def run_prediction(symbol):
    logging.info(f"\n🔮 Predictive Test for {symbol}...")
    df = fetch_historical_data(symbol)
    features = load_features(symbol)

    # Fill missing features
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    df = df.fillna(0)

    # Save time for logging
    times = df['time'].copy()

    # Only use feature columns for prediction
    X = df[features]

    # Load model
    model = joblib.load(f"{MODEL_PATH}{symbol}_model.pkl")

    # Predict
    probs = model.predict_proba(X)
    signals = model.predict(X)

    count = 0
    for i in range(len(X)):
        conf = probs[i][signals[i]]
        if conf >= 0.7:  # Confidence threshold
            time_str = times.iloc[i].strftime("%Y-%m-%d %H:%M:%S")
            direction = "BUY" if signals[i] == 0 else "SELL"
            logging.info(f"🕒 {time_str} | 📈 Signal: {direction} | 🤖 Confidence: {conf:.2f}")
            count += 1

    logging.info(f"✅ Predictive signals: {count}")

def main():
    initialize_mt5()
    for symbol in SYMBOLS:
        try:
            run_prediction(symbol)
        except Exception as e:
            logging.error(f"❌ Error for {symbol}: {e}")

if __name__ == "__main__":
    main()
