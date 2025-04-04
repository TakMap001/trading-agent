# live_trading.py — aligned with SL/TP logic used in data_collection

import pandas as pd
import joblib
import numpy as np
import MetaTrader5 as mt5
import requests
import time
import logging
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER
from config import MODEL_PATH

class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.encode('utf-8', errors='replace').decode('utf-8') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt', encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)

START_BALANCE = 100
CONFIDENCE_THRESHOLD = 0.85
MAX_BALANCE = 1_000_000
SYMBOLS = ['BTCUSDm', 'ETHUSDm', 'XAUUSDm']
TRADE_INTERVAL_HOURS = 1

SYMBOL_SETTINGS = {
    'XAUUSDm': {'min_lot': 0.01, 'sl_multiplier': 2.5, 'tp_multiplier': 2.8, 'min_sl_distance': 1.0},
    'BTCUSDm': {'min_lot': 0.01, 'sl_multiplier': 2.0, 'tp_multiplier': 2.5, 'min_sl_distance': 50.0},
    'ETHUSDm': {'min_lot': 0.1, 'sl_multiplier': 2.0, 'tp_multiplier': 2.5, 'min_sl_distance': 10.0},
}

last_trade_time = {}

class TradeTracker:
    def __init__(self):
        self.open_trades = {}

    def add_trade(self, symbol, ticket, direction, entry_price, sl, tp, volume):
        self.open_trades[ticket] = {
            'symbol': symbol,
            'direction': direction,
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'volume': volume,
            'opened_at': datetime.now()
        }

    def remove_trade(self, ticket):
        return self.open_trades.pop(ticket, None)

tracker = TradeTracker()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_lot_size(balance, symbol):
    """
    Dynamic lot size:
    - For BTCUSDm and XAUUSDm: 0.01 lots per $100
    - For ETHUSDm: 0.1 lots per $100
    """
    if symbol == 'ETHUSDm':
        lot_size = round((balance // 100) * 0.1, 2)
        return max(0.1, lot_size)
    else:  # BTCUSDm and XAUUSDm
        lot_size = round((balance // 100) * 0.01, 2)
        return max(0.01, lot_size)

def load_feature_list(symbol):
    path = f"models/{symbol}_features.txt"
    with open(path, "r", encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    if not mt5.login(MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER):
        raise RuntimeError("MT5 login failed")
    logging.info("✅ MT5 initialized and logged in")

def get_latest_data(symbol, features):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)
    if rates is None or len(rates) == 0:
        return None
    row = pd.Series(rates[0])
    return row.reindex(features, fill_value=0).fillna(0).infer_objects(copy=False)

def check_existing_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return positions if positions else []

def execute_trade(symbol, model, balance):
    features = load_feature_list(symbol)
    row = get_latest_data(symbol, features)
    if row is None:
        return balance

    existing_positions = check_existing_positions(symbol)
    if existing_positions:
        logging.info(f"Existing positions found for {symbol}, skipping new trade")
        return balance

    X_row = row.values.reshape(1, -1)
    prob = model.predict_proba(X_row)[0]
    prediction = int(prob[1] > CONFIDENCE_THRESHOLD)
    confidence = prob[1] if prediction == 1 else 1 - prob[1]
    if confidence < CONFIDENCE_THRESHOLD:
        return balance

    lot_size = get_lot_size(balance, symbol)
    tick_info = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    if tick_info is None or symbol_info is None:
        return balance

    point = symbol_info.point
    price = tick_info.ask if prediction == 0 else tick_info.bid
    direction = "BUY" if prediction == 0 else "SELL"

    settings = SYMBOL_SETTINGS.get(symbol, {})
    atr = row.get('ATR', 1.0)
    sl_distance = max(settings.get('sl_multiplier', 2.0) * atr, settings.get('min_sl_distance', 1.0))
    tp_distance = settings.get('tp_multiplier', 3.0) * atr

    if direction == "BUY":
        sl = round(price - sl_distance, 6)
        tp = round(price + tp_distance, 6)
    else:
        sl = round(price + sl_distance, 6)
        tp = round(price - tp_distance, 6)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI_Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"TRADE EXECUTED [✓]: {symbol} {direction} {lot_size} lots")
        tracker.add_trade(symbol, result.order, direction, price, sl, tp, lot_size)
        send_telegram_message(
            f"🚀 *Trade Executed*\n"
            f"*{symbol}* {direction}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Size: {lot_size} lots\n"
            f"Entry: {price}\n"
            f"SL: {sl} | TP: {tp}\n"
            f"Ticket: #{result.order}"
        )
    else:
        logging.error(f"TRADE FAILED [✗]: {result.comment}")
        send_telegram_message(f"❌ *Trade Failed* {symbol}\nError: {result.comment}")

    return balance

def main():
    initialize_mt5()
    balance = START_BALANCE
    models = {}
    for symbol in SYMBOLS:
        try:
            models[symbol] = joblib.load(f"models/{symbol}_model.pkl")
            logging.info(f"Model loaded for {symbol} [✓]")
        except Exception as e:
            logging.error(f"Error loading model for {symbol}: {e}")

    while balance < MAX_BALANCE:
        for symbol, model in models.items():
            balance = execute_trade(symbol, model, balance)
        time.sleep(3600 * TRADE_INTERVAL_HOURS)

if __name__ == '__main__':
    main()
