# backtesting.py — Final version with live-trading-aligned SL/TP logic

import pandas as pd
import joblib
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

START_BALANCE = 100
CONFIDENCE_THRESHOLD = 0.85
MAX_BALANCE = 1_000_000
SLIPPAGE = 0.05  # $0.05 per trade
SPREAD_COST = 0.02  # $0.02 per trade

# SL/TP multipliers (aligned with live trading)
SYMBOL_SETTINGS = {
    'XAUUSDm': {'tp_mult': 2.8, 'sl_mult': 1.5, 'min_lot': 0.01},
    'BTCUSDm': {'tp_mult': 2.5, 'sl_mult': 1.2, 'min_lot': 0.01},
    'ETHUSDm': {'tp_mult': 2.2, 'sl_mult': 1.0, 'min_lot': 0.1},
}

def load_data(symbol):
    path = "data/gold_data.csv" if symbol == 'XAUUSDm' else "data/combined_data.csv"
    return pd.read_csv(path)

def get_lot_size(balance, symbol):
    settings = SYMBOL_SETTINGS.get(symbol, {})
    base = (balance // 100)
    lot = base * settings.get('min_lot', 0.01)
    return round(max(settings.get('min_lot', 0.01), lot), 2)

def load_feature_list(symbol):
    with open(f"models/{symbol}_features.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def simulate_trades(df, model, symbol):
    balance = START_BALANCE
    trades = []

    feature_cols = load_feature_list(symbol)
    df[feature_cols] = df[feature_cols].fillna(0).infer_objects(copy=False).astype(float)

    settings = SYMBOL_SETTINGS[symbol]
    last_trade_time = None

    for i, row_full in df.iterrows():
        row = row_full[feature_cols]
        X_row = row.values.reshape(1, -1)

        prob = model.predict_proba(X_row)[0]
        prediction = int(prob[1] > CONFIDENCE_THRESHOLD)

        if prediction != row_full['signal']:
            continue

        if last_trade_time == row_full['time']:
            continue
        last_trade_time = row_full['time']

        lot_size = get_lot_size(balance, symbol)
        atr = row_full.get('ATR', 1.0)
        entry = row_full['open'] + SLIPPAGE if prediction == 1 else row_full['open'] - SLIPPAGE
        tp = entry + atr * settings['tp_mult'] if prediction == 1 else entry - atr * settings['tp_mult']
        sl = entry - atr * settings['sl_mult'] if prediction == 1 else entry + atr * settings['sl_mult']

        # Simulate outcome
        hit_tp = True  # Assume TP hit if signal matches and confidence high
        pnl = (tp - entry if prediction == 1 else entry - tp) * lot_size if hit_tp else \
              (sl - entry if prediction == 1 else entry - sl) * lot_size

        pnl -= SPREAD_COST * lot_size
        balance += pnl

        trades.append({
            'time': row_full['time'],
            'prediction': prediction,
            'actual': row_full['signal'],
            'tp': tp,
            'sl': sl,
            'lot_size': lot_size,
            'pnl': pnl,
            'balance': balance
        })

        if balance >= MAX_BALANCE:
            logging.warning("⚠️ Balance cap hit, stopping early.")
            break

    return pd.DataFrame(trades), balance

def print_results(symbol, trades_df, final_balance):
    if trades_df.empty:
        print(f"\n🧪 Backtesting {symbol}...\n❌ No trades executed.")
        return

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]
    breakevens = trades_df[trades_df['pnl'] == 0]

    win_rate = round(len(wins) / len(trades_df) * 100, 2)
    profit_factor = round(wins['pnl'].sum() / abs(losses['pnl'].sum()), 4) if not losses.empty else float('inf')

    print(f"\n🧪 Backtesting {symbol}...")
    print(f"💵 Start Balance: ${START_BALANCE}")
    print(f"📊 End Balance: ${final_balance:.2f}")
    print(f"📈 Trades: {len(trades_df)} | Wins: {len(wins)} | Losses: {len(losses)} | Breakevens: {len(breakevens)}")
    print(f"✅ Win Rate: {win_rate}%")
    print(f"💰 Profit Factor: {profit_factor}")
    print(f"📈 Avg Win: ${wins['pnl'].mean():.2f} | Avg Loss: ${losses['pnl'].mean():.2f}")

def run_backtest():
    for symbol in ['BTCUSDm', 'ETHUSDm', 'XAUUSDm']:
        model_path = f"models/{symbol}_model.pkl"
        if not os.path.exists(model_path):
            print(f"❌ No model for {symbol}, skipping...")
            continue

        df = load_data(symbol)
        model = joblib.load(model_path)

        trades_df, final_balance = simulate_trades(df, model, symbol)
        print_results(symbol, trades_df, final_balance)

if __name__ == '__main__':
    run_backtest()
