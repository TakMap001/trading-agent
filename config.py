# config.py
MT5_ACCOUNT = 209276416
MT5_PASSWORD = "Sh@n@@2024"
MT5_SERVER = "Exness-MT5Trial9"

ALPHA_VANTAGE_API_KEY = "NVGR3AD8EXD584GP"

# FRED API key
FRED_API_KEY = "b24b4de0bc11508883a2d2f3b9a72655"

# NEWSAPI API key
NEWSAPI_API_KEY = "4018e1c94c1d4aeabedfbeace710ecda"

# BINANCE 
BINANCE_API_KEY = "9Uen2ZFoj2jfbkm3FTEZOQEEuFb4x7KLF67COgkckFrYP1PoNDP81X0fVUszYIod"
BINANCE_SECRET_KEY = "vQ55QtNPAgqwaT2JAESglAXpI3mzJsGOdrxGeDKkFiorLAtTS7slT1fwL6Bbe49p"

# REDDIT
REDDIT_CLIENT_ID = "hrfskrfpLYJzWPxMCUgAMg"
REDDIT_SECRET = "uU1DBR9HsQ6E6dpnVeMnS7Cj_9epVw"
REDDIT_USER_AGENT = "bot:TradingSentimentBot_v1:v1.0 (by /u/Agreeable_Ladder_217)"

TELEGRAM_BOT_TOKEN = "7622018027:AAESHennbiG3McjA4xUAcI5ZQxzC2QTwggA"
TELEGRAM_CHAT_ID = "782962404"

DATA_PATH = "data/combined_data.csv"
MODEL_PATH = "models/"
LOG_PATH = "logs/"
TIMEZONE = "Etc/UTC"

# === Assets to Trade ===
ASSETS = ["XAUUSDm", "BTCUSDm", "ETHUSDm"]

# === Dynamic Lot Sizing Rules ===
def get_lot_size(symbol, balance):
    if symbol == "ETHUSDm":
        base = 0.1
    else:
        base = 0.01
    lots = (balance // 100) * base
    return round(max(lots, base), 2)  # Minimum lot = base


