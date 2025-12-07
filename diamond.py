import os
import time
import re
import requests
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from telegram import ParseMode, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

BOT_TOKEN = "8274059753:AAEFq6rWegHWj4GBRfFiXelGViF6fq2nnRw"
TWELVE_API_KEY = "5f65d8f4bedb4232a449d9ef3bb594c4"
   # https://twelvedata.com/apikey

def extract_symbol_timeframe(text):
    symbol, tf = None, None
    symbol_list = [
        "XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD",
        "USDCHF","GBPNZD","EURJPY","XAGUSD","EURGBP","BTCUSD","ETHUSD"
    ]
    tf_list = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines[:8]:
        m = re.search(r"\b([A-Z]{6,7})\s*[▼]?\s*([MHDW][1-9][0-9]?|MN)\b", line.upper())
        if m:
            symbol, tf = m.group(1), m.group(2)
            break
    if not symbol:
        for line in lines:
            for s in symbol_list:
                if s in line.upper():
                    symbol = s; break
            if symbol: break
    if not tf:
        for line in lines:
            for t in tf_list:
                if t in line.upper():
                    tf = t; break
            if tf: break
    return symbol, tf

def twelvedata_interval(tf):
    mapping = {
        "M1":"1min", "M5":"5min", "M15":"15min", "M30":"30min", 
        "H1":"1h", "H4":"4h", "D1":"1day", "W1":"1week", "MN":"1month"
    }
    return mapping.get(tf.upper(), "15min")

def higher_tf(tf):
    hierarchy = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]
    ix = hierarchy.index(tf) if tf in hierarchy else 2
    return hierarchy[min(ix+1, len(hierarchy)-1)]

def get_twelvedata_ohlc(symbol, interval, n=500):
    fxmap = {
        "XAUUSD":"XAU/USD", "EURUSD":"EUR/USD", "GBPUSD":"GBP/USD", "USDJPY":"USD/JPY",
        "AUDUSD":"AUD/USD", "NZDUSD":"NZD/USD", "USDCAD":"USD/CAD", "USDCHF":"USD/CHF",
        "GBPNZD":"GBP/NZD","EURJPY":"EUR/JPY","XAGUSD":"XAG/USD","EURGBP":"EUR/GBP"
    }
    cryptomap = {"BTCUSD":"BTC/USD", "ETHUSD":"ETH/USD"}
    if symbol in fxmap: apisymbol = fxmap[symbol]
    elif symbol in cryptomap: apisymbol = cryptomap[symbol]
    else: apisymbol = symbol  # fallback
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": apisymbol,
        "interval": interval,
        "outputsize": n,
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "values" not in data or not data["values"]:
        return None
    closes = [float(item["close"]) for item in reversed(data["values"])]  # oldest to newest
    return closes

def ema(prices, period):
    prices = np.array(prices)
    if len(prices) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    a = np.convolve(prices, weights[::-1], mode='valid')
    return float(a[-1])

def rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) < period + 1:
        return None
    diff = np.diff(prices)
    up = np.where(diff > 0, diff, 0)
    down = np.where(diff < 0, -diff, 0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-10)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def macd(prices, fast=12, slow=26, signal=9):
    prices = pd.Series(prices)
    if len(prices) < slow + signal:
        return None, None, None
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])

def advanced_trade_signal(prices_main, prices_htf, symbol, tf, htf):
    if not prices_main or len(prices_main) < 55 or not prices_htf or len(prices_htf) < 55:
        return None
    ema21 = ema(prices_main, 21)
    ema50 = ema(prices_main, 50)
    price = prices_main[-1]
    ema21_h = ema(prices_htf, 21)
    ema50_h = ema(prices_htf, 50)
    rsi_val = rsi(prices_main, 14)
    macd_line, signal_line, macd_hist = macd(prices_main)
    step = 15 if symbol.startswith("XAU") else (100 if symbol.startswith("BTC") else 8)
    multi = 30 if symbol.startswith("XAU") else (300 if symbol.startswith("BTC") else 20)
    def ok(val): return val is not None and not np.isnan(val) and not np.isinf(val)
    side, sl, tp = "NEUTRAL", None, None

    # Multi-TF trend + indicator filter logic (outputs ONLY side/entry/sl/tp)
    if all(map(ok, [ema21, ema50, ema21_h, ema50_h, rsi_val, macd_line, signal_line])):
        if (ema21 > ema50 and price > ema21 and ema21_h > ema50_h and rsi_val > 45 and rsi_val < 70
            and macd_line > signal_line):
            side = "BUY"
            sl = round(price - step, 2)
            tp = round(price + multi, 2)
        elif (ema21 < ema50 and price < ema21 and ema21_h < ema50_h and rsi_val < 55 and rsi_val > 30
              and macd_line < signal_line):
            side = "SELL"
            sl = round(price + step, 2)
            tp = round(price - multi, 2)
    return {
        "entry": round(price,2),
        "side": side,
        "sl": sl,
        "tp": tp,
        "tf": tf,
        "htf": htf,
        "symbol": symbol
    }

def image_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    photo_file = update.message.photo[-1].get_file()
    img_path = f"./chart_{user.id}_{int(time.time())}.jpg"
    photo_file.download(img_path)
    update.message.reply_text("⏳ Analyzing screenshot...")

    ocr_text = pytesseract.image_to_string(Image.open(img_path), config="--psm 6")
    os.remove(img_path)
    symbol, tf = extract_symbol_timeframe(ocr_text)
    if not symbol or not tf:
        update.message.reply_text(
            "❗ Couldn't detect symbol or timeframe. Reply: SYMBOL TIMEFRAME (e.g. XAUUSD M30)"
        )
        return
    interval = twelvedata_interval(tf)
    htf = higher_tf(tf)
    interval_htf = twelvedata_interval(htf)
    prices = get_twelvedata_ohlc(symbol, interval, n=500)
    prices_htf = get_twelvedata_ohlc(symbol, interval_htf, n=200)
    result = advanced_trade_signal(prices, prices_htf, symbol, tf, htf)
    if not result or result["side"] == "NEUTRAL":
        update.message.reply_text(
            "❗ No trade setup (trend/indicator confluence not strong enough with current chart)."
        )
        return
    msg = (
        f"📊 Advanced EMA+RSI+MACD+MTF Signal\n"
        f"*Symbol:* `{result['symbol']}`\n"
        f"*Timeframe:* `{result['tf']}`\n"
        f"*HTF Confirm:* `{result['htf']}`\n"
        f"*Side:* `{result['side']}`\n"
        f"*Entry:* `{result['entry']}`\n"
        f"*Stop Loss:* `{result['sl']}`\n"
        f"*Take Profit:* `{result['tp']}`"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Send a screenshot from MT4/MT5/TradingView. Bot will extract symbol/timeframe and reply ONLY with: Symbol, Timeframe, HTF Confirm, Side, Entry, Stop Loss, Take Profit."
    )

updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(Filters.photo, image_handler))
updater.start_polling()
updater.idle()