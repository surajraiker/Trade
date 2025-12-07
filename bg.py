import os
import time
import re
from PIL import Image
import pytesseract
from telegram import ParseMode, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

BOT_TOKEN = "8274059753:AAF9SQI-UzvIzj6_cCQJhhOjgm2TBEJ5gBU"  # Replace with your token

def ocr_from_image(img_path):
    image = Image.open(img_path)
    text = pytesseract.image_to_string(image, config="--psm 6")
    return text

def calc_dynamic_sl_tp(entry, price_scale, side):
    # SL = 0.1% (or higher for crypto/stocks), TP = 0.2% (edit as needed)
    sl_gap = price_scale * 0.001
    tp_gap = price_scale * 0.002
    if side == 'BUY':
        sl = entry - sl_gap
        tp = entry + tp_gap
    else:
        sl = entry + sl_gap
        tp = entry - tp_gap
    # Round SL/TP to match price's decimal precision
    decimals = len(str(entry).split('.')[-1])
    sl = round(sl, decimals)
    tp = round(tp, decimals)
    return sl, tp

def extract_bidask_signal(text):
    text_low = text.lower()
    # Try to find explicit BUY/SELL paired with a number
    buy_pattern = re.findall(r"(?:buy|long)\D{0,6}?([-+]?\d+(?:[.,]\d+){1,2})", text_low)
    sell_pattern = re.findall(r"(?:sell|short)\D{0,6}?([-+]?\d+(?:[.,]\d+){1,2})", text_low)
    all_prices = [float(p.replace(',','')) for p in re.findall(r"([-+]?\d+(?:[.,]\d+){1,2})", text.replace(',','')) if len(p.split('.')[-1]) in [2,3,4,5]]
    entry, side = None, None

    if buy_pattern and float(buy_pattern[0]) > 0:
        side = "BUY"
        entry = float(buy_pattern[0])
    elif sell_pattern and float(sell_pattern[0]) > 0:
        side = "SELL"
        entry = float(sell_pattern[0])
    elif len(all_prices) >= 2:
        # Two close prices, infer side
        p1, p2 = all_prices[0], all_prices[1]
        if abs(p1 - p2) / max(abs(p1), abs(p2)) < 0.003:  # likely bid/ask
            entry = min(p1, p2)
            side = "BUY"
        else:
            entry = p1
            side = "BUY" if p2 > p1 else "SELL"
    elif all_prices:
        entry = all_prices[0]
        side = "BUY"  # fallback
    if entry is None or side is None:
        return {}

    # Price scale: choose entry, or use average of first 2 prices to estimate percent pip
    price_scale = entry
    sl, tp = calc_dynamic_sl_tp(entry, price_scale, side)
    return {
        'side': side, 'entry': entry,
        'sl': sl, 'tp': tp,
        'confidence': 'OCR Dynamic'
    }

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Send a chart screenshot from any platform (MT4, MT5, TradingView, Broker App, etc). "
        "The bot will auto-detect BUY/SELL entry, compute side, stop loss, and take profit dynamically—no manual setup needed."
    )

def image_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    photo_file = update.message.photo[-1].get_file()
    file_path = f"./chart_{user.id}_{int(time.time())}.jpg"
    photo_file.download(file_path)
    update.message.reply_text("Analyzing screenshot...")

    ocr_text = ocr_from_image(file_path)
    os.remove(file_path)
    result = extract_bidask_signal(ocr_text)
    if all(result.get(x) for x in ("side", "entry", "sl", "tp")):
        update.message.reply_text(
            f"📊 *Trade Signal (OCR)*\n"
            f"*Side*: `{result['side']}`\n"
            f"*Entry*: `{result['entry']}`\n"
            f"*Stop Loss*: `{result['sl']}`\n"
            f"*Take Profit*: `{result['tp']}`",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        debug_info = ocr_text if ocr_text else "(no OCR output)"
        update.message.reply_text(
            "❗ Couldn't auto-detect prices or trade. Please crop the screenshot to show clear price boxes or annotate BUY/SELL levels!\n"
            f"OCR Text:\n```\n{debug_info[:350]}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )

updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(Filters.photo, image_handler))
updater.start_polling()
updater.idle()