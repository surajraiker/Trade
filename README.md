# Telegram Chart Signal Bot

This bot extracts trading signals from chart screenshots using OCR and posts symbol, timeframe, HTF confirm, side, entry, stop loss, and take profit to Telegram.

## Features

- OCR from chart screenshot
- Parses symbol & timeframe
- Analyzes market conditions using EMA, RSI, MACD
- Posts a concise trading signal to Telegram

## Install

```bash
# Clone the repository
git clone https://github.com/<yourusername>/telegram-chart-signal-bot.git
cd telegram-chart-signal-bot

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for image text extraction)
# Ubuntu
sudo apt-get update
sudo apt-get install tesseract-ocr
# Mac
brew install tesseract
# Windows
# Download from https://github.com/tesseract-ocr/tesseract

```

## Run Locally

Set your bot token and API key as environment variables:

```bash
export BOT_TOKEN='your-telegram-bot-token'
export TWELVE_API_KEY='your-twelve-data-key'
python bot.py
```

## Deploy to railway.app (Free Cloud Hosting)

1. Sign up and log in at [railway.app](https://railway.app/).
2. Click "New Project" > "Deploy from GitHub" and select your repo.
3. In **Variables**, set `BOT_TOKEN` and `TWELVE_API_KEY`.
4. Confirm the start command is `python bot.py`.
5. Click "Deploy." Your bot runs in the cloud!

## GitHub Actions (CI Testing)

You can automate tests on push using GitHub Actions.
See `.github/workflows/test.yml` in this repo.

## Security

**Never commit your Telegram bot token or API keys.**
Always use environment variables or GitHub repo secrets.
