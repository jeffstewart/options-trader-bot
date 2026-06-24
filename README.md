# News Trading Bot

An event-driven trading bot that monitors Alpaca's real-time news stream,
filters for material events using keyword matching, scores sentiment with
Claude AI, and executes bracket orders on your Alpaca paper trading account.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys
cp .env.example .env
# Edit .env with your Alpaca paper keys + Anthropic key

# 3. Run the bot
python bot.py
```

## How it works

```
Alpaca news stream
      ↓
Keyword pre-filter  ← fast, cheap, no API call
      ↓ (match)
Claude AI scorer    ← headline + body → sentiment / confidence / tickers
      ↓ (confidence ≥ 65%)
Safety checks       ← market hours, cooldown, position size
      ↓
Alpaca bracket order  (entry + stop-loss + take-profit)
      ↓
trades.csv log
```

## Key settings (top of bot.py)

| Setting | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE` | 0.65 | Claude must be ≥65% confident |
| `MAX_POSITION_USD` | $1,000 | Max dollars per trade |
| `STOP_LOSS_PCT` | 3% | Stop-loss below entry |
| `TAKE_PROFIT_PCT` | 6% | Take-profit above entry |
| `COOLDOWN_SECS` | 300s | Min time between trades on same ticker |

## Adding keywords

Edit the `BULLISH_KEYWORDS` and `BEARISH_KEYWORDS` sets in `bot.py`.
These are lowercase substring matches against the full news text.

## Trade log

Each executed trade is appended to `trades.csv` with:
timestamp, ticker, side, qty, price, stop, target, confidence, magnitude, reasoning

## API keys you need

- **Alpaca paper trading** — free at https://app.alpaca.markets/paper-trading/overview
- **Anthropic API** — https://console.anthropic.com/

## Safety notes

- This bot is for educational/research use on paper trading only.
- Do not use real money without extensive backtesting and risk review.
- The bot will not trade outside market hours (checked via Alpaca clock API).
- Bracket orders automatically cap your loss to `STOP_LOSS_PCT` per trade.