"""Find how far back Alpaca free tier allows stock bar data."""
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

# Test different date ranges to find the free tier cutoff
test_ranges = [
    ("6 months ago", 180),
    ("3 months ago", 90),
    ("60 days ago",  60),
    ("45 days ago",  45),
    ("30 days ago",  30),
]

for label, days_ago in test_ranges:
    end   = datetime.now(timezone.utc) - timedelta(days=days_ago)
    start = end - timedelta(days=5)
    try:
        resp = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",   # try IEX feed explicitly
        ))
        bars = resp.get("AAPL", [])
        print(f"✅ {label} ({start.date()} → {end.date()}): {len(bars)} bars")
    except Exception as e:
        print(f"❌ {label} ({start.date()} → {end.date()}): {e}")