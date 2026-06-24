"""Debug get_price_at for AAPL around Memorial Day 2026-05-26."""
import os
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

ticker = "AAPL"
dt = datetime(2026, 5, 26, tzinfo=timezone.utc)

start = dt - timedelta(days=14)
end   = dt + timedelta(days=8)

print(f"Fetching bars for {ticker} from {start.date()} to {end.date()}")

resp = client.get_stock_bars(StockBarsRequest(
    symbol_or_symbols=ticker,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
))

bars = resp.get(ticker, [])
print(f"\nGot {len(bars)} bars:")
for b in bars:
    print(f"  t={b.timestamp}  t.date()={b.timestamp.date()}  close={b.close}  tzinfo={b.timestamp.tzinfo}")

print(f"\ndt={dt}  dt.date()={dt.date()}  tzinfo={dt.tzinfo}")

# Test same_day comparison
print("\nSame-day matches:")
for b in bars:
    match = b.timestamp.date() == dt.date()
    print(f"  {b.timestamp.date()} == {dt.date()} → {match}")