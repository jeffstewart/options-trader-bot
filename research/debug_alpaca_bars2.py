"""Inspect BarSet response structure."""
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

end   = datetime.now(timezone.utc) - timedelta(days=180)
start = end - timedelta(days=5)

resp = client.get_stock_bars(StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed="iex",
))

print(f"Type: {type(resp)}")
print(f"Attrs: {[a for a in dir(resp) if not a.startswith('_')]}")

# Try data attribute
d = getattr(resp, 'data', None)
print(f"\nresp.data type: {type(d)}")
if isinstance(d, dict):
    print(f"resp.data keys: {list(d.keys())[:3]}")
    for k, v in list(d.items())[:1]:
        print(f"  resp.data['{k}'] type: {type(v)}, len: {len(v) if hasattr(v,'__len__') else 'n/a'}")
        if hasattr(v, '__iter__'):
            for i, bar in enumerate(v):
                if i >= 2: break
                print(f"    bar[{i}] type={type(bar)}")
                print(f"    bar[{i}] attrs={[a for a in dir(bar) if not a.startswith('_')][:8]}")
                print(f"    bar[{i}] timestamp={getattr(bar,'timestamp',None)}  close={getattr(bar,'close',None)}")