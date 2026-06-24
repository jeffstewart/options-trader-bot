"""Quick diagnostic to inspect the NewsClient response structure."""
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
ALPACA_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

client = NewsClient(ALPACA_KEY, ALPACA_SECRET)

end   = datetime.now(timezone.utc)
start = end - timedelta(days=3)

print(f"Fetching news from {start.date()} to {end.date()}")

req = NewsRequest(start=start, end=end, limit=5)
resp = client.get_news(req)

print(f"\nResponse type : {type(resp)}")
print(f"Response attrs: {[a for a in dir(resp) if not a.startswith('_')]}")

# Try common attribute names
for attr in ("news", "data", "items", "articles", "results"):
    val = getattr(resp, attr, "NOT FOUND")
    if val != "NOT FOUND":
        print(f"\nresp.{attr} = {type(val)} len={len(val) if hasattr(val, '__len__') else 'n/a'}")
        if hasattr(val, "__iter__"):
            for i, item in enumerate(val):
                if i >= 2: break
                print(f"  item[{i}] type={type(item)} attrs={[a for a in dir(item) if not a.startswith('_')][:10]}")
                print(f"  item[{i}] headline={getattr(item, 'headline', 'N/A')[:80]}")
                print(f"  item[{i}] created_at={getattr(item, 'created_at', 'N/A')}")

# Also try iterating resp directly
print("\nTrying direct iteration of resp:")
try:
    for i, item in enumerate(resp):
        if i >= 2: break
        print(f"  item[{i}] = {type(item)} headline={getattr(item, 'headline', 'N/A')[:80]}")
except Exception as e:
    print(f"  Not iterable: {e}")

print("\nRaw resp:", repr(resp)[:500])