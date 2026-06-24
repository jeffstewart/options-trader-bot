"""Debug Alpaca news pagination."""
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

client = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

end   = datetime.now(timezone.utc) - timedelta(days=5)
start = end - timedelta(days=14)

print(f"Fetching {start.date()} → {end.date()}")

# Page 1
req  = NewsRequest(start=start, end=end, limit=50, sort="desc",
                   include_content=False, exclude_contentless=True)
resp = client.get_news(req)

items = (resp.data or {}).get("news", [])
token = getattr(resp, "next_page_token", None)

print(f"Page 1: {len(items)} articles")
print(f"next_page_token: {repr(token)}")
print(f"resp.data keys: {list((resp.data or {}).keys())}")

# Check all top-level keys on resp.data for a pagination hint
if resp.data:
    for k, v in resp.data.items():
        if k != "news":
            print(f"  resp.data['{k}'] = {repr(v)[:100]}")

# Try page 2 if token exists
if token:
    req2  = NewsRequest(start=start, end=end, limit=50, sort="desc",
                        page_token=token, include_content=False, exclude_contentless=True)
    resp2 = client.get_news(req2)
    items2 = (resp2.data or {}).get("news", [])
    print(f"\nPage 2: {len(items2)} articles")
    print(f"next_page_token: {repr(getattr(resp2, 'next_page_token', None))}")
else:
    print("\nNo next_page_token — checking if total available > 50:")
    # Try with a smaller limit to see if there are more
    req_small = NewsRequest(start=start, end=end, limit=5, sort="desc",
                            include_content=False, exclude_contentless=True)
    resp_small = client.get_news(req_small)
    small_items = (resp_small.data or {}).get("news", [])
    small_token = getattr(resp_small, "next_page_token", None)
    print(f"  limit=5 returned {len(small_items)} items, next_page_token={repr(small_token)}")