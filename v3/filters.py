"""
filters.py (v3) — deterministic pre-score filters.
Pure string/regex matching running before scoring to eliminate noise at zero LLM cost:
  - Macro and index wraps
  - Listicles and opinion roundups
  - Reactive recaps ("Why is X surging?")
  - Index inclusion / reconstitution
  - Analyst ratings and price target revisions
  - Technical chart chatter
  - Acquirer-side speculative M&A
"""

import html
import re

PRESCORE_SKIP_PATTERNS = (
    # macro / index wraps
    "stock futures", "dow futures", "nasdaq futures", "s&p futures", "futures rise", "futures fall",
    "futures point", "futures slip", "premarket", "pre-market", "market wrap", "this week in markets",
    "stock market today", "closing bell", "opening bell", "stocks close higher", "stocks close lower",
    "treasury yield", "10-year yield",
    # listicles / opinion roundups
    "stocks to watch", "stocks to buy", "stocks to avoid", "stocks to consider",
    "best stocks to", "top stocks to",
    # reactive recaps
    "what's going on with", "whats going on with", "what is going on with", "what's driving",
    # index reconstitution / inclusion
    "russell 3000", "russell 2000", "russell 1000", "russell microcap",
    "s&p 600", "s&p 400", "s&p smallcap", "s&p midcap",
    "dow jones industrial average", "nasdaq biotechnology index",
    "to join the s&p", "joins the s&p", "joins s&p", "added to the s&p",
    "to join the russell", "joins the russell", "added to the russell", "added to russell",
    "to join the nasdaq", "join the nasdaq-100", "join the dow jones", "joins the dow jones",
    # returns-calculator recap variant
    "would have made owning",
    # pundit/TV segment noise
    "live on cnbc", "posts on x", "jim cramer", "dan ives",
    # pre-event speculation
    "earnings preview", "in the spotlight ahead of", "earnings date",
)

PRESCORE_SKIP_REGEXES = (
    re.compile(r"\bif you (?:had )?invested\b", re.I),
    re.compile(r"\$[\d,]+\s+invested\s+in\b", re.I),
    re.compile(r"\d+\s+years?\s+ago\s+would\s+be\s+worth", re.I),
    re.compile(r"\boutperformed\b[^.]{0,40}\bover the (?:past|last)\s+\d+\s+(?:year|month)", re.I),
    # intraday listicle
    re.compile(r"\b\d+\s[\w\s-]{0,30}stocks moving in \w+day", re.I),
    # reactive already-moved recap questions
    re.compile(r"why is .{0,40}(stock|shares).{0,20}(surg|gain|jump|ris|soar|climb|trend|mov)", re.I),
    re.compile(r"(stock|shares)\s+(is |are )?(surg\w+|rall\w+|climb\w+|soar\w+|jump\w+|trending higher|gaining).{0,30}(what|here's why|why)", re.I),
    # acquirer-side speculative deal talk
    re.compile(r"\bin (talks|discussions) to (buy|acquire)\b", re.I),
    re.compile(r"\bweighs\b", re.I),
)

SOFT_CATALYST_PATTERNS = (
    # analyst ratings / price targets
    "price target", " pt ", "raises pt", "lowers pt", "analyst", "analysts", "upgraded",
    "downgraded", "rating to", "reiterat", "initiates coverage", "initiated coverage",
    "buy rating", "sell rating", "outperform rating", "underperform rating", "fab five",
    "their forecast", "their price", "their estimates",
    # technical / chart signals
    "trading signal", "flashed", "circuit breaker", "breakout", "death cross", "golden cross",
    "oversold", "overbought", "moving average", "chart pattern", "technical indicator",
    # acquirer-side M&A
    "to acquire", "agreed to acquire", "in discussions to buy", "in talks to buy",
    "is acquiring", "to purchase", "deal to buy", "bid for",
)


def normalize_headline(text: str) -> str:
    """Decode HTML entities + normalize curly quotes before filter matching."""
    t = html.unescape(html.unescape(text or ""))
    return (t.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"'))


def prescore_noise(headline: str) -> "str | None":
    """Return the matched noise pattern or None."""
    hn = normalize_headline(headline)
    h = hn.lower()
    sub = next((p for p in PRESCORE_SKIP_PATTERNS if p in h), None)
    if sub:
        return sub
    rx = next((r for r in PRESCORE_SKIP_REGEXES if r.search(hn)), None)
    return rx.pattern if rx else None


def soft_catalyst_hit(headline: str, body: str) -> "str | None":
    """Return the matched soft-catalyst pattern or None."""
    text = normalize_headline(f"{headline or ''} {body or ''}").lower()
    return next((p for p in SOFT_CATALYST_PATTERNS if p in text), None)

