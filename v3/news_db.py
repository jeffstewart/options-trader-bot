"""
news_db.py (v3) — embedded SQLite news archive and context retrieval.
Features:
  - Fast, zero-dependency embedded database via standard sqlite3 with WAL mode.
  - Ingests and indexes incoming articles and their associated ticker symbols.
  - Point-in-time ticker history retrieval (strictly before as_of_ts, zero lookahead bias).
  - Trailing market-wide sentiment aggregation and macro headline extraction.
  - Startup / backtest news backfill from Alpaca REST API.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import requests

try:
    from v3 import config as cfg
except ImportError:
    import config as cfg

log = logging.getLogger(__name__)


def _get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    path = Path(db_path or cfg.NEWS_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize database tables and indexes."""
    with _get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                headline TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                url TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_created ON articles(created_at);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_symbols (
                article_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                PRIMARY KEY (article_id, symbol),
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON article_symbols(symbol);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_scores (
                article_id TEXT PRIMARY KEY,
                scored_at TEXT NOT NULL,
                model TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL,
                magnitude REAL,
                reasoning TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
            );
        """)
        conn.commit()


def store_article(item: dict, db_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Store an article and its associated symbols.
    Returns True if newly inserted, False if already exists or invalid.
    """
    article_id = str(item.get("id") or item.get("article_id") or "")
    headline = (item.get("headline") or "").strip()
    if not headline:
        return False

    raw_ts = item.get("created_at") or item.get("updated_at")
    if isinstance(raw_ts, datetime):
        ts_str = raw_ts.astimezone(timezone.utc).isoformat()
    elif isinstance(raw_ts, str):
        ts_str = raw_ts.replace("Z", "+00:00")
    else:
        ts_str = datetime.now(timezone.utc).isoformat()

    if not article_id:
        import hashlib
        article_id = hashlib.sha256(f"{ts_str}_{headline}".encode("utf-8")).hexdigest()[:16]

    summary = (item.get("summary") or item.get("body") or "").strip()
    source = item.get("source") or "Alpaca/Benzinga"
    url = item.get("url") or ""
    symbols = [s.strip().upper() for s in (item.get("symbols") or []) if s and isinstance(s, str)]

    try:
        with _get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO articles (article_id, created_at, updated_at, headline, summary, source, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (article_id, ts_str, ts_str, headline, summary, source, url))

            if cur.rowcount == 0:
                # Article already existed
                return False

            if symbols:
                symbol_tuples = [(article_id, sym) for sym in set(symbols)]
                cur.executemany("""
                    INSERT OR IGNORE INTO article_symbols (article_id, symbol)
                    VALUES (?, ?)
                """, symbol_tuples)

            conn.commit()
            return True
    except Exception as e:
        log.warning("Failed to store article %s: %s", article_id, e)
        return False


def record_score(article_id: str, model: str, sentiment: str, confidence: float,
                 magnitude: float, reasoning: str, db_path: Optional[Union[str, Path]] = None) -> None:
    """Store an LLM score result for a given article."""
    scored_at = datetime.now(timezone.utc).isoformat()
    try:
        with _get_connection(db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO article_scores (article_id, scored_at, model, sentiment, confidence, magnitude, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (article_id, scored_at, model, sentiment, confidence, magnitude, reasoning))
            conn.commit()
    except Exception as e:
        log.debug("Failed to record score for article %s: %s", article_id, e)


def get_ticker_news_history(symbol: str, as_of_ts: Optional[datetime] = None,
                            days: int = 7, limit: int = 10,
                            db_path: Optional[Union[str, Path]] = None) -> list[dict]:
    """
    Query prior news referring to a ticker, strictly before as_of_ts.
    Returns list of dicts ordered chronologically (oldest to newest).
    """
    symbol = symbol.strip().upper()
    now_utc = as_of_ts if as_of_ts is not None else datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    start_utc = now_utc - timedelta(days=days)
    end_str = now_utc.isoformat()
    start_str = start_utc.isoformat()

    with _get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT a.article_id, a.created_at, a.headline, a.summary, a.source,
                   s.sentiment, s.confidence, s.magnitude, s.reasoning
            FROM articles a
            JOIN article_symbols sym ON a.article_id = sym.article_id
            LEFT JOIN article_scores s ON a.article_id = s.article_id
            WHERE sym.symbol = ?
              AND a.created_at < ?
              AND a.created_at >= ?
            ORDER BY a.created_at ASC
            LIMIT ?
        """, (symbol, end_str, start_str, limit)).fetchall()

        return [dict(r) for r in rows]


def get_market_sentiment_context(as_of_ts: Optional[datetime] = None, hours: int = 24,
                                 db_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Aggregate broad market sentiment and extract macro headlines over trailing window.
    """
    now_utc = as_of_ts if as_of_ts is not None else datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    start_utc = now_utc - timedelta(hours=hours)
    end_str = now_utc.isoformat()
    start_str = start_utc.isoformat()

    with _get_connection(db_path) as conn:
        # Total article count
        total_articles = conn.execute("""
            SELECT COUNT(*) AS cnt FROM articles
            WHERE created_at < ? AND created_at >= ?
        """, (end_str, start_str)).fetchone()["cnt"]

        # Scored sentiment breakdown
        score_rows = conn.execute("""
            SELECT s.sentiment, COUNT(*) as cnt
            FROM article_scores s
            JOIN articles a ON s.article_id = a.article_id
            WHERE a.created_at < ? AND a.created_at >= ?
            GROUP BY s.sentiment
        """, (end_str, start_str)).fetchall()

        sentiment_counts = {r["sentiment"]: r["cnt"] for r in score_rows}
        scored_total = sum(sentiment_counts.values())

        # Top macro / index headlines
        macro_symbols = ("SPY", "QQQ", "DIA", "IWM", "VXX", "UVXY")
        macro_placeholders = ",".join("?" for _ in macro_symbols)
        macro_rows = conn.execute(f"""
            SELECT DISTINCT a.headline, a.created_at
            FROM articles a
            JOIN article_symbols sym ON a.article_id = sym.article_id
            WHERE sym.symbol IN ({macro_placeholders})
              AND a.created_at < ? AND a.created_at >= ?
            ORDER BY a.created_at DESC
            LIMIT 5
        """, (*macro_symbols, end_str, start_str)).fetchall()

        macro_headlines = [r["headline"] for r in macro_rows]

        return {
            "total_articles": total_articles,
            "scored_total": scored_total,
            "bullish_count": sentiment_counts.get("bullish", 0),
            "bearish_count": sentiment_counts.get("bearish", 0),
            "neutral_count": sentiment_counts.get("neutral", 0),
            "macro_headlines": macro_headlines,
        }


def backfill_news_from_alpaca(api_key: Optional[str] = None, api_secret: Optional[str] = None,
                              start_ts: Optional[datetime] = None, end_ts: Optional[datetime] = None,
                              days: int = 7, db_path: Optional[Union[str, Path]] = None,
                              limit_per_page: int = 50) -> int:
    """
    Backfill historical news from Alpaca News REST API into SQLite.
    Returns total number of new articles inserted.
    """
    key = api_key or cfg.ALPACA_KEY
    secret = api_secret or cfg.ALPACA_SECRET
    if not key or not secret:
        log.warning("Alpaca API credentials missing; cannot backfill news.")
        return 0

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    now_utc = end_ts or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    if start_ts is None:
        start_utc = now_utc - timedelta(days=days)
    else:
        start_utc = start_ts
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)

    url = "https://data.alpaca.markets/v1beta1/news"
    page_token = None
    inserted_total = 0
    pages = 0

    log.info("Starting Alpaca news backfill from %s to %s ...",
             start_utc.strftime("%Y-%m-%d %H:%M"), now_utc.strftime("%Y-%m-%d %H:%M"))

    init_db(db_path)

    while True:
        params = {
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sort": "asc",
            "limit": str(limit_per_page),
            "include_content": "false",
        }
        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                log.warning("Alpaca news API error %d: %s", resp.status_code, resp.text[:200])
                break

            data = resp.json()
            items = data.get("news", [])
            if not items:
                break

            for item in items:
                if store_article(item, db_path=db_path):
                    inserted_total += 1

            pages += 1
            page_token = data.get("next_page_token")
            if not page_token:
                break

            if pages % 10 == 0:
                log.info("Backfill progress: %d pages, %d new articles saved...", pages, inserted_total)

        except Exception as e:
            log.warning("Backfill request exception: %s", e)
            break

    log.info("Alpaca news backfill complete: %d new articles stored across %d pages.", inserted_total, pages)
    return inserted_total

