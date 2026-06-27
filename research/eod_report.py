"""
eod_report.py — end-of-day recap of the trading bot. Run after the market close.
Gathers: account P&L vs SPY/QQQ, news_call's live fills (the leg we're watching), positions
closed today, the stock-SHADOW leg's performance vs index, and cap/position usage.
Self-contained → a scheduled session just runs this and relays the output.

Usage:  cd /Users/jeff/Claude/Trader && .venv/bin/python eod_report.py
"""
import os, csv
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta, date
from collections import Counter
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")
from alpaca.trading.client import TradingClient
import yahoo_data
import config as c

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
TODAY = date.today().isoformat()


def hdr(t): print(f"\n── {t} " + "─" * max(0, 56 - len(t)))


def _idx_today(sym):
    bars = yahoo_data.get_yahoo_bars(sym, datetime.now(timezone.utc) - timedelta(days=8),
                                     datetime.now(timezone.utc) + timedelta(days=1))
    return (bars[-1]["c"] / bars[-2]["c"] - 1) * 100 if len(bars) >= 2 else None


def account_section():
    hdr("ACCOUNT & vs MARKET (today)")
    a = tc.get_account()
    eq, le = float(a.equity), float(a.last_equity)
    pnl = eq - le; pct = pnl / le * 100 if le else 0.0
    print(f"  equity ${eq:,.2f}   cash ${float(a.cash):,.2f}   today P&L ${pnl:+,.2f} ({pct:+.2f}%)")
    for sym in ("SPY", "QQQ"):
        r = _idx_today(sym)
        if r is not None:
            print(f"  {sym} today {r:+.2f}%   → account vs {sym}: {pct - r:+.2f}%")


def trades_today():
    hdr("REAL TRADES OPENED TODAY")
    rows = [r for r in csv.DictReader(open("trades.csv")) if r["timestamp"][:10] == TODAY]
    print(f"  {len(rows)} real entries today   by strategy: {dict(Counter(r['strategy'] for r in rows))}")
    nc = [r for r in rows if r["strategy"] == "news_call"]
    print(f"  ⭐ news_call LIVE fills today: {len(nc)}" + ("  (still none — watch the cap/signal flow)" if not nc else ""))
    for r in nc:
        print(f"     {r['timestamp'][11:16]} {r['source_ticker']:6} {r.get('option_symbol','')}  qty{r['qty']}  conf={r['confidence']} mag={r['magnitude']}")


def closes_today():
    hdr("POSITIONS CLOSED TODAY")
    rows = [r for r in csv.DictReader(open("closed_trades.csv")) if r["timestamp"][:10] == TODAY]
    tot = sum(float(r["pnl_usd"]) for r in rows)
    print(f"  {len(rows)} closed   realized ${tot:+,.2f}   reasons: {dict(Counter(r['reason'] for r in rows))}")
    for r in sorted(rows, key=lambda x: float(x["pnl_usd"]))[:6]:
        print(f"     {r['strategy']:9} {r['underlying']:6} ${float(r['pnl_usd']):+,.0f} ({r['reason']})")


def shadow_stock_section():
    hdr("STOCK SHADOW (paper) — activity + vs index")
    try:
        rows = [r for r in csv.DictReader(open("shadow_trades.csv")) if r.get("strategy") == "stock_shadow"]
    except FileNotFoundError:
        rows = []
    opened_today = [r for r in rows if r.get("action") == "open" and r["timestamp"][:10] == TODAY]
    print(f"  stock-shadow opened today: {len(opened_today)}   (all-time stock-shadow rows: {len(rows)})")
    try:
        import stock_vs_index_live as sv
        sh = sv.shadow_stock_trades()
        if sh:
            start = min(t["entry_date"] for t in sh); end = max(t["exit_date"] for t in sh)
            spy, qqq = sv.load_index("SPY", start, end), sv.load_index("QQQ", start, end)
            sv.compare(sh, "stock SHADOW vs index", spy, qqq)
        else:
            print("  (no completed shadow stock trades to compare yet)")
    except Exception as e:
        print(f"  [shadow compare error: {e}]")


GATE_WAIT_TIMEOUT_MS = c.GATE_WAIT_TIMEOUT_SECS * 1000   # auto-synced to the live gate wait_for


def gate_section():
    """Confirm/veto gate reliability today. KEY: a row whose gemini_ms exceeds the bot's 12s wait_for
    was an ACTED-ON fallback even if the gate later returned 'confirm' (the bot gave up at 12s) — the
    decision column alone undercounts these. Fallbacks now matter more: in an UPTREND the bot trades on
    Ollama alone, but a DOWNTREND high-conviction bypass with no confirm is SKIPPED (2026-06-25 change)."""
    hdr("CONFIRM/VETO GATE (today)")
    try:
        rows = [r for r in csv.DictReader(open("gemini_decisions.csv")) if r["timestamp"][:10] == TODAY]
    except FileNotFoundError:
        print("  (no gemini_decisions.csv)"); return
    if not rows:
        print("  no gate calls today (no real-leg-eligible bullish signals)"); return
    confirms = vetoes = timeouts = other_fb = 0
    lats = []
    for r in rows:
        gms = float(r.get("gemini_ms") or 0)
        dec = r.get("decision", "")
        if gms > 0:
            lats.append(gms)
        if gms > GATE_WAIT_TIMEOUT_MS:          # bot's wait_for gave up → fallback, regardless of verdict
            timeouts += 1
        elif dec == "fallback":                 # cap / rate-limit / error → no usable verdict
            other_fb += 1
        elif dec == "confirm":
            confirms += 1
        elif dec == "veto":
            vetoes += 1
    n = len(rows); fb = timeouts + other_fb
    print(f"  {n} gate calls   {confirms} confirm / {vetoes} veto / {fb} fallback ({fb / n * 100:.0f}% fallback)")
    if fb:
        print(f"     └ fallbacks: {timeouts} timeout(>{GATE_WAIT_TIMEOUT_MS // 1000}s) · {other_fb} cap/ratelimit/error")
    if lats:
        lats.sort()
        p50 = lats[len(lats) // 2]; p90 = lats[min(len(lats) - 1, int(len(lats) * 0.9))]
        print(f"     latency: median {p50 / 1000:.1f}s · p90 {p90 / 1000:.1f}s · max {max(lats) / 1000:.1f}s  "
              f"(wait_for {GATE_WAIT_TIMEOUT_MS // 1000}s — fallbacks climb as p90 nears it)")


def cap_section():
    hdr("CAP / POSITIONS")
    pos = tc.get_all_positions()
    print(f"  open positions: {len(pos)} / {c.MAX_OPEN_POSITIONS} cap → {c.MAX_OPEN_POSITIONS - len(pos)} free slots")


def health_section():
    """Unattended-run health: are the daemons up, and did the watchdog have to intervene today?
    Lets a scan of N days of EOD reports surface crashes/restarts without digging logs."""
    hdr("BOT HEALTH (unattended)")
    for n in ("bot", "dashboard", "resume_after_reset"):
        try:
            pid = int(open(c.DATA_DIR / f"{n}.pid").read().strip())
            os.kill(pid, 0); st = "UP"
        except Exception:
            st = "DOWN"
        print(f"  {n:20} {st}")
    try:
        lines = open(c.LOG_DIR / "watchdog.log").read().splitlines()
        today = [l for l in lines if l.startswith(f"[{TODAY}")]
        restarts = [l for l in today if "restarting" in l]
        rotations = [l for l in today if "rotated" in l]
        print(f"  watchdog today: {len(restarts)} restart(s), {len(rotations)} log rotation(s)")
        for l in restarts[-4:]:
            print(f"     {l}")
    except Exception:
        print("  watchdog log: (none yet)")


def main():
    print(f"════════════ EOD REPORT — {TODAY} ════════════")
    for fn in (health_section, account_section, trades_today, closes_today, gate_section, shadow_stock_section, cap_section):
        try:
            fn()
        except Exception as e:
            print(f"  [section error: {e}]")
    print("\n════════════ END ════════════")


if __name__ == "__main__":
    main()
