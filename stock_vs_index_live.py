"""
stock_vs_index_live.py — did the stock strategy's name-selection beat just buying the index with
the same money over the same holding windows?  Reports TWO legs:
  • REAL   — the actual stock positions the bot opened (trades.csv entries → closed_trades.csv /
             Alpaca open positions).
  • SHADOW — the paper stock leg (shadow_trades.csv 'stock_shadow' closes + shadow_state.json
             open shadows). Populates going forward now that the real stock leg is shadow-only.

For each trade, the counterfactual = the SAME dollars in SPY/QQQ over the SAME entry→exit window.
SPY/QQQ from yahoo_data (Alpaca free plan 403s on recent SIP). Aggregates + how often stock won.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python stock_vs_index_live.py
"""
import os, csv, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta, date

from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")
from alpaca.trading.client import TradingClient
import yahoo_data

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)


def _ts(s):
    try: return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception: return None


def load_index(sym, start, end):
    bars = yahoo_data.get_yahoo_bars(sym, datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=6),
                                     datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=2))
    return sorted([(b["t"].date(), float(b["c"])) for b in bars])


def last_close(tk):
    try:
        bars = yahoo_data.get_yahoo_bars(tk, datetime.now(timezone.utc) - timedelta(days=8), datetime.now(timezone.utc) + timedelta(days=1))
        return float(bars[-1]["c"]) if bars else None
    except Exception:
        return None


def px_before(series, d):
    c = [v for dt, v in series if dt <= d]; return c[-1] if c else (series[0][1] if series else None)
def px_after(series, d):
    c = [v for dt, v in series if dt >= d]; return c[0] if c else (series[-1][1] if series else None)


def real_stock_trades():
    entries = []
    for r in csv.DictReader(open("trades.csv")):
        if r.get("strategy") != "stock":
            continue
        t = _ts(r.get("timestamp"))
        if not t:
            continue
        try: entries.append({"ts": t, "tk": r.get("source_ticker", ""), "qty": float(r.get("qty", 0) or 0), "px": float(r.get("cost_basis", 0) or 0)})
        except Exception: pass
    closed_pool = [{"tk": r["underlying"], "epx": float(r["entry_price"]), "xts": _ts(r["timestamp"]), "pnl": float(r["pnl_usd"])}
                   for r in csv.DictReader(open("closed_trades.csv")) if r.get("strategy") == "stock"]
    open_pos = {}
    for p in tc.get_all_positions():
        if "equity" in str(getattr(p, "asset_class", "")).lower():
            open_pos[p.symbol] = {"epx": float(p.avg_entry_price), "upl": float(p.unrealized_pl or 0)}
    used, trades, now = set(), [], datetime.now(timezone.utc)
    for e in sorted(entries, key=lambda x: x["ts"]):
        cost = e["qty"] * e["px"]; m = None
        op = open_pos.get(e["tk"])
        if op and abs(op["epx"] - e["px"]) / max(e["px"], 1e-9) < 0.02:
            m = {"status": "open", "exit_date": now.date(), "pnl": op["upl"]}
        if m is None:
            best = None
            for i, c in enumerate(closed_pool):
                if i in used or c["tk"] != e["tk"] or abs(c["epx"] - e["px"]) / max(e["px"], 1e-9) > 0.02: continue
                if c["xts"] and c["xts"] < e["ts"]: continue
                if best is None or (c["xts"] and best[1]["xts"] and c["xts"] < best[1]["xts"]): best = (i, c)
            if best:
                used.add(best[0]); m = {"status": "closed", "exit_date": best[1]["xts"].date(), "pnl": best[1]["pnl"]}
        if m: trades.append({"tk": e["tk"], "entry_date": e["ts"].date(), "cost": cost, **m})
    return trades


def shadow_stock_trades():
    if not os.path.exists("shadow_trades.csv"):
        return []
    rows = [r for r in csv.DictReader(open("shadow_trades.csv")) if r.get("strategy") == "stock_shadow"]
    opens = {}
    for i, r in enumerate(rows):
        if r.get("action") == "open":
            opens.setdefault(r["symbol"], []).append((i, _ts(r["timestamp"])))
    used, trades = set(), []
    for r in rows:
        if r.get("action") != "close":
            continue
        sym = r["symbol"]; xts = _ts(r["timestamp"]); ent = None
        for (i, ots) in opens.get(sym, []):
            if i in used or (ots and xts and ots > xts): continue
            ent = ots; used.add(i); break
        try: qty = float(r.get("qty", 0) or 0); epx = float(r.get("entry_price", 0) or 0)
        except Exception: continue
        trades.append({"tk": r["underlying"], "entry_date": (ent.date() if ent else xts.date()),
                       "exit_date": xts.date(), "cost": qty * epx, "pnl": float(r.get("pnl_usd", 0) or 0), "status": "closed"})
    # open shadows
    try:
        st = json.load(open("shadow_state.json"))
        for v in st.values():
            if not isinstance(v, dict) or v.get("strategy") != "stock_shadow": continue
            tk = v["underlying"]; epx = float(v["entry_price"]); qty = float(v["qty"]); edt = _ts(v.get("entry_dt"))
            cur = last_close(tk)
            trades.append({"tk": tk, "entry_date": (edt.date() if edt else date.today()), "exit_date": date.today(),
                           "cost": qty * epx, "pnl": ((cur - epx) * qty if cur else 0.0), "status": "open"})
    except Exception:
        pass
    return trades


def compare(trades, label, spy, qqq):
    if not trades:
        print(f"\n══ {label} ══  no trades yet"); return
    agg = {"pnl": 0.0, "SPY": 0.0, "QQQ": 0.0, "cost": 0.0}; bs = bq = 0
    for t in trades:
        agg["pnl"] += t["pnl"]; agg["cost"] += t["cost"]
        for name, series in (("SPY", spy), ("QQQ", qqq)):
            p0, p1 = px_after(series, t["entry_date"]), px_before(series, t["exit_date"])
            ip = t["cost"] * (p1 / p0 - 1) if (p0 and p1) else 0.0
            agg[name] += ip
            if name == "SPY" and t["pnl"] > ip: bs += 1
            if name == "QQQ" and t["pnl"] > ip: bq += 1
    c = agg["cost"]; n = len(trades); nc = sum(t["status"] == "closed" for t in trades)
    pc = lambda v: f"${format(v,'+,.0f')} ({v/c*100:+.2f}%)" if c else "$0"
    print(f"\n══ {label} ══  {n} trades ({nc} closed / {n-nc} open), deployed ${c:,.0f}")
    print(f"  stock {pc(agg['pnl']):>20}   SPY {pc(agg['SPY']):>20}   QQQ {pc(agg['QQQ']):>20}")
    print(f"  beat SPY {bs}/{n} ({bs/n*100:.0f}%)  beat QQQ {bq}/{n} ({bq/n*100:.0f}%)   |   "
          f"vs SPY ${agg['pnl']-agg['SPY']:+,.0f}   vs QQQ ${agg['pnl']-agg['QQQ']:+,.0f}")


def main():
    real, shadow = real_stock_trades(), shadow_stock_trades()
    allt = real + shadow
    if not allt:
        print("no stock trades (real or shadow) yet."); return
    start = min(t["entry_date"] for t in allt); end = max(t["exit_date"] for t in allt)
    spy, qqq = load_index("SPY", start, end), load_index("QQQ", start, end)
    print(f"Stock strategy vs index — window {start} → {end}")
    compare(real,   "REAL stock (actual positions)", spy, qqq)
    compare(shadow, "SHADOW stock (paper, since 2026-06-15)", spy, qqq)


if __name__ == "__main__":
    main()
