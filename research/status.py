"""
status.py — one-command, on-demand snapshot of the live bot. Everything you'd want to glance at without
digging: system health, account + open positions, today's activity, multi-day P&L, gate health, and —
key for an unattended run — DATA-CAPTURE FRESHNESS (is each file still being written?), Groq-backtest
coverage, and shadow counts. Reuses the EOD report's sections + adds the operational ones.

Run:  ./manage.sh report      (or, from data/:  USE_YAHOO_BARS=1 ../.venv/bin/python ../research/status.py)
"""
import os, csv, json, subprocess, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone
from collections import defaultdict, Counter
import config as c
import eod_report as eod          # reuse account/trades/closes/gate/cap sections + tc + TODAY + hdr

hdr, TODAY, tc = eod.hdr, eod.TODAY, eod.tc


def _age_str(path):
    try:
        m = (time.time() - os.path.getmtime(path)) / 60.0
        return f"{m:.0f}m ago" if m < 1440 else f"{m/1440:.1f}d ago"
    except Exception:
        return "?"


def system_section():
    hdr("SYSTEM")
    for n in ("bot", "dashboard", "resume_after_reset", "groq_backtest"):
        try:
            pid = int(open(c.DATA_DIR / f"{n}.pid").read().strip())
            os.kill(pid, 0)
            up = subprocess.run(["ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            print(f"  {n:20} UP   (uptime {up})")
        except Exception:
            print(f"  {n:20} DOWN")
    caf = subprocess.run(["pgrep", "-x", "caffeinate"], capture_output=True).returncode == 0
    print(f"  {'caffeinate':20} {'UP (no-sleep)' if caf else 'MISSING — Mac can sleep!'}")
    try:
        df = subprocess.run(["df", "-h", str(c.BASE_DIR)], capture_output=True, text=True).stdout.splitlines()[1].split()
        print(f"  {'disk':20} {df[3]} free ({df[4]} used)")
    except Exception:
        pass
    try:
        today = [l for l in open(c.LOG_DIR / "watchdog.log") if l.startswith(f"[{TODAY}")]
        print(f"  {'watchdog today':20} {sum('restarting' in l for l in today)} restart(s), {sum('rotated' in l for l in today)} rotation(s)")
    except Exception:
        print(f"  {'watchdog today':20} (no log yet)")
    try:
        bk = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/TraderBackup/LAST_BACKUP.txt")
        print(f"  {'last iCloud backup':20} {open(bk).read().strip()}")
    except Exception:
        print(f"  {'last iCloud backup':20} (none yet)")


def positions_section():
    hdr("OPEN POSITIONS")
    pos = tc.get_all_positions()
    if not pos:
        print("  (none)"); return
    for p in sorted(pos, key=lambda x: float(x.unrealized_pl or 0)):
        print(f"  {p.symbol:24} {p.side.value:5} qty {str(p.qty):>5}  "
              f"uPL ${float(p.unrealized_pl or 0):>+8.2f} ({float(p.unrealized_plpc or 0)*100:>+5.1f}%)")


def performance_section():
    hdr("REALIZED P&L by strategy (7d / all-time)")
    rows = list(csv.DictReader(open("closed_trades.csv")))
    cutoff = datetime.now(timezone.utc).timestamp() - 7 * 86400
    def isrecent(r):
        try:
            return datetime.fromisoformat(r["timestamp"]).timestamp() >= cutoff
        except Exception:
            return False
    def agg(rs):
        d = defaultdict(list)
        for r in rs:
            try:
                d[r["strategy"]].append(float(r["pnl_usd"]))
            except Exception:
                pass
        return d
    a7, aa = agg([r for r in rows if isrecent(r)]), agg(rows)
    for s in sorted(aa, key=lambda k: sum(aa[k])):
        p7, pa = a7.get(s, []), aa[s]
        win = sum(1 for x in pa if x > 0) / len(pa) * 100
        s7 = f"7d ${sum(p7):>+6.0f}/{len(p7):<2}" if p7 else "7d      —  "
        print(f"  {s:12} {s7}   all ${sum(pa):>+7.0f}/{len(pa):<3} win {win:>3.0f}%")
    tot = sum(x for p in aa.values() for x in p)
    print(f"  {'TOTAL':12} {'':14}   all ${tot:>+7.0f}")


def data_freshness_section():
    hdr("DATA CAPTURE (rows · last write) — confirm it's still collecting")
    for f in ("trades.csv", "closed_trades.csv", "position_paths.csv", "regime_decisions.csv",
              "gemini_decisions.csv", "shadow_trades.csv", "scorer_ab.csv"):
        if not os.path.exists(f):
            print(f"  {f:22} —      (not created yet)")
        else:
            rows = max(0, sum(1 for _ in open(f)) - 1)
            print(f"  {f:22} {rows:>6} rows · {_age_str(f)}")


def backtest_section():
    hdr("GROQ primary-scorer backtest coverage")
    try:
        d = json.load(open("groq_primary_backtest_cache.json"))
        cnt = Counter(k.split(":", 1)[0] for k in d if d[k])
        for m in ("llama-3.3-70b", "gpt-oss-20b", "gpt-oss-120b"):
            print(f"  {m:16} {cnt.get(m, 0):>4}/1872 ({cnt.get(m, 0) / 1872 * 100:.0f}%)")
    except Exception as e:
        print(f"  [error: {e}]")


def shadows_section():
    hdr("SHADOW captures (paper counterfactuals)")
    try:
        rows = list(csv.DictReader(open("shadow_trades.csv")))
        opened = Counter(r["strategy"] for r in rows if r.get("action") == "open")
        for s, n in sorted(opened.items(), key=lambda x: -x[1]):
            print(f"  {s:24} {n} opened")
        if not opened:
            print("  (none yet)")
    except Exception:
        print("  (none yet)")


def main():
    print(f"════════════ TRADER STATUS — {datetime.now():%Y-%m-%d %H:%M} ════════════")
    try:
        print(f"  market: {'OPEN' if tc.get_clock().is_open else 'closed'}")
    except Exception:
        pass
    for fn in (system_section, eod.account_section, positions_section, eod.trades_today,
               eod.closes_today, performance_section, eod.gate_section, data_freshness_section,
               backtest_section, shadows_section, eod.cap_section):
        try:
            fn()
        except Exception as e:
            print(f"  [section error in {fn.__name__}: {e}]")
    print("\n════════════ END ════════════")


if __name__ == "__main__":
    main()
