"""
news_call_options_test.py — the live trade review (2026-07-16) showed news_call's losses are the
OPTIONS WRAPPER (median ~10.6% round-trip spread + overnight IV crush on Δ0.40/DTE~10 calls), not the
signal: the same signals expressed as stock ran ≈flat, and even perfect exits couldn't go meaningfully
positive. The PEAD-options sweep (pead_options_test.py) already proved geometry is decisive — ATM/short
loses to theta+crush, ITM/longer-DTE (Δ0.70/45d) flips broad-positive. This is the SAME sweep for the
NEWS_CALL population (non-earnings bullish, live dyn gate): current live geometry vs delta × DTE grid,
plus the stock expression baseline. NET of costs (Black-Scholes + IV-crush + spreads).

Per the pead lesson: judge cells on MEDIAN + win% (robust), not totals (tail-skewed).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/news_call_options_test.py   (run from data/)
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
import exit_sweep_unified as es
import config as cfg
import bot

HOLD = cfg.NEWS_CALL_MAX_HOLD_DAYS          # 3d — the live news_call time cap
STOCK_HOLD = cfg.NEWS_STOCK_MAX_HOLD_DAYS   # 5d — the live stock-leg cap


def row(label, s):
    med = es.pctl([p for p in s["pnls"]], 50) if s["pnls"] else 0
    print(f"  {label:16}{s['n']:>5}{s['total']:>+11,.0f}{s['mean']:>+8.0f}{med:>+9.0f}"
          f"{s['win']:>6.0f}{s['sharpe']:>+8.2f}{s['x2']:>5}")
    return s


def main():
    _, end, days, cache = es.BULL
    rows = es.load_signals(cache, end, days, es.PROMPT, want_ticker=True)
    news = [r for r in rows if not bot._is_earnings_headline(r["headline"])]
    reg = es.build_regime(end, days, 200)
    gate = es.dyn_gate
    print(f"news_call options test — {len(news)} non-earnings bullish signals · {HOLD}d hold · NET of costs")
    print(f"(judge on median + win%, not total — totals are tail-skewed; see pead_options_test.py)")

    print(f"\n  {'expression':16}{'n':>5}{'total':>11}{'$/tr':>8}{'med$/tr':>9}{'win%':>6}{'sharpe':>8}{'2x+':>5}")

    stock = row(f"STOCK {STOCK_HOLD}d/10%", es.stat(es.sim_stock(news, reg, cfg.STOCK_TRAIL_PCT
                if hasattr(cfg, "STOCK_TRAIL_PCT") else 0.10, STOCK_HOLD, gate)))

    live = row("Δ0.40/10d LIVE", es.stat(es.sim_option(news, reg, 0.40, HOLD, gate, 0.40, 10,
                                                        cfg.MAX_POSITION_USD)))

    print(f"\n  ── geometry sweep (40% premium trail, {HOLD}d hold) ──")
    for dte in (21, 30, 45):
        for delta in (0.50, 0.60, 0.70, 0.80):
            row(f"Δ{delta:.2f}/{dte}d", es.stat(es.sim_option(news, reg, 0.40, HOLD, gate, delta, dte,
                                                               cfg.MAX_POSITION_USD)))

    # The winning PEAD geometry also benefited from a hold long enough for the drift; try the news
    # signals with a longer leash at the ITM cell (does the news edge extend past 3d, or decay?)
    print(f"\n  ── ITM Δ0.70/45d at longer holds (does the news edge extend or decay?) ──")
    for hold in (3, 5, 10):
        row(f"Δ0.70/45d {hold}d", es.stat(es.sim_option(news, reg, 0.40, hold, gate, 0.70, 45,
                                                         cfg.MAX_POSITION_USD)))


if __name__ == "__main__":
    main()
