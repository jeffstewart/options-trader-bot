"""
scorer_pnl_matched.py — compare scorers on P&L at a MATCHED NUMBER OF TRADES.

jeff's point (2026-07-29): the existing comparison gates every model at mag>=0.35 AND conf>=0.70,
but those numbers were calibrated to sonnet5's scale. At that fixed cutoff kimi admits 272 articles,
sonnet5 admits 89 and ollama admits 495 -- so any P&L difference conflates "ranks articles better"
with "happens to put its numbers where our threshold sits". Selectivity and skill get mixed
together and neither is measurable.

THE FIX: rank each model's own bullish signals by its own composite score, then walk down that
ranking until exactly N TRADES have been simulated. Every model is then judged on its top-N picks,
whatever raw numbers it used to express them. Differences at matched N are ranking skill.

WHY "N TRADES" AND NOT "TOP N SIGNALS": the simulator drops rows for reasons unrelated to the
scorer -- no price data at that timestamp, invalid/delisted ticker, a same-day duplicate on the
same underlying, an unaffordable contract. Taking the top 100 signals can yield 60 trades for one
model and 85 for another, silently un-matching the comparison. So we truncate on realised trades.

Ranking scalar defaults to magnitude x confidence: a single value that respects both fields the
live gate uses, and the same shape v1's scale_position_usd weights position size by. `--rank mag`
and `--rank conf` are available as robustness checks -- if the verdict flips between them, the
result is a ranking artifact and should not be trusted.

Geometry is identical to lotto_sonnet5_selectivity_grid.py (deep-OTM delta 0.20, DTE 14, 7-day max
hold, tiered-profit exit, $100 budget) so numbers are comparable with that script's output.

Usage (run from data/):
    USE_YAHOO_BARS=1 python ../research/scorer_pnl_matched.py
    USE_YAHOO_BARS=1 python ../research/scorer_pnl_matched.py --rank mag --n 50 100
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as _bt              # noqa: E402
import config as _cfg               # noqa: E402
import news_call_sweep_unified as nc  # noqa: E402
import small_account_sonnet5_sweep as s5  # noqa: E402
from benchmark import compute_stats  # noqa: E402
from regime_filter import build_regime  # noqa: E402

DATA_DIR = Path("/Users/jeff/Claude/Trader/data")
BUDGET = 100
DEFAULT_NS = [25, 50, 100, 150]

# (label, cache file, key prefix). All three were scored on the SAME 1,683-article universe with
# the SAME anthropic_scorer prompt, except ollama:unified_v1 which is kept as the incumbent
# reference -- it is the configuration v2 actually ran, on a different prompt.
SOURCES = [
    ("kimi-k2.6",        "kimi_backtest_cache.json",     "k2.6"),
    ("sonnet5",          "anthropic_backtest_cache.json", "sonnet5"),
    ("ollama-t0.1-ap",   "ollama_temp_cache.json",       "t01"),
    ("ollama-t1.0-ap",   "ollama_temp_cache.json",       "t10"),
]


def load_rows(cache_file: str, prefix: str, end_dt, days: int) -> list[dict]:
    """Bullish rows in the shape the simulator wants, joined to dual_score_cache for the article
    timestamp. Mirrors small_account_sonnet5_sweep.load_sonnet5_rows so every source is built the
    same way -- the loader must not be a source of difference."""
    cache = json.loads((DATA_DIR / cache_file).read_text())
    dual = json.loads((DATA_DIR / "dual_score_cache.json").read_text())
    start_dt = end_dt - timedelta(days=days)
    rows = []
    for k, v in cache.items():
        p, _, ck = k.partition(":")
        if p != prefix or not v or v.get("sent") != "bullish":
            continue
        dv = dual.get(ck)
        if not isinstance(dv, dict):
            continue
        ca = (dv.get("_article", {}) or {}).get("created_at")
        if not ca:
            continue
        try:
            created_at = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start_dt <= created_at <= end_dt):
            continue
        tickers = [t for t in (v.get("tick") or [])
                   if isinstance(t, str) and t not in ("BTC", "ETH")]
        if not tickers:
            continue
        try:
            mag, conf = float(v.get("mag") or 0), float(v.get("conf") or 0)
        except (TypeError, ValueError):
            continue
        # llama at t=1.0 emitted nan confidences and a magnitude of -0.18; a nan silently sorts
        # unpredictably and would corrupt the ranking, so drop non-finite/out-of-schema rows.
        if not (0.0 <= mag <= 1.0 and 0.0 <= conf <= 1.0):
            continue
        rows.append({"created_at": created_at, "magnitude": mag, "confidence": conf,
                     "tickers": tickers})
    rows.sort(key=lambda r: r["created_at"])
    return rows


def load_unified_rows(prompt: str, end_dt, days: int, restrict: set) -> list[dict]:
    """unified_scores.json rows, restricted to a given article-hash set. The restriction is the
    whole point in prompt-A/B mode: unified_v1 covers 8,275 articles while the anthropic-prompt arm
    covers the 1,683-article universe, and an unrestricted unified_v1 would bring a 4x larger
    candidate pool to a comparison that is supposed to differ only in prompt wording."""
    uni = json.loads((DATA_DIR / "unified_scores.json").read_text())
    dual = json.loads((DATA_DIR / "dual_score_cache.json").read_text())
    start_dt = end_dt - timedelta(days=days)
    rows = []
    for k, v in uni.items():
        pfx, _, ck = k.partition(":")
        if pfx != prompt or not v or v.get("sentiment") != "bullish":
            continue
        if restrict and ck not in restrict:
            continue
        dv = dual.get(ck)
        if not isinstance(dv, dict):
            continue
        ca = (dv.get("_article", {}) or {}).get("created_at")
        if not ca:
            continue
        try:
            created_at = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start_dt <= created_at <= end_dt):
            continue
        tickers = [t for t in (v.get("tickers") or [])
                   if isinstance(t, str) and t not in ("BTC", "ETH")]
        if not tickers:
            continue
        try:
            mag, conf = float(v.get("magnitude") or 0), float(v.get("confidence") or 0)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= mag <= 1.0 and 0.0 <= conf <= 1.0):
            continue
        rows.append({"created_at": created_at, "magnitude": mag,
                     "confidence": conf, "tickers": tickers})
    rows.sort(key=lambda r: r["created_at"])
    return rows


def simulate_top_n(rows: list[dict], reg, target_n: int, rank: str) -> list[dict]:
    """Walk the score-ranked list, simulating until target_n TRADES exist (not target_n signals).
    Returns fewer than target_n only if the model runs out of tradeable signals."""
    key = {"magconf": lambda r: r["magnitude"] * r["confidence"],
           "mag":     lambda r: r["magnitude"],
           "conf":    lambda r: r["confidence"]}[rank]
    ranked = sorted(rows, key=key, reverse=True)
    trades, seen = [], set()
    for r in ranked:
        if len(trades) >= target_n:
            break
        d = r["created_at"].date()
        if reg and not reg(d):
            continue
        tk = r["tickers"][0]
        if not _bt.is_valid_stock_ticker(tk):
            continue
        dedupe = f"{d}_{tk}"
        if dedupe in seen:
            continue
        seen.add(dedupe)
        sp = _bt.get_price_at(tk, r["created_at"])
        if not sp:
            continue
        t = _bt.simulate_option_pnl(tk, r["created_at"], sp, BUDGET,
                                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                    option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
        if t:
            trades.append(t)
    return trades


def run_tables(loaded, reg, args):
    # Preserve and patch the simulator globals to the lotto geometry, exactly as
    # lotto_sonnet5_selectivity_grid.py does, so results are comparable across scripts.
    tiered = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    save_mult = _cfg.MAX_CONTRACT_BUDGET_MULT
    try:
        _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = 0.20, 14, 7
        _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, tiered
        _cfg.MAX_CONTRACT_BUDGET_MULT = 1.0

        for target_n in args.n:
            print(f"\n{'=' * 88}\nMATCHED AT n={target_n} TRADES · lotto Δ0.20/DTE14 · "
                  f"${BUDGET} budget · ranked by {args.rank}\n{'=' * 88}")
            # `pool` and `pctile` are the fairness controls. Matching trade COUNT still lets a
            # promiscuous model be more selective: taking the top 25 from a 1,130-signal pool is
            # the top 2%, from a 270-signal pool it is the top 9%. On a tail-driven strategy the
            # bigger pool simply gets more lottery tickets, which can masquerade as better ranking.
            # The bootstrap CI on total$ shows whether any gap survives resampling at all.
            print(f"{'model':<22} {'n':>4} {'pool':>6} {'pctile':>7} {'total$':>9} {'win%':>6} "
                  f"{'sharpe':>7} {'avg$':>7} {'top3%':>6} {'total$ 90% CI':>22}")
            for label, rows in loaded.items():
                trades = simulate_top_n(rows, reg, target_n, args.rank)
                if not trades:
                    print(f"{label:<22} {'--':>4}  (no tradeable signals)")
                    continue
                s = compute_stats(trades)
                srt = sorted(trades, key=lambda t: -t["pnl_usd"])
                tot = sum(t["pnl_usd"] for t in srt)
                top3 = sum(t["pnl_usd"] for t in srt[:3])
                flag = "" if len(trades) >= target_n else "  ⚠️ ran out"
                pnls = [t["pnl_usd"] for t in trades]
                boot = sorted(sum(random.choices(pnls, k=len(pnls))) for _ in range(2000))
                lo, hi = boot[100], boot[1899]
                pctile = 100 * len(trades) / max(len(rows), 1)
                print(f"{label:<22} {s['trades']:>4} {len(rows):>6} {pctile:>6.1f}% "
                      f"${s['total_pnl']:>+8,.0f} {s['win_rate']:>5.1f}% {s['sharpe']:>+6.2f} "
                      f"${s['total_pnl'] / s['trades']:>+6.0f} "
                      f"{(top3 / tot * 100) if tot else 0:>5.0f}% "
                      f"[{lo:>+8,.0f},{hi:>+8,.0f}]{flag}")
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
        _cfg.MAX_CONTRACT_BUDGET_MULT = save_mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default="magconf", choices=["magconf", "mag", "conf"])
    ap.add_argument("--n", type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--prompt-ab", action="store_true",
                    help="llama3.2 unified_v1 vs anthropic_scorer prompt: same model, same "
                         "temperature 0.1, same articles, only the prompt differs")
    args = ap.parse_args()

    _, end_dt, days, _ = nc.BULL
    reg = build_regime(end_dt, days, 200)

    if args.prompt_ab:
        # Restrict BOTH arms to the articles the anthropic-prompt arm has actually scored, so
        # neither side gets extra candidates. This is the only fully controlled comparison
        # available: identical model, temperature, articles and simulator; prompt is the lone
        # free variable.
        ap_cache = json.loads((DATA_DIR / "ollama_temp_cache.json").read_text())
        scored = {k.partition(":")[2] for k, v in ap_cache.items()
                  if k.startswith("t01:") and v}
        loaded = {
            "llama/anthropic-prompt": load_rows("ollama_temp_cache.json", "t01", end_dt, days),
            "llama/unified_v1":       load_unified_rows("unified_v1", end_dt, days, scored),
        }
        print(f"\nPROMPT A/B — llama3.2, temp 0.1, restricted to the {len(scored)} articles "
              f"scored with both prompts")
        for label, rows in loaded.items():
            print(f"   {label:<24} {len(rows):>5} bullish in window")
        run_tables(loaded, reg, args)
        return

    loaded = {}
    for label, cf, prefix in SOURCES:
        if not (DATA_DIR / cf).exists():
            print(f"  (skipping {label}: {cf} not found)")
            continue
        rows = load_rows(cf, prefix, end_dt, days)
        if rows:
            loaded[label] = rows
    # The incumbent: what v2 actually ran (unified_v1 prompt), via the existing loader.
    try:
        loaded["ollama-unified_v1"] = nc.load_scored_from_unified(
            "dual_score_cache.json", end_dt, days, "unified_v1")
    except Exception as e:
        print(f"  (skipping ollama-unified_v1: {e})")

    print(f"\nBullish signals available in window ({days}d to {end_dt.date()}), ranked by {args.rank}:")
    for label, rows in loaded.items():
        print(f"   {label:<22} {len(rows):>5}")

    run_tables(loaded, reg, args)

    print("\npctile = what fraction of that model's OWN bullish pool the n trades represent. A model\n"
          "with a bigger pool is being allowed to cherry-pick harder, so compare pctile before\n"
          "reading a total$ gap as ranking skill.\n"
          "top3% is the share of total P&L from the 3 best trades — a lotto strategy is\n"
          "tail-driven by construction, so a model 'winning' on total$ with top3 near 100%\n"
          "won one lottery, not the comparison. Read win% and avg$ alongside it.")


if __name__ == "__main__":
    main()
