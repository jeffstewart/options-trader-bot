"""
scorer_divergence.py — WHEN do Ollama (llama3.2) and Groq (llama-3.3-70b) diverge on the
SAME headline? Reads scorer_ab.csv (the live A/B log). Not a quality verdict (too little data
yet) — a PICTURE of where they differ, with the decision-relevant angle front and center:
how often the model choice would FLIP an actual trade.

Usage:  .venv/bin/python scorer_divergence.py
"""
import csv, statistics
from pathlib import Path
import config as cfg


def passes_trade_gate(sent, mag, conf):
    """The live long-trade gate: bullish AND mag≥MIN AND conf≥dynamic floor."""
    return (sent == "bullish" and mag >= cfg.MIN_MAGNITUDE
            and conf >= cfg.BASE_CONFIDENCE + (1 - mag) * cfg.CONFIDENCE_SLOPE)


def lotto_gate(sent, mag, conf):
    return sent == "bullish" and mag >= cfg.LOTTO_MIN_MAGNITUDE and conf >= cfg.LOTTO_MIN_CONFIDENCE


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = []
    with open("scorer_ab.csv") as fh:
        for r in csv.DictReader(fh):
            if r["groq_status"] != "ok" or not r["ollama_sentiment"] or not r["groq_sentiment"]:
                continue
            om, gm = f(r["ollama_mag"]), f(r["groq_mag"])
            oc, gc = f(r["ollama_conf"]), f(r["groq_conf"])
            if None in (om, gm, oc, gc):
                continue
            rows.append({**r, "om": om, "gm": gm, "oc": oc, "gc": gc})
    n = len(rows)
    print(f"═══ SCORER DIVERGENCE — Ollama(llama3.2) vs Groq(llama-3.3-70b) ═══")
    print(f"comparable rows (both scored OK): {n}\n")
    if n < 10:
        print("  (too few rows)"); return

    sents = ["bullish", "bearish", "neutral"]
    norm = lambda s: s if s in sents else "neutral"

    # ── 1. Sentiment confusion matrix ──────────────────────────────────────
    print("  [1] SENTIMENT confusion  (rows=Ollama, cols=Groq):")
    print(f"      {'ollama\\groq':<12}" + "".join(f"{s[:7]:>9}" for s in sents) + f"{'row tot':>9}")
    agree = 0
    for os_ in sents:
        cells = []
        for gs in sents:
            c = sum(1 for r in rows if norm(r["ollama_sentiment"]) == os_ and norm(r["groq_sentiment"]) == gs)
            cells.append(c)
            if os_ == gs:
                agree += c
        print(f"      {os_:<12}" + "".join(f"{c:>9}" for c in cells) + f"{sum(cells):>9}")
    print(f"      → sentiment agreement: {agree}/{n} = {agree/n*100:.0f}%   "
          f"(disagree on {n-agree})\n")

    # ── 2. Magnitude / confidence divergence ───────────────────────────────
    dmag = [abs(r["om"] - r["gm"]) for r in rows]
    dconf = [abs(r["oc"] - r["gc"]) for r in rows]
    print("  [2] SCORE divergence (where both gave numbers):")
    print(f"      |Δmagnitude|  mean={statistics.mean(dmag):.3f}  median={statistics.median(dmag):.3f}  "
          f"max={max(dmag):.2f}")
    print(f"      |Δconfidence| mean={statistics.mean(dconf):.3f}  median={statistics.median(dconf):.3f}  "
          f"max={max(dconf):.2f}")
    print(f"      Groq runs {'HIGHER' if statistics.mean(r['gm'] for r in rows) > statistics.mean(r['om'] for r in rows) else 'LOWER'} "
          f"magnitude on avg (ollama {statistics.mean(r['om'] for r in rows):.2f} vs groq {statistics.mean(r['gm'] for r in rows):.2f})\n")

    # ── 3. THE decision-relevant one: would the model choice FLIP a trade? ──
    flips = [r for r in rows
             if passes_trade_gate(r["ollama_sentiment"], r["om"], r["oc"])
             != passes_trade_gate(r["groq_sentiment"], r["gm"], r["gc"])]
    o_only = [r for r in flips if passes_trade_gate(r["ollama_sentiment"], r["om"], r["oc"])]
    g_only = [r for r in flips if passes_trade_gate(r["groq_sentiment"], r["gm"], r["gc"])]
    o_trades = sum(1 for r in rows if passes_trade_gate(r["ollama_sentiment"], r["om"], r["oc"]))
    g_trades = sum(1 for r in rows if passes_trade_gate(r["groq_sentiment"], r["gm"], r["gc"]))
    print("  [3] TRADE-DECISION divergence (live long gate: bullish & mag≥.35 & dyn-conf):")
    print(f"      Ollama would trade {o_trades};  Groq would trade {g_trades}")
    print(f"      DISAGREE on {len(flips)}/{n} ({len(flips)/n*100:.0f}%) — "
          f"Ollama-only {len(o_only)}, Groq-only {len(g_only)}")
    lf = [r for r in rows
          if lotto_gate(r["ollama_sentiment"], r["om"], r["oc"])
          != lotto_gate(r["groq_sentiment"], r["gm"], r["gc"])]
    print(f"      (lotto gate .70/.85: flips on {len(lf)})\n")

    # ── 4. show the actual flip headlines ──────────────────────────────────
    if flips:
        print("  [4] the trade-flip headlines (what each model saw differently):")
        for r in flips[:14]:
            o = f"O:{r['ollama_sentiment'][:4]} m{r['om']:.2f} c{r['oc']:.2f}"
            g = f"G:{r['groq_sentiment'][:4]} m{r['gm']:.2f} c{r['gc']:.2f}"
            who = "ollama→TRADE" if passes_trade_gate(r["ollama_sentiment"], r["om"], r["oc"]) else "groq→TRADE"
            print(f"      [{who:11}] {o} | {g}  «{r['headline'][:62]}»")


if __name__ == "__main__":
    main()
