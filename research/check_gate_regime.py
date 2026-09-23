"""
check_gate_regime.py — run this locally against the full weeks-long CSV to see whether
passes_news_call_gate ever actually varies, and whether that changes partway through
the collection window (which would indicate the collector's placement relative to
signal_checks() changed mid-collection, mixing two different sampling regimes in one file).

Usage:
    python3 check_gate_regime.py ~/Claude/Trader/data/contract_grid_snapshots.csv
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "contract_grid_snapshots.csv"

df = pd.read_csv(path, low_memory=False, parse_dates=["signal_ts"],
                  usecols=["signal_id", "signal_ts", "magnitude", "confidence",
                           "passes_news_call_gate", "passes_lotto_gate"])
sig = df.groupby("signal_id").first().sort_values("signal_ts")
sig["date"] = sig["signal_ts"].dt.date
sig["mag_x_conf"] = sig["magnitude"] * sig["confidence"]

print(f"n signals: {len(sig)}")
print(f"\npasses_news_call_gate overall: {sig['passes_news_call_gate'].value_counts().to_dict()}")
print(f"passes_lotto_gate overall:     {sig['passes_lotto_gate'].value_counts().to_dict()}")

print("\nPer-day breakdown (watch for a day where False first appears/disappears -- that's")
print("the signature of a mid-collection change in what gets logged):")
daily = sig.groupby("date").agg(
    n=("magnitude", "size"),
    pct_news_call_false=("passes_news_call_gate", lambda s: (~s.astype(bool)).mean()),
    min_mag_x_conf=("mag_x_conf", "min"),
    max_mag_x_conf=("mag_x_conf", "max"),
    min_magnitude=("magnitude", "min"),
    min_confidence=("confidence", "min"),
)
with pd.option_context("display.max_rows", None, "display.width", 140):
    print(daily.to_string())

print("\nIf pct_news_call_false is 0.0 on every single day, the two thresholds (GRID_LOG_THRESHOLD")
print("vs. the news_call_ok boundary) may simply not diverge for the mag/confidence combinations")
print("your scorer actually produces -- check this against config.py's actual numbers directly,")
print("independent of any timing question.")