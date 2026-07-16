import json, statistics

groq = json.load(open("/Users/jeff/Claude/Trader/data/groq_primary_backtest_cache.json"))
dual = json.load(open("/Users/jeff/Claude/Trader/data/dual_score_cache.json"))

def safe_mag(v):
    """Extract magnitude as float regardless of schema variations."""
    for field in ("mag", "magnitude"):
        raw = v.get(field)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try: return float(raw)
            except ValueError: pass
        if isinstance(raw, dict):
            # e.g. {"value": 0.6} or {"score": 0.6}
            for sub in ("value", "score", "magnitude", "mag"):
                if sub in raw:
                    try: return float(raw[sub])
                    except (ValueError, TypeError): pass
    return 0.0

def safe_sent(v):
    for field in ("sent", "sentiment"):
        s = v.get(field, "")
        if s: return s
    return ""

mags = {"gpt-oss-20b": [], "gpt-oss-120b": [], "local": []}

for k, v in groq.items():
    if not v or not isinstance(v, dict): continue
    mag, sent = safe_mag(v), safe_sent(v)
    for prefix in ["gpt-oss-20b", "gpt-oss-120b"]:
        if k.startswith(prefix + ":") and sent == "bullish":
            mags[prefix].append(mag)

for k, v in dual.items():
    if not v or not isinstance(v, dict): continue
    mag, sent = safe_mag(v), safe_sent(v)
    if k.startswith("local:") and sent == "bullish":
        mags["local"].append(mag)

def dist(name, mags_list, target_n=None):
    if not mags_list: print(f"\n{name}: no data"); return
    mags_s = sorted(mags_list, reverse=True)
    n_bull = len(mags_list)
    if target_n is None: target_n = n_bull
    cutoff = mags_s[min(target_n-1, n_bull-1)]
    print(f"\n{name}  ({n_bull} bullish signals)")
    print(f"  min={min(mags_list):.2f}  median={statistics.median(mags_list):.2f}  "
          f"mean={sum(mags_list)/n_bull:.2f}  max={max(mags_list):.2f}")
    if target_n < n_bull:
        print(f"  threshold for top-{target_n} picks: mag>={cutoff:.2f}")
    print("  distribution:")
    buckets = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for lo, hi in zip(buckets, buckets[1:]):
        n = sum(1 for m in mags_list if lo <= m < hi)
        bar = "█" * (n // 5)
        print(f"    {lo:.1f}-{hi:.1f}  {n:4d}  {bar}")

ollama_n = len(mags["local"])
dist("local (ollama)",  mags["local"])
dist("gpt-oss-20b",        mags["gpt-oss-20b"],  target_n=ollama_n)
dist("gpt-oss-120b",       mags["gpt-oss-120b"], target_n=ollama_n)