"""
resume_after_reset.py — keeps groq_vs_ollama_backtest.py running across Groq's daily request-cap
(RPD) resets. Groq free-tier per-model RPD resets 00:00 UTC; this daemon resumes the scoring right
after each reset (00:10 UTC buffer) and stops once both models are fully scored.

Each cycle: kill any running/stalled backtest → clear None (parse-fail) holes → if complete, exit →
else relaunch the backtest daemon. Resumes IMMEDIATELY on start too (the cap may already have reset).

Stop early: `touch resume_after_reset.stop`, or kill this daemon (pid in resume_after_reset.pid).

Run:  .venv/bin/python spawn_daemon.py resume_after_reset.pid resume_after_reset.log \
          .venv/bin/python -u resume_after_reset.py
"""
import os, sys, json, time, subprocess
from datetime import datetime, timezone, timedelta

DIR = "/Users/jeff/Claude/Trader"                       # 2026-06-24 restructure: code in core/+research/,
CACHE = os.path.join(DIR, "data", "groq_primary_backtest_cache.json")   # data/caches, logs/logs
STOP = os.path.join(DIR, "data", "resume_after_reset.stop")
PY = os.path.join(DIR, ".venv/bin/python")
TOTAL_PER_MODEL = 1900                 # universe ~1906; treat ≥this as complete (a few articles drop out)
# Completion gates on 8b + gpt-oss-120b — scout dropped (deprecation), 70b can't finish on free tier.
# 70b still gets scored opportunistically; its partial data shows in reports.
MODELS = ("llama-3.1-8b", "gpt-oss-120b")
MAX_CYCLES = 14


def counts():
    if not os.path.exists(CACHE):
        return {m: 0 for m in MODELS}, 0
    d = json.load(open(CACHE))
    c = {m: sum(1 for k, v in d.items() if v and k.startswith(m + ":")) for m in MODELS}
    return c, sum(1 for v in d.values() if v is None)


def clear_nones():
    if not os.path.exists(CACHE):
        return 0
    d = json.load(open(CACHE)); before = len(d)
    d = {k: v for k, v in d.items() if v is not None}
    json.dump(d, open(CACHE, "w"))
    return before - len(d)


def kill_backtest():
    subprocess.run(["pkill", "-f", "groq_vs_ollama_backtest.py"], cwd=DIR)
    time.sleep(2)


def launch_backtest():
    env = dict(os.environ, USE_YAHOO_BARS="1")
    # spawn_daemon (core/) sets CWD=data/, so the backtest's bare cache refs resolve into data/.
    subprocess.run([PY, os.path.join(DIR, "core", "spawn_daemon.py"),
                    os.path.join(DIR, "data", "groq_backtest.pid"),
                    os.path.join(DIR, "logs", "groq_backtest.log"),
                    PY, "-u", os.path.join(DIR, "research", "groq_vs_ollama_backtest.py")],
                   cwd=os.path.join(DIR, "data"), env=env)


def next_reset(now):
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=10, second=0, microsecond=0)
    if now.hour == 0 and now.minute < 10:
        nxt = now.replace(minute=10, second=0, microsecond=0)
    return nxt


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC] {msg}", flush=True)


def cycle(tag):
    kill_backtest()
    n = clear_nones()
    c, _ = counts()
    status = " · ".join(f"{m.split('-')[-1]} {c[m]}/{TOTAL_PER_MODEL}" for m in MODELS)
    log(f"{tag}: {status} (cleared {n} None holes)")
    if all(c[m] >= TOTAL_PER_MODEL for m in MODELS):
        log("BOTH MODELS COMPLETE — scheduler done.")
        return True
    launch_backtest()
    log("relaunched backtest daemon")
    return False


def main():
    log("scheduler started — resuming now (cap may already have reset), then after each 00:10 UTC reset")
    if cycle("resume-now"):
        return
    for i in range(MAX_CYCLES):
        target = next_reset(datetime.now(timezone.utc))
        log(f"sleeping until {target:%Y-%m-%d %H:%M} UTC (cycle {i+1}/{MAX_CYCLES})")
        while datetime.now(timezone.utc) < target:
            if os.path.exists(STOP):
                kill_backtest(); log("STOP file found — scheduler exiting (backtest stopped)."); return
            time.sleep(min(600, (target - datetime.now(timezone.utc)).total_seconds()))
        if os.path.exists(STOP):
            kill_backtest(); log("STOP file found — scheduler exiting."); return
        if cycle(f"post-reset cycle {i+1}"):
            return
    log(f"reached MAX_CYCLES={MAX_CYCLES} — scheduler stopping (70b may still be partial on free tier).")


if __name__ == "__main__":
    main()
