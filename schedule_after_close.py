"""
schedule_after_close.py — wait until after the market close, then run the
overnight LLM-router jobs WITHOUT contending with the live bot for Ollama
(the bot goes idle when the market is closed, so Ollama is free).

Sequence (sequential — one Ollama consumer at a time):
  1. Finish the bull-window router rescore (resumes router_llm_cache.json)
  2. Rescore the 2022 bear-window articles → router_llm_bear_cache.json
  3. Re-run the bull LLM-router eval on the now-complete cache
  4. Run the 2022 cross-regime LLM-router eval  ← the decisive test

Launch:  nohup .venv/bin/python schedule_after_close.py >> after_close.log 2>&1 &
Note: dies if the machine sleeps/reboots before close — re-launch if so.
"""
import os, subprocess, time, datetime, zoneinfo

ET = zoneinfo.ZoneInfo("America/New_York")
PY = ".venv/bin/python"


def log(m):
    print(f"{datetime.datetime.now(ET):%Y-%m-%d %H:%M:%S} ET  {m}", flush=True)


def main():
    now = datetime.datetime.now(ET)
    target = now.replace(hour=16, minute=5, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)   # safety: already past close → next day
    secs = (target - now).total_seconds()
    log(f"waiting {int(secs)}s until {target:%Y-%m-%d %H:%M} ET (after close)")
    time.sleep(max(0, secs))

    rescore_env = dict(os.environ)                       # Ollama jobs
    yahoo_env   = dict(os.environ, USE_YAHOO_BARS="1")   # eval jobs

    jobs = [
        ("bull rescore (resume)",
         [PY, "router_rescore.py", "--delay", "0"], rescore_env),
        ("2022 bear rescore",
         [PY, "router_rescore.py", "--articles-from", "bear_dual_cache.json",
          "--output-file", "router_llm_bear_cache.json", "--delay", "0"], rescore_env),
        ("bull LLM-router eval (full cache)",
         [PY, "router_llm_eval.py"], yahoo_env),
        ("2022 cross-regime LLM-router eval",
         [PY, "router_llm_eval.py", "--cache", "router_llm_bear_cache.json",
          "--end-date", "2022-06-30", "--days", "90"], yahoo_env),
    ]
    for name, cmd, env in jobs:
        log(f"▶ START: {name}")
        rc = subprocess.run(cmd, env=env).returncode
        log(f"■ DONE ({rc}): {name}")
    log("✅ ALL AFTER-CLOSE JOBS COMPLETE")


if __name__ == "__main__":
    main()
