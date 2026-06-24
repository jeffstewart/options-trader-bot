"""
run_local_sweep.py — overnight local prompt/model sweep on this 8 GB M1.

Sequencing matters on 8 GB RAM:
  1. WAIT for the after-close scheduler (rescores + evals) to finish — so we don't
     run two Ollama jobs at once (OOM) or two Yahoo writers at once (cache corruption).
  2. STOP the bot to free RAM (market is closed → it can't execute anything anyway;
     restarted in a finally so it's always back up before the open).
  3. Sweep models SEQUENTIALLY, `ollama stop` between each — never two resident.
       llama3.2 (3B) → llama3.1:8b → qwen2.5:7b (different family: Alibaba Qwen)
     Each graded by signal_lab vs the llama3.2 baseline (prompt_exp.py).
  4. Restart the bot.

Launch:  nohup .venv/bin/python run_local_sweep.py >> local_sweep.log 2>&1 &
"""
import os, subprocess, time, datetime, zoneinfo

ET = zoneinfo.ZoneInfo("America/New_York")
PY = ".venv/bin/python"
MODELS  = ["llama3.2", "llama3.1:8b", "qwen2.5:7b"]
PROMPTS = "baseline,direct_return,materiality,catalyst_typed"
LIMIT   = "250"


def log(m):
    print(f"{datetime.datetime.now(ET):%Y-%m-%d %H:%M:%S} ET  {m}", flush=True)


def alive(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def main():
    # 1. Wait for the after-close chain (scheduler + its rescore/eval children) to end.
    log("waiting for after-close scheduler (rescores+evals) to finish before sweeping…")
    while alive("schedule_after_close") or alive("router_rescore") or alive("router_llm_eval"):
        time.sleep(60)
    log("after-close chain done — Ollama + Yahoo free; starting local sweep")

    # 2. Stop the bot to free RAM (safe: market closed; restarted in finally).
    subprocess.run(["pkill", "-f", "bot.py"])
    time.sleep(3)
    log("stopped bot.py to free memory for the sweep")

    try:
        for m in MODELS:
            subprocess.run(["ollama", "pull", m])           # no-op if already present
            logname = f"prompt_exp_{m.replace(':', '_').replace('.', '')}.log"
            log(f"▶ sweeping {m}  → {logname}")
            with open(logname, "w") as f:
                subprocess.run(
                    [PY, "prompt_exp.py", "--provider", "ollama", "--model", m,
                     "--prompts", PROMPTS, "--limit", LIMIT, "--delay", "0"],
                    env=dict(os.environ, USE_YAHOO_BARS="1"),
                    stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(["ollama", "stop", m])           # unload before next model
            log(f"■ done + unloaded {m}")
    finally:
        # 3. Always bring the bot back (idle until the open, but monitoring-ready).
        subprocess.Popen(f"nohup {PY} bot.py >> bot.log 2>&1 &", shell=True)
        log("restarted bot.py")

    log("✅ LOCAL SWEEP COMPLETE — results in prompt_exp_*.log (grep 'IC=')")


if __name__ == "__main__":
    main()
