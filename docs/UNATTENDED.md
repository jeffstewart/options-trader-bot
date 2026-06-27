# Unattended operation (2-week data-collection run)

Set up 2026-06-26 so the bot runs without daily babysitting and captures enough data to diagnose
what to change when you return.

## One thing to install (sandbox can't edit crontab — run this in your terminal)

```bash
( crontab -l 2>/dev/null; \
  echo '*/5 * * * * /Users/jeff/Claude/Trader/manage.sh watchdog >> /Users/jeff/Claude/Trader/logs/watchdog_cron.log 2>&1' ) \
  | crontab -
crontab -l   # confirm: you should see the EOD line AND the new watchdog line
```

That's it. The watchdog runs every 5 min and is idempotent.

## What keeps it alive

- **`manage.sh watchdog`** (cron, every 5 min): restarts any down daemon (bot / dashboard / scheduler),
  keeps a `caffeinate` running (prevents Mac sleep — otherwise trading + capture freeze), and
  copy-truncates `bot/dashboard/ollama/groq_backtest` logs to the last 50k lines once they exceed 150 MB.
  Actions are logged to `logs/watchdog.log`. **Reboot recovery:** cron runs after boot, so within 5 min
  of a reboot the watchdog restarts everything (requires the Mac to be powered on; enable auto-login if
  it isn't already).
- **In-process supervisor**: every bot coroutine (news/stock streams, trailing-stop monitor, SEC + NewsAPI
  pollers) runs under `_supervised()` — a crash restarts that task instead of killing the bot.
- **`caffeinate -dimsu`** keeps the machine awake; the watchdog re-launches it if it ever dies.

## What it captures (all in `data/`, never rotated)

| File | What |
|---|---|
| `trades.csv` / `closed_trades.csv` | every open / close with P&L + reason |
| `position_paths.csv` | real ~30s premium path per position, `phase=open` and **`phase=postclose`** (for exit-rule replay) |
| `gemini_decisions.csv` | every gate call: decision, latency, mags |
| `regime_decisions.csv` | **NEW** — regime context per long-beta signal (in_uptrend, bypass, mag) → validate the bypass's forward P&L |
| `shadow_trades.csv` | paper counterfactuals: stock / sub-threshold news_call / **regime_block** (would-be trades the downtrend gate blocked) / veto / lotto-OOW / pead |
| `scorer_ab.csv` | Groq-vs-Ollama A/B |
| `groq_primary_backtest_cache.json` | the primary-scorer backtest, accumulating via the scheduler |
| `eod_reports/eod_YYYY-MM-DD.txt` | daily snapshot (account, trades, gate stats) via the EOD cron |

## Stopping / pausing it (IMPORTANT gotcha)

`./manage.sh stop` alone will NOT stop it — the watchdog cron restarts everything within 5 min.
To actually stop:
```bash
crontab -l | grep -v 'manage.sh watchdog' | crontab -   # remove the watchdog first
./manage.sh stop                                          # then stop the daemons
pkill -x caffeinate                                       # (optional) let the Mac sleep again
```
To resume later: re-add the watchdog cron line (top of this doc), then `./manage.sh start && ./manage.sh sched`.

## Health check when you return

```bash
cd /Users/jeff/Claude/Trader
./manage.sh status                 # all 4 daemons UP?
tail -20 logs/watchdog.log         # any restarts / rotations while away?
ls -t eod_reports/ | head          # daily reports kept generating?
grep -c crashed logs/bot.log       # in-process supervisor catches (should be ~0)
```

## Open questions queued for when the subscription is back

1. **Validate the directional gate end-to-end** — (a) bypass P&L: join `regime_decisions.csv` (bypass=1)
   to closed trades (post-2026-06-25 is the real test); (b) **the gate's counterfactual**: the new
   `regime_block_shadow` rows in `shadow_trades.csv` show how the blocked would-be trades actually did —
   if they're net losers the gate is working; if winners, the 0.85 bypass bar is too strict. Together
   these calibrate the regime/bypass thresholds.
2. **Re-evaluate the exit ratchet on real quotes** — `position_paths.csv` (open + postclose) replays
   live-40%-flat vs the deployed ratchet vs tighter, on actual 30s quotes.
3. **Move the gate to a fast Groq production model** — once the primary-scorer backtest hits ~75%
   coverage (needs the magnitude recalibration it quantifies). Fixes the mistral free-tier latency.
4. **Catalyst-score gate** — losers skew speculative ("in discussions/weighs/plans"); winners are
   resolved/material. Consider gating on the `catalyst` score.
