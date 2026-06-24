# Trader Bot — Logic Flow (as of 2026-06-08)

Traced from `bot.py` / `config.py`. Three parts: **(1) news in → signal**,
**(2) signal → trade decision per strategy**, **(3) monitor → close**.

Live config snapshot: `MIN_MAGNITUDE=0.35`, `BASE_CONFIDENCE=0.70`, `CONFIDENCE_SLOPE=0.25`,
`MAX_POSITION_USD=$1000`, `COOLDOWN_SECS=300`, `MAX_OPEN_POSITIONS=30`,
`DAILY_LOSS_LIMIT=$2000`, `MONITOR_INTERVAL=30s`, regime = SPY > 200d SMA,
`NEWS_CALL_ENABLED=False`, `LOTTO 0.70/0.85`, `MATERIALITY_SHADOW_ENABLED=True`.

---

## PART 1 — News ingestion → `process_signal`

```
┌─ 1. Alpaca/Benzinga WebSocket news ("*") ─┐   real-time push
├─ 2. SEC EDGAR 8-K RSS poller (every 120s) ┤   pull
└─ 3. NewsAPI poller (every 180s, if key)  ─┘   pull
                    │  (headline, body, source, article_ts)
                    ▼
            ┌───────────────────┐
            │  process_signal   │
            └───────────────────┘

[ separate stream ] StockDataStream trades  ──►  _price_cache{ticker: price}
   (25 pre-subscribed liquid symbols; real-time prices used by the monitor)
```

---

## PART 2 — Scoring, gates & per-strategy dispatch

```
process_signal(headline, body, source)
   │
   ├─ market_is_open()? ── NO ─► return (bot only trades during RTH)
   │        │ YES
   ▼
 score_with_ollama(headline, body)        ← local llama3.2, SYSTEM_PROMPT
   │   returns JSON {tickers, sentiment, confidence, magnitude, reasoning}
   │   PROMPT is BULLISH-BIASED: "Only flag bullish sentiment — we trade long calls only"
   │   (⇒ bearish signals are rare; bear_short/pairs fire infrequently)
   ▼
 sentiment?
   │
   ├── "bearish" ───────────────────────────────────────────────┐
   │     gate: signal_checks (mag ≥ 0.35 AND conf ≥ 0.70+(1-mag)·0.25)
   │       │ pass
   │       ├─ record_pairs_signal("bear")  +  maybe_enter_pairs()   ← ANY regime
   │       ├─ log_router_decision (shadow, no order)
   │       └─ for up to 2 tickers (not cooldown/crypto):
   │              execute_bear_short_fn → SELL short stock     [bear_short]  ANY regime
   │                                                                          └─► return
   │
   ├── "neutral"/unknown ─► return
   │
   └── "bullish"
         gate: signal_checks (same mag/conf gate)
           │ pass
           ├─ record_pairs_signal("bull")  +  maybe_enter_pairs()   ← ANY regime
           ├─ log_router_decision (shadow)
           │
           ├─ REGIME GATE: market_in_uptrend()  (SPY > 200d SMA)?
           │        NO ─► return  (skip ALL long-beta strategies; pairs/bear already ran)
           │        │ YES
           ▼
         position_usd = MAX_POSITION_USD × mag × conf   (floored at 10%)
           │
           ├─ tickers EMPTY (macro-bullish)? ─► QQQ call            [qqq_macro]
           │                                     (if QQQ not on cooldown) └─► return
           │
           └─ for up to 2 tickers (skip cooldown / BTC,ETH):
                 ├─ news_call:  ATM call Δ0.50      [news_call]  ── DISABLED (NEWS_CALL_ENABLED=False)
                 ├─ stock:      buy stock           [stock]      ── always (bullish+uptrend)
                 ├─ pead:       IF earnings headline ─► buy stock [pead]  (_is_earnings_headline)
                 └─ lotto:      IF mag≥0.70 AND conf≥0.85 ─► OTM call Δ0.25  [lotto]
                                   └─ + shadow materiality score → lotto_materiality.csv
```

### Gates that can block a trade (in order)
1. **Market open** (RTH only).
2. **signal_checks** — `mag ≥ 0.35` AND `conf ≥ 0.70 + (1−mag)·0.25` (dynamic floor: weaker
   magnitude needs higher confidence).
3. **Regime** (`SPY > 200d SMA`) — long-beta strategies only; bear_short & pairs ignore it.
4. **Cooldown** — 300 s per ticker (`_recent_trades`).
5. **Daily loss limit** — halts all new trades once realized losses ≥ $2000/day.
6. **Position caps / dedup** — `MAX_OPEN_POSITIONS=30` (lotto has its own sub-cap and is
   exempt from the main cap); never opens a 2nd position in the same contract / `ticker__strategy`.

### Pairs entry (`maybe_enter_pairs`, market-neutral, ANY regime)
```
once per UTC day, when the day has BOTH a bull and a bear signal (different tickers):
   LONG  top-scored bull ticker   [pairs_long]   +   SHORT top-scored bear ticker  [pairs_short]
   equal $ (sized off the bull leg). One pair per day.
```

### Strategy reference
| strategy | instrument | extra trigger | regime gate | trailing stop | time cap |
|---|---|---|---|---|---|
| `news_call` | ATM call Δ0.50 | — (DISABLED) | uptrend | profit-tiered | none |
| `stock` | long stock | — | uptrend | flat 10% | `NEWS_STOCK_MAX_HOLD_DAYS` |
| `pead` | long stock | **earnings headline** | uptrend | tiered + breakeven lock | `PEAD_MAX_HOLD_DAYS` |
| `lotto` | OTM call Δ0.25 | mag≥0.70 & conf≥0.85 | uptrend | wide tiered | `LOTTO_MAX_HOLD_DAYS` |
| `qqq_macro` | QQQ call | bullish + NO ticker | uptrend | profit-tiered | none |
| `bear_short` | short stock | bearish | **none** | flat on trough | none |
| `pairs_long/short` | long + short stock | bull+bear same day | **none** | flat | none live* |

\* `PAIRS_MAX_HOLD` exists in config but is **not enforced** in the live monitor (documented divergence).
A bullish signal always opens `stock`; it adds `pead` only on an **earnings** headline and `lotto`
only on a **high-conviction** (mag≥0.70 & conf≥0.85) signal — so up to 3 positions/ticker, but
typically just `stock` (× up to 2 tickers).

---

## PART 3 — Monitor & close (`trailing_stop_monitor`, every 30 s)

```
every 30s, for each open position in _monitored_positions:
   │
   ├─ current price:  stock → _price_cache[ticker] or REST ;  option → option-quote mid
   │     (none available → skip this cycle)
   │
   ├─ ① TIME-STOP: held ≥ max_hold_days AND in EOD window (last 15 min)? ─► CLOSE
   │        caps: stock=2d · lotto=7d · pead=45d ; pairs/bear_short/qqq_macro/news_call=none
   │        (before the EOD window it falls through to ③ so it's still trailing-stop protected;
   │         exits at the CLOSE on the final day, NOT carried to next-morning open — gap risk)
   │
   ├─ ② SHORT positions (bear_short, pairs_short):
   │        track TROUGH (lowest price);  stop = trough × (1 + short_trail)
   │        price ≥ stop  (a RALLY) ─► CLOSE = BUY-to-cover
   │
   └─ ③ LONG positions (stock, pead, lotto, qqq_macro, option):
            track PEAK (highest price);  stop = peak × (1 − trail)
            trail width by strategy:  stock=flat 10% · pead=tiered · lotto=wide tiered
                                      pairs_long=flat · option/qqq=profit-tiered
            (pead breakeven-lock: once gain ≥ threshold, stop ≥ entry×1.005)
            price ≤ stop  (a DROP) ─► CLOSE

   On any CLOSE trigger:
       ├─ market_is_open()?  NO ─► DEFER (log once), retry next cycle   ◄── the "stop breached
       │                                                                     but market closed" path
       └─ YES ─► place close:
                   stock      → close_stock_position    (market SELL)
                   option     → close_option_position   (stop-limit ABOVE market;
                                Alpaca rejects uncovered option market-sells — workaround)
                   short      → close_short_position     (market BUY to cover)
                 │
                 ├─ confirmed?  NO ─► keep monitoring, retry next cycle
                 └─ YES ─► log_closed_trade → closed_trades.csv
                            ├─ if pnl<0: _daily_loss_usd += |pnl|  (feeds the $2000 halt)
                            └─ remove from _monitored_positions

   each cycle also: _write_bot_state() → bot_state.json
       (stop/peak/entry_dt/strategy/qty/materiality → dashboard + restart recovery)
```

### Protection caveat (important)
There are **no exchange-held stops**. This Alpaca account cannot hold uncovered option
sell-stops, so the **in-process 30 s monitor is the ONLY protection** for every position.
If the bot process is down, open positions are unprotected until it restarts (on restart it
reloads positions from Alpaca + recovers strategy/peak/entry_dt from `trades.csv`/`bot_state.json`).

---

## One-glance summary
- **In:** 3 news feeds → `process_signal` (RTH only).
- **Score:** one local llama3.2 call, bullish-biased prompt → sentiment + magnitude + confidence + tickers.
- **Decide:** mag/conf gate → sentiment branch → (bullish) 200d-SMA regime gate → fan out to
  stock + pead (+ lotto if high-conviction); (bearish) bear_short; (bull+bear same day) pairs.
  news_call currently OFF; lotto also logs a shadow materiality score for the live A/B.
- **Manage:** 30 s in-process trailing-stop + time-stop monitor; closes deferred while market is
  closed; realized closes logged + counted toward the daily-loss halt.
```
```
