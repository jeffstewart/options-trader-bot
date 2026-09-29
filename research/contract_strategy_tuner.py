"""
contract_strategy_tuner.py

Framework for evaluating and tuning:
  1) CONTRACT SELECTION  - which contract to buy given a signal's option chain
  2) NO-TRADE GATE       - whether to skip a signal entirely (like the bot's
                            existing news_call/lotto gates, but the FEATURES
                            and thresholds it uses are discovered from data,
                            not prescribed - see section 5)
  3) EXIT STRATEGY        - when to sell, given the contract's tracked price path

...against the contract_grid_snapshots.csv data produced by the trader bot,
with a proper train/test split so nothing is validated on data it was tuned on.

CONSERVATIVE FILL ASSUMPTIONS:
  - Entry fill = ASK at the first snapshot for a signal (minutes_since_signal ~ 0)
  - Exit fill  = BID at whatever snapshot the exit rule triggers (never mid/ask)
  - If an exit rule never triggers within the tracked window, exit at the BID
    of the LAST available snapshot ("still holding at data cutoff").

TRAIN/TEST SPLIT:
  Signals are split CHRONOLOGICALLY (train = earlier signals, test = later
  signals), not randomly - a random split would let any fitted rule "see the
  future" relative to some test points, which a live bot never gets. ALL
  fitting (best selection+exit combo, feature significance, gate thresholds
  or model coefficients) happens on TRAIN ONLY. TEST is touched exactly once,
  to report out-of-sample performance.

SECTION 5 DESIGN NOTE (why features aren't hand-picked):
  Rather than assuming which signal/contract attributes predict a losing
  trade (confidence? spread? delta?), this script runs three independent
  significance checks on a BROAD, auto-detected set of entry-time features
  and only builds the gate from whatever survives:
    (a) univariate Spearman correlation vs P&L, BH-FDR corrected across all
        features tested (guards against false discoveries from testing many
        features on a small n)
    (b) L1-regularized logistic regression (sparsity does the feature
        selection - irrelevant features get a zero coefficient)
    (c) Random Forest permutation importance with an actual label-shuffle
        significance test (p-value = how often a shuffled-label model
        matches or beats the real importance)
  A feature has to show up in at least two of these to be treated as real.

IMPORTANT CAVEAT:
  This data currently covers ~3.5 days, all down-market, no live trades yet.
  n=118 signals total means a chronological 70/30 split leaves ~35 test
  signals, and train-side feature significance is being computed on ~80.
  That is a THIN sample for 15+ candidate features - expect the significant
  set to be unstable and possibly empty; an empty result is a legitimate
  finding ("not enough data yet to say"), not a bug. Re-run this script as
  more data (esp. up-market) accumulates.

Usage:
    python3 contract_strategy_tuner.py /path/to/contract_grid_snapshots.csv
"""

import sys
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the full dataset (convenience function for small files or interactive exploration).
    For large snapshot files (>100MB), prefer load_entry_snapshot_and_stats and
    load_selected_paths to stream in chunks and avoid memory exhaustion.
    """
    df = pd.read_csv(path, low_memory=False)
    if "signal_ts" in df.columns:
        df["signal_ts"] = pd.to_datetime(df["signal_ts"])
    if "snapshot_ts" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    df = df.sort_values(["signal_id", "symbol", "minutes_since_signal"]).reset_index(drop=True)
    return df


def load_entry_snapshot_and_stats(path: str, chunksize: int = 500_000) -> tuple[pd.DataFrame, int, pd.Series]:
    """Pass 1: Stream the CSV in chunks.
    Extract the first snapshot row per (signal_id, symbol) to form entry_df.
    Simultaneously track total rows and signal snapshot counts without keeping
    the full multi-gigabyte table in memory.
    """
    entry_chunks = []
    seen = set()
    total_rows = 0
    signal_counts = Counter()

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        signal_counts.update(chunk["signal_id"].value_counts().to_dict())

        c_entry = chunk.drop_duplicates(subset=["signal_id", "symbol"])
        mask = ~c_entry.set_index(["signal_id", "symbol"]).index.isin(seen)
        new_entries = c_entry[mask]
        if len(new_entries) > 0:
            seen.update(zip(new_entries["signal_id"], new_entries["symbol"]))
            entry_chunks.append(new_entries)

    entry_df = pd.concat(entry_chunks, ignore_index=True)
    if "signal_ts" in entry_df.columns:
        entry_df["signal_ts"] = pd.to_datetime(entry_df["signal_ts"])
    if "snapshot_ts" in entry_df.columns:
        entry_df["snapshot_ts"] = pd.to_datetime(entry_df["snapshot_ts"])
    entry_df["row_in_group"] = 0

    # Downcast float64 to float32 to minimize memory footprint
    for col in entry_df.select_dtypes(include=["float64"]).columns:
        entry_df[col] = entry_df[col].astype("float32")

    rows_per_signal = pd.Series(signal_counts)
    return entry_df, total_rows, rows_per_signal


def load_selected_paths(path: str, selected_pairs: set, chunksize: int = 500_000) -> pd.DataFrame:
    """Pass 2: Stream the CSV in chunks, extracting snapshot rows ONLY for
    the contracts that were actually selected by at least one selection rule.
    Only loads the essential path columns: signal_id, symbol, minutes_since_signal, bid, ask.
    """
    selected_signals = {s for s, _ in selected_pairs}
    path_cols = ["signal_id", "symbol", "minutes_since_signal", "bid", "ask"]
    dtypes = {
        "minutes_since_signal": "float32",
        "bid": "float32",
        "ask": "float32",
    }

    path_chunks = []
    for chunk in pd.read_csv(path, usecols=path_cols, chunksize=chunksize, dtype=dtypes, low_memory=False):
        sub = chunk[chunk["signal_id"].isin(selected_signals)]
        if len(sub) > 0:
            pairs = list(zip(sub["signal_id"], sub["symbol"]))
            mask = [p in selected_pairs for p in pairs]
            matched = sub[mask]
            if len(matched) > 0:
                path_chunks.append(matched)

    if path_chunks:
        paths_df = pd.concat(path_chunks, ignore_index=True)
        paths_df = paths_df.sort_values(["signal_id", "symbol", "minutes_since_signal"]).reset_index(drop=True)
    else:
        paths_df = pd.DataFrame({
            "signal_id": pd.Series(dtype="str"),
            "symbol": pd.Series(dtype="str"),
            "minutes_since_signal": pd.Series(dtype="float32"),
            "bid": pd.Series(dtype="float32"),
            "ask": pd.Series(dtype="float32"),
        })

    paths_df = prep_contract_paths(paths_df)
    return paths_df


def prep_contract_paths(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        for c in ["entry_price", "bid_peak", "drawdown_from_peak", "ret_from_entry"]:
            df[c] = pd.Series(dtype="float32")
        for c in ["row_in_group", "group_size"]:
            df[c] = pd.Series(dtype="int64")
        return df

    g = df.groupby(["signal_id", "symbol"], sort=False)
    df["entry_price"] = g["ask"].transform("first")
    df["bid_peak"] = g["bid"].cummax()
    df["drawdown_from_peak"] = df["bid"] / df["bid_peak"] - 1.0
    df["ret_from_entry"] = df["bid"] / df["entry_price"] - 1.0
    df["row_in_group"] = g.cumcount()
    df["group_size"] = g["symbol"].transform("size")
    return df


def entry_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """First row per (signal_id, symbol) = the chain as it looked at signal time."""
    return df[df["row_in_group"] == 0].copy()


def get_signal_meta(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("signal_id")[["signal_ts", "ticker", "magnitude", "confidence",
                                  "catalyst", "passes_lotto_gate", "regime_mom_ok"]]
        .first()
    )


# ----------------------------------------------------------------------------
# 2. TRAIN / TEST SPLITTING (signal-level, chronological)
# ----------------------------------------------------------------------------

def chronological_split(signal_meta: pd.DataFrame, test_frac: float = 0.3):
    ordered = signal_meta.sort_values("signal_ts")
    n_test = max(1, int(round(len(ordered) * test_frac)))
    test_ids = set(ordered.tail(n_test).index)
    train_ids = set(ordered.head(len(ordered) - n_test).index)
    return train_ids, test_ids


def walk_forward_splits(signal_meta: pd.DataFrame, n_splits: int = 5, min_train_frac: float = 0.4):
    """Expanding-window walk-forward CV - use once you have ~60+ signals for
    multiple folds to be meaningful."""
    ordered = signal_meta.sort_values("signal_ts")
    ids = ordered.index.tolist()
    n = len(ids)
    min_train = max(5, int(n * min_train_frac))
    remaining = n - min_train
    fold_size = max(1, remaining // n_splits)
    splits, start = [], min_train
    for _ in range(n_splits):
        end = min(start + fold_size, n)
        if start >= n:
            break
        train_ids = set(ids[:start])
        test_ids = set(ids[start:end])
        if test_ids:
            splits.append((train_ids, test_ids))
        start = end
    return splits


# ----------------------------------------------------------------------------
# 3. CONTRACT SELECTION RULES
# ----------------------------------------------------------------------------

def select_by_delta_target(entry_df, target_delta, min_oi=0, max_spread_pct=999):
    d = entry_df[(entry_df["oi_at_signal"].fillna(0) >= min_oi) &
                 (entry_df["spread_pct"] <= max_spread_pct)].dropna(subset=["delta"]).copy()
    d["delta_dist"] = (d["delta"] - target_delta).abs()
    return d.loc[d.groupby("signal_id")["delta_dist"].idxmin()].set_index("signal_id")["symbol"]


def select_by_otm_pct(entry_df, target_otm_pct, min_oi=0, max_spread_pct=999):
    d = entry_df[(entry_df["oi_at_signal"].fillna(0) >= min_oi) &
                 (entry_df["spread_pct"] <= max_spread_pct)].copy()
    d["otm_pct"] = d["strike"] / d["stock_price_at_signal"] - 1.0
    d["otm_dist"] = (d["otm_pct"] - target_otm_pct).abs()
    return d.loc[d.groupby("signal_id")["otm_dist"].idxmin()].set_index("signal_id")["symbol"]


def select_cheapest_liquid(entry_df, min_oi=10, max_spread_pct=0.15):
    d = entry_df[(entry_df["oi_at_signal"].fillna(0) >= min_oi) &
                 (entry_df["spread_pct"] <= max_spread_pct)].copy()
    return d.loc[d.groupby("signal_id")["ask"].idxmin()].set_index("signal_id")["symbol"]


SELECTION_RULES = {
    "delta_0.70_liquid": lambda e: select_by_delta_target(e, 0.70, min_oi=10, max_spread_pct=0.20),
    "delta_0.50_liquid": lambda e: select_by_delta_target(e, 0.50, min_oi=10, max_spread_pct=0.20),
    "delta_0.30_liquid": lambda e: select_by_delta_target(e, 0.30, min_oi=10, max_spread_pct=0.20),
    "otm_5pct_liquid":   lambda e: select_by_otm_pct(e, 0.05, min_oi=10, max_spread_pct=0.20),
    "otm_10pct_liquid":  lambda e: select_by_otm_pct(e, 0.10, min_oi=10, max_spread_pct=0.20),
    "cheapest_lotto":    lambda e: select_cheapest_liquid(e, min_oi=10, max_spread_pct=0.15),
}


# ----------------------------------------------------------------------------
# 4. EXIT STRATEGIES
# ----------------------------------------------------------------------------

def exit_fixed_horizon(paths, horizon_min, max_gap_mult=3.0):
    """Exit at the first snapshot with minutes_since_signal >= horizon_min, else the
    last available snapshot. GUARDS AGAINST a real data artifact: the poller skips
    entirely while the market is closed, so if a signal fires late in the session,
    'the first snapshot >= horizon_min' can land the next trading session's open -
    hours or days later - silently turning a same-session timed exit into an
    overnight/weekend hold. If the horizon-satisfying snapshot is more than
    max_gap_mult x horizon_min minutes away, that's treated as this kind of gap:
    fall back to the last snapshot still within that window (closest same-session
    read) and label it distinctly so it's visible instead of silently mislabeled
    as a normal 'horizon' exit."""
    cap = horizon_min * max_gap_mult
    hit = paths[paths["minutes_since_signal"] >= horizon_min]
    first_hit = hit.groupby(["signal_id", "symbol"], sort=False).first()

    within_cap = paths[paths["minutes_since_signal"] <= cap]
    last_within_cap = within_cap.groupby(["signal_id", "symbol"], sort=False).last()
    last_row = paths.groupby(["signal_id", "symbol"], sort=False).last()

    out = last_row[["bid", "entry_price", "minutes_since_signal"]].copy()
    out["exit_reason"] = "window_end"

    normal = first_hit[first_hit["minutes_since_signal"] <= cap]
    out.update(normal[["bid", "entry_price", "minutes_since_signal"]])
    out.loc[out.index.isin(normal.index), "exit_reason"] = "horizon"

    gap_ids = first_hit.index.difference(normal.index)
    if len(gap_ids) > 0:
        gap_fallback = last_within_cap.loc[last_within_cap.index.intersection(gap_ids)]
        out.update(gap_fallback[["bid", "entry_price", "minutes_since_signal"]])
        out.loc[out.index.isin(gap_fallback.index), "exit_reason"] = "horizon_session_gap_fallback"

    return out.rename(columns={"bid": "exit_price", "minutes_since_signal": "minutes_held"}).reset_index()


def exit_take_profit_stop_loss(paths, tp_pct, sl_pct):
    trig = paths[(paths["ret_from_entry"] >= tp_pct) | (paths["ret_from_entry"] <= -sl_pct)]
    first_trig = trig.groupby(["signal_id", "symbol"], sort=False).first()
    last_row = paths.groupby(["signal_id", "symbol"], sort=False).last()
    out = last_row[["bid", "entry_price", "minutes_since_signal"]].copy()
    out.update(first_trig[["bid", "entry_price", "minutes_since_signal"]])
    reason = np.where(
        out.index.isin(first_trig.index),
        np.where(first_trig.reindex(out.index)["ret_from_entry"] >= tp_pct, "take_profit", "stop_loss"),
        "window_end",
    )
    out["exit_reason"] = reason
    return out.rename(columns={"bid": "exit_price", "minutes_since_signal": "minutes_held"}).reset_index()


def exit_trailing_stop(paths, trail_pct):
    trig = paths[paths["drawdown_from_peak"] <= -trail_pct]
    first_trig = trig.groupby(["signal_id", "symbol"], sort=False).first()
    last_row = paths.groupby(["signal_id", "symbol"], sort=False).last()
    out = last_row[["bid", "entry_price", "minutes_since_signal"]].copy()
    out.update(first_trig[["bid", "entry_price", "minutes_since_signal"]])
    out["exit_reason"] = np.where(out.index.isin(first_trig.index), "trailing_stop", "window_end")
    return out.rename(columns={"bid": "exit_price", "minutes_since_signal": "minutes_held"}).reset_index()


def exit_trailing_stop_with_floor(paths, trail_pct, min_hold_min):
    eligible = paths[paths["minutes_since_signal"] >= min_hold_min]
    trig = eligible[eligible["drawdown_from_peak"] <= -trail_pct]
    first_trig = trig.groupby(["signal_id", "symbol"], sort=False).first()
    last_row = paths.groupby(["signal_id", "symbol"], sort=False).last()
    out = last_row[["bid", "entry_price", "minutes_since_signal"]].copy()
    out.update(first_trig[["bid", "entry_price", "minutes_since_signal"]])
    out["exit_reason"] = np.where(out.index.isin(first_trig.index), "trailing_stop", "window_end")
    return out.rename(columns={"bid": "exit_price", "minutes_since_signal": "minutes_held"}).reset_index()


EXIT_STRATEGIES = {}
for h in [15, 30, 60, 120]:
    EXIT_STRATEGIES[f"fixed_{h}min"] = (exit_fixed_horizon, {"horizon_min": h})
for tp, sl in [(0.20, 0.10), (0.30, 0.15), (0.50, 0.20), (0.30, 0.30)]:
    EXIT_STRATEGIES[f"tp{int(tp*100)}_sl{int(sl*100)}"] = (exit_take_profit_stop_loss, {"tp_pct": tp, "sl_pct": sl})
for trail in [0.10, 0.15, 0.20, 0.30]:
    EXIT_STRATEGIES[f"trail{int(trail*100)}"] = (exit_trailing_stop, {"trail_pct": trail})
for trail, floor in [(0.15, 5), (0.20, 5), (0.20, 10)]:
    EXIT_STRATEGIES[f"trail{int(trail*100)}_floor{floor}m"] = (exit_trailing_stop_with_floor, {"trail_pct": trail, "min_hold_min": floor})


def simulate(paths: pd.DataFrame, exit_name: str) -> pd.DataFrame:
    exit_fn, kwargs = EXIT_STRATEGIES[exit_name]
    exits = exit_fn(paths, **kwargs)
    exits["pnl_pct"] = exits["exit_price"] / exits["entry_price"] - 1.0
    return exits


# ----------------------------------------------------------------------------
# 5. FEATURE DISCOVERY + NO-TRADE GATE
#    No feature list is prescribed. A broad set of entry-time attributes is
#    auto-detected from the schema, then three independent methods vote on
#    which ones actually matter. Only those build the gate.
# ----------------------------------------------------------------------------

# Columns that are IDs, timestamps, text, or engineering artifacts (not real
# features) get excluded. Everything else numeric/boolean at entry time is a
# candidate - this list is about data hygiene, not about picking "the good
# predictors" (that's what section 5 figures out).
# These are the LLM scorer's own outputs. They are NOT guaranteed a spot in the
# no-trade gate because they're assumed to matter - they're guaranteed a spot
# because the whole point of the scorer is to be predictive, so its own outputs
# deserve a chance to be tested/used even when a small-sample significance vote
# doesn't (yet) single them out. See the printed "LLM SCORE FEATURES" section
# for why they may look statistically weak right now: this dataset only contains
# signals that already passed passes_news_call_gate, which pre-filters on these
# same scores - so the observed range here is narrower than the scorer's full
# output range, and a real relationship can easily look flat within a narrow,
# already-filtered band (restriction of range). That's a data-collection
# limitation, not evidence the scores don't matter.
ALWAYS_CONSIDERED_FOR_GATE = ["magnitude", "confidence", "catalyst"]

NON_FEATURE_COLS = {
    "signal_id", "signal_ts", "ticker", "headline", "source", "scorer_model",
    "snapshot_ts", "symbol", "expiry", "minutes_since_signal", "row_in_group",
    "group_size", "entry_price", "bid_peak", "drawdown_from_peak", "ret_from_entry",
    "stock_price",  # duplicate of stock_price_at_signal at the entry row
    "passes_news_call_gate",  # constant True by construction in this data (already gates entry into the file)
}


def build_broad_feature_table(entry_df: pd.DataFrame, picked: pd.Series):
    """One row per signal_id: every entry-time numeric/boolean attribute of the
    contract that was actually picked, plus signal-level scores/regime state.
    Zero-variance columns are dropped automatically (they can't predict anything
    and would just add noise to the significance tests)."""
    picked_df = picked.reset_index()  # signal_id, symbol
    d = picked_df.merge(entry_df, on=["signal_id", "symbol"], how="left").set_index("signal_id")

    # standard, non-arbitrary derived ratios (not "features I think matter" -
    # just putting raw prices on a comparable scale across different contracts)
    d["otm_pct"] = d["strike"] / d["stock_price_at_signal"] - 1.0
    d["spy_trend"] = d["spy_price"] / d["spy_sma"] - 1.0

    candidate_cols = [c for c in d.columns if c not in NON_FEATURE_COLS]
    feats = pd.DataFrame(index=d.index)
    for c in candidate_cols:
        col = d[c]
        if col.dtype == bool:
            feats[c] = col.astype(int)
        elif pd.api.types.is_numeric_dtype(col):
            feats[c] = col
        # non-numeric (e.g. leftover text) columns are silently skipped

    nunique = feats.nunique(dropna=True)
    dropped_zero_var = sorted(nunique[nunique <= 1].index.tolist())
    feats = feats[nunique[nunique > 1].index]

    # oi_at_signal missingness means "no OI reported" -> treat as 0, matches
    # the liquidity-filter convention used elsewhere in this script
    if "oi_at_signal" in feats.columns:
        feats["oi_at_signal"] = feats["oi_at_signal"].fillna(0)

    return feats, dropped_zero_var


def univariate_significance(feats: pd.DataFrame, pnl: pd.Series) -> pd.DataFrame:
    """Spearman correlation of each feature vs P&L, with Benjamini-Hochberg
    FDR correction across all features tested (testing 15+ features on ~80
    rows WILL produce spurious p<0.05 hits by chance alone if uncorrected)."""
    aligned = feats.join(pnl.rename("pnl_pct"), how="inner")
    rows = []
    for col in feats.columns:
        x, y = aligned[col], aligned["pnl_pct"]
        mask = x.notna() & y.notna()
        if mask.sum() < 8 or x[mask].nunique() < 2:
            rows.append({"feature": col, "spearman_rho": np.nan, "p_value": np.nan, "n": int(mask.sum())})
            continue
        rho, p = spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "spearman_rho": rho, "p_value": p, "n": int(mask.sum())})
    res = pd.DataFrame(rows)

    valid = res["p_value"].notna()
    res["q_value"] = np.nan
    if valid.sum() > 0:
        sub = res[valid].sort_values("p_value").reset_index()
        m = len(sub)
        ranks = np.arange(1, m + 1)
        q = (sub["p_value"].values * m / ranks)
        q = np.minimum.accumulate(q[::-1])[::-1]  # enforce BH monotonicity
        q = np.clip(q, 0, 1)
        res.loc[sub["index"], "q_value"] = q

    return res.sort_values("p_value")


def l1_logistic_selection(feats: pd.DataFrame, pnl: pd.Series, cv: int = 5):
    """L1-regularized logistic regression predicting P(pnl>0). Sparsity does
    the feature selection - a feature with coef=0 was not useful for
    separating winners from losers once the others are accounted for."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    X = feats.fillna(feats.median())
    y = (pnl.reindex(X.index) > 0).astype(int)
    if y.nunique() < 2:
        return None

    n_splits = min(cv, y.value_counts().min())
    if n_splits < 2:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegressionCV(Cs=10, cv=n_splits, penalty="l1", solver="liblinear",
                                max_iter=2000, scoring="neg_log_loss")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        clf.fit(Xs, y)

    coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    return {"clf": clf, "scaler": scaler, "feature_cols": X.columns.tolist(),
            "median": X.median(), "coefs": coefs}


def rf_permutation_significance(feats: pd.DataFrame, pnl: pd.Series,
                                 n_shuffles: int = 100, random_state: int = 0) -> pd.DataFrame:
    """Random Forest permutation importance (predicting pnl_pct directly, a
    regression target - richer signal than win/loss), with an actual
    significance test: refit on label-shuffled data n_shuffles times and see
    how often a shuffled model's importance for a feature matches or beats
    the real one. This is the slow, thorough check - catches nonlinear /
    interaction effects the linear L1 model can't see."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    aligned = feats.join(pnl.rename("pnl_pct"), how="inner").dropna(subset=["pnl_pct"])
    X = aligned[feats.columns].fillna(aligned[feats.columns].median())
    y = aligned["pnl_pct"].values
    rng = np.random.RandomState(random_state)

    if len(X) < 15:
        return pd.DataFrame({"feature": feats.columns, "rf_importance": np.nan, "p_value": np.nan})

    rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, n_jobs=-1, random_state=random_state)
    rf.fit(X, y)
    observed = permutation_importance(rf, X, y, n_repeats=20, n_jobs=-1, random_state=random_state).importances_mean

    null_importances = np.zeros((n_shuffles, X.shape[1]))
    for i in range(n_shuffles):
        y_shuff = rng.permutation(y)
        rf_null = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                         n_jobs=-1, random_state=rng.randint(1_000_000))
        rf_null.fit(X, y_shuff)
        pi_null = permutation_importance(rf_null, X, y_shuff, n_repeats=3, n_jobs=-1, random_state=random_state)
        null_importances[i] = pi_null.importances_mean

    p_values = (np.sum(null_importances >= observed, axis=0) + 1) / (n_shuffles + 1)
    return pd.DataFrame({"feature": X.columns, "rf_importance": observed, "p_value": p_values}).sort_values("p_value")


def discover_significant_features(feats: pd.DataFrame, pnl: pd.Series, alpha: float = 0.15) -> dict:
    """Runs all three checks, returns the full report plus the 'agreed significant'
    feature list (must clear the bar on >=2 of the 3 methods)."""
    uni = univariate_significance(feats, pnl)
    l1 = l1_logistic_selection(feats, pnl)
    rf = rf_permutation_significance(feats, pnl)

    uni_sig = set(uni.loc[uni["q_value"] < alpha, "feature"])
    l1_sig = set(l1["coefs"][l1["coefs"].abs() > 1e-6].index) if l1 else set()
    rf_sig = set(rf.loc[rf["p_value"] < alpha, "feature"])

    vote_counts = pd.Series(0, index=feats.columns)
    for s in (uni_sig, l1_sig, rf_sig):
        vote_counts.loc[list(s)] += 1

    # order agreed-significant by strength of evidence, not alphabetically -
    # spread_pct and theta have been the two most consistently robust features
    # across every run so far, and an alphabetical sort was silently dropping
    # them whenever max_total_dims truncated the list (e.g. 'bid'/'delta' sort
    # before 'spread_pct'/'theta' but are far less robust) - rank by vote count
    # first, |Spearman rho| as the tie-break among equally-voted features.
    abs_rho = uni.set_index("feature")["spearman_rho"].abs().reindex(feats.columns).fillna(0)
    order = pd.DataFrame({"votes": vote_counts, "abs_rho": abs_rho}).sort_values(
        ["votes", "abs_rho"], ascending=[False, False]).index
    agreed = [f for f in order if vote_counts[f] >= 2]

    return {"univariate": uni, "l1": l1, "rf_permutation": rf,
            "uni_sig": uni_sig, "l1_sig": l1_sig, "rf_sig": rf_sig,
            "vote_counts": vote_counts.sort_values(ascending=False),
            "agreed_significant": agreed}


def fit_rule_gate_from_discovery(feats: pd.DataFrame, pnl: pd.Series, discovery: dict,
                                  min_keep_frac: float = 0.4, max_total_dims: int = 5):
    """Builds a simple, interpretable threshold rule. Two kinds of features can
    end up in it:
      - FORCED: the LLM scorer's own outputs (ALWAYS_CONSIDERED_FOR_GATE) -
        always tested, regardless of whether they cleared the significance vote.
      - DISCOVERED: whatever else survived >=2/3 significance checks, filling
        any remaining slots up to max_total_dims.
    If nothing forced or discovered is usable, falls back to the single
    strongest univariate feature (marked exploratory). Direction (keep high vs
    keep low) comes from the sign of the Spearman correlation - even for
    forced features where that correlation wasn't strong enough to be
    'significant', the sign is still the best available guess at direction."""
    uni = discovery["univariate"].set_index("feature")

    forced = [f for f in ALWAYS_CONSIDERED_FOR_GATE if f in feats.columns]
    remaining_slots = max(0, max_total_dims - len(forced))
    discovered = [f for f in discovery["agreed_significant"] if f not in forced][:remaining_slots]
    origin = {f: "forced (LLM score)" for f in forced}
    origin.update({f: "discovered (>=2/3 significance methods)" for f in discovered})
    chosen = forced + discovered

    exploratory = False
    if not chosen:
        exploratory = True
        chosen = uni["p_value"].dropna().sort_values().head(1).index.tolist()
        origin = {f: "exploratory (nothing else available)" for f in chosen}
    if not chosen:
        return None

    n = len(feats)
    min_keep = max(5, int(round(n * min_keep_frac)))

    grids = {}
    for f in chosen:
        rho = uni.loc[f, "spearman_rho"] if f in uni.index else 0
        direction = "min" if rho >= 0 else "max"  # positive rho -> higher is better -> use as a floor
        qs = [0.0, 0.25, 0.5] if direction == "min" else [1.0, 0.75, 0.5]
        grids[f] = [(direction, feats[f].quantile(q)) for q in qs]

    best = None
    for combo in product(*grids.values()):
        keep = pd.Series(True, index=feats.index)
        thresholds = {}
        for f, (direction, val) in zip(chosen, combo):
            keep &= (feats[f] >= val) if direction == "min" else (feats[f] <= val)
            thresholds[f] = (direction, val)
        n_keep = keep.sum()
        if n_keep < min_keep:
            continue
        avg_pnl = pnl[keep.reindex(pnl.index, fill_value=False)].mean()
        if best is None or avg_pnl > best["avg_pnl"]:
            best = {"thresholds": thresholds, "feature_origin": origin,
                    "avg_pnl": avg_pnl, "n_keep": int(n_keep), "exploratory": exploratory}
    return best


def apply_rule_gate(feats: pd.DataFrame, rule: dict) -> pd.Series:
    if rule is None:
        return pd.Series(True, index=feats.index)
    keep = pd.Series(True, index=feats.index)
    for f, (direction, val) in rule["thresholds"].items():
        keep &= (feats[f] >= val) if direction == "min" else (feats[f] <= val)
    return keep


def fit_model_gate(feats: pd.DataFrame, pnl: pd.Series, l1_result: dict, min_keep_frac: float = 0.4):
    """Reuses the L1 logistic fit from discovery (same sparsity-based feature
    selection) as the model-based gate, choosing a probability threshold on
    TRAIN that maximizes avg P&L among kept trades subject to a minimum
    keep-fraction, same anti-degenerate guard as the rule gate."""
    if l1_result is None:
        return None
    X = feats[l1_result["feature_cols"]].fillna(l1_result["median"])
    Xs = l1_result["scaler"].transform(X)
    proba_train = l1_result["clf"].predict_proba(Xs)[:, 1]

    n = len(X)
    min_keep = max(5, int(round(n * min_keep_frac)))
    best = None
    for q in np.linspace(0.0, 0.9, 19):
        thresh = np.quantile(proba_train, q)
        keep = proba_train >= thresh
        if keep.sum() < min_keep:
            continue
        avg_pnl = pnl.reindex(X.index)[keep].mean()
        if best is None or avg_pnl > best["avg_pnl"]:
            best = {"threshold": thresh, "avg_pnl": avg_pnl, "n_keep": int(keep.sum())}
    if best is None:
        return None
    return {"clf": l1_result["clf"], "scaler": l1_result["scaler"],
            "feature_cols": l1_result["feature_cols"], "median": l1_result["median"],
            "threshold": best["threshold"]}


def apply_model_gate(feats: pd.DataFrame, gate) -> pd.Series:
    if gate is None:
        return pd.Series(True, index=feats.index)
    X = feats[gate["feature_cols"]].fillna(gate["median"])
    Xs = gate["scaler"].transform(X)
    proba = gate["clf"].predict_proba(Xs)[:, 1]
    return pd.Series(proba >= gate["threshold"], index=feats.index)


# ----------------------------------------------------------------------------
# 6. FULL PIPELINE
# ----------------------------------------------------------------------------

def evaluate_split(exits: pd.DataFrame, ids) -> dict:
    sub = exits[exits["signal_id"].isin(ids)]
    n = len(sub)
    if n == 0:
        return {"n_signals": 0, "win_rate": np.nan, "avg_pnl_pct": np.nan, "median_pnl_pct": np.nan,
                "p10_pnl_pct": np.nan, "p90_pnl_pct": np.nan, "worst_pnl_pct": np.nan, "best_pnl_pct": np.nan}
    pnl = sub["pnl_pct"]
    return {"n_signals": n, "win_rate": (pnl > 0).mean(),
            "avg_pnl_pct": pnl.mean(), "median_pnl_pct": pnl.median(),
            "p10_pnl_pct": pnl.quantile(0.10), "p90_pnl_pct": pnl.quantile(0.90),
            "worst_pnl_pct": pnl.min(), "best_pnl_pct": pnl.max()}


def run_combo_with_discovery(df, entry_df, selection_name, exit_name, train_ids, test_ids):
    picked = SELECTION_RULES[selection_name](entry_df)
    picked_df = picked.reset_index()
    paths = df.merge(picked_df, on=["signal_id", "symbol"], how="inner")
    exits = simulate(paths, exit_name)
    pnl_by_signal = exits.set_index("signal_id")["pnl_pct"]

    feats, dropped_zero_var = build_broad_feature_table(entry_df, picked)

    train_feats = feats.loc[feats.index.intersection(train_ids)]
    train_pnl = pnl_by_signal.loc[pnl_by_signal.index.intersection(train_ids)]

    print(f"\n  Candidate features ({feats.shape[1]}): {list(feats.columns)}")
    if dropped_zero_var:
        print(f"  Dropped (zero variance in this data): {dropped_zero_var}")

    discovery = discover_significant_features(train_feats, train_pnl)
    print("\n  --- Feature significance (TRAIN only) ---")
    print("  Univariate (Spearman vs P&L, BH-corrected):")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(discovery["univariate"].to_string(index=False))
    print("\n  L1-logistic non-zero coefficients:")
    if discovery["l1"] is not None:
        nz = discovery["l1"]["coefs"][discovery["l1"]["coefs"].abs() > 1e-6]
        print(nz.to_string() if len(nz) else "    (none survived L1 shrinkage)")
    else:
        print("    (could not fit - insufficient class balance)")
    print("\n  RF permutation importance (label-shuffle p-values):")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(discovery["rf_permutation"].to_string(index=False))
    print(f"\n  Vote counts (out of 3 methods):\n{discovery['vote_counts'].to_string()}")
    print(f"\n  AGREED SIGNIFICANT (>=2/3 methods): {discovery['agreed_significant'] or '(none - see caveat)'}")

    # --- LLM score features get their own callout, regardless of whether they
    # cleared the significance vote above - see ALWAYS_CONSIDERED_FOR_GATE ---
    score_feats = [f for f in ALWAYS_CONSIDERED_FOR_GATE if f in feats.columns]
    if score_feats:
        print("\n  --- LLM SCORE FEATURES (magnitude/confidence/catalyst) ---")
        print("  These are the scorer's own outputs. They're always included as gate")
        print("  candidates below regardless of the vote above - see the comment on")
        print("  ALWAYS_CONSIDERED_FOR_GATE for why a 'not significant' result here")
        print("  doesn't necessarily mean the score doesn't matter.")
        uni_scores = discovery["univariate"].set_index("feature").loc[score_feats]
        print(uni_scores.to_string())
        print("\n  Observed range in TRAIN data (narrower than this = restricted range,")
        print("  likely because passes_news_call_gate already filtered on these scores")
        print("  before any row reached this file):")
        print(train_feats[score_feats].agg(["min", "max"]).to_string())

    rule_gate = fit_rule_gate_from_discovery(train_feats, train_pnl, discovery)
    model_gate = fit_model_gate(train_feats, train_pnl, discovery["l1"])

    keep_rule = apply_rule_gate(feats, rule_gate)
    keep_model = apply_model_gate(feats, model_gate)

    rows = []
    for gate_label, keep_mask in [("no_gate", pd.Series(True, index=feats.index)),
                                   ("rule_gate", keep_rule),
                                   ("model_gate", keep_model)]:
        kept_ids = set(keep_mask[keep_mask].index)
        for split_label, ids in [("train", train_ids), ("test", test_ids)]:
            stats = evaluate_split(exits, ids & kept_ids)
            stats.update({"gate": gate_label, "split": split_label})
            rows.append(stats)

    return pd.DataFrame(rows), discovery, rule_gate, model_gate, exits, keep_rule, keep_model


def notable_trades(df: pd.DataFrame, exits: pd.DataFrame, ids, keep_rule: pd.Series,
                    keep_model: pd.Series, n: int = 5) -> pd.DataFrame:
    """Identifying details (ticker/headline/exit reason) for the N best and N
    worst trades in a split, with flags for whether each gate kept them - so a
    standout trade (good or bad) can actually be looked up and sanity-checked
    instead of just trusted as a number in an average."""
    sub = exits[exits["signal_id"].isin(ids)].copy()
    info_cols = [c for c in ["ticker", "headline", "signal_ts"] if c in df.columns]
    if info_cols:
        info = df.groupby("signal_id")[info_cols].first()
        sub = sub.merge(info, on="signal_id", how="left")
    for col in ["ticker", "headline"]:
        if col not in sub.columns:
            sub[col] = ""
    sub["kept_by_rule_gate"] = sub["signal_id"].map(keep_rule.to_dict()).fillna(False)
    sub["kept_by_model_gate"] = sub["signal_id"].map(keep_model.to_dict()).fillna(False)
    sub = sub.sort_values("pnl_pct", ascending=False)
    cols = ["signal_id", "ticker", "headline", "pnl_pct", "exit_reason", "minutes_held",
            "kept_by_rule_gate", "kept_by_model_gate"]
    cols = [c for c in cols if c in sub.columns]
    best = sub.head(n)[cols]
    worst = sub.tail(n)[cols].sort_values("pnl_pct")
    return best, worst


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "contract_grid_snapshots.csv"
    print(f"Loading {path} (Pass 1: entry snapshots and dataset diagnostics) ...")
    entry_df, total_rows, rows_per_signal = load_entry_snapshot_and_stats(path)
    signal_meta = get_signal_meta(entry_df)
    print(f"Loaded {total_rows:,} total rows across {len(signal_meta)} signals, {entry_df['ticker'].nunique()} tickers.")

    train_ids, test_ids = chronological_split(signal_meta, test_frac=0.3)
    print(f"Chronological split: {len(train_ids)} train signals, {len(test_ids)} test signals.")

    # --- Coverage diagnostics: is the data what you think it is? ---
    ts_min, ts_max = signal_meta["signal_ts"].min(), signal_meta["signal_ts"].max()
    span_days = max((ts_max - ts_min).total_seconds() / 86400, 1e-6)
    print(f"\nSignal date range: {ts_min} to {ts_max}  ({span_days:.1f} days span)")
    print(f"Signals per day (avg over span): {len(signal_meta) / span_days:.1f}")

    signal_days = pd.to_datetime(signal_meta["signal_ts"]).dt.date
    all_days = pd.date_range(ts_min.date(), ts_max.date(), freq="D").date
    missing_days = sorted(set(all_days) - set(signal_days))
    if missing_days:
        print(f"Days with ZERO signals in this range ({len(missing_days)} of {len(all_days)} days) "
              f"- check for weekends/holidays vs actual bot downtime: {missing_days}")
    else:
        print("No calendar days with zero signals in range.")

    print(f"Rows (snapshots) per signal: mean={rows_per_signal.mean():.0f}, "
          f"median={rows_per_signal.median():.0f}, min={rows_per_signal.min()}, max={rows_per_signal.max()}")
    print("  If this differs a lot from a previous run, the bot's polling interval or tracked-window")
    print("  length may have changed mid-collection - that would make exit-horizon rules (fixed_Nmin,")
    print("  trailing stops) less comparable between older and newer signals within this same file.")

    for flag in ["regime_uptrend", "regime_mom_ok", "regime_chop_ok"]:
        if flag in entry_df.columns:
            vc = entry_df.groupby("signal_id")[flag].first().value_counts()
            print(f"  {flag} across signals: {dict(vc)}")

    # Identify candidate contracts selected by any selection rule
    all_selected_pairs = set()
    for sel_name, rule_fn in SELECTION_RULES.items():
        picked = rule_fn(entry_df)
        all_selected_pairs.update(zip(picked.index, picked.values))

    print(f"\nPass 2: Loading price paths for {len(all_selected_pairs):,} selected contracts ...")
    paths_df = load_selected_paths(path, all_selected_pairs)
    print(f"Extracted {len(paths_df):,} path snapshots ({paths_df.memory_usage(deep=True).sum() / 1e6:.1f} MB in RAM).")

    if len(paths_df) == 0:
        print("\nNo price paths found for candidate contracts in dataset (missing bid/ask data or empty chain). Exiting.")
        sys.exit(0)

    print("\nRanking selection x exit combos on TRAIN data only ...")
    train_rank_rows = []
    for sel_name in SELECTION_RULES:
        picked = SELECTION_RULES[sel_name](entry_df)
        picked_df = picked.reset_index()
        paths = paths_df.merge(picked_df, on=["signal_id", "symbol"], how="inner")
        for exit_name in EXIT_STRATEGIES:
            exits = simulate(paths, exit_name)
            stats = evaluate_split(exits, train_ids)
            stats.update({"selection_rule": sel_name, "exit_rule": exit_name})
            train_rank_rows.append(stats)
    train_rank = pd.DataFrame(train_rank_rows).sort_values("avg_pnl_pct", ascending=False)
    train_rank.to_csv("train_ranked_combos.csv", index=False)

    best_row = train_rank.iloc[0]
    best_sel, best_exit = best_row["selection_rule"], best_row["exit_rule"]
    print(f"Best combo on TRAIN: {best_sel} + {best_exit} "
          f"(train avg_pnl={best_row['avg_pnl_pct']:.3%}, n={best_row['n_signals']:.0f})")

    print(f"\nRunning feature discovery + gate fitting for {best_sel} + {best_exit} ...")
    results, discovery, rule_gate, model_gate, exits, keep_rule, keep_model = run_combo_with_discovery(
        paths_df, entry_df, best_sel, best_exit, train_ids, test_ids
    )
    results.to_csv("train_test_gate_validation.csv", index=False)

    print("\n  --- TRAIN vs TEST, with and without each gate ---")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(results[["gate", "split", "n_signals", "win_rate", "avg_pnl_pct", "median_pnl_pct",
                        "p10_pnl_pct", "p90_pnl_pct", "worst_pnl_pct", "best_pnl_pct"]].to_string(index=False))
    print("\n  win_rate can look low next to a positive avg_pnl_pct for options - that's expected,")
    print("  not a contradiction: a few large winners (see best_pnl_pct / p90) can outweigh many")
    print("  small/total losses (worst_pnl_pct is capped near -100%, upside isn't capped the same way).")
    print("  Check p90/best vs worst/p10 before trusting an average built on a handful of signals.")

    print("\n  Fitted rule gate:", rule_gate)
    print("  Fitted model gate probability threshold:", None if model_gate is None else model_gate["threshold"])

    print("\n  --- Notable trades in TEST (identify outliers before trusting an average) ---")
    best_trades, worst_trades = notable_trades(entry_df, exits, test_ids, keep_rule, keep_model, n=5)
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.max_colwidth", 40):
        print("  Best 5:")
        print(best_trades.to_string(index=False))
        print("\n  Worst 5:")
        print(worst_trades.to_string(index=False))

    print("\nSaved: train_ranked_combos.csv, train_test_gate_validation.csv")
    print("\nA gate that helps on train but not test is a real, useful finding at this")
    print("stage - it means the discovered features are overfit to n~80, not that gating")
    print("itself is wrong. Re-run as more data accumulates and watch the agreed-significant")
    print("feature set stabilize (or not).")