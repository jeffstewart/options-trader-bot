"""
pricing.py — Black-Scholes option pricing + volatility utilities.

Used by the backtest to price long calls realistically off the DENSE stock
bars (the historical option tape is far too sparse — typically 2-10 prints
over a 6-week hold even for SPY/NVDA — to price options directly).

The model captures the three forces that move a long call:
  • delta/gamma — via the stock price path (S)
  • theta       — falls out naturally as time-to-expiry (T) shrinks each day
  • vega / IV crush — via an explicit implied-vol path that starts elevated at
                       a news event and decays back toward the underlying's
                       realized volatility (see iv_crush_path)

No scipy dependency — uses statistics.NormalDist for the normal CDF and its
inverse.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()          # standard normal
TRADING_DAYS = 252


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def norm_inv(p: float) -> float:
    """Inverse standard-normal CDF (quantile function)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return _N.inv_cdf(p)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """
    Black-Scholes price of a European call.
      S     spot price
      K     strike
      T     time to expiry in YEARS
      sigma annualized implied volatility (e.g. 0.45 = 45%)
      r     risk-free rate
    Degenerate inputs collapse to discounted intrinsic value.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K * math.exp(-r * max(T, 0.0)))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """Delta (N(d1)) of a European call."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return norm_cdf(d1)


def _norm_pdf(x: float) -> float:
    """Standard-normal probability density N'(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_call_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """Gamma (∂delta/∂S, per $1 move in spot). Identical for calls and puts.
    Matches Alpaca's snapshot convention. 0 on degenerate inputs."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, sigma, r)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_call_vega(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """Vega per 1-POINT (1% = 0.01) change in IV — matches Alpaca's snapshot convention.
    Identical for calls and puts. 0 on degenerate inputs."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, sigma, r)
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_call_theta(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """Theta per CALENDAR DAY (negative for long calls) — matches Alpaca's snapshot
    convention. 0 on degenerate inputs."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = _bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * sqrtT
    annual = (-(S * _norm_pdf(d1) * sigma) / (2.0 * sqrtT)
              - r * K * math.exp(-r * T) * norm_cdf(d2))
    return annual / 365.0


def bs_put_price(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """
    Black-Scholes price of a European put.
    Degenerate inputs collapse to discounted intrinsic value.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, K * math.exp(-r * max(T, 0.0)) - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def put_strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                         r: float = 0.04) -> float:
    """
    Strike for a put with the given (positive) target delta magnitude, e.g.
    target_delta=0.40 → a ~ -0.40-delta put (slightly OTM, below spot).
    Put delta = N(d1) - 1, so |delta| = N(-d1) = target → d1 = -norm_inv(target).
    """
    if T <= 0 or sigma <= 0:
        return S
    sqrtT = math.sqrt(T)
    d1 = -norm_inv(target_delta)
    ln_S_over_K = d1 * sigma * sqrtT - (r + 0.5 * sigma * sigma) * T
    return S / math.exp(ln_S_over_K)


def strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                     r: float = 0.04) -> float:
    """
    Invert delta = N(d1) to find the strike that yields `target_delta`.
    Returns a continuous strike (caller may round to a listed increment).
    """
    if T <= 0 or sigma <= 0:
        return S
    sqrtT = math.sqrt(T)
    d1 = norm_inv(target_delta)
    # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)  →  solve for K
    ln_S_over_K = d1 * sigma * sqrtT - (r + 0.5 * sigma * sigma) * T
    return S / math.exp(ln_S_over_K)


def implied_vol_call(price: float, S: float, K: float, T: float,
                     r: float = 0.04, lo: float = 1e-3, hi: float = 5.0) -> float | None:
    """
    Back out implied vol from an observed call price via bisection.
    Used to calibrate the IV model against real option prints.
    Returns None if the price is outside the arbitrage bounds.
    """
    if T <= 0 or price <= 0 or S <= 0:
        return None
    intrinsic = max(0.0, S - K * math.exp(-r * T))
    if price < intrinsic - 1e-6 or price >= S:
        return None  # below intrinsic or above spot — not invertible
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        diff = bs_call_price(S, K, T, mid, r) - price
        if abs(diff) < 1e-4:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def realized_vol(closes: list[float]) -> float | None:
    """
    Annualized close-to-close realized volatility from a list of daily closes
    (chronological order). Returns None if insufficient data.
    """
    if len(closes) < 5:
        return None
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 4:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def iv_crush_path(base_iv: float, days_held: float, news_multiplier: float,
                  halflife_days: float) -> float:
    """
    Implied vol on a given day of the hold, modeling post-news IV crush.

    At entry (days_held=0) IV is elevated to base_iv * news_multiplier; it
    decays exponentially back toward base_iv with the given half-life. This is
    the term the old delta-approximation backtest entirely ignored — and it is
    the main reason buying calls on news can lose even when the stock rises.
    """
    elevated = base_iv * news_multiplier
    if halflife_days <= 0:
        return base_iv
    decay = 0.5 ** (days_held / halflife_days)
    return base_iv + (elevated - base_iv) * decay
