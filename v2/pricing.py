"""
pricing.py (v2) — minimal Black-Scholes subset needed for the contract-mispricing check
(execution.py's pick_call_contract). Kept as a separate copy from core/pricing.py since v2 is
deliberately self-contained (no imports from core/) -- same pattern as sec_edgar.py.

No scipy dependency -- uses statistics.NormalDist for the normal CDF, same as core/pricing.py.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    """Black-Scholes price of a European call. Degenerate inputs collapse to discounted intrinsic."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K * math.exp(-r * max(T, 0.0)))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def implied_vol_call(price: float, S: float, K: float, T: float,
                     r: float = 0.04, lo: float = 1e-3, hi: float = 5.0) -> float | None:
    """Back out implied vol from an observed call price via bisection. Returns None if the price
    is outside the arbitrage bounds (below intrinsic or above spot -- not invertible)."""
    if T <= 0 or price <= 0 or S <= 0:
        return None
    intrinsic = max(0.0, S - K * math.exp(-r * T))
    if price < intrinsic - 1e-6 or price >= S:
        return None
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
