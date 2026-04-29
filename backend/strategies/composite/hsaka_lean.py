"""
HSAKA Lean Strategy — Production Module
2-component confluence: Market Structure + Swing Failure Pattern (SFP)

Key implementation details (matching validated code):
- Structure uses argrelextrema + PERSISTENT state (bias[idx:] = trend)
- Each component scores ±2 (max score 4)
- ct=2 means structure alone can generate signal
- NO ffill

Proven edge (8/13 assets significant on 4h):
- BTC 4h: OOS 2.008, p=0.009, 93.7% invested
- ETH 4h: OOS 1.581, p=0.024
- RENDER: OOS 2.828, TAO 2.026, FET 1.758, ARB 1.759
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def market_structure(df: pd.DataFrame, sw: int = 10) -> np.ndarray:
    """
    Persistent market structure via argrelextrema swing detection.
    Once a trend is detected, it persists until the next swing point
    changes direction. This is the key difference from naive window-based
    structure detection.
    
    Returns: array of +1 (bullish) or -1 (bearish) or 0 (undetermined)
    """
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Find swing highs and lows using scipy
    sh_idx = argrelextrema(high, np.greater_equal, order=sw)[0]
    sl_idx = argrelextrema(low, np.less_equal, order=sw)[0]

    # Build sorted list of all swing points
    all_swings = (
        [(idx, high[idx], "H") for idx in sh_idx] +
        [(idx, low[idx], "L") for idx in sl_idx]
    )
    all_swings.sort(key=lambda x: x[0])

    bias = np.zeros(n)
    prev_high = prev_low = None
    trend = 0

    for idx, price, stype in all_swings:
        if stype == "H":
            if prev_high is not None:
                trend = 1 if price > prev_high else -1
            prev_high = price
        else:
            if prev_low is not None:
                trend = 1 if price > prev_low else -1
            prev_low = price
        bias[idx:] = trend  # Persistent!

    return bias


def swing_failure_pattern(df: pd.DataFrame, sfp_lb: int = 20) -> np.ndarray:
    """
    Swing Failure Pattern — false breakout reversal signal.
    Bearish SFP: wick above prev high, close below it.
    Bullish SFP: wick below prev low, close above it.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(high)
    sig = np.zeros(n)

    for i in range(sfp_lb, n):
        prev_high = np.max(high[i - sfp_lb:i])
        prev_low = np.min(low[i - sfp_lb:i])

        if high[i] > prev_high and close[i] < prev_high:
            sig[i] = -1  # bearish SFP
        elif low[i] < prev_low and close[i] > prev_low:
            sig[i] = 1   # bullish SFP

    return sig


def generate_signal(df: pd.DataFrame, sw: int = 10, sfp_lb: int = 20,
                    ct: int = 2) -> np.ndarray:
    """
    Generate HSAKA Lean signal.
    
    Score = structure * 2 + sfp * 2 (max ±4)
    Signal = +1 when score >= ct, -1 when score <= -ct, else 0.
    
    With ct=2: structure alone (±2) is sufficient to generate signal.
    SFP adds conviction but isn't required.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
        sw: structure window for argrelextrema order parameter
        sfp_lb: lookback period for swing failure pattern
        ct: confluence threshold (2-4)
    
    Returns:
        signal: array of +1 (long), -1 (short), 0 (flat)
    """
    struct = market_structure(df, sw)
    sfp = swing_failure_pattern(df, sfp_lb)

    score = struct * 2 + sfp * 2
    signal = np.where(score >= ct, 1, np.where(score <= -ct, -1, 0))

    return signal


# Default per-asset optimal parameters (from walk-forward optimization)
OPTIMAL_PARAMS = {
    "btc":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "eth":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "sol":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "link":   {"sw": 10, "sfp_lb": 20, "ct": 2},
    "avax":   {"sw": 10, "sfp_lb": 20, "ct": 2},
    "sui":    {"sw": 14, "sfp_lb": 20, "ct": 2},
    "inj":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "arb":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "op":     {"sw": 10, "sfp_lb": 20, "ct": 2},
    "render": {"sw": 14, "sfp_lb": 20, "ct": 2},
    "tia":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "tao":    {"sw": 10, "sfp_lb": 20, "ct": 2},
    "fet":    {"sw": 10, "sfp_lb": 20, "ct": 2},
}


if __name__ == "__main__":
    import sys
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    tf = sys.argv[2] if len(sys.argv) > 2 else "4h"

    path = f"/home/linuxuser/Desktop/maestro/data/ohlcv/binance_{asset}_usdt_{tf}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])

    params = OPTIMAL_PARAMS.get(asset, {"sw": 10, "sfp_lb": 20, "ct": 2})
    signal = generate_signal(df, **params)

    longs = np.sum(signal == 1)
    shorts = np.sum(signal == -1)
    flat = np.sum(signal == 0)
    invested = np.mean(np.abs(signal)) * 100

    print(f"HSAKA Lean | {asset.upper()} {tf} | sw={params['sw']} sfp={params['sfp_lb']} ct={params['ct']}")
    print(f"Long: {longs} | Short: {shorts} | Flat: {flat} | Invested: {invested:.1f}%")
    print(f"Current signal: {int(signal[-1])}")
