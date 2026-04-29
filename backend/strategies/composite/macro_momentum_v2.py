"""
MacroMomentum V2 — Enhanced with Dip-Buying & Pyramiding
Entry: trend regime (SMA + ROC + M2 accel) then buy dips (RSI, EMA, BB, funding, MSTR)
Exit: trim on RSI/BB/ATR extension, trailing stop or macro regime off
Position sizing: pyramiding with macro score scaling
"""
import numpy as np
import pandas as pd

NAME = "MacroMomentumV2"
CATEGORY = "composite"
DESCRIPTION = "Macro-aware crypto momentum with dip-buying, pyramiding, and multi-signal exits"
REQUIRES_DERIVATIVES = False

DEFAULT_PARAMS = dict(
    sma_slow=120,
    momentum_period=25,
    roc_threshold=0.0,
    rsi_period=14,
    rsi_entry=45,
    rsi_exit=75,
    ema_period=21,
    bb_period=20,
    bb_std=2.0,
    trail_stop_pct=0.09,
    initial_size=0.3,
    pyramid_size=0.2,
    max_position=1.0,
    trim_pct=0.25,
    atr_exit_mult=2.0,
    macro_weight=1.0,
)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_bb(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signals(
    df: pd.DataFrame,
    macro_score: pd.Series = None,
    m2_accelerating: pd.Series = None,
    funding_zscore: pd.Series = None,
    mstr_excess: pd.Series = None,
    **params,
) -> pd.DataFrame:
    """
    Full V2 signal generation with pyramiding simulation.
    Returns DataFrame with columns: signal, position_size, entry_type
    """
    p = {**DEFAULT_PARAMS, **params}
    close = df["close"].copy()
    n = len(close)

    # Indicators
    sma_slow = close.rolling(p["sma_slow"]).mean()
    roc = close.pct_change(p["momentum_period"])
    rsi = _compute_rsi(close, p["rsi_period"])
    ema = close.ewm(span=p["ema_period"], adjust=False).mean()
    bb_lower, bb_mid, bb_upper = _compute_bb(close, p["bb_period"], p["bb_std"])
    atr = _compute_atr(df, 14)

    # Trend regime
    trend_up = close > sma_slow
    momentum_pos = roc > p["roc_threshold"]

    # M2 acceleration (external or default True)
    if m2_accelerating is not None:
        m2_acc = m2_accelerating.reindex(df.index, method="ffill").fillna(True).astype(bool)
    else:
        m2_acc = pd.Series(True, index=df.index)

    regime = trend_up & momentum_pos & m2_acc

    # Macro score multiplier
    if macro_score is not None:
        ms = macro_score.reindex(df.index, method="ffill").fillna(3).clip(0, 6)
        macro_mult = 0.5 + (ms / 6.0) * 0.5 * p["macro_weight"]  # [0.5, 1.0]
    else:
        macro_mult = pd.Series(1.0, index=df.index)

    # Dip entry conditions
    rsi_dip = rsi < p["rsi_entry"]
    ema_dip = close < ema  # price below EMA = mean reversion opportunity
    bb_dip = close <= bb_lower
    
    funding_dip = pd.Series(False, index=df.index)
    if funding_zscore is not None:
        fz = funding_zscore.reindex(df.index, method="ffill").fillna(0)
        funding_dip = fz < -1.0

    mstr_dip = pd.Series(False, index=df.index)
    if mstr_excess is not None:
        me = mstr_excess.reindex(df.index, method="ffill").fillna(0)
        mstr_dip = me < -0.05

    # Regime onset: regime just turned on (was off yesterday)
    regime_onset = regime & ~regime.shift(1).fillna(False)
    
    any_dip = rsi_dip | ema_dip | bb_dip | funding_dip | mstr_dip | regime_onset

    # Exit trim conditions
    rsi_hot = rsi > p["rsi_exit"]
    bb_hot = close > bb_upper
    # ATR extension computed in loop (needs avg entry)

    # Shift all conditions by 1 to avoid lookahead
    regime = regime.shift(1).fillna(False)
    any_dip = any_dip.shift(1).fillna(False)
    rsi_hot = rsi_hot.shift(1).fillna(False)
    bb_hot = bb_hot.shift(1).fillna(False)
    rsi_dip = rsi_dip.shift(1).fillna(False)
    ema_dip = ema_dip.shift(1).fillna(False)
    bb_dip = bb_dip.shift(1).fillna(False)
    funding_dip = funding_dip.shift(1).fillna(False)
    mstr_dip = mstr_dip.shift(1).fillna(False)
    m2_acc_shifted = m2_acc.shift(1).fillna(True)
    trend_up_shifted = trend_up.shift(1).fillna(False)
    atr_shifted = atr.shift(1).fillna(0)

    # Simulate pyramiding
    signals = np.zeros(n, dtype=int)
    pos_sizes = np.zeros(n, dtype=float)
    entry_types = [""] * n

    pos = 0.0  # current position size
    avg_entry = 0.0
    peak_equity = 1.0
    equity = 1.0
    total_cost = 0.0  # weighted cost basis

    warmup = max(p["sma_slow"], p["bb_period"], p["momentum_period"]) + 2

    for i in range(warmup, n):
        price = close.iloc[i]
        prev_price = close.iloc[i - 1] if i > 0 else price

        # Update equity
        if pos > 0 and prev_price > 0:
            daily_ret = (price - prev_price) / prev_price
            equity *= (1 + daily_ret * pos)
        peak_equity = max(peak_equity, equity)

        # Check trailing stop
        if pos > 0:
            drawdown = 1 - equity / peak_equity
            if drawdown >= p["trail_stop_pct"]:
                signals[i] = -1
                entry_types[i] = "trailing_stop"
                pos = 0.0
                avg_entry = 0.0
                total_cost = 0.0
                pos_sizes[i] = 0.0
                continue

        # Check macro regime exit
        if pos > 0 and not m2_acc_shifted.iloc[i] and not trend_up_shifted.iloc[i]:
            signals[i] = -1
            entry_types[i] = "macro_exit"
            pos = 0.0
            avg_entry = 0.0
            total_cost = 0.0
            pos_sizes[i] = 0.0
            continue

        # Trim conditions (if in position)
        if pos > 0:
            trimmed = False
            # RSI overbought trim
            if rsi_hot.iloc[i]:
                trim = pos * p["trim_pct"]
                pos -= trim
                entry_types[i] = "trim_rsi"
                trimmed = True
            # BB upper trim
            elif bb_hot.iloc[i]:
                trim = pos * p["trim_pct"]
                pos -= trim
                entry_types[i] = "trim_bb"
                trimmed = True
            # ATR extension trim
            elif avg_entry > 0 and atr_shifted.iloc[i] > 0:
                extension = (price - avg_entry) / atr_shifted.iloc[i]
                if extension > p["atr_exit_mult"]:
                    trim = pos * p["trim_pct"]
                    pos -= trim
                    entry_types[i] = "trim_atr"
                    trimmed = True

            if trimmed:
                pos = max(pos, 0.0)
                if pos < 0.01:
                    pos = 0.0
                    avg_entry = 0.0
                    total_cost = 0.0
                    signals[i] = -1
                else:
                    signals[i] = 1
                pos_sizes[i] = pos
                continue

        # Entry / pyramid
        if regime.iloc[i] and any_dip.iloc[i]:
            mm = float(macro_mult.iloc[i]) if hasattr(macro_mult, "iloc") else macro_mult
            max_pos = p["max_position"] * mm

            if pos == 0:
                # Initial entry
                add = p["initial_size"] * mm
                add = min(add, max_pos)
                pos = add
                avg_entry = price
                total_cost = price * add
                signals[i] = 1
                # Determine entry type
                if rsi_dip.iloc[i]:
                    entry_types[i] = "entry_rsi"
                elif ema_dip.iloc[i]:
                    entry_types[i] = "entry_ema"
                elif bb_dip.iloc[i]:
                    entry_types[i] = "entry_bb"
                elif funding_dip.iloc[i]:
                    entry_types[i] = "entry_funding"
                elif mstr_dip.iloc[i]:
                    entry_types[i] = "entry_mstr"
            elif pos < max_pos:
                # Pyramid
                add = p["pyramid_size"] * mm
                add = min(add, max_pos - pos)
                if add > 0.01:
                    total_cost += price * add
                    pos += add
                    avg_entry = total_cost / pos
                    signals[i] = 1
                    entry_types[i] = "pyramid"

        if pos > 0:
            signals[i] = max(signals[i], 1)
        pos_sizes[i] = pos

    result = pd.DataFrame(
        {"signal": signals, "position_size": pos_sizes, "entry_type": entry_types},
        index=df.index,
    )
    return result


def generate_signals_simple(df: pd.DataFrame, **params) -> pd.DataFrame:
    """Simple version using only price data (no external signals)."""
    return generate_signals(df, **params)
