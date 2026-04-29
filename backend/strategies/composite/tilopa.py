"""Young Tilopa Strategy — Liquidity-Driven Price Action

Combines 5 core components to identify high-probability setups:
1. Market Structure (Engulfment & FTR zones)
2. Liquidity Voids / Illiquid Moves
3. Compression Detection
4. Volume Profile Concepts (POC, LVN)
5. Flag Limits (impulsive move origins)

Philosophy: Price is driven by liquidity. The strategy hunts for moments when
price returns to supply/demand zones (FTR, flag limits, illiquid move zones)
while structural bias and compression indicate a high-probability setup.

Signal generation uses confluence scoring - multiple components must align.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

NAME = "YoungTilopa"
CATEGORY = "composite"
DESCRIPTION = "Liquidity-driven price action with confluence scoring"
REQUIRES_DERIVATIVES = False


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    return atr


def detect_swing_points(df: pd.DataFrame, window: int = 5):
    """Detect swing highs and lows using scipy."""
    high = df['high'].values
    low = df['low'].values
    
    swing_high_idx = argrelextrema(high, np.greater_equal, order=window)[0]
    swing_low_idx = argrelextrema(low, np.less_equal, order=window)[0]
    
    # Create series
    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)
    
    swing_highs.iloc[swing_high_idx] = high[swing_high_idx]
    swing_lows.iloc[swing_low_idx] = low[swing_low_idx]
    
    return swing_highs, swing_lows


def detect_market_structure(df: pd.DataFrame, swing_highs: pd.Series, swing_lows: pd.Series):
    """
    Component 1: Market Structure (HH/HL vs LH/LL) + Engulfment + FTR zones.
    
    Returns:
        structural_bias: +1 (bullish), -1 (bearish), 0 (neutral)
        ftr_zones: List of (start_idx, end_idx, zone_low, zone_high, direction)
    """
    structural_bias = pd.Series(0, index=df.index)
    
    # Track swing highs and lows
    high_swings = swing_highs.dropna()
    low_swings = swing_lows.dropna()
    
    ftr_zones = []
    
    # Combine and sort
    all_swings = []
    for idx, val in high_swings.items():
        all_swings.append((idx, val, 'H'))
    for idx, val in low_swings.items():
        all_swings.append((idx, val, 'L'))
    all_swings.sort(key=lambda x: x[0])
    
    if len(all_swings) < 4:
        return structural_bias, ftr_zones
    
    prev_high = None
    prev_low = None
    trend = 0
    
    for i, (idx, price, stype) in enumerate(all_swings):
        if stype == 'H':
            if prev_high is not None:
                # Check for HH or LH
                if price > prev_high:
                    trend = 1  # HH -> bullish
                    # FTR zone = consolidation before breakout
                    prev_swing_idx = all_swings[i-1][0] if i > 0 else idx
                    ftr_start = df.index.get_loc(prev_swing_idx)
                    ftr_end = df.index.get_loc(idx)
                    if ftr_end - ftr_start > 2:
                        zone_low = df['low'].iloc[ftr_start:ftr_end].min()
                        zone_high = df['high'].iloc[ftr_start:ftr_end].max()
                        ftr_zones.append((ftr_start, ftr_end, zone_low, zone_high, 1))
                else:
                    trend = -1  # LH -> bearish
                    prev_swing_idx = all_swings[i-1][0] if i > 0 else idx
                    ftr_start = df.index.get_loc(prev_swing_idx)
                    ftr_end = df.index.get_loc(idx)
                    if ftr_end - ftr_start > 2:
                        zone_low = df['low'].iloc[ftr_start:ftr_end].min()
                        zone_high = df['high'].iloc[ftr_start:ftr_end].max()
                        ftr_zones.append((ftr_start, ftr_end, zone_low, zone_high, -1))
            prev_high = price
        
        elif stype == 'L':
            if prev_low is not None:
                if price > prev_low:
                    trend = 1  # HL -> bullish
                else:
                    trend = -1  # LL -> bearish
            prev_low = price
        
        # Assign trend from this swing forward
        loc = df.index.get_loc(idx)
        structural_bias.iloc[loc:] = trend
    
    return structural_bias, ftr_zones


def detect_illiquid_moves(df: pd.DataFrame, atr: pd.Series, 
                          body_threshold: float = 1.5, 
                          vol_factor: float = 0.7):
    """
    Component 2: Liquidity Voids / Illiquid Moves.
    
    Detect candles with large body relative to ATR and low volume.
    These are areas price will likely revisit to "fill the void".
    
    Returns:
        illiquid_zones: List of (idx, zone_low, zone_high, direction)
    """
    body = abs(df['close'] - df['open'])
    vol_sma = df['volume'].rolling(20).mean()
    
    illiquid = (body / atr > body_threshold) & (df['volume'] < vol_sma * vol_factor)
    
    illiquid_zones = []
    for idx in df[illiquid].index:
        loc = df.index.get_loc(idx)
        zone_low = min(df['open'].iloc[loc], df['close'].iloc[loc])
        zone_high = max(df['open'].iloc[loc], df['close'].iloc[loc])
        direction = 1 if df['close'].iloc[loc] > df['open'].iloc[loc] else -1
        illiquid_zones.append((loc, zone_low, zone_high, direction))
    
    return illiquid_zones


def detect_compression(df: pd.DataFrame, atr: pd.Series, lookback: int = 14, threshold: float = 0.75):
    """
    Component 3: Compression Detection.
    
    Measure compression as declining ATR or narrowing Bollinger Bands.
    Compression = ATR ratio (current ATR / ATR N periods ago) < threshold.
    
    Returns:
        compression: Boolean series indicating compressed states
    """
    atr_ratio = atr / atr.shift(lookback)
    compression = atr_ratio < threshold
    
    # Alternative: Bollinger Band Width
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    bb_width = (std * 2) / sma
    bb_compression = bb_width < bb_width.rolling(lookback).mean() * 0.8
    
    # Combine both
    compression = compression | bb_compression
    
    return compression.fillna(False)


def calculate_volume_profile(df: pd.DataFrame, lookback: int = 50):
    """
    Component 4: Volume Profile (POC and LVN).
    
    Calculate Point of Control (price with highest volume) and
    Low Volume Nodes (price areas with abnormally low volume).
    
    Returns:
        poc: Series of POC prices
        lvn: Boolean series indicating low volume nodes
    """
    poc = pd.Series(np.nan, index=df.index)
    lvn = pd.Series(False, index=df.index)
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        # Simple POC approximation: volume-weighted average price
        vwap = (window['volume'] * (window['high'] + window['low'] + window['close']) / 3).sum() / window['volume'].sum()
        poc.iloc[i] = vwap
        
        # LVN: current volume significantly below recent average
        vol_mean = window['volume'].mean()
        vol_std = window['volume'].std()
        is_lvn = df['volume'].iloc[i] < (vol_mean - vol_std)
        lvn.iloc[i] = is_lvn
    
    return poc, lvn


def detect_flag_limits(df: pd.DataFrame, atr: pd.Series, impulse_threshold: float = 2.0, min_consecutive: int = 2):
    """
    Component 5: Flag Limits.
    
    Detect impulsive moves (consecutive candles with high range in same direction)
    and mark the origin candle(s) as flag limit zones.
    
    Returns:
        flag_zones: List of (idx, zone_low, zone_high, direction)
    """
    body = abs(df['close'] - df['open'])
    direction = np.sign(df['close'] - df['open'])
    
    # Detect large moves
    large_move = body > (atr * impulse_threshold)
    
    flag_zones = []
    
    i = 0
    while i < len(df) - min_consecutive:
        # Check for consecutive large moves in same direction
        if large_move.iloc[i]:
            consecutive = 1
            curr_dir = direction.iloc[i]
            
            j = i + 1
            while j < len(df) and large_move.iloc[j] and direction.iloc[j] == curr_dir:
                consecutive += 1
                j += 1
            
            if consecutive >= min_consecutive:
                # The candle before the impulse = flag limit
                if i > 0:
                    flag_idx = i - 1
                    zone_low = df['low'].iloc[flag_idx]
                    zone_high = df['high'].iloc[flag_idx]
                    flag_zones.append((flag_idx, zone_low, zone_high, int(curr_dir)))
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return flag_zones


def generate_signals(df: pd.DataFrame, 
                     swing_window: int = 7, 
                     confluence_threshold: int = 4,
                     compression_lookback: int = 14,
                     ftr_atr_mult: float = 1.5) -> pd.Series:
    """
    Generate +1 (long), -1 (short), 0 (flat) signals based on confluence.
    
    Parameters:
        swing_window: Window for swing high/low detection
        confluence_threshold: Minimum score to generate signal (3-5 recommended)
        compression_lookback: Lookback period for compression detection
        ftr_atr_mult: ATR multiplier for FTR zone width
    
    Returns:
        signals: Series with +1 (long), -1 (short), 0 (flat)
    """
    signals = pd.Series(0, index=df.index)
    
    if len(df) < 100:
        return signals
    
    # Calculate ATR
    atr = calculate_atr(df)
    
    # Component 1: Market Structure
    swing_highs, swing_lows = detect_swing_points(df, swing_window)
    structural_bias, ftr_zones = detect_market_structure(df, swing_highs, swing_lows)
    
    # Component 2: Illiquid Moves
    illiquid_zones = detect_illiquid_moves(df, atr)
    
    # Component 3: Compression
    compression = detect_compression(df, atr, compression_lookback)
    
    # Component 4: Volume Profile
    poc, lvn = calculate_volume_profile(df)
    
    # Component 5: Flag Limits
    flag_zones = detect_flag_limits(df, atr)
    
    # Confluence Scoring
    for i in range(swing_window * 2, len(df)):
        confluence_score = 0
        signal_direction = 0
        
        current_price = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr == 0:
            continue
        
        # 1. Structural Bias (+2)
        bias = structural_bias.iloc[i]
        if bias != 0:
            confluence_score += 2
            signal_direction = bias
        
        # 2. At FTR Zone (+2)
        at_ftr = False
        for ftr_start, ftr_end, zone_low, zone_high, ftr_dir in ftr_zones:
            zone_expanded_low = zone_low - current_atr * ftr_atr_mult
            zone_expanded_high = zone_high + current_atr * ftr_atr_mult
            
            if zone_expanded_low <= current_price <= zone_expanded_high and i > ftr_end:
                at_ftr = True
                if ftr_dir == bias:  # Direction agrees
                    confluence_score += 2
                break
        
        # 3. Compression (+1)
        if compression.iloc[i]:
            confluence_score += 1
        
        # 4. At Flag Limit (+1)
        at_flag = False
        for flag_idx, flag_low, flag_high, flag_dir in flag_zones:
            flag_zone_low = flag_low - current_atr
            flag_zone_high = flag_high + current_atr
            
            if flag_zone_low <= current_price <= flag_zone_high and i > flag_idx:
                at_flag = True
                if flag_dir == bias:
                    confluence_score += 1
                break
        
        # 5. Illiquid Move Approach (+1)
        illiquid_approach = False
        for illiq_idx, illiq_low, illiq_high, illiq_dir in illiquid_zones:
            if illiq_low <= current_price <= illiq_high and i > illiq_idx:
                illiquid_approach = True
                if illiq_dir == bias:
                    confluence_score += 1
                break
        
        # 6. Volume Confluence (+1)
        if lvn.iloc[i] and not pd.isna(poc.iloc[i]):
            # Price near POC + LVN = confluence
            if abs(current_price - poc.iloc[i]) < current_atr:
                confluence_score += 1
        
        # Generate Signal
        if confluence_score >= confluence_threshold and signal_direction != 0:
            signals.iloc[i] = signal_direction
    
    # Forward-fill signals (maintain position)
    signals = signals.replace(0, np.nan).ffill().fillna(0).astype(int)
    
    return signals
