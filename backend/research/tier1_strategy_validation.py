#!/usr/bin/env python3
"""
Tier 1 Strategy Walk-Forward Validation
========================================
Tests the top 3 book-derived strategies with rigorous methodology:

1. Funding Rate Carry (Carver) — harvest funding from crowded trades
2. NR4/NR7 Trend Continuation (Crabel) — volatility compression breakout
3. EWMAC Multi-Speed Trend Following (Carver) — multi-TF trend with vol scaling

Also includes:
4. OU Process Mean Reversion (Chan) — half-life based MR on spreads
5. Dual Momentum (Antonacci) — absolute + relative momentum

Methodology:
- 14-fold walk-forward (train folds 1-7, test folds 8-14)
- Non-overlapping trades
- Realistic costs (10bps per round trip)
- Bonferroni correction
- Multiple holding periods scaled to strategy type
- Proper annualization

Data sources:
- Spot candles: ~/Desktop/maestro/data/spot/{tf}/
- Funding rates: ~/Desktop/maestro/data/derivatives/*_funding_full.csv
- Derivatives: ~/Desktop/maestro/data/derivatives_5m/compiled/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
import sys
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

SPOT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/spot")
DERIV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/derivatives")
DERIV_5M_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/derivatives_5m/compiled")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 10
MIN_TRADES = 30
N_FOLDS = 14

ASSETS_FULL = [
    'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'AVAX', 'DOT',
    'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'OP', 'ARB', 'SUI',
    'DOGE', 'XRP', 'RENDER', 'FET', 'TIA', 'SEI', 'DYDX', 'INJ'
]

# Assets with funding data
FUNDING_ASSETS = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'OP', 'ARB',
                  'SUI', 'FET', 'RENDER', 'TIA', 'TAO', 'INJ']

# ============================================================
# DATA LOADING
# ============================================================

def load_spot(asset, tf):
    """Load spot candle data."""
    path = SPOT_DIR / tf / f"{asset}_spot_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('timestamp', 'date', 'time', 'datetime'):
            col_map[c] = 'timestamp'
        elif cl == 'open': col_map[c] = 'open'
        elif cl == 'high': col_map[c] = 'high'
        elif cl == 'low': col_map[c] = 'low'
        elif cl == 'close': col_map[c] = 'close'
        elif cl in ('volume', 'vol'): col_map[c] = 'volume'
    df = df.rename(columns=col_map)
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for r in required:
        if r not in df.columns:
            return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['close'])
    return df


def load_funding(asset):
    """Load funding rate data."""
    path = DERIV_DIR / f"{asset.lower()}_funding_full.csv"
    if not path.exists():
        path = DERIV_DIR / f"{asset.lower()}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Normalize funding rate column
    if 'funding_rate' in df.columns:
        pass
    elif 'c' in df.columns:
        df['funding_rate'] = df['c']
    else:
        return None
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', 'funding_rate']]


def load_spot_daily(asset):
    """Load daily spot data, trying multiple sources."""
    # Try 1d directory first
    for pattern in [
        SPOT_DIR / '1d' / f"{asset}_spot_1d.csv",
        SPOT_DIR / f"{asset}_spot_daily.csv",
    ]:
        if pattern.exists():
            return load_spot_generic(pattern)
    # Fall back to 4h and resample
    df_4h = load_spot(asset, '4h')
    if df_4h is not None:
        df_4h = df_4h.set_index('timestamp')
        daily = df_4h.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        return daily
    return None


def load_spot_generic(path):
    """Generic spot loader."""
    df = pd.read_csv(path)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('timestamp', 'date', 'time', 'datetime'): col_map[c] = 'timestamp'
        elif cl == 'open': col_map[c] = 'open'
        elif cl == 'high': col_map[c] = 'high'
        elif cl == 'low': col_map[c] = 'low'
        elif cl == 'close': col_map[c] = 'close'
        elif cl in ('volume', 'vol'): col_map[c] = 'volume'
    df = df.rename(columns=col_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['close'])
    return df


# ============================================================
# INDICATOR HELPERS
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx


# ============================================================
# STRATEGY 1: FUNDING RATE CARRY
# ============================================================

def strategy_funding_carry(spot_df, funding_df, params):
    """
    Harvest funding rate carry by going opposite to crowded trades.
    
    When funding is very positive (longs pay shorts) → short (collect funding)
    When funding is very negative (shorts pay longs) → long (collect funding)
    
    The key insight: you get PAID the funding rate while holding the position.
    Edge comes from both funding income AND mean-reversion of overextended sentiment.
    """
    # Merge spot and funding on date
    spot = spot_df.copy()
    spot['date'] = spot['timestamp'].dt.date
    fund = funding_df.copy()
    fund['date'] = fund['timestamp'].dt.date
    
    merged = spot.merge(fund[['date', 'funding_rate']], on='date', how='left')
    merged['funding_rate'] = merged['funding_rate'].ffill()
    
    if merged['funding_rate'].isna().all():
        return None, None
    
    # Rolling average funding rate
    avg_window = params['avg_window']
    merged['funding_avg'] = merged['funding_rate'].rolling(avg_window, min_periods=1).mean()
    
    # Z-score of funding rate
    funding_std = merged['funding_rate'].rolling(params['zscore_window'], min_periods=20).std()
    funding_mean = merged['funding_rate'].rolling(params['zscore_window'], min_periods=20).mean()
    merged['funding_zscore'] = (merged['funding_rate'] - funding_mean) / (funding_std + 1e-10)
    
    threshold = params['threshold']
    signals = pd.Series(0, index=merged.index)
    
    if params['mode'] == 'simple':
        # Simple: trade when funding exceeds threshold
        signals[merged['funding_avg'] > threshold] = -1  # short (collect positive funding)
        signals[merged['funding_avg'] < -threshold] = 1   # long (collect negative funding)
    elif params['mode'] == 'zscore':
        # Z-score based: more adaptive
        signals[merged['funding_zscore'] > params['z_threshold']] = -1
        signals[merged['funding_zscore'] < -params['z_threshold']] = 1
    elif params['mode'] == 'extreme':
        # Only trade extreme funding (top/bottom decile historically)
        q_high = merged['funding_rate'].rolling(params['zscore_window']).quantile(0.9)
        q_low = merged['funding_rate'].rolling(params['zscore_window']).quantile(0.1)
        signals[merged['funding_rate'] > q_high] = -1
        signals[merged['funding_rate'] < q_low] = 1
    
    # Calculate funding P&L per bar (daily funding = sum of 3x 8h payments)
    # Binance: funding paid every 8h. Daily funding ≈ 3x the rate
    # If short and funding positive, we RECEIVE funding
    funding_pnl = pd.Series(0.0, index=merged.index)
    funding_pnl = -signals * merged['funding_rate'] / 100  # funding is in % terms
    
    return signals, funding_pnl


# ============================================================
# STRATEGY 2: NR4/NR7 TREND CONTINUATION
# ============================================================

def strategy_nr_trend(df, params):
    """
    NR4/NR7 + trend filter + close position + volume confirmation.
    
    4 conditions must align:
    1. Narrowest range in N bars (volatility compression)
    2. Price above/below trend EMA (directional bias)
    3. Close in top/bottom third of bar (conviction)
    4. Volume below average (quiet before storm)
    """
    nr_lookback = params['nr_lookback']  # 4 or 7
    trend_ema = params['trend_ema']      # 50
    vol_lookback = params['vol_lookback']  # 20
    
    # Calculate bar range
    df_calc = df.copy()
    df_calc['range'] = df_calc['high'] - df_calc['low']
    
    # NR identification
    df_calc['min_range'] = df_calc['range'].rolling(nr_lookback).min()
    df_calc['is_nr'] = (df_calc['range'] <= df_calc['min_range'] * 1.01)  # 1% tolerance
    
    # Trend filter
    df_calc['ema'] = calc_ema(df_calc['close'], trend_ema)
    df_calc['uptrend'] = df_calc['close'] > df_calc['ema']
    df_calc['downtrend'] = df_calc['close'] < df_calc['ema']
    
    # Close position within bar
    df_calc['close_pct'] = (df_calc['close'] - df_calc['low']) / (df_calc['range'] + 1e-10)
    
    # Volume filter (quiet bar = below average)
    df_calc['avg_vol'] = df_calc['volume'].rolling(vol_lookback).mean()
    df_calc['quiet'] = df_calc['volume'] < df_calc['avg_vol'] * params.get('vol_threshold', 1.0)
    
    # ADX filter (optional - only trade in trending markets)
    if params.get('use_adx', False):
        adx = calc_adx(df_calc['high'], df_calc['low'], df_calc['close'], 14)
        df_calc['trending'] = adx > params.get('adx_min', 20)
    else:
        df_calc['trending'] = True
    
    signals = pd.Series(0, index=df_calc.index)
    
    # Long: NR + uptrend + close in top 1/3 + quiet volume + trending
    long_setup = (
        df_calc['is_nr'] &
        df_calc['uptrend'] &
        (df_calc['close_pct'] > params.get('close_pct_long', 0.67)) &
        df_calc['quiet'] &
        df_calc['trending']
    )
    
    # Short: NR + downtrend + close in bottom 1/3 + quiet volume + trending
    short_setup = (
        df_calc['is_nr'] &
        df_calc['downtrend'] &
        (df_calc['close_pct'] < params.get('close_pct_short', 0.33)) &
        df_calc['quiet'] &
        df_calc['trending']
    )
    
    # Entry on NEXT bar (break of NR high/low)
    # Simplified: signal on NR bar, entry next bar
    signals[long_setup] = 1
    signals[short_setup] = -1
    
    return signals


# ============================================================
# STRATEGY 3: EWMAC MULTI-SPEED TREND FOLLOWING
# ============================================================

def strategy_ewmac(df, params):
    """
    Carver's EWMAC: Exponentially Weighted Moving Average Crossover.
    
    Key differences from basic MA cross:
    1. Multiple speed pairs combined (captures trends at all speeds)
    2. Volatility-scaled position sizing
    3. Continuous forecast (not binary)
    
    Speed pairs: (fast, slow) = (2,8), (4,16), (8,32), (16,64), (32,128)
    Forecast = (fast_ema - slow_ema) / volatility
    Combined forecast = average of all speed pair forecasts, capped at ±20
    """
    speed_pairs = params.get('speed_pairs', [(2,8), (4,16), (8,32), (16,64)])
    vol_lookback = params.get('vol_lookback', 25)
    forecast_cap = params.get('forecast_cap', 20)
    forecast_threshold = params.get('forecast_threshold', 0)  # min forecast to trade
    
    close = df['close']
    
    # Calculate volatility (Carver uses percentage returns std)
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback).std()
    
    # Calculate forecast for each speed pair
    forecasts = []
    for fast, slow in speed_pairs:
        ema_fast = calc_ema(close, fast)
        ema_slow = calc_ema(close, slow)
        
        # Raw forecast: normalized by volatility and price
        raw_forecast = (ema_fast - ema_slow) / (close * vol + 1e-10)
        
        # Scale to target avg abs forecast of 10
        abs_avg = raw_forecast.abs().rolling(min(252, len(close)//2), min_periods=50).mean()
        scaled = raw_forecast / (abs_avg + 1e-10) * 10
        
        # Cap at ±forecast_cap
        capped = scaled.clip(-forecast_cap, forecast_cap)
        forecasts.append(capped)
    
    # Combine forecasts (equal weight)
    combined = pd.concat(forecasts, axis=1).mean(axis=1)
    
    # Re-cap combined forecast
    combined = combined.clip(-forecast_cap, forecast_cap)
    
    # Convert to binary signals for walk-forward testing
    signals = pd.Series(0, index=df.index)
    signals[combined > forecast_threshold] = 1
    signals[combined < -forecast_threshold] = -1
    
    return signals


# ============================================================  
# STRATEGY 4: KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
# ============================================================

def strategy_kama(df, params):
    """
    Kaufman's Adaptive Moving Average.
    
    Key: Efficiency Ratio (ER) = direction / volatility
    - ER near 1.0 = trending (fast smoothing)
    - ER near 0.0 = choppy (slow smoothing)
    
    This adapts automatically to regime without explicit detection.
    """
    er_period = params.get('er_period', 10)
    fast_sc = 2 / (params.get('fast_period', 2) + 1)  # fast smoothing constant
    slow_sc = 2 / (params.get('slow_period', 30) + 1)  # slow smoothing constant
    
    close = df['close'].values
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Need at least er_period bars
    if n < er_period + 1:
        return pd.Series(0, index=df.index)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        # Efficiency Ratio
        direction = abs(close[i] - close[i - er_period])
        volatility = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))
        
        if volatility == 0:
            er = 0
        else:
            er = direction / volatility
        
        # Smoothing constant: adapts between fast and slow
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    kama_series = pd.Series(kama, index=df.index)
    
    # Signals: price crosses KAMA
    signals = pd.Series(0, index=df.index)
    
    # Add a filter band (% above/below KAMA to trigger)
    band = params.get('band_pct', 0.005)  # 0.5% band
    
    price = df['close']
    signals[(price > kama_series * (1 + band)) & (price.shift(1) <= kama_series.shift(1) * (1 + band))] = 1
    signals[(price < kama_series * (1 - band)) & (price.shift(1) >= kama_series.shift(1) * (1 - band))] = -1
    
    return signals


# ============================================================
# STRATEGY 5: DUAL MOMENTUM
# ============================================================

def strategy_dual_momentum(df, params):
    """
    Antonacci's Dual Momentum:
    1. Absolute momentum: Is the asset trending up? (return > 0 over lookback)
    2. Relative momentum: Is it outperforming? (vs benchmark, not applicable for single asset)
    
    For single-asset: use absolute momentum + trend strength filter.
    Long when lookback return > 0 AND recent momentum accelerating.
    """
    lookback = params.get('lookback', 63)  # ~3 months for daily, or scaled
    accel_period = params.get('accel_period', 21)  # acceleration check
    
    close = df['close']
    
    # Absolute momentum
    mom = close.pct_change(lookback)
    
    # Momentum acceleration (recent > older)
    mom_recent = close.pct_change(accel_period)
    mom_older = close.shift(accel_period).pct_change(accel_period)
    accelerating = mom_recent > mom_older
    
    # Volatility filter (avoid high-vol regimes)
    vol = close.pct_change().rolling(lookback).std()
    vol_ma = vol.rolling(lookback * 2).mean()
    low_vol = vol < vol_ma * params.get('vol_filter', 1.5)
    
    signals = pd.Series(0, index=df.index)
    
    if params.get('mode', 'absolute') == 'absolute':
        # Simple absolute momentum
        signals[mom > params.get('min_return', 0)] = 1
        signals[mom < -params.get('min_return', 0)] = -1
    elif params['mode'] == 'accelerating':
        # Absolute + acceleration
        signals[(mom > 0) & accelerating] = 1
        signals[(mom < 0) & ~accelerating] = -1
    elif params['mode'] == 'filtered':
        # Absolute + acceleration + vol filter
        signals[(mom > 0) & accelerating & low_vol] = 1
        signals[(mom < 0) & ~accelerating] = -1
    
    return signals


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def walk_forward_test(df, signals, holding_bars, cost_bps=COST_BPS, 
                      funding_pnl=None, bars_per_year=365):
    """Walk-forward with non-overlapping trades, optional funding P&L."""
    n = len(df)
    if n < 100:
        return None
    
    fold_size = n // N_FOLDS
    if fold_size < 20:
        return None
    
    test_start = fold_size * 7
    test_df = df.iloc[test_start:].copy()
    test_signals = signals.iloc[test_start:].copy()
    if funding_pnl is not None:
        test_funding = funding_pnl.iloc[test_start:].copy()
    
    # Generate non-overlapping trades
    trades = []
    i = 0
    while i < len(test_df) - holding_bars:
        sig = test_signals.iloc[i]
        if sig != 0:
            entry_price = test_df['close'].iloc[i]
            exit_price = test_df['close'].iloc[i + holding_bars]
            
            # Price return
            if sig == 1:
                price_ret = (exit_price / entry_price) - 1
            else:
                price_ret = (entry_price / exit_price) - 1
            
            # Funding return (accumulated over holding period)
            fund_ret = 0
            if funding_pnl is not None:
                fund_ret = test_funding.iloc[i:i+holding_bars].sum()
            
            # Total return minus costs
            total_ret = price_ret + fund_ret - cost_bps / 10000
            
            trades.append({
                'entry_idx': i,
                'direction': sig,
                'price_return': price_ret,
                'funding_return': fund_ret,
                'total_return': total_ret,
            })
            
            i += holding_bars
        else:
            i += 1
    
    if len(trades) < MIN_TRADES:
        return None
    
    returns = np.array([t['total_return'] for t in trades])
    price_returns = np.array([t['price_return'] for t in trades])
    funding_returns = np.array([t['funding_return'] for t in trades])
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret < 1e-10:
        return None
    
    # Compute full metrics suite
    from metrics import compute_metrics
    
    test_bars = len(test_df)
    test_years = test_bars / bars_per_year if bars_per_year > 0 else 1
    tpy = len(trades) / test_years if test_years > 0 else len(trades)
    
    m = compute_metrics(returns, bars_per_year=bars_per_year, 
                        total_bars=test_bars, trades_per_year=tpy)
    
    long_rets = [t['total_return'] for t in trades if t['direction'] == 1]
    short_rets = [t['total_return'] for t in trades if t['direction'] == -1]
    
    # Merge with strategy-specific fields
    m.update({
        'n_long': len(long_rets),
        'n_short': len(short_rets),
        'mean_long': np.mean(long_rets) if long_rets else 0,
        'mean_short': np.mean(short_rets) if short_rets else 0,
        'mean_price_ret': np.mean(price_returns),
        'mean_funding_ret': np.mean(funding_returns),
        'total_funding_pnl': np.sum(funding_returns),
        'test_bars': test_bars,
    })
    return m


# ============================================================
# PARAMETER CONFIGURATIONS
# ============================================================

FUNDING_PARAMS = [
    {'name': 'simple_0.03', 'mode': 'simple', 'threshold': 0.03, 'avg_window': 3, 'zscore_window': 90},
    {'name': 'simple_0.05', 'mode': 'simple', 'threshold': 0.05, 'avg_window': 3, 'zscore_window': 90},
    {'name': 'simple_0.01_7d', 'mode': 'simple', 'threshold': 0.01, 'avg_window': 7, 'zscore_window': 90},
    {'name': 'zscore_1.5', 'mode': 'zscore', 'threshold': 0, 'z_threshold': 1.5, 'avg_window': 7, 'zscore_window': 90},
    {'name': 'zscore_2.0', 'mode': 'zscore', 'threshold': 0, 'z_threshold': 2.0, 'avg_window': 7, 'zscore_window': 90},
    {'name': 'extreme_90', 'mode': 'extreme', 'threshold': 0, 'avg_window': 7, 'zscore_window': 180},
]

NR_PARAMS = [
    # NR4 variants
    {'name': 'nr4_ema50', 'nr_lookback': 4, 'trend_ema': 50, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': False, 'close_pct_long': 0.67, 'close_pct_short': 0.33},
    {'name': 'nr4_ema50_adx', 'nr_lookback': 4, 'trend_ema': 50, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': True, 'adx_min': 20, 'close_pct_long': 0.67, 'close_pct_short': 0.33},
    {'name': 'nr4_ema20', 'nr_lookback': 4, 'trend_ema': 20, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': False, 'close_pct_long': 0.60, 'close_pct_short': 0.40},
    {'name': 'nr4_ema50_relaxed', 'nr_lookback': 4, 'trend_ema': 50, 'vol_lookback': 20, 'vol_threshold': 1.2, 'use_adx': False, 'close_pct_long': 0.55, 'close_pct_short': 0.45},
    # NR7 variants
    {'name': 'nr7_ema50', 'nr_lookback': 7, 'trend_ema': 50, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': False, 'close_pct_long': 0.67, 'close_pct_short': 0.33},
    {'name': 'nr7_ema50_adx', 'nr_lookback': 7, 'trend_ema': 50, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': True, 'adx_min': 20, 'close_pct_long': 0.67, 'close_pct_short': 0.33},
    {'name': 'nr7_ema20', 'nr_lookback': 7, 'trend_ema': 20, 'vol_lookback': 20, 'vol_threshold': 1.0, 'use_adx': False, 'close_pct_long': 0.60, 'close_pct_short': 0.40},
]

EWMAC_PARAMS = [
    {'name': 'ewmac_2speed', 'speed_pairs': [(4,16), (16,64)], 'vol_lookback': 25, 'forecast_cap': 20, 'forecast_threshold': 0},
    {'name': 'ewmac_3speed', 'speed_pairs': [(2,8), (8,32), (32,128)], 'vol_lookback': 25, 'forecast_cap': 20, 'forecast_threshold': 0},
    {'name': 'ewmac_4speed', 'speed_pairs': [(2,8), (4,16), (8,32), (16,64)], 'vol_lookback': 25, 'forecast_cap': 20, 'forecast_threshold': 0},
    {'name': 'ewmac_5speed', 'speed_pairs': [(2,8), (4,16), (8,32), (16,64), (32,128)], 'vol_lookback': 25, 'forecast_cap': 20, 'forecast_threshold': 0},
    {'name': 'ewmac_fast', 'speed_pairs': [(2,8), (4,16), (8,32)], 'vol_lookback': 15, 'forecast_cap': 20, 'forecast_threshold': 2},
    {'name': 'ewmac_slow', 'speed_pairs': [(8,32), (16,64), (32,128), (64,256)], 'vol_lookback': 50, 'forecast_cap': 20, 'forecast_threshold': 0},
]

KAMA_PARAMS = [
    {'name': 'kama_10_2_30', 'er_period': 10, 'fast_period': 2, 'slow_period': 30, 'band_pct': 0.005},
    {'name': 'kama_10_2_30_wide', 'er_period': 10, 'fast_period': 2, 'slow_period': 30, 'band_pct': 0.01},
    {'name': 'kama_20_2_30', 'er_period': 20, 'fast_period': 2, 'slow_period': 30, 'band_pct': 0.005},
    {'name': 'kama_10_3_50', 'er_period': 10, 'fast_period': 3, 'slow_period': 50, 'band_pct': 0.005},
    {'name': 'kama_5_2_30_tight', 'er_period': 5, 'fast_period': 2, 'slow_period': 30, 'band_pct': 0.003},
]

DUAL_MOM_PARAMS = [
    {'name': 'abs_63d', 'lookback': 63, 'accel_period': 21, 'mode': 'absolute', 'min_return': 0, 'vol_filter': 1.5},
    {'name': 'abs_42d', 'lookback': 42, 'accel_period': 14, 'mode': 'absolute', 'min_return': 0, 'vol_filter': 1.5},
    {'name': 'abs_21d', 'lookback': 21, 'accel_period': 7, 'mode': 'absolute', 'min_return': 0, 'vol_filter': 1.5},
    {'name': 'accel_63d', 'lookback': 63, 'accel_period': 21, 'mode': 'accelerating', 'min_return': 0, 'vol_filter': 1.5},
    {'name': 'accel_42d', 'lookback': 42, 'accel_period': 14, 'mode': 'accelerating', 'min_return': 0, 'vol_filter': 1.5},
    {'name': 'filtered_63d', 'lookback': 63, 'accel_period': 21, 'mode': 'filtered', 'min_return': 0, 'vol_filter': 1.5},
]

# Holding periods per strategy type and timeframe
HOLDING_CONFIGS = {
    'funding_daily': {'3d': 3, '1w': 7, '2w': 14},
    'nr_daily': {'3d': 3, '1w': 7, '2w': 14},
    'nr_4h': {'1d': 6, '3d': 18, '1w': 42},
    'ewmac_daily': {'1w': 7, '2w': 14, '1m': 30},
    'ewmac_4h': {'3d': 18, '1w': 42, '2w': 84},
    'kama_daily': {'3d': 3, '1w': 7, '2w': 14},
    'kama_4h': {'1d': 6, '3d': 18, '1w': 42},
    'dual_mom_daily': {'1w': 7, '2w': 14, '1m': 30},
}


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():
    print("=" * 80)
    print("TIER 1 BOOK STRATEGY WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    all_results = []
    test_count = 0
    valid_count = 0
    start_time = time.time()
    
    # --------------------------------------------------------
    # STRATEGY 1: FUNDING RATE CARRY
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("STRATEGY 1: FUNDING RATE CARRY (Carver)")
    print(f"{'='*60}")
    
    for asset in FUNDING_ASSETS:
        funding = load_funding(asset)
        if funding is None:
            print(f"  {asset}: No funding data")
            continue
        
        spot = load_spot_daily(asset)
        if spot is None:
            print(f"  {asset}: No daily spot data")
            continue
        
        print(f"\n  {asset} (spot: {len(spot)} bars, funding: {len(funding)} bars)")
        
        for params in FUNDING_PARAMS:
            signals, funding_pnl = strategy_funding_carry(spot, funding, params)
            if signals is None:
                continue
            
            n_signals = (signals != 0).sum()
            if n_signals < 5:
                continue
            
            for hold_name, hold_bars in HOLDING_CONFIGS['funding_daily'].items():
                test_count += 1
                result = walk_forward_test(spot, signals, hold_bars, COST_BPS,
                                          funding_pnl=funding_pnl, bars_per_year=365)
                if result is None:
                    continue
                valid_count += 1
                
                row = {
                    'strategy': 'funding_carry',
                    'variant': params['name'],
                    'strategy_type': 'carry',
                    'timeframe': '1d',
                    'asset': asset,
                    'holding_period': hold_name,
                    'holding_bars': hold_bars,
                    **result,
                }
                all_results.append(row)
                
                marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀" if result['sharpe'] < -1 else "  "
                if result['sharpe'] > 0.5 or result['sharpe'] < -1:
                    print(f"    {marker} {params['name']} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                          f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                          f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}, "
                          f"fund_pnl={result['total_funding_pnl']*100:.2f}%")
    
    # --------------------------------------------------------
    # STRATEGY 2: NR4/NR7 TREND CONTINUATION
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("STRATEGY 2: NR4/NR7 TREND CONTINUATION (Crabel)")
    print(f"{'='*60}")
    
    for tf, hold_config_key, bpy in [('1d', 'nr_daily', 365), ('4h', 'nr_4h', 365*6)]:
        print(f"\n  --- Timeframe: {tf} ---")
        for asset in ASSETS_FULL:
            if tf == '1d':
                df = load_spot_daily(asset)
            else:
                df = load_spot(asset, tf)
            if df is None:
                continue
            
            print(f"\n  {asset} ({len(df):,} bars)")
            
            for params in NR_PARAMS:
                signals = strategy_nr_trend(df, params)
                n_signals = (signals != 0).sum()
                if n_signals < 5:
                    continue
                
                for hold_name, hold_bars in HOLDING_CONFIGS[hold_config_key].items():
                    test_count += 1
                    result = walk_forward_test(df, signals, hold_bars, COST_BPS,
                                              bars_per_year=bpy)
                    if result is None:
                        continue
                    valid_count += 1
                    
                    row = {
                        'strategy': 'nr_trend',
                        'variant': params['name'],
                        'strategy_type': 'breakout',
                        'timeframe': tf,
                        'asset': asset,
                        'holding_period': hold_name,
                        'holding_bars': hold_bars,
                        **result,
                    }
                    all_results.append(row)
                    
                    marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀" if result['sharpe'] < -1 else "  "
                    if result['sharpe'] > 0.5 or result['sharpe'] < -1:
                        print(f"    {marker} {params['name']} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                              f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                              f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}")
    
    # --------------------------------------------------------
    # STRATEGY 3: EWMAC MULTI-SPEED TREND
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("STRATEGY 3: EWMAC MULTI-SPEED TREND (Carver)")
    print(f"{'='*60}")
    
    for tf, hold_config_key, bpy in [('1d', 'ewmac_daily', 365), ('4h', 'ewmac_4h', 365*6)]:
        print(f"\n  --- Timeframe: {tf} ---")
        for asset in ASSETS_FULL:
            if tf == '1d':
                df = load_spot_daily(asset)
            else:
                df = load_spot(asset, tf)
            if df is None:
                continue
            
            print(f"\n  {asset} ({len(df):,} bars)")
            
            for params in EWMAC_PARAMS:
                signals = strategy_ewmac(df, params)
                n_signals = (signals != 0).sum()
                if n_signals < 5:
                    continue
                
                for hold_name, hold_bars in HOLDING_CONFIGS[hold_config_key].items():
                    test_count += 1
                    result = walk_forward_test(df, signals, hold_bars, COST_BPS,
                                              bars_per_year=bpy)
                    if result is None:
                        continue
                    valid_count += 1
                    
                    row = {
                        'strategy': 'ewmac',
                        'variant': params['name'],
                        'strategy_type': 'trend',
                        'timeframe': tf,
                        'asset': asset,
                        'holding_period': hold_name,
                        'holding_bars': hold_bars,
                        **result,
                    }
                    all_results.append(row)
                    
                    marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀" if result['sharpe'] < -1 else "  "
                    if result['sharpe'] > 0.5 or result['sharpe'] < -1:
                        print(f"    {marker} {params['name']} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                              f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                              f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}")
    
    # --------------------------------------------------------
    # STRATEGY 4: KAMA ADAPTIVE TREND
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("STRATEGY 4: KAMA ADAPTIVE TREND (Kaufman)")
    print(f"{'='*60}")
    
    for tf, hold_config_key, bpy in [('1d', 'kama_daily', 365), ('4h', 'kama_4h', 365*6)]:
        print(f"\n  --- Timeframe: {tf} ---")
        for asset in ASSETS_FULL:
            if tf == '1d':
                df = load_spot_daily(asset)
            else:
                df = load_spot(asset, tf)
            if df is None:
                continue
            
            if len(df) < 200:
                continue
            
            print(f"\n  {asset} ({len(df):,} bars)")
            
            for params in KAMA_PARAMS:
                signals = strategy_kama(df, params)
                n_signals = (signals != 0).sum()
                if n_signals < 5:
                    continue
                
                for hold_name, hold_bars in HOLDING_CONFIGS[hold_config_key].items():
                    test_count += 1
                    result = walk_forward_test(df, signals, hold_bars, COST_BPS,
                                              bars_per_year=bpy)
                    if result is None:
                        continue
                    valid_count += 1
                    
                    row = {
                        'strategy': 'kama',
                        'variant': params['name'],
                        'strategy_type': 'adaptive_trend',
                        'timeframe': tf,
                        'asset': asset,
                        'holding_period': hold_name,
                        'holding_bars': hold_bars,
                        **result,
                    }
                    all_results.append(row)
                    
                    marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀" if result['sharpe'] < -1 else "  "
                    if result['sharpe'] > 0.5 or result['sharpe'] < -1:
                        print(f"    {marker} {params['name']} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                              f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                              f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}")
    
    # --------------------------------------------------------
    # STRATEGY 5: DUAL MOMENTUM
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("STRATEGY 5: DUAL MOMENTUM (Antonacci)")
    print(f"{'='*60}")
    
    for asset in ASSETS_FULL:
        df = load_spot_daily(asset)
        if df is None:
            continue
        if len(df) < 200:
            continue
        
        print(f"\n  {asset} ({len(df):,} bars)")
        
        for params in DUAL_MOM_PARAMS:
            signals = strategy_dual_momentum(df, params)
            n_signals = (signals != 0).sum()
            if n_signals < 5:
                continue
            
            for hold_name, hold_bars in HOLDING_CONFIGS['dual_mom_daily'].items():
                test_count += 1
                result = walk_forward_test(df, signals, hold_bars, COST_BPS,
                                          bars_per_year=365)
                if result is None:
                    continue
                valid_count += 1
                
                row = {
                    'strategy': 'dual_momentum',
                    'variant': params['name'],
                    'strategy_type': 'momentum',
                    'timeframe': '1d',
                    'asset': asset,
                    'holding_period': hold_name,
                    'holding_bars': hold_bars,
                    **result,
                }
                all_results.append(row)
                
                marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀" if result['sharpe'] < -1 else "  "
                if result['sharpe'] > 0.5 or result['sharpe'] < -1:
                    print(f"    {marker} {params['name']} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                          f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                          f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}")
    
    # ============================================================
    # RESULTS ANALYSIS
    # ============================================================
    
    if not all_results:
        print("\nNo valid results!")
        return
    
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "tier1_strategy_results.csv"
    results_df.to_csv(out_path, index=False)
    
    total_tests = len(results_df)
    bonferroni_alpha = 0.05 / total_tests if total_tests > 0 else 0.05
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests attempted: {test_count}")
    print(f"Valid tests (≥{MIN_TRADES} trades): {total_tests}")
    print(f"Bonferroni α: {bonferroni_alpha:.2e}")
    
    raw_passes = results_df[(results_df['p_value'] < 0.05) & (results_df['sharpe'] > 0)]
    bonf_passes = results_df[(results_df['p_value'] < bonferroni_alpha) & (results_df['sharpe'] > 0)]
    
    print(f"\nRaw passes (p<0.05, Sharpe>0): {len(raw_passes)} ({len(raw_passes)/total_tests*100:.1f}%)")
    print(f"Bonferroni survivors: {len(bonf_passes)} ({len(bonf_passes)/total_tests*100:.1f}%)")
    
    # By strategy
    print(f"\n--- BY STRATEGY ---")
    for strat in ['funding_carry', 'nr_trend', 'ewmac', 'kama', 'dual_momentum']:
        subset = results_df[results_df['strategy'] == strat]
        if len(subset) == 0:
            continue
        avg = subset['sharpe'].mean()
        med = subset['sharpe'].median()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        pos = (subset['sharpe'] > 0).sum()
        print(f"  {strat:20s}: N={len(subset):4d}, avg={avg:+.3f}, med={med:+.3f}, "
              f"pos={pos}/{len(subset)} ({pos/len(subset)*100:.0f}%), raw={raw:3d}, Bonf={bonf}")
    
    # By strategy type
    print(f"\n--- BY STRATEGY TYPE ---")
    for stype in sorted(results_df['strategy_type'].unique()):
        subset = results_df[results_df['strategy_type'] == stype]
        avg = subset['sharpe'].mean()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        print(f"  {stype:20s}: N={len(subset):4d}, avg Sharpe={avg:+.3f}, raw={raw:3d}, Bonf={bonf}")
    
    # By timeframe
    print(f"\n--- BY TIMEFRAME ---")
    for tf in sorted(results_df['timeframe'].unique()):
        subset = results_df[results_df['timeframe'] == tf]
        avg = subset['sharpe'].mean()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        print(f"  {tf}: N={len(subset)}, avg Sharpe={avg:+.3f}, raw={raw}, Bonf={bonf}")
    
    # By holding period
    print(f"\n--- BY HOLDING PERIOD ---")
    for hp in sorted(results_df['holding_period'].unique()):
        subset = results_df[results_df['holding_period'] == hp]
        avg = subset['sharpe'].mean()
        print(f"  {hp:6s}: N={len(subset)}, avg Sharpe={avg:+.3f}")
    
    # Top assets
    print(f"\n--- TOP 10 ASSETS ---")
    asset_sharpe = results_df.groupby('asset')['sharpe'].mean().sort_values(ascending=False)
    for asset, sharpe in asset_sharpe.head(10).items():
        n = len(results_df[results_df['asset'] == asset])
        print(f"  {asset:8s}: avg Sharpe={sharpe:+.3f} (N={n})")
    
    # Top 30 individual results
    print(f"\n--- TOP 30 INDIVIDUAL RESULTS ---")
    top = results_df.nlargest(30, 'sharpe')
    for _, r in top.iterrows():
        bonf_flag = "✅" if r['p_value'] < bonferroni_alpha else "  "
        fund_str = f" fund={r['total_funding_pnl']*100:+.1f}%" if r.get('total_funding_pnl', 0) != 0 else ""
        print(f"  {bonf_flag} {r['asset']:8s} {r['strategy']:20s} {r['variant']:20s} {r['timeframe']} "
              f"hold={r['holding_period']:6s} Sharpe={r['sharpe']:+.2f} N={r['n_trades']:5d} "
              f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} p={r['p_value']:.2e} "
              f"ret={r['total_return']:+.1%}{fund_str}")
    
    # Funding-specific analysis
    print(f"\n--- FUNDING CARRY BREAKDOWN ---")
    funding_results = results_df[results_df['strategy'] == 'funding_carry']
    if len(funding_results) > 0:
        print(f"  Avg price return per trade: {funding_results['mean_price_ret'].mean()*100:+.3f}%")
        print(f"  Avg funding return per trade: {funding_results['mean_funding_ret'].mean()*100:+.3f}%")
        print(f"  Total funding P&L (avg): {funding_results['total_funding_pnl'].mean()*100:+.2f}%")
        # Best funding assets
        fund_by_asset = funding_results.groupby('asset')['sharpe'].mean().sort_values(ascending=False)
        print(f"  Best funding assets:")
        for a, s in fund_by_asset.items():
            print(f"    {a}: avg Sharpe {s:+.3f}")
    
    # Bottom 10
    print(f"\n--- BOTTOM 10 ---")
    bottom = results_df.nsmallest(10, 'sharpe')
    for _, r in bottom.iterrows():
        print(f"  💀 {r['asset']:8s} {r['strategy']:20s} {r['variant']:20s} "
              f"Sharpe={r['sharpe']:+.2f} N={r['n_trades']}")
    
    # Long vs short
    print(f"\n--- LONG vs SHORT ---")
    print(f"  Avg long return:  {results_df['mean_long'].mean()*100:+.3f}%")
    print(f"  Avg short return: {results_df['mean_short'].mean()*100:+.3f}%")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Results saved to: {out_path}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_validation()
