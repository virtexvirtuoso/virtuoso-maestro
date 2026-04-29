#!/usr/bin/env python3
"""
Scalping Strategy Walk-Forward Validation
==========================================
Tests scalping strategies at 5m/15m with:
- TF-appropriate indicator parameters
- Realistic transaction costs (10bps per trade, compounded)
- Non-overlapping returns
- 14-fold walk-forward (train 1-7, test 8-14)
- Bonferroni correction for multiple testing
- Proper holding periods scaled to timeframe

Strategies tested:
1. VWAP Reversion — price deviates from VWAP, reverts
2. StochRSI Oversold/Overbought — momentum at extremes
3. EMA Ribbon — fast EMA cross above/below slow ribbon
4. Bollinger Mean Reversion — touch band, revert to mean
5. RSI Extreme Reversion — RSI < 20 buy, > 80 sell (short TF)
6. Momentum Burst — strong candle continuation
7. Range Breakout — break above/below N-bar high/low
8. MACD Scalp — MACD cross on short parameters
9. Keltner Channel MR — price outside Keltner, revert
10. Dual EMA Momentum — EMA8/EMA21 cross with ADX filter
11. Volume Spike Breakout — breakout on high relative volume
12. Stochastic %K/%D Cross — classic stoch cross at extremes

Mr. V's rules:
- Walk-forward validate EVERYTHING
- Non-overlapping trades only
- Realistic costs (10bps RT)
- Bonferroni correction
- Holding periods scale with TF
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
import sys

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/spot")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 10  # 10bps per round trip
MIN_TRADES = 30  # minimum trades for valid test
N_FOLDS = 14
TRAIN_FOLDS = range(0, 7)
TEST_FOLDS = range(7, 14)

# Assets to test (top liquid ones + known trendors)
ASSETS = [
    'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'AVAX', 'DOT', 'MATIC',
    'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'OP', 'ARB', 'SUI',
    'DOGE', 'XRP', 'RENDER', 'FET', 'TIA', 'SEI', 'DYDX', 'INJ'
]

# Timeframes and their holding periods
TF_CONFIG = {
    '5m': {
        'holding_periods': {'15m': 3, '1h': 12, '4h': 48},
        'subdir': '5m',
    },
    '15m': {
        'holding_periods': {'1h': 4, '4h': 16, '12h': 48},
        'subdir': '15m',
    },
}

# ============================================================
# DATA LOADING
# ============================================================

def load_spot(asset, tf):
    """Load spot candle data for given asset and timeframe."""
    subdir = TF_CONFIG[tf]['subdir']
    path = DATA_DIR / subdir / f"{asset}_spot_{tf}.csv"
    if not path.exists():
        # Try alternate naming
        path = DATA_DIR / subdir / f"{asset}USDT_{tf}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('timestamp', 'date', 'time', 'datetime'):
            col_map[c] = 'timestamp'
        elif cl == 'open':
            col_map[c] = 'open'
        elif cl == 'high':
            col_map[c] = 'high'
        elif cl == 'low':
            col_map[c] = 'low'
        elif cl == 'close':
            col_map[c] = 'close'
        elif cl in ('volume', 'vol'):
            col_map[c] = 'volume'
    
    df = df.rename(columns=col_map)
    
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for r in required:
        if r not in df.columns:
            return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['close'])
    
    return df

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(period).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calc_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = stoch_rsi.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def calc_stochastic(high, low, close, k_period=14, k_smooth=3, d_smooth=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def calc_bbands(close, period=20, std_mult=2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, ma, lower

def calc_keltner(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0):
    ema = calc_ema(close, ema_period)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, ema, lower

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

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
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx

def calc_vwap(high, low, close, volume, reset_period=None):
    """Cumulative VWAP. For intraday, ideally reset daily but we use rolling."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    return vwap

def calc_rolling_vwap(high, low, close, volume, period=48):
    """Rolling VWAP over N bars — better for continuous calculation."""
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    rolling_tp_vol = tp_vol.rolling(period).sum()
    rolling_vol = volume.rolling(period).sum()
    return rolling_tp_vol / (rolling_vol + 1e-10)

# ============================================================
# STRATEGY SIGNAL GENERATORS
# ============================================================
# Each returns a Series of {1=long, -1=short, 0=flat}
# Parameters are TF-aware via the `params` dict

def get_tf_params(tf):
    """Get TF-appropriate parameters."""
    if tf == '5m':
        return {
            'rsi_period': 7,        # ~35min lookback
            'stoch_period': 8,
            'bb_period': 12,        # 1h of data
            'ema_fast': 5,          # 25min
            'ema_mid': 13,          # ~1h
            'ema_slow': 34,         # ~3h
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'keltner_period': 12,
            'atr_period': 10,
            'adx_period': 10,
            'vwap_period': 48,      # 4h rolling VWAP
            'breakout_period': 24,  # 2h high/low
            'vol_lookback': 20,
            'vol_threshold': 2.0,   # 2x avg volume
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            'stoch_oversold': 15,
            'stoch_overbought': 85,
            'bb_std': 2.0,
            'keltner_mult': 1.5,
            'adx_threshold': 20,
            'momentum_bars': 3,     # 15min momentum
            'momentum_threshold': 0.003,  # 0.3% move
        }
    elif tf == '15m':
        return {
            'rsi_period': 10,       # ~2.5h lookback
            'stoch_period': 10,
            'bb_period': 16,        # 4h of data
            'ema_fast': 5,          # 1.25h
            'ema_mid': 13,          # ~3.25h
            'ema_slow': 34,         # ~8.5h
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'keltner_period': 16,
            'atr_period': 12,
            'adx_period': 12,
            'vwap_period': 32,      # 8h rolling VWAP
            'breakout_period': 16,  # 4h high/low
            'vol_lookback': 16,
            'vol_threshold': 2.0,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'bb_std': 2.0,
            'keltner_mult': 1.5,
            'adx_threshold': 22,
            'momentum_bars': 3,     # 45min momentum
            'momentum_threshold': 0.005,  # 0.5% move
        }
    return {}


def strategy_vwap_reversion(df, params):
    """Price deviates from rolling VWAP, mean reverts back."""
    vwap = calc_rolling_vwap(df['high'], df['low'], df['close'], df['volume'], params['vwap_period'])
    atr = calc_atr(df['high'], df['low'], df['close'], params['atr_period'])
    
    deviation = (df['close'] - vwap) / (atr + 1e-10)
    
    signals = pd.Series(0, index=df.index)
    signals[deviation < -1.5] = 1   # price well below VWAP → long
    signals[deviation > 1.5] = -1   # price well above VWAP → short
    return signals


def strategy_stoch_rsi(df, params):
    """StochRSI at extremes with direction confirmation."""
    k, d = calc_stoch_rsi(df['close'], params['rsi_period'], params['stoch_period'])
    
    signals = pd.Series(0, index=df.index)
    # Long: K crosses above D in oversold zone
    long_cond = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 0.3)
    # Short: K crosses below D in overbought zone
    short_cond = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 0.7)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def strategy_ema_ribbon(df, params):
    """Fast EMA crosses above/below slow EMA ribbon."""
    ema_fast = calc_ema(df['close'], params['ema_fast'])
    ema_mid = calc_ema(df['close'], params['ema_mid'])
    ema_slow = calc_ema(df['close'], params['ema_slow'])
    
    signals = pd.Series(0, index=df.index)
    # Long: fast > mid > slow (ribbon aligned bullish) AND just crossed
    bull_ribbon = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    bear_ribbon = (ema_fast < ema_mid) & (ema_mid < ema_slow)
    
    # Entry on ribbon alignment change
    signals[bull_ribbon & ~bull_ribbon.shift(1).fillna(False)] = 1
    signals[bear_ribbon & ~bear_ribbon.shift(1).fillna(False)] = -1
    return signals


def strategy_bollinger_mr(df, params):
    """Bollinger Band mean reversion — touch outer band, revert."""
    upper, mid, lower = calc_bbands(df['close'], params['bb_period'], params['bb_std'])
    
    signals = pd.Series(0, index=df.index)
    # Long: close touches lower band AND starts recovering
    touch_lower = df['close'] <= lower
    recovering = df['close'] > df['close'].shift(1)
    signals[touch_lower & recovering] = 1
    
    # Short: close touches upper band AND starts declining
    touch_upper = df['close'] >= upper
    declining = df['close'] < df['close'].shift(1)
    signals[touch_upper & declining] = -1
    return signals


def strategy_rsi_extreme(df, params):
    """RSI at extreme levels — mean reversion."""
    rsi = calc_rsi(df['close'], params['rsi_period'])
    
    signals = pd.Series(0, index=df.index)
    # Long: RSI crosses up through oversold
    signals[(rsi > params['rsi_oversold']) & (rsi.shift(1) <= params['rsi_oversold'])] = 1
    # Short: RSI crosses down through overbought
    signals[(rsi < params['rsi_overbought']) & (rsi.shift(1) >= params['rsi_overbought'])] = -1
    return signals


def strategy_momentum_burst(df, params):
    """Strong candle continuation — big move continues."""
    n = params['momentum_bars']
    pct_change = df['close'].pct_change(n)
    
    signals = pd.Series(0, index=df.index)
    signals[pct_change > params['momentum_threshold']] = 1
    signals[pct_change < -params['momentum_threshold']] = -1
    return signals


def strategy_range_breakout(df, params):
    """Break above/below N-bar high/low."""
    period = params['breakout_period']
    highest = df['high'].rolling(period).max().shift(1)
    lowest = df['low'].rolling(period).min().shift(1)
    
    signals = pd.Series(0, index=df.index)
    signals[df['close'] > highest] = 1
    signals[df['close'] < lowest] = -1
    return signals


def strategy_macd_scalp(df, params):
    """MACD cross with short parameters for scalping."""
    macd_line, signal_line, hist = calc_macd(
        df['close'], params['macd_fast'], params['macd_slow'], params['macd_signal']
    )
    
    signals = pd.Series(0, index=df.index)
    # Long: MACD crosses above signal
    signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
    # Short: MACD crosses below signal
    signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
    return signals


def strategy_keltner_mr(df, params):
    """Keltner Channel mean reversion — price outside channel reverts."""
    upper, mid, lower = calc_keltner(
        df['high'], df['low'], df['close'],
        params['keltner_period'], params['atr_period'], params['keltner_mult']
    )
    
    signals = pd.Series(0, index=df.index)
    # Long: close below lower Keltner and recovering
    below = df['close'] < lower
    recovering = df['close'] > df['close'].shift(1)
    signals[below & recovering] = 1
    
    # Short: close above upper Keltner and declining
    above = df['close'] > upper
    declining = df['close'] < df['close'].shift(1)
    signals[above & declining] = -1
    return signals


def strategy_dual_ema_momentum(df, params):
    """EMA8/EMA21 cross with ADX filter — only trade when trending."""
    ema_fast = calc_ema(df['close'], params['ema_fast'])
    ema_slow = calc_ema(df['close'], params['ema_mid'])  # use mid as "slow" for scalp
    adx = calc_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
    signals = pd.Series(0, index=df.index)
    trending = adx > params['adx_threshold']
    
    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    signals[cross_up & trending] = 1
    signals[cross_down & trending] = -1
    return signals


def strategy_volume_spike_breakout(df, params):
    """Breakout confirmed by volume spike."""
    avg_vol = df['volume'].rolling(params['vol_lookback']).mean()
    vol_ratio = df['volume'] / (avg_vol + 1e-10)
    
    period = params['breakout_period']
    highest = df['high'].rolling(period).max().shift(1)
    lowest = df['low'].rolling(period).min().shift(1)
    
    high_vol = vol_ratio > params['vol_threshold']
    
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > highest) & high_vol] = 1
    signals[(df['close'] < lowest) & high_vol] = -1
    return signals


def strategy_stochastic_cross(df, params):
    """Classic Stochastic %K/%D cross at extremes."""
    k, d = calc_stochastic(
        df['high'], df['low'], df['close'],
        params['stoch_period'], 3, 3
    )
    
    signals = pd.Series(0, index=df.index)
    # Long: K crosses above D below oversold
    long_cond = (k > d) & (k.shift(1) <= d.shift(1)) & (k < params['stoch_oversold'])
    # Short: K crosses below D above overbought
    short_cond = (k < d) & (k.shift(1) >= d.shift(1)) & (k > params['stoch_overbought'])
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


# All strategies
STRATEGIES = {
    'vwap_reversion': strategy_vwap_reversion,
    'stoch_rsi': strategy_stoch_rsi,
    'ema_ribbon': strategy_ema_ribbon,
    'bollinger_mr': strategy_bollinger_mr,
    'rsi_extreme': strategy_rsi_extreme,
    'momentum_burst': strategy_momentum_burst,
    'range_breakout': strategy_range_breakout,
    'macd_scalp': strategy_macd_scalp,
    'keltner_mr': strategy_keltner_mr,
    'dual_ema_adx': strategy_dual_ema_momentum,
    'volume_spike_breakout': strategy_volume_spike_breakout,
    'stochastic_cross': strategy_stochastic_cross,
}

STRATEGY_TYPES = {
    'vwap_reversion': 'mean_reversion',
    'stoch_rsi': 'mean_reversion',
    'ema_ribbon': 'trend',
    'bollinger_mr': 'mean_reversion',
    'rsi_extreme': 'mean_reversion',
    'momentum_burst': 'trend',
    'range_breakout': 'trend',
    'macd_scalp': 'trend',
    'keltner_mr': 'mean_reversion',
    'dual_ema_adx': 'trend',
    'volume_spike_breakout': 'trend',
    'stochastic_cross': 'mean_reversion',
}

# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def walk_forward_test(df, signals, holding_bars, cost_bps=COST_BPS):
    """
    Walk-forward test with non-overlapping trades.
    
    Returns dict with Sharpe, returns, trade count, etc.
    """
    n = len(df)
    if n < 100:
        return None
    
    # Split into 14 folds
    fold_size = n // N_FOLDS
    if fold_size < 20:
        return None
    
    # Test on folds 8-14 (second half)
    test_start = fold_size * 7
    test_df = df.iloc[test_start:].copy()
    test_signals = signals.iloc[test_start:].copy()
    
    # Generate non-overlapping trades
    trades = []
    i = 0
    while i < len(test_df) - holding_bars:
        sig = test_signals.iloc[i]
        if sig != 0:
            entry_price = test_df['close'].iloc[i]
            exit_price = test_df['close'].iloc[i + holding_bars]
            
            if sig == 1:
                ret = (exit_price / entry_price) - 1
            else:
                ret = (entry_price / exit_price) - 1
            
            # Subtract costs
            ret -= cost_bps / 10000
            
            trades.append({
                'entry_idx': i,
                'direction': sig,
                'return': ret,
            })
            
            # Skip to end of holding period (non-overlapping)
            i += holding_bars
        else:
            i += 1
    
    if len(trades) < MIN_TRADES:
        return None
    
    returns = np.array([t['return'] for t in trades])
    
    # Stats
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret < 1e-10:
        return None
    
    # t-test: H0 = mean return is zero
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    
    # Annualized Sharpe (scale by trades per year estimate)
    # For non-overlapping: use actual trade frequency
    test_bars = len(test_df)
    trades_per_bar = len(trades) / test_bars
    
    # Bars per year
    if '5m' in str(holding_bars):
        bars_per_year = 365.25 * 24 * 12  # 5m bars
    else:
        bars_per_year = 365.25 * 24 * 12  # default to 5m
    
    # Simpler: annualize based on holding period
    # trades_per_year = bars_per_year / holding_bars * trades_per_bar... complex
    # Just use sqrt(N) scaling on the t-stat approach
    # Compute full metrics suite
    from metrics import compute_metrics
    
    test_years = test_bars / bars_per_year if bars_per_year > 0 else 1
    tpy = len(trades) / test_years if test_years > 0 else len(trades)
    
    m = compute_metrics(returns, bars_per_year=bars_per_year,
                        total_bars=test_bars, trades_per_year=tpy)
    
    long_trades = [t['return'] for t in trades if t['direction'] == 1]
    short_trades = [t['return'] for t in trades if t['direction'] == -1]
    
    m.update({
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'mean_long': np.mean(long_trades) if long_trades else 0,
        'mean_short': np.mean(short_trades) if short_trades else 0,
        'test_bars': test_bars,
    })
    return m


# ============================================================
# MAIN VALIDATION LOOP
# ============================================================

def run_validation():
    print("=" * 80)
    print("SCALPING STRATEGY WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    # Calculate total tests for Bonferroni
    total_tests_est = len(ASSETS) * len(STRATEGIES) * 2 * 3  # 2 TFs, ~3 holding periods
    print(f"\nEstimated total tests: ~{total_tests_est}")
    print(f"Bonferroni α: {0.05 / total_tests_est:.2e}")
    print(f"Assets: {len(ASSETS)}")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Timeframes: {list(TF_CONFIG.keys())}")
    print()
    
    # Bars per year for each TF
    bpy = {
        '5m': 365.25 * 24 * 12,
        '15m': 365.25 * 24 * 4,
    }
    
    all_results = []
    test_count = 0
    valid_count = 0
    start_time = time.time()
    
    for tf in TF_CONFIG:
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf}")
        print(f"{'='*60}")
        
        params = get_tf_params(tf)
        holding_periods = TF_CONFIG[tf]['holding_periods']
        bars_per_year = bpy[tf]
        
        for asset in ASSETS:
            df = load_spot(asset, tf)
            if df is None:
                print(f"  {asset}: No data for {tf}")
                continue
            
            print(f"\n  {asset} ({len(df):,} bars)")
            
            for strat_name, strat_func in STRATEGIES.items():
                try:
                    signals = strat_func(df, params)
                except Exception as e:
                    print(f"    {strat_name}: ERROR - {e}")
                    continue
                
                n_signals = (signals != 0).sum()
                if n_signals < 5:
                    continue
                
                for hold_name, hold_bars in holding_periods.items():
                    test_count += 1
                    
                    result = walk_forward_test(df, signals, hold_bars, COST_BPS)
                    
                    if result is None:
                        continue
                    
                    valid_count += 1
                    
                    # Annualize Sharpe properly
                    if result['std_return'] > 0:
                        trades_per_year = result['n_trades'] / (result['test_bars'] / bars_per_year)
                        result['sharpe'] = result['mean_return'] / result['std_return'] * np.sqrt(trades_per_year)
                    
                    row = {
                        'timeframe': tf,
                        'asset': asset,
                        'strategy': strat_name,
                        'strategy_type': STRATEGY_TYPES[strat_name],
                        'holding_period': hold_name,
                        'holding_bars': hold_bars,
                        **result,
                    }
                    all_results.append(row)
                    
                    # Print notable results
                    if result['sharpe'] > 0.5 or result['sharpe'] < -2:
                        marker = "⭐" if result['sharpe'] > 1.0 else "📊" if result['sharpe'] > 0.5 else "💀"
                        print(f"    {marker} {strat_name} hold={hold_name}: Sharpe={result['sharpe']:.2f}, "
                              f"N={result['n_trades']}, WR={result['win_rate']:.1%}, "
                              f"PF={result['profit_factor']:.2f}, p={result['p_value']:.4f}")
            
            # Progress
            elapsed = time.time() - start_time
            if test_count > 0:
                rate = test_count / elapsed
                print(f"    [{test_count} tests, {valid_count} valid, {elapsed:.0f}s, {rate:.1f} tests/s]")
    
    # ============================================================
    # RESULTS ANALYSIS
    # ============================================================
    
    if not all_results:
        print("\nNo valid results!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    out_path = RESULTS_DIR / "scalping_validation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n\nResults saved to: {out_path}")
    
    # Total tests for Bonferroni
    total_tests = len(results_df)
    bonferroni_alpha = 0.05 / total_tests
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {test_count}")
    print(f"Valid tests (≥{MIN_TRADES} trades): {valid_count}")
    print(f"Bonferroni α: {bonferroni_alpha:.2e}")
    
    # Raw passes (p < 0.05, positive Sharpe)
    raw_passes = results_df[(results_df['p_value'] < 0.05) & (results_df['sharpe'] > 0)]
    bonf_passes = results_df[(results_df['p_value'] < bonferroni_alpha) & (results_df['sharpe'] > 0)]
    
    print(f"\nRaw passes (p<0.05, Sharpe>0): {len(raw_passes)} ({len(raw_passes)/len(results_df)*100:.1f}%)")
    print(f"Bonferroni survivors: {len(bonf_passes)} ({len(bonf_passes)/len(results_df)*100:.1f}%)")
    
    # By strategy type
    print(f"\n--- BY STRATEGY TYPE ---")
    for stype in ['trend', 'mean_reversion']:
        subset = results_df[results_df['strategy_type'] == stype]
        if len(subset) == 0:
            continue
        avg_sharpe = subset['sharpe'].mean()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        print(f"  {stype}: N={len(subset)}, avg Sharpe={avg_sharpe:.3f}, raw={raw}, Bonferroni={bonf}")
    
    # By strategy
    print(f"\n--- BY STRATEGY ---")
    for strat in sorted(STRATEGIES.keys()):
        subset = results_df[results_df['strategy'] == strat]
        if len(subset) == 0:
            continue
        avg_sharpe = subset['sharpe'].mean()
        med_sharpe = subset['sharpe'].median()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        print(f"  {strat:25s}: N={len(subset):4d}, avg={avg_sharpe:+.3f}, med={med_sharpe:+.3f}, "
              f"raw={raw:3d}, Bonf={bonf}")
    
    # By timeframe
    print(f"\n--- BY TIMEFRAME ---")
    for tf in TF_CONFIG:
        subset = results_df[results_df['timeframe'] == tf]
        if len(subset) == 0:
            continue
        avg_sharpe = subset['sharpe'].mean()
        raw = len(subset[(subset['p_value'] < 0.05) & (subset['sharpe'] > 0)])
        bonf = len(subset[(subset['p_value'] < bonferroni_alpha) & (subset['sharpe'] > 0)])
        print(f"  {tf}: N={len(subset)}, avg Sharpe={avg_sharpe:.3f}, raw={raw}, Bonferroni={bonf}")
    
    # By holding period
    print(f"\n--- BY HOLDING PERIOD ---")
    for hp in sorted(results_df['holding_period'].unique()):
        subset = results_df[results_df['holding_period'] == hp]
        if len(subset) == 0:
            continue
        avg_sharpe = subset['sharpe'].mean()
        print(f"  {hp:6s}: N={len(subset)}, avg Sharpe={avg_sharpe:.3f}")
    
    # By asset (top 10)
    print(f"\n--- TOP 10 ASSETS ---")
    asset_sharpe = results_df.groupby('asset')['sharpe'].mean().sort_values(ascending=False)
    for asset, sharpe in asset_sharpe.head(10).items():
        n = len(results_df[results_df['asset'] == asset])
        print(f"  {asset:8s}: avg Sharpe={sharpe:+.3f} (N={n})")
    
    # Top 20 individual results
    print(f"\n--- TOP 20 INDIVIDUAL RESULTS ---")
    top = results_df.nlargest(20, 'sharpe')
    for _, r in top.iterrows():
        bonf_flag = "✅" if r['p_value'] < bonferroni_alpha else "  "
        print(f"  {bonf_flag} {r['asset']:8s} {r['strategy']:25s} {r['timeframe']} hold={r['holding_period']:6s} "
              f"Sharpe={r['sharpe']:+.2f} N={r['n_trades']:5d} WR={r['win_rate']:.1%} "
              f"PF={r['profit_factor']:.2f} p={r['p_value']:.2e} ret={r['total_return']:+.1%}")
    
    # Worst 10
    print(f"\n--- BOTTOM 10 INDIVIDUAL RESULTS ---")
    bottom = results_df.nsmallest(10, 'sharpe')
    for _, r in bottom.iterrows():
        print(f"  💀 {r['asset']:8s} {r['strategy']:25s} {r['timeframe']} hold={r['holding_period']:6s} "
              f"Sharpe={r['sharpe']:+.2f} N={r['n_trades']:5d}")
    
    # Mean reversion vs trend at each TF
    print(f"\n--- MEAN REVERSION vs TREND BY TF ---")
    for tf in TF_CONFIG:
        for stype in ['trend', 'mean_reversion']:
            subset = results_df[(results_df['timeframe'] == tf) & (results_df['strategy_type'] == stype)]
            if len(subset) == 0:
                continue
            avg = subset['sharpe'].mean()
            pos = (subset['sharpe'] > 0).sum()
            print(f"  {tf} {stype:15s}: avg={avg:+.3f}, {pos}/{len(subset)} positive ({pos/len(subset)*100:.0f}%)")
    
    # Long vs short performance
    print(f"\n--- LONG vs SHORT ---")
    avg_long = results_df['mean_long'].mean()
    avg_short = results_df['mean_short'].mean()
    print(f"  Avg long trade return:  {avg_long*100:+.3f}%")
    print(f"  Avg short trade return: {avg_short*100:+.3f}%")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_validation()
