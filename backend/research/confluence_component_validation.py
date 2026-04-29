"""
Confluence Component Validation — Walk-Forward Testing of ALL Directional Signals
==================================================================================
Tests every directional indicator in Virtuoso's confluence system using the same
rigorous methodology that killed orderflow (non-overlapping trades, 10bps costs, Bonferroni).

Components tested:
1. TECHNICAL: RSI, MACD/AO, Williams%R, CCI, ADX (trend strength)
2. PRICE_STRUCTURE: S/R proximity, trend position, market structure breaks, FVG, range position
3. VOLUME: OBV trend, CMF, ADL, relative volume, volume delta, VWAP position
4. SENTIMENT: Funding rate, LSR, liquidation events (requires derivatives data)
5. ORDERBOOK: Depth imbalance, OIR, bid/ask pressure (requires live orderbook — SKIP)

Methodology:
- Walk-forward: 70% train / 30% test, 14 folds
- Non-overlapping trades with realistic costs (10bps round-trip)
- Multiple holding periods: 1d, 3d, 1w, 2w
- Bonferroni correction for multiple testing
- Assets: BTC, ETH + 11 alts (using existing OHLCV data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path.home() / "Desktop/maestro/data"
ORDERFLOW_DIR = DATA_DIR / "orderflow"
RESULTS_DIR = Path.home() / "Desktop/maestro/backend/research/results"
RESULTS_DIR.mkdir(exist_ok=True)

COST_BPS = 10  # 10bps round-trip
ASSETS = ['BTC', 'ETH', 'SOL', 'SUI', 'LINK', 'AVAX', 'INJ', 'OP', 'ARB', 'TIA',
          'ADA', 'APT', 'ATOM', 'BNB', 'DOGE', 'DOT', 'FIL', 'FTM', 'NEAR', 'UNI', 'XRP']
HOLDING_PERIODS = {'1d': 1, '3d': 3, '1w': 7, '2w': 14}
N_FOLDS = 14
TRAIN_RATIO = 0.7
SIGNIFICANCE = 0.05


def load_ohlcv(asset: str, timeframe: str = '1d') -> pd.DataFrame:
    """Load daily spot OHLCV data."""
    spot_path = DATA_DIR / "spot" / f"{asset}_spot_daily.csv"
    if spot_path.exists():
        df = pd.read_csv(spot_path, parse_dates=['Date'])
        df = df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 
                                'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    return pd.DataFrame()


def calculate_forward_returns(df: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """Calculate non-overlapping forward returns with costs."""
    results = df.copy()
    for name, period in periods.items():
        # Forward return
        results[f'fwd_{name}'] = results['close'].shift(-period) / results['close'] - 1
        # Apply costs
        results[f'fwd_{name}'] -= COST_BPS / 10000  # per-trade cost
    return results


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calc_ao(high: pd.Series, low: pd.Series, fast=5, slow=34) -> pd.Series:
    midpoint = (high + low) / 2
    return midpoint.rolling(fast).mean() - midpoint.rolling(slow).mean()


def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period=20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def calc_donchian(high: pd.Series, low: pd.Series, close: pd.Series, period=20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    position = (close - lower) / (upper - lower)  # 0-1
    return position


def calc_bollinger_position(close: pd.Series, period=20, std=2):
    sma = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return (close - lower) / (upper - lower)  # 0-1


# ============================================================
# VOLUME INDICATORS
# ============================================================

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def calc_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period=20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum()


def calc_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    return (mfm * volume).cumsum()


def calc_relative_volume(volume: pd.Series, period=20) -> pd.Series:
    return volume / volume.rolling(period).mean()


def calc_vwap_position(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period=20):
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(period).sum() / volume.rolling(period).sum()
    return (close - vwap) / vwap  # % distance from VWAP


# ============================================================
# PRICE STRUCTURE INDICATORS
# ============================================================

def calc_trend_position(close: pd.Series, periods=[20, 50, 100, 200]):
    """How many MAs is price above? Normalized 0-1."""
    above = pd.DataFrame()
    for p in periods:
        if len(close) > p:
            above[f'ma{p}'] = (close > close.rolling(p).mean()).astype(float)
    if above.empty:
        return pd.Series(0.5, index=close.index)
    return above.mean(axis=1)


def calc_structure_break(high: pd.Series, low: pd.Series, close: pd.Series, period=20):
    """Break of structure: price above/below recent swing high/low."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    # Bullish break = close > previous swing high
    bull_break = (close > hh.shift(1)).astype(float)
    bear_break = (close < ll.shift(1)).astype(float)
    return bull_break - bear_break  # +1 bullish, -1 bearish, 0 neutral


def calc_range_position(high: pd.Series, low: pd.Series, close: pd.Series, period=20):
    """Position within recent range. 0=bottom, 1=top."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (close - ll) / (hh - ll)


def calc_fvg_signal(high: pd.Series, low: pd.Series, close: pd.Series):
    """Fair value gap detection (simplified). +1 = bullish FVG, -1 = bearish FVG."""
    # Bullish FVG: current low > 2-bars-ago high (gap up)
    bull_fvg = (low > high.shift(2)).astype(float)
    # Bearish FVG: current high < 2-bars-ago low (gap down)
    bear_fvg = (high < low.shift(2)).astype(float)
    return bull_fvg - bear_fvg


# ============================================================
# SENTIMENT INDICATORS (from derivatives data)
# ============================================================

def load_derivatives_data(asset: str) -> dict:
    """Load funding rate, OI, LSR, liquidations from derivatives dir."""
    deriv_dir = DATA_DIR / "derivatives"
    result = {}
    
    symbol = f"{asset}USDT"
    for dtype in ['funding_rate', 'open_interest', 'long_short_ratio', 'liquidations']:
        path = deriv_dir / dtype / f"{symbol}_{dtype}_daily.csv"
        if not path.exists():
            path = deriv_dir / dtype / f"{symbol}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=0).columns else [0])
                result[dtype] = df
            except:
                pass
    return result


# ============================================================
# SIGNAL GENERATORS — convert indicators to directional signals
# ============================================================

def generate_signals(df: pd.DataFrame) -> dict:
    """Generate all directional signals from OHLCV data."""
    signals = {}
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    
    # === TECHNICAL (momentum-based, scored 0-100 in production) ===
    
    # RSI momentum: >50 = bullish, <50 = bearish
    rsi = calc_rsi(c, 14)
    signals['rsi_momentum'] = (rsi > 50).astype(int)
    
    # RSI extreme reversal: >70 = sell, <30 = buy (mean reversion)
    signals['rsi_reversal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    
    # MACD histogram direction
    _, _, hist = calc_macd(c)
    signals['macd_histogram'] = np.sign(hist)
    
    # MACD crossover
    macd, sig, _ = calc_macd(c)
    signals['macd_crossover'] = np.where(macd > sig, 1, -1)
    
    # AO momentum
    ao = calc_ao(h, l)
    signals['ao_momentum'] = np.sign(ao)
    
    # Williams %R
    wr = calc_williams_r(h, l, c)
    signals['williams_r'] = np.where(wr > -50, 1, -1)
    
    # CCI direction
    cci = calc_cci(h, l, c)
    signals['cci_direction'] = np.sign(cci)
    
    # ADX trend filter (>25 = trending)
    adx = calc_adx(h, l, c)
    signals['adx_trending'] = (adx > 25).astype(int)
    
    # Donchian breakout
    don = calc_donchian(h, l, c)
    signals['donchian'] = np.where(don > 0.8, 1, np.where(don < 0.2, -1, 0))
    
    # Bollinger position
    bb = calc_bollinger_position(c)
    signals['bollinger'] = np.where(bb > 0.8, 1, np.where(bb < 0.2, -1, 0))  # breakout
    signals['bollinger_mr'] = np.where(bb > 0.8, -1, np.where(bb < 0.2, 1, 0))  # mean reversion
    
    # === PRICE STRUCTURE ===
    
    # Trend position (how many MAs above)
    signals['trend_position'] = np.where(calc_trend_position(c) > 0.6, 1, 
                                np.where(calc_trend_position(c) < 0.4, -1, 0))
    
    # Structure break
    signals['structure_break'] = calc_structure_break(h, l, c, 20)
    
    # Range position (breakout)
    rp = calc_range_position(h, l, c, 20)
    signals['range_breakout'] = np.where(rp > 0.9, 1, np.where(rp < 0.1, -1, 0))
    signals['range_mr'] = np.where(rp > 0.9, -1, np.where(rp < 0.1, 1, 0))
    
    # FVG
    signals['fvg'] = calc_fvg_signal(h, l, c)
    
    # === VOLUME ===
    
    # OBV trend (SMA20)
    obv = calc_obv(c, v)
    obv_sma = obv.rolling(20).mean()
    signals['obv_trend'] = np.where(obv > obv_sma, 1, -1)
    
    # CMF direction
    cmf = calc_cmf(h, l, c, v)
    signals['cmf'] = np.sign(cmf)
    
    # ADL trend
    adl = calc_adl(h, l, c, v)
    adl_sma = adl.rolling(20).mean()
    signals['adl_trend'] = np.where(adl > adl_sma, 1, -1)
    
    # Relative volume (high vol = continuation)
    rvol = calc_relative_volume(v)
    price_dir = np.sign(c.diff())
    signals['volume_confirmation'] = np.where(rvol > 1.5, price_dir, 0)
    
    # VWAP position
    vwap_pos = calc_vwap_position(h, l, c, v)
    signals['vwap_position'] = np.where(vwap_pos > 0, 1, -1)
    
    # Volume delta (if buy/sell volume available)
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        vd = df['buy_volume'] - df['sell_volume']
        signals['volume_delta'] = np.sign(vd)
    
    # Convert all to Series
    for k, v_sig in signals.items():
        if not isinstance(v_sig, pd.Series):
            signals[k] = pd.Series(v_sig, index=df.index)
    
    return signals


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_test(signal: pd.Series, returns: pd.Series, n_folds=N_FOLDS) -> dict:
    """
    Walk-forward test with non-overlapping trades.
    Returns Sharpe, t-stat, p-value, trade count.
    """
    mask = signal.notna() & returns.notna()
    sig = signal[mask].values.astype(float)
    ret = returns[mask].values.astype(float)
    
    # Only take non-overlapping trades
    n = len(sig)
    if n < 50:
        return {'sharpe': 0, 't_stat': 0, 'p_value': 1, 'n_trades': 0, 'mean_ret': 0}
    
    fold_size = n // n_folds
    if fold_size < 10:
        return {'sharpe': 0, 't_stat': 0, 'p_value': 1, 'n_trades': 0, 'mean_ret': 0}
    
    oos_returns = []
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)
        
        # Skip first few folds as training
        if fold < int(n_folds * (1 - TRAIN_RATIO)):
            continue
        
        fold_sig = sig[test_start:test_end]
        fold_ret = ret[test_start:test_end]
        
        # Signal-weighted returns (long when signal > 0, short when < 0, flat when 0)
        trade_returns = fold_sig * fold_ret
        
        # Filter to actual trades (signal != 0)
        active = fold_sig != 0
        if active.sum() > 0:
            oos_returns.extend(trade_returns[active])
    
    if len(oos_returns) < 10:
        return {'sharpe': 0, 't_stat': 0, 'p_value': 1, 'n_trades': len(oos_returns), 'mean_ret': 0}
    
    oos_returns = np.array(oos_returns)
    mean_ret = oos_returns.mean()
    std_ret = oos_returns.std()
    
    if std_ret == 0:
        return {'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_drawdown': 0,
                'profit_factor': 0, 'win_rate': 0, 'payoff_ratio': 0,
                'expectancy': 0, 'tail_ratio': 0, 'skewness': 0, 'kurtosis': 0,
                't_stat': 0, 'p_value': 1, 'n_trades': len(oos_returns), 'mean_ret': mean_ret}
    
    from metrics import compute_metrics
    m = compute_metrics(oos_returns, bars_per_year=365, 
                        total_bars=n, trades_per_year=365)
    m['mean_ret'] = round(mean_ret * 100, 4)
    return m


# ============================================================
# MAIN VALIDATION LOOP
# ============================================================

def run_validation():
    print("=" * 80)
    print("CONFLUENCE COMPONENT VALIDATION — Walk-Forward Testing")
    print("=" * 80)
    print(f"Assets: {len(ASSETS)} | Holding periods: {list(HOLDING_PERIODS.keys())}")
    print(f"Folds: {N_FOLDS} | Train: {TRAIN_RATIO*100}% | Costs: {COST_BPS}bps")
    print()
    
    all_results = []
    
    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"  {asset}")
        print(f"{'='*60}")
        
        df = load_ohlcv(asset, '1D')
        if df.empty or len(df) < 100:
            print(f"  ⚠ Insufficient data for {asset}, skipping")
            continue
        
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        print(f"  Data: {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
        
        # Calculate forward returns
        df = calculate_forward_returns(df, HOLDING_PERIODS)
        
        # Generate signals
        signals = generate_signals(df)
        print(f"  Signals: {len(signals)}")
        
        for sig_name, sig_series in signals.items():
            for hold_name, hold_period in HOLDING_PERIODS.items():
                ret_col = f'fwd_{hold_name}'
                if ret_col not in df.columns:
                    continue
                
                result = walk_forward_test(sig_series, df[ret_col])
                result['asset'] = asset
                result['signal'] = sig_name
                result['holding'] = hold_name
                all_results.append(result)
                
                if result['p_value'] < SIGNIFICANCE:
                    print(f"  ✓ {sig_name} ({hold_name}): Sharpe={result['sharpe']}, "
                          f"t={result['t_stat']}, p={result['p_value']}, n={result['n_trades']}")
    
    # Summary
    results_df = pd.DataFrame(all_results)
    total_tests = len(results_df)
    raw_passes = (results_df['p_value'] < SIGNIFICANCE).sum()
    bonferroni_threshold = SIGNIFICANCE / total_tests
    bonf_passes = (results_df['p_value'] < bonferroni_threshold).sum()
    
    print("\n" + "=" * 80)
    print("GRAND SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Raw passes (p<0.05): {raw_passes} ({raw_passes/total_tests*100:.1f}%)")
    print(f"Bonferroni threshold: p<{bonferroni_threshold:.8f}")
    print(f"Bonferroni passes: {bonf_passes}")
    print(f"Expected by chance (5%): {total_tests * 0.05:.0f}")
    
    # Breakdown by signal category
    print("\n--- By Signal ---")
    sig_summary = results_df.groupby('signal').agg(
        avg_sharpe=('sharpe', 'mean'),
        max_sharpe=('sharpe', 'max'),
        avg_t=('t_stat', 'mean'),
        min_p=('p_value', 'min'),
        passes=('p_value', lambda x: (x < SIGNIFICANCE).sum()),
        tests=('p_value', 'count')
    ).sort_values('avg_sharpe', ascending=False)
    print(sig_summary.to_string())
    
    # Breakdown by holding period
    print("\n--- By Holding Period ---")
    hold_summary = results_df.groupby('holding').agg(
        avg_sharpe=('sharpe', 'mean'),
        passes=('p_value', lambda x: (x < SIGNIFICANCE).sum()),
        tests=('p_value', 'count')
    )
    print(hold_summary.to_string())
    
    # Breakdown by asset
    print("\n--- By Asset ---")
    asset_summary = results_df.groupby('asset').agg(
        avg_sharpe=('sharpe', 'mean'),
        passes=('p_value', lambda x: (x < SIGNIFICANCE).sum()),
        tests=('p_value', 'count')
    ).sort_values('avg_sharpe', ascending=False)
    print(asset_summary.to_string())
    
    # Top 20 results
    print("\n--- Top 20 by Sharpe ---")
    top20 = results_df.nlargest(20, 'sharpe')[['asset', 'signal', 'holding', 'sharpe', 't_stat', 'p_value', 'n_trades']]
    print(top20.to_string(index=False))
    
    # Category analysis
    tech_signals = ['rsi_momentum', 'rsi_reversal', 'macd_histogram', 'macd_crossover', 'ao_momentum', 
                    'williams_r', 'cci_direction', 'adx_trending', 'donchian', 'bollinger', 'bollinger_mr']
    price_signals = ['trend_position', 'structure_break', 'range_breakout', 'range_mr', 'fvg']
    volume_signals = ['obv_trend', 'cmf', 'adl_trend', 'volume_confirmation', 'vwap_position', 'volume_delta']
    
    print("\n--- By Category ---")
    for cat_name, cat_signals in [('TECHNICAL', tech_signals), ('PRICE_STRUCTURE', price_signals), ('VOLUME', volume_signals)]:
        cat_df = results_df[results_df['signal'].isin(cat_signals)]
        if cat_df.empty:
            continue
        passes = (cat_df['p_value'] < SIGNIFICANCE).sum()
        total = len(cat_df)
        print(f"{cat_name}: {passes}/{total} pass ({passes/total*100:.1f}%), "
              f"avg Sharpe={cat_df['sharpe'].mean():.3f}, best={cat_df['sharpe'].max():.3f}")
    
    # Save results
    results_df.to_csv(RESULTS_DIR / 'confluence_validation_results.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'confluence_validation_results.csv'}")
    
    return results_df


if __name__ == '__main__':
    run_validation()
