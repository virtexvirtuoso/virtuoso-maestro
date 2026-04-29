#!/usr/bin/env python3
"""
Focused Derivatives Signal Validation — Sub-hourly with proper price data.
Tests 15m and 5m derivatives against matching spot prices.
Uses stricter thresholds to keep trade counts reasonable.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

DERIV_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/derivatives_5m/compiled"))
SPOT_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/spot"))
RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/backend/research/results"))
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Key assets with good liquidity and full data overlap
ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'ARBUSDT', 'OPUSDT',
    'FETUSDT', 'SUIUSDT', 'TIAUSDT', 'SEIUSDT',
]

# Timeframe configs: (deriv_tf, spot_tf, holding_periods_in_bars, tf_minutes, lookback)
TIMEFRAME_CONFIGS = {
    '5m_5m':   ('5m',  '5m',  [3, 12, 48],    5,  60),   # hold 15m/1h/4h, lookback 5hrs
    '15m_15m': ('15m', '15m', [4, 16, 48],    15,  40),   # hold 1h/4h/12h, lookback 10hrs
    '1h_1h':   ('1h',  '1h',  [4, 12, 24],    60,  50),   # hold 4h/12h/1d, lookback 50hrs
}

HOLDING_LABELS = {
    '5m_5m':   ['15m', '1h', '4h'],
    '15m_15m': ['1h', '4h', '12h'],
    '1h_1h':   ['4h', '12h', '1d'],
}

COST_BPS = 10
MIN_TRADES = 30
ALPHA = 0.05


def load_deriv(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    f = DERIV_PATH / f"{symbol}_{tf}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Forward-fill nulls in ratio columns
    for col in ['top_trader_accounts_lsr', 'top_trader_positions_lsr',
                'global_accounts_lsr', 'taker_buy_sell_ratio']:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def load_spot(ticker: str, tf: str) -> Optional[pd.DataFrame]:
    f = SPOT_PATH / tf / f"{ticker}_spot_{tf}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get('date', col_map.get('timestamp', df.columns[0]))
    close_col = col_map.get('close', df.columns[4])
    df = df.rename(columns={ts_col: 'timestamp', close_col: 'close'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', 'close']]


def merge_data(symbol: str, deriv_tf: str, spot_tf: str) -> Optional[pd.DataFrame]:
    ticker = symbol.replace('USDT', '')
    deriv = load_deriv(symbol, deriv_tf)
    spot = load_spot(ticker, spot_tf)
    if deriv is None or spot is None:
        return None
    merged = pd.merge_asof(
        deriv.sort_values('timestamp'),
        spot.sort_values('timestamp'),
        on='timestamp', direction='backward', suffixes=('', '_spot')
    )
    merged = merged.dropna(subset=['close'])
    if len(merged) < 500:
        return None
    # Use second half as OOS
    mid = len(merged) // 2
    return merged.iloc[mid:].reset_index(drop=True)


def generate_signals(df: pd.DataFrame, lookback: int) -> dict:
    """Generate all signals with STRICT thresholds to reduce trade frequency."""
    signals = {}
    oi = df['oi_usd']
    close = df['close']

    # Rolling stats
    oi_mean = oi.rolling(lookback).mean()
    oi_std = oi.rolling(lookback).std()
    oi_z = (oi - oi_mean) / oi_std
    oi_change = oi.pct_change(1)
    price_change = close.pct_change(1)

    # OI signals — STRICT thresholds
    signals['oi_momentum_strict'] = (oi_change > oi_change.rolling(lookback).quantile(0.8)).astype(int)
    signals['oi_breakout'] = (oi_z > 2.5).astype(int)
    signals['oi_flush'] = (oi_z < -2.5).astype(int)
    signals['oi_divergence_bull'] = ((price_change < -price_change.rolling(lookback).std()) & (oi_change > 0)).astype(int)

    # LSR signals
    lsr = df.get('global_accounts_lsr')
    if lsr is not None and lsr.notna().sum() > lookback:
        lsr_z = (lsr - lsr.rolling(lookback).mean()) / lsr.rolling(lookback).std()
        signals['lsr_extreme_long'] = (lsr_z < -2).astype(int)  # crowd very short → go long
        signals['lsr_extreme_short'] = (lsr_z > 2).astype(int)  # crowd very long → stay flat (0)

    # Top trader signals
    top_lsr = df.get('top_trader_positions_lsr')
    if top_lsr is not None and top_lsr.notna().sum() > lookback:
        top_z = (top_lsr - top_lsr.rolling(lookback).mean()) / top_lsr.rolling(lookback).std()
        signals['top_trader_bullish'] = (top_z > 1.5).astype(int)

    # Taker signals — STRICT
    taker = df.get('taker_buy_sell_ratio')
    if taker is not None and taker.notna().sum() > lookback:
        taker_ma = taker.rolling(lookback).mean()
        taker_std = taker.rolling(lookback).std()
        taker_z = (taker - taker_ma) / taker_std
        signals['taker_surge'] = (taker_z > 2).astype(int)
        signals['taker_capitulation'] = (taker_z < -2).astype(int)  # selling exhaustion → long
        signals['taker_regime'] = (taker_ma > 1.1).astype(int)

    # Combined — require multiple conditions
    if 'taker_surge' in signals:
        signals['oi_taker_confirm'] = ((oi_change > 0) & (taker_z > 1.5)).astype(int)
    if lsr is not None and lsr.notna().sum() > lookback:
        signals['oi_lsr_smart'] = ((oi_change > oi_change.rolling(lookback).quantile(0.7)) &
                                    (lsr_z < 0)).astype(int)

    return signals


def evaluate_signal(signal: pd.Series, prices: pd.Series, hold_bars: int, tf_minutes: int) -> dict:
    """Evaluate with non-overlapping trades and transaction costs."""
    cost = COST_BPS / 10000

    # Find non-overlapping trade entries
    entries = []
    i = 0
    while i < len(signal) - hold_bars:
        if signal.iloc[i] == 1:
            entries.append(i)
            i += hold_bars  # skip holding period
        else:
            i += 1

    if len(entries) < MIN_TRADES:
        return {'n_trades': len(entries), 'valid': False}

    # Calculate returns
    returns = []
    for idx in entries:
        if idx + hold_bars < len(prices):
            ret = prices.iloc[idx + hold_bars] / prices.iloc[idx] - 1 - cost
            returns.append(ret)

    if len(returns) < MIN_TRADES:
        return {'n_trades': len(returns), 'valid': False}

    returns = np.array(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0 or np.isnan(std_ret):
        return {'n_trades': len(returns), 'valid': False}

    # Annualization
    trades_per_year = len(returns) / ((len(signal) * tf_minutes) / (365.25 * 24 * 60))
    ann_factor = np.sqrt(max(trades_per_year, 1))

    sharpe = (mean_ret / std_ret) * ann_factor
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    win_rate = (returns > 0).mean()

    return {
        'n_trades': len(returns),
        'mean_return': mean_ret,
        'std_return': std_ret,
        'sharpe': sharpe,
        't_stat': t_stat,
        'p_value': p_value,
        'win_rate': win_rate,
        'valid': True,
    }


def main():
    print("=" * 80)
    print("FOCUSED DERIVATIVES SIGNAL VALIDATION")
    print("=" * 80)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {len(ASSETS)}")
    print(f"Timeframes: {list(TIMEFRAME_CONFIGS.keys())}")
    print(f"Signals: 12 (strict thresholds)")
    print(f"Cost: {COST_BPS} bps per round trip")
    print("=" * 80)

    all_results = []
    total_combos = len(TIMEFRAME_CONFIGS) * len(ASSETS) * 3 * 12  # rough estimate
    done = 0

    for tf_name, (deriv_tf, spot_tf, hold_bars_list, tf_min, lookback) in TIMEFRAME_CONFIGS.items():
        hold_labels = HOLDING_LABELS[tf_name]
        print(f"\n{'=' * 60}")
        print(f"TIMEFRAME: {tf_name} (lookback={lookback} bars = {lookback * tf_min / 60:.0f}h)")
        print(f"{'=' * 60}")

        for asset in ASSETS:
            df = merge_data(asset, deriv_tf, spot_tf)
            if df is None:
                done += len(hold_bars_list) * 12
                continue

            signals = generate_signals(df, lookback)

            for sig_name, sig_series in signals.items():
                for hold_bars, hold_label in zip(hold_bars_list, hold_labels):
                    done += 1
                    result = evaluate_signal(sig_series, df['close'], hold_bars, tf_min)

                    if not result['valid']:
                        status = '-'
                        print(f"[{done:4d}] {status} {tf_name:7s} {hold_label:4s} {asset:12s} {sig_name:25s} N={result['n_trades']:5d} (insufficient)")
                        continue

                    r = result
                    p_bonf = min(r['p_value'] * total_combos, 1.0)
                    pass_raw = r['p_value'] < ALPHA and r['sharpe'] > 0
                    pass_bonf = p_bonf < ALPHA and r['sharpe'] > 0

                    if r['sharpe'] > 0:
                        status = '★' if pass_bonf else '●' if pass_raw else '◇'
                    else:
                        status = '✗'

                    print(f"[{done:4d}] {status} {tf_name:7s} {hold_label:4s} {asset:12s} {sig_name:25s} "
                          f"N={r['n_trades']:5d} Sharpe={r['sharpe']:7.2f} p={r['p_value']:.4f} wr={r['win_rate']:.1%}")

                    all_results.append({
                        'signal': sig_name, 'timeframe': tf_name, 'holding_period': hold_label,
                        'asset': asset, 'n_trades': r['n_trades'], 'mean_return': r['mean_return'],
                        'sharpe': r['sharpe'], 't_stat': r['t_stat'], 'p_value': r['p_value'],
                        'p_bonferroni': p_bonf, 'win_rate': r['win_rate'],
                        'pass_raw': pass_raw, 'pass_bonferroni': pass_bonf,
                    })

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_PATH / 'derivatives_validation_focused.csv', index=False)

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")

        valid = results_df[results_df['n_trades'] >= MIN_TRADES]
        print(f"\nTotal valid tests: {len(valid)}")
        print(f"Positive Sharpe: {(valid['sharpe'] > 0).sum()} ({(valid['sharpe'] > 0).mean()*100:.1f}%)")
        print(f"Raw significant + positive: {valid['pass_raw'].sum()}")
        print(f"Bonferroni survivors: {valid['pass_bonferroni'].sum()}")

        print(f"\n### BY TIMEFRAME ###")
        by_tf = valid.groupby('timeframe').agg(
            avg_sharpe=('sharpe', 'mean'),
            pct_positive=('sharpe', lambda x: (x > 0).mean()),
            raw_pass=('pass_raw', 'sum'),
            bonf_pass=('pass_bonferroni', 'sum'),
            count=('sharpe', 'count'),
        )
        print(by_tf.to_string())

        print(f"\n### BY SIGNAL ###")
        by_sig = valid.groupby('signal').agg(
            avg_sharpe=('sharpe', 'mean'),
            pct_positive=('sharpe', lambda x: (x > 0).mean()),
            raw_pass=('pass_raw', 'sum'),
            count=('sharpe', 'count'),
        ).sort_values('avg_sharpe', ascending=False)
        print(by_sig.to_string())

        print(f"\n### TOP 20 (by Sharpe) ###")
        top = valid.nlargest(20, 'sharpe')[['signal', 'timeframe', 'holding_period', 'asset', 'sharpe', 'p_value', 'n_trades', 'win_rate']]
        print(top.to_string())

        print(f"\n### WORST 10 (most negative) ###")
        worst = valid.nsmallest(10, 'sharpe')[['signal', 'timeframe', 'holding_period', 'asset', 'sharpe', 'n_trades']]
        print(worst.to_string())

    print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
