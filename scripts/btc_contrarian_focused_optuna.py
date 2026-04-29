#!/usr/bin/env python3
"""
Focused Optuna optimization around known-good parameters.

The original regime classifier achieved Sharpe 7.93 with these params:
- long_threshold: -0.0005
- short_threshold: 0.001
- zscore_lookback: 24
- zscore_threshold: 2.0
- ADX threshold: 25
- position_size: 0.1

This script optimizes in a VERY NARROW range around these values.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data'
OHLCV_PATH = f'{DATA_DIR}/merged/binance_btc_usdt_4h.csv'
FUNDING_PATH = f'{DATA_DIR}/derivatives/btc_funding.csv'


def load_data() -> pd.DataFrame:
    df = pd.read_csv(OHLCV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df['timestamp'] >= '2023-05-01'].reset_index(drop=True)

    if 'funding_rate' in df.columns:
        df = df.drop(columns=['funding_rate'])

    funding = pd.read_csv(FUNDING_PATH)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
    funding = funding.rename(columns={'fundingRate': 'funding_rate_real'})

    df = pd.merge_asof(df, funding[['timestamp', 'funding_rate_real']],
                       on='timestamp', direction='backward')
    df['funding_rate'] = df['funding_rate_real'].fillna(0)

    return df


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    close = df['close']
    high = df['high']
    low = df['low']

    # ADX (fixed period=14 like original)
    period = 14
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(period).mean()

    df['atr'] = atr
    df['atr_pct'] = df['atr'].rolling(168).rank(pct=True)

    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['sma_200'] = close.rolling(200).mean()

    df['trend_alignment'] = (
        (close > df['sma_20']).astype(int) +
        (close > df['sma_50']).astype(int) +
        (close > df['sma_200']).astype(int) +
        (df['sma_20'] > df['sma_50']).astype(int) +
        (df['sma_50'] > df['sma_200']).astype(int)
    )

    # Regime classification (fixed thresholds like original)
    df['regime'] = 'RANGING'
    trending_up = (df['adx'] > 25) & (df['trend_alignment'] >= 4)
    trending_down = (df['adx'] > 25) & (df['trend_alignment'] <= 1)
    volatile = df['atr_pct'] > 0.8

    df.loc[trending_up, 'regime'] = 'TRENDING_UP'
    df.loc[trending_down, 'regime'] = 'TRENDING_DOWN'
    df.loc[volatile, 'regime'] = 'VOLATILE'

    # Funding z-score
    lookback = params['zscore_lookback']
    df['funding_ma'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).mean()
    df['funding_std'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    df['funding_zscore'] = (df['funding_rate'] - df['funding_ma']) / df['funding_std']

    return df


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']
    z_score = df['funding_zscore']

    long_thresh = params['long_threshold']
    short_thresh = params['short_threshold']
    z_thresh = params['zscore_threshold']

    extreme_negative = (funding < long_thresh) | (z_score < -z_thresh)
    extreme_positive = (funding > short_thresh) | (z_score > z_thresh)

    for i in range(len(df)):
        regime = df['regime'].iloc[i]

        if regime == 'RANGING':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
            elif extreme_positive.iloc[i]:
                signals.iloc[i] = -1

        elif regime == 'VOLATILE':
            if funding.iloc[i] < long_thresh * 2 or z_score.iloc[i] < -z_thresh - 0.5:
                signals.iloc[i] = 1
            elif funding.iloc[i] > short_thresh * 2 or z_score.iloc[i] > z_thresh + 0.5:
                signals.iloc[i] = -1

        elif regime == 'TRENDING_UP':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1

        elif regime == 'TRENDING_DOWN':
            if extreme_positive.iloc[i]:
                signals.iloc[i] = -1

    return signals


def backtest(df: pd.DataFrame, signals: pd.Series, params: dict) -> dict:
    n = len(df)
    position_size = params['position_size']
    commission = 0.0004

    pnl = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    total_trades = 0
    wins = 0

    for i in range(1, n):
        signal = signals.iloc[i-1]
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]

        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size

        if signal != 0 and signal != current_pos:
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                if pnl.iloc[i] > 0:
                    wins += 1
            current_pos = signal
            total_trades += 1
            pnl.iloc[i] -= commission * abs(signal) * position_size
        elif signal == 0 and current_pos != 0:
            pnl.iloc[i] -= commission * abs(current_pos) * position_size
            if pnl.iloc[i] > 0:
                wins += 1
            current_pos = 0.0
            total_trades += 1

    cumulative = (1 + pnl).cumprod() - 1
    returns = pnl[pnl != 0]

    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252 * 6) * returns.mean() / returns.std()
    else:
        sharpe = 0

    cummax = (1 + pnl).cumprod().cummax()
    dd = (1 + pnl).cumprod() / cummax - 1
    max_dd = dd.min()

    return {
        'total_return': cumulative.iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0
    }


def walk_forward(df: pd.DataFrame, params: dict, n_splits: int = 5) -> dict:
    n = len(df)
    fold_size = n // n_splits

    sharpes = []
    returns = []
    trades = []

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)
        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * 0.7)
        test_df = fold_data.iloc[train_size:].reset_index(drop=True)

        if len(test_df) < 50:
            continue

        test_df = calculate_indicators(test_df, params)
        test_signals = generate_signals(test_df, params)
        result = backtest(test_df, test_signals, params)

        sharpes.append(result['sharpe_ratio'])
        returns.append(result['total_return'])
        trades.append(result['total_trades'])

    return {'sharpes': sharpes, 'returns': returns, 'trades': trades}


def create_objective(df: pd.DataFrame):
    """
    Objective: avg_sharpe * sqrt(compounded_return) + return_bonus

    Focus on RETURNS while maintaining positive Sharpe.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            # VERY NARROW ranges around original values
            'long_threshold': trial.suggest_float('long_threshold', -0.0007, -0.0003),  # Original: -0.0005
            'short_threshold': trial.suggest_float('short_threshold', 0.0008, 0.0012),  # Original: 0.001
            'zscore_lookback': trial.suggest_int('zscore_lookback', 20, 28),  # Original: 24
            'zscore_threshold': trial.suggest_float('zscore_threshold', 1.8, 2.2),  # Original: 2.0
            'position_size': trial.suggest_float('position_size', 0.08, 0.15),  # Original: 0.1
        }

        results = walk_forward(df, params)

        sharpes = results['sharpes']
        returns = results['returns']
        trades = results['trades']

        if len(sharpes) == 0 or min(trades) < 3:
            return -100

        avg_sharpe = np.mean(sharpes)
        compounded = np.prod([1 + r for r in returns]) - 1
        consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)

        # Objective emphasizes RETURNS
        if compounded >= 0:
            objective = avg_sharpe + compounded * 100 + consistency * 2
        else:
            objective = avg_sharpe * 0.5 + compounded * 100

        return objective

    return objective


def main():
    print("=" * 80)
    print("FOCUSED OPTUNA: Narrow Search Around Known-Good Parameters")
    print("=" * 80)

    df = load_data()
    print(f"Data: {len(df):,} bars")

    # First, test original parameters
    print("\n" + "-" * 40)
    print("BASELINE: Original Parameters")
    print("-" * 40)

    original_params = {
        'long_threshold': -0.0005,
        'short_threshold': 0.001,
        'zscore_lookback': 24,
        'zscore_threshold': 2.0,
        'position_size': 0.1,
    }

    original_results = walk_forward(df, original_params)
    orig_sharpe = np.mean(original_results['sharpes'])
    orig_return = np.prod([1 + r for r in original_results['returns']]) - 1
    orig_consistency = sum(1 for s in original_results['sharpes'] if s > 0) / len(original_results['sharpes'])

    print(f"Original Avg Sharpe: {orig_sharpe:.3f}")
    print(f"Original Return: {orig_return*100:.2f}%")
    print(f"Original Consistency: {orig_consistency*100:.0f}%")
    print(f"Fold Sharpes: {[f'{s:.2f}' for s in original_results['sharpes']]}")

    # Run optimization
    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = create_objective(df)

    print("\n" + "=" * 80)
    print("Running focused optimization (100 trials)...")
    print("=" * 80)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Best parameters
    print("\n" + "=" * 80)
    print("BEST PARAMETERS (Focused Search)")
    print("=" * 80)

    best_params = study.best_params
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    print(f"\nBest objective: {study.best_value:.4f}")

    # Validate best params
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    best_results = walk_forward(df, best_params)

    print(f"\n{'Fold':<6} {'Sharpe':>10} {'Return':>10} {'Trades':>8}")
    print("-" * 40)

    for i, (s, r, t) in enumerate(zip(best_results['sharpes'], best_results['returns'], best_results['trades'])):
        print(f"  {i:<4} {s:>10.3f} {r*100:>9.1f}% {t:>8}")

    best_sharpe = np.mean(best_results['sharpes'])
    best_return = np.prod([1 + r for r in best_results['returns']]) - 1
    best_consistency = sum(1 for s in best_results['sharpes'] if s > 0) / len(best_results['sharpes'])

    print("-" * 40)
    print(f"  AVG  {best_sharpe:>10.3f} {best_return*100:>9.1f}%")
    print(f"  Consistency: {best_consistency*100:.0f}%")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Optimized")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Original':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 65)
    print(f"{'Avg Sharpe':<20} {orig_sharpe:>15.3f} {best_sharpe:>15.3f} {(best_sharpe-orig_sharpe):>14.3f}")
    print(f"{'Compounded Return':<20} {orig_return*100:>14.2f}% {best_return*100:>14.2f}% {(best_return-orig_return)*100:>13.2f}%")
    print(f"{'Consistency':<20} {orig_consistency*100:>14.0f}% {best_consistency*100:>14.0f}%")

    # Save
    output = {
        'original_params': original_params,
        'original_sharpe': orig_sharpe,
        'original_return': orig_return,
        'original_consistency': orig_consistency,
        'best_params': best_params,
        'best_objective': study.best_value,
        'best_sharpe': best_sharpe,
        'best_return': best_return,
        'best_consistency': best_consistency,
        'fold_sharpes': best_results['sharpes'],
        'fold_returns': best_results['returns'],
        'fold_trades': best_results['trades']
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/optuna_focused_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
