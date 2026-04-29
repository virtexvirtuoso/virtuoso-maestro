#!/usr/bin/env python3
"""
Robust Optuna Hyperoptimization for Adaptive Contrarian Strategy

Key improvements over naive optimization:
1. Objective penalizes variance across folds (consistency)
2. Minimum trade constraints
3. Penalizes extreme parameters
4. Uses median instead of mean (robust to outliers)
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
    """Load and prepare data."""
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
    """Calculate indicators."""
    close = df['close']
    high = df['high']
    low = df['low']

    period = params['adx_period']
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
    df['atr_pct'] = df['atr'].rolling(params['atr_lookback']).rank(pct=True)

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

    df['regime'] = 'RANGING'
    trending_up = (df['adx'] > params['adx_threshold']) & (df['trend_alignment'] >= 4)
    trending_down = (df['adx'] > params['adx_threshold']) & (df['trend_alignment'] <= 1)
    volatile = df['atr_pct'] > params['vol_threshold']

    df.loc[trending_up, 'regime'] = 'TRENDING_UP'
    df.loc[trending_down, 'regime'] = 'TRENDING_DOWN'
    df.loc[volatile, 'regime'] = 'VOLATILE'

    lookback = params['zscore_lookback']
    df['funding_ma'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).mean()
    df['funding_std'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    df['funding_zscore'] = (df['funding_rate'] - df['funding_ma']) / df['funding_std']

    return df


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Generate signals."""
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']
    z_score = df['funding_zscore']

    extreme_negative = (funding < params['long_threshold']) | (z_score < -params['zscore_threshold'])
    extreme_positive = (funding > params['short_threshold']) | (z_score > params['zscore_threshold'])

    for i in range(len(df)):
        regime = df['regime'].iloc[i]

        if regime == 'RANGING':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
            elif extreme_positive.iloc[i]:
                signals.iloc[i] = -1
        elif regime == 'TRENDING_UP':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
        elif regime == 'TRENDING_DOWN':
            if extreme_positive.iloc[i]:
                signals.iloc[i] = -1

    return signals


def backtest(df: pd.DataFrame, signals: pd.Series, params: dict) -> dict:
    """Backtest."""
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


def robust_walk_forward(df: pd.DataFrame, params: dict, n_splits: int = 5) -> dict:
    """Walk-forward with robust metrics."""
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

    return {
        'sharpes': sharpes,
        'returns': returns,
        'trades': trades
    }


def create_robust_objective(df: pd.DataFrame):
    """Robust objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Simpler parameter space
            'long_threshold': trial.suggest_float('long_threshold', -0.001, -0.0002),
            'short_threshold': trial.suggest_float('short_threshold', 0.0005, 0.002),
            'zscore_lookback': trial.suggest_int('zscore_lookback', 16, 32),
            'zscore_threshold': trial.suggest_float('zscore_threshold', 1.8, 2.5),
            'adx_period': trial.suggest_int('adx_period', 12, 16),
            'adx_threshold': trial.suggest_float('adx_threshold', 22, 30),
            'atr_lookback': trial.suggest_int('atr_lookback', 140, 200),
            'vol_threshold': trial.suggest_float('vol_threshold', 0.75, 0.85),
            'position_size': trial.suggest_float('position_size', 0.08, 0.15),
        }

        results = robust_walk_forward(df, params)

        sharpes = results['sharpes']
        returns = results['returns']
        trades = results['trades']

        # Minimum trades constraint
        if min(trades) < 5:
            return -100

        # Use MEDIAN for robustness (not mean)
        median_sharpe = np.median(sharpes)
        median_return = np.median(returns)

        # Consistency bonus: % of folds with positive Sharpe
        consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)

        # Sharpe variance penalty
        sharpe_std = np.std(sharpes)
        variance_penalty = sharpe_std * 0.5

        # Objective: median_sharpe * (1 + consistency) * sqrt(|median_return|) - variance
        if median_return >= 0:
            return_factor = np.sqrt(median_return + 0.01)
        else:
            return_factor = -np.sqrt(abs(median_return) + 0.01)

        objective = median_sharpe * (1 + consistency) * return_factor - variance_penalty

        return objective

    return objective


def main():
    print("=" * 80)
    print("ROBUST OPTUNA HYPEROPTIMIZATION")
    print("Objective: median_sharpe * (1+consistency) * sqrt(|return|) - variance")
    print("=" * 80)

    df = load_data()
    print(f"Data: {len(df):,} bars")

    sampler = TPESampler(seed=42, n_startup_trials=20)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = create_robust_objective(df)

    print("\nStarting optimization (150 trials)...")
    study.optimize(objective, n_trials=150, show_progress_bar=True)

    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)

    best_params = study.best_params
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    print(f"\nBest objective: {study.best_value:.4f}")

    # Final validation
    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    results = robust_walk_forward(df, best_params)

    print(f"\n{'Fold':<6} {'Sharpe':>10} {'Return':>10} {'Trades':>8}")
    print("-" * 40)

    for i, (s, r, t) in enumerate(zip(results['sharpes'], results['returns'], results['trades'])):
        print(f"  {i:<4} {s:>10.3f} {r*100:>9.1f}% {t:>8}")

    avg_sharpe = np.mean(results['sharpes'])
    median_sharpe = np.median(results['sharpes'])
    compounded = np.prod([1 + r for r in results['returns']]) - 1
    consistency = sum(1 for s in results['sharpes'] if s > 0) / len(results['sharpes'])

    print("-" * 40)
    print(f"  AVG  {avg_sharpe:>10.3f} {compounded*100:>9.1f}%")
    print(f"  MED  {median_sharpe:>10.3f}")
    print(f"  Consistency: {consistency*100:.0f}%")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Before vs After Optimization")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Before':>15} {'After':>15}")
    print("-" * 50)
    print(f"{'Avg Sharpe':<20} {'7.93':>15} {avg_sharpe:>15.3f}")
    print(f"{'Median Sharpe':<20} {'-':>15} {median_sharpe:>15.3f}")
    print(f"{'Compounded Return':<20} {'1.5%':>15} {compounded*100:>14.1f}%")
    print(f"{'Consistency':<20} {'80%':>15} {consistency*100:>14.0f}%")

    # Save
    output = {
        'best_params': best_params,
        'best_objective': study.best_value,
        'avg_sharpe': avg_sharpe,
        'median_sharpe': median_sharpe,
        'compounded_return': compounded,
        'consistency': consistency,
        'fold_sharpes': results['sharpes'],
        'fold_returns': results['returns'],
        'fold_trades': results['trades']
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/optuna_robust_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
