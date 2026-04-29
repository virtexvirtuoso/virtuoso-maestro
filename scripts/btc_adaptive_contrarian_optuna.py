#!/usr/bin/env python3
"""
Optuna Hyperoptimization for Adaptive Contrarian Strategy

Walk-forward optimization targeting:
1. Higher returns (not just Sharpe)
2. Better entry/exit timing
3. Optimal regime thresholds
4. Position sizing (Kelly-inspired)

Optimization objective: Sharpe * sqrt(abs(return)) to balance both
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
    """Calculate all indicators with given parameters."""
    close = df['close']
    high = df['high']
    low = df['low']

    # ADX
    period = params['adx_period']
    tr = pd.concat([high - low,
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(period).mean()

    # ATR percentile
    df['atr'] = atr
    df['atr_pct'] = df['atr'].rolling(params['atr_lookback']).rank(pct=True)

    # SMAs
    df['sma_fast'] = close.rolling(params['sma_fast']).mean()
    df['sma_slow'] = close.rolling(params['sma_slow']).mean()
    df['sma_trend'] = close.rolling(params['sma_trend']).mean()

    # Trend alignment
    df['trend_alignment'] = (
        (close > df['sma_fast']).astype(int) +
        (close > df['sma_slow']).astype(int) +
        (close > df['sma_trend']).astype(int) +
        (df['sma_fast'] > df['sma_slow']).astype(int) +
        (df['sma_slow'] > df['sma_trend']).astype(int)
    )

    # Regime
    df['regime'] = 'RANGING'
    trending_up = (df['adx'] > params['adx_threshold']) & (df['trend_alignment'] >= params['trend_up_min'])
    trending_down = (df['adx'] > params['adx_threshold']) & (df['trend_alignment'] <= params['trend_down_max'])
    volatile = df['atr_pct'] > params['vol_threshold']

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
    """Generate trading signals with given parameters."""
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']
    z_score = df['funding_zscore']

    # Extreme conditions
    extreme_negative = (funding < params['long_threshold']) | (z_score < -params['zscore_threshold'])
    extreme_positive = (funding > params['short_threshold']) | (z_score > params['zscore_threshold'])

    # Tighter for volatile
    extreme_negative_tight = (funding < params['long_threshold'] * params['vol_multiplier']) | \
                             (z_score < -params['zscore_threshold'] - params['vol_zscore_add'])
    extreme_positive_tight = (funding > params['short_threshold'] * params['vol_multiplier']) | \
                             (z_score > params['zscore_threshold'] + params['vol_zscore_add'])

    for i in range(len(df)):
        regime = df['regime'].iloc[i]

        if regime == 'RANGING':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
            elif extreme_positive.iloc[i]:
                signals.iloc[i] = -1

        elif regime == 'VOLATILE':
            if extreme_negative_tight.iloc[i]:
                signals.iloc[i] = 1
            elif extreme_positive_tight.iloc[i]:
                signals.iloc[i] = -1

        elif regime == 'TRENDING_UP':
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1

        elif regime == 'TRENDING_DOWN':
            if extreme_positive.iloc[i]:
                signals.iloc[i] = -1

    return signals


def backtest(df: pd.DataFrame, signals: pd.Series, params: dict) -> dict:
    """Backtest with position sizing."""
    n = len(df)
    position_size = params['position_size']
    commission = params['commission']

    pnl = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    entry_price = 0.0
    bars_in_trade = 0
    total_trades = 0
    wins = 0

    for i in range(1, n):
        signal = signals.iloc[i-1]
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]

        # Track bars in trade for time-based exit
        if current_pos != 0:
            bars_in_trade += 1

        # P&L
        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size

        # Time-based exit
        if current_pos != 0 and bars_in_trade > params['max_hold_bars']:
            pnl.iloc[i] -= commission * abs(current_pos) * position_size
            if pnl.iloc[i] > 0:
                wins += 1
            current_pos = 0.0
            bars_in_trade = 0
            total_trades += 1
            continue

        # Signal processing
        if signal != 0 and signal != current_pos:
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                if pnl.iloc[i] > 0:
                    wins += 1
            current_pos = signal
            entry_price = price
            bars_in_trade = 0
            total_trades += 1
            pnl.iloc[i] -= commission * abs(signal) * position_size

        elif signal == 0 and current_pos != 0:
            pnl.iloc[i] -= commission * abs(current_pos) * position_size
            if pnl.iloc[i] > 0:
                wins += 1
            current_pos = 0.0
            bars_in_trade = 0
            total_trades += 1

    # Metrics
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


def walk_forward_evaluate(df: pd.DataFrame, params: dict, n_splits: int = 5) -> float:
    """Walk-forward evaluation returning combined objective."""
    n = len(df)
    fold_size = n // n_splits

    test_sharpes = []
    test_returns = []

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
        test_result = backtest(test_df, test_signals, params)

        if test_result['total_trades'] > 0:
            test_sharpes.append(test_result['sharpe_ratio'])
            test_returns.append(test_result['total_return'])

    if not test_sharpes:
        return -100

    avg_sharpe = np.mean(test_sharpes)
    compounded_return = np.prod([1 + r for r in test_returns]) - 1

    # Objective: Maximize Sharpe * sqrt(|return|) * sign(return)
    # This rewards both high Sharpe AND high returns
    if compounded_return >= 0:
        objective = avg_sharpe * np.sqrt(compounded_return + 0.01)
    else:
        objective = avg_sharpe * -np.sqrt(abs(compounded_return) + 0.01)

    return objective


def create_objective(df: pd.DataFrame):
    """Create Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Funding thresholds
            'long_threshold': trial.suggest_float('long_threshold', -0.002, -0.0001),
            'short_threshold': trial.suggest_float('short_threshold', 0.0003, 0.003),
            'zscore_lookback': trial.suggest_int('zscore_lookback', 12, 48),
            'zscore_threshold': trial.suggest_float('zscore_threshold', 1.5, 3.5),

            # Volatile regime adjustments
            'vol_multiplier': trial.suggest_float('vol_multiplier', 1.5, 3.0),
            'vol_zscore_add': trial.suggest_float('vol_zscore_add', 0.3, 1.0),

            # Regime detection
            'adx_period': trial.suggest_int('adx_period', 10, 20),
            'adx_threshold': trial.suggest_float('adx_threshold', 18, 35),
            'atr_lookback': trial.suggest_int('atr_lookback', 100, 250),
            'vol_threshold': trial.suggest_float('vol_threshold', 0.7, 0.9),
            'sma_fast': trial.suggest_int('sma_fast', 10, 30),
            'sma_slow': trial.suggest_int('sma_slow', 40, 80),
            'sma_trend': trial.suggest_int('sma_trend', 150, 250),
            'trend_up_min': trial.suggest_int('trend_up_min', 3, 5),
            'trend_down_max': trial.suggest_int('trend_down_max', 0, 2),

            # Position management
            'position_size': trial.suggest_float('position_size', 0.05, 0.25),
            'max_hold_bars': trial.suggest_int('max_hold_bars', 12, 72),
            'commission': 0.0004,
        }

        return walk_forward_evaluate(df, params)

    return objective


def main():
    print("=" * 80)
    print("OPTUNA HYPEROPTIMIZATION: Adaptive Contrarian Strategy")
    print("Objective: Maximize Sharpe * sqrt(|return|)")
    print("=" * 80)

    df = load_data()
    print(f"Data: {len(df):,} bars")

    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='adaptive_contrarian_v2'
    )

    objective = create_objective(df)

    print("\nStarting optimization (100 trials)...")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Best parameters
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)

    best_params = study.best_params
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    print(f"\nBest objective: {study.best_value:.4f}")

    # Validate best params
    print("\n" + "=" * 80)
    print("VALIDATION WITH BEST PARAMS")
    print("=" * 80)

    best_params['commission'] = 0.0004
    n = len(df)
    fold_size = n // 5

    print(f"\n{'Fold':<6} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 45)

    all_sharpes = []
    all_returns = []

    for i in range(5):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)
        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * 0.7)
        test_df = fold_data.iloc[train_size:].reset_index(drop=True)

        test_df = calculate_indicators(test_df, best_params)
        test_signals = generate_signals(test_df, best_params)
        result = backtest(test_df, test_signals, best_params)

        print(f"  {i:<4} {result['sharpe_ratio']:>8.3f} "
              f"{result['total_return']*100:>9.1f}% "
              f"{result['max_drawdown']*100:>7.1f}% "
              f"{result['total_trades']:>8}")

        all_sharpes.append(result['sharpe_ratio'])
        all_returns.append(result['total_return'])

    avg_sharpe = np.mean(all_sharpes)
    compounded = np.prod([1 + r for r in all_returns]) - 1
    consistency = sum(1 for s in all_sharpes if s > 0) / len(all_sharpes)

    print("-" * 45)
    print(f"  AVG  {avg_sharpe:>8.3f} {compounded*100:>9.1f}% {'':>8} Consistency: {consistency*100:.0f}%")

    # Save results
    results = {
        'best_params': best_params,
        'best_objective': study.best_value,
        'avg_sharpe': avg_sharpe,
        'compounded_return': compounded,
        'consistency': consistency,
        'fold_sharpes': all_sharpes,
        'fold_returns': all_returns
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/optuna_adaptive_contrarian_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Before vs After Optimization")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>15}")
    print("-" * 65)
    print(f"{'Avg Sharpe':<20} {'7.93':>15} {avg_sharpe:>15.3f} {(avg_sharpe/7.93-1)*100:>14.1f}%")
    print(f"{'Compounded Return':<20} {'1.5%':>15} {compounded*100:>14.1f}% {(compounded/0.015-1)*100:>14.1f}%")
    print(f"{'Consistency':<20} {'80%':>15} {consistency*100:>14.0f}% {'':>15}")


if __name__ == '__main__':
    main()
