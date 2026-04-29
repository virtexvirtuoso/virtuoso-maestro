#!/usr/bin/env python3
"""
BTC Walk-Forward Optuna Optimization

Optimizes top BTC strategies using walk-forward analysis with Optuna.
Based on screening results: Momentum, ADXSmas, EMA_Cross, VWAP

Usage:
    python scripts/btc_optuna_walkforward.py --timeframe 1d --trials 50 --splits 5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import argparse
import json
import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from engine_v2.strategy_adapter import SignalOutput, VectorBTStrategy
from engine_v2.walk_forward_optuna import WalkForwardConfig, WalkForwardOptuna
from engine_v2.vectorbt_engine import BacktestConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'


# =============================================================================
# STRATEGY ADAPTERS FOR TOP BTC PERFORMERS
# =============================================================================

class MomentumTrendFilterStrategy(VectorBTStrategy):
    """
    Momentum + Trend Filter Strategy
    Top BTC performer: Sharpe 0.95, Return +1170%
    """

    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'momentum_period': 14,
            'momentum_threshold': 0.02,
            'trend_period': 50,
        }

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'momentum_period': ('int', 5, 30),
            'momentum_threshold': ('float', 0.005, 0.05),
            'trend_period': ('int', 20, 100),
        }

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)

        # Momentum
        momentum = close.pct_change(params['momentum_period'])

        # Trend filter
        sma_trend = close.rolling(params['trend_period']).mean()
        uptrend = close > sma_trend
        downtrend = close < sma_trend

        # Signals: momentum + trend confirmation
        entries = (momentum > params['momentum_threshold']) & uptrend
        exits = (momentum < 0) | ~uptrend

        short_entries = (momentum < -params['momentum_threshold']) & downtrend
        short_exits = (momentum > 0) | ~downtrend

        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'momentum': momentum, 'sma_trend': sma_trend}
        )


class ADXSmasStrategy(VectorBTStrategy):
    """
    ADX + SMAs Strategy
    Top BTC performer: Sharpe 0.88, Return +1098% (4h)
    """

    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'adx_threshold': 25,
            'fast_sma': 10,
            'slow_sma': 30,
        }

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'adx_period': ('int', 7, 28),
            'adx_threshold': ('int', 15, 40),
            'fast_sma': ('int', 5, 20),
            'slow_sma': ('int', 20, 60),
        }

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)

        # Calculate ADX
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params['adx_period']).mean()

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(params['adx_period']).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(params['adx_period']).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(params['adx_period']).mean()

        # SMAs
        fast_sma = close.rolling(params['fast_sma']).mean()
        slow_sma = close.rolling(params['slow_sma']).mean()

        # Strong trend + crossover
        strong_trend = adx > params['adx_threshold']

        entries = strong_trend & (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        exits = (fast_sma < slow_sma) | (adx < params['adx_threshold'] * 0.7)

        short_entries = strong_trend & (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        short_exits = (fast_sma > slow_sma) | (adx < params['adx_threshold'] * 0.7)

        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'adx': adx, 'fast_sma': fast_sma, 'slow_sma': slow_sma}
        )


class SimpleEMACrossStrategy(VectorBTStrategy):
    """
    Simple EMA Cross Strategy
    BTC performer: Sharpe 0.65, Return +868%
    """

    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'fast_period': 12,
            'slow_period': 26,
        }

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'fast_period': ('int', 5, 30),
            'slow_period': ('int', 15, 60),
        }

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)

        ema_fast = close.ewm(span=params['fast_period'], adjust=False).mean()
        ema_slow = close.ewm(span=params['slow_period'], adjust=False).mean()

        fast_above = ema_fast > ema_slow

        entries = fast_above & ~fast_above.shift(1).fillna(False)
        exits = ~fast_above & fast_above.shift(1).fillna(True)

        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'ema_fast': ema_fast, 'ema_slow': ema_slow}
        )


class VWAPStrategy(VectorBTStrategy):
    """
    VWAP Strategy
    BTC performer: Sharpe 0.83, Return +371%
    """

    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'vwap_period': 20,
            'band_mult': 2.0,
        }

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'vwap_period': ('int', 10, 50),
            'band_mult': ('float', 1.0, 3.0),
        }

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        volume = self._get_volume(data)

        # VWAP
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).rolling(params['vwap_period']).sum()
        cumulative_vol = volume.rolling(params['vwap_period']).sum()
        vwap = cumulative_tp_vol / cumulative_vol

        # Bands
        vwap_std = typical_price.rolling(params['vwap_period']).std()
        upper_band = vwap + params['band_mult'] * vwap_std
        lower_band = vwap - params['band_mult'] * vwap_std

        # Signals
        entries = (close > vwap) & (close.shift(1) <= vwap.shift(1))
        exits = (close < vwap) | (close > upper_band)

        short_entries = (close < vwap) & (close.shift(1) >= vwap.shift(1))
        short_exits = (close > vwap) | (close < lower_band)

        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'vwap': vwap, 'upper_band': upper_band, 'lower_band': lower_band}
        )


class IchimokuStrategy(VectorBTStrategy):
    """
    Ichimoku Cloud Strategy
    Best overall in Freqtrade validation: +137% return
    """

    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_period': 52,
        }

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'tenkan_period': ('int', 5, 15),
            'kijun_period': ('int', 15, 40),
            'senkou_period': ('int', 30, 80),
        }

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)

        # Tenkan-sen
        tenkan_high = high.rolling(params['tenkan_period']).max()
        tenkan_low = low.rolling(params['tenkan_period']).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen
        kijun_high = high.rolling(params['kijun_period']).max()
        kijun_low = low.rolling(params['kijun_period']).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A and B
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(params['kijun_period'])
        senkou_b_high = high.rolling(params['senkou_period']).max()
        senkou_b_low = low.rolling(params['senkou_period']).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(params['kijun_period'])

        # Cloud
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        # Signals
        entries = (close > cloud_top) & (close.shift(1) <= cloud_top.shift(1))
        exits = close < cloud_top

        short_entries = (close < cloud_bottom) & (close.shift(1) >= cloud_bottom.shift(1))
        short_exits = close > cloud_bottom

        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'cloud_top': cloud_top, 'cloud_bottom': cloud_bottom}
        )


# =============================================================================
# MAIN OPTIMIZATION RUNNER
# =============================================================================

STRATEGIES = {
    'MomentumTrendFilter': MomentumTrendFilterStrategy(),
    'ADXSmas': ADXSmasStrategy(),
    'EMA_Cross': SimpleEMACrossStrategy(),
    'VWAP': VWAPStrategy(),
    'Ichimoku': IchimokuStrategy(),
}


def load_btc_data(timeframe: str) -> pd.DataFrame:
    """Load BTC data for specified timeframe."""
    filepath = os.path.join(DATA_DIR, f'binance_btc_usdt_{timeframe}.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BTC data not found: {filepath}")

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def run_walkforward_optimization(
    strategy_name: str,
    strategy: VectorBTStrategy,
    data: pd.DataFrame,
    n_splits: int = 5,
    n_trials: int = 50,
) -> dict:
    """Run walk-forward optimization for a strategy."""

    logger.info(f"Starting walk-forward optimization for {strategy_name}")
    logger.info(f"  Data: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
    logger.info(f"  Splits: {n_splits}, Trials/split: {n_trials}")

    config = WalkForwardConfig(
        num_splits=n_splits,
        train_splits=2,
        test_splits=1,
        n_trials=n_trials,
        optimization_metric='sharpe_ratio',
        use_vwr_ranking=True,
        pruning_enabled=True,
        n_startup_trials=10,
        use_dashboard_storage=True,
    )

    backtest_config = BacktestConfig(
        cash=100000.0,
        commission=0.001,
        slippage=0.0005,
    )

    engine = WalkForwardOptuna(
        data=data,
        strategy=strategy,
        config=config,
        backtest_config=backtest_config,
        logger=logger,
    )

    result = engine.run()

    # Compile results
    fold_summaries = []
    for i, (fold_result, params) in enumerate(zip(result.fold_results, result.optimal_params_per_fold)):
        fold_summaries.append({
            'fold': i,
            'sharpe': fold_result.sharpe_ratio,
            'return': fold_result.total_return,
            'max_dd': fold_result.max_drawdown,
            'trades': fold_result.num_trades,
            'win_rate': fold_result.win_rate,
            'params': params,
        })

    return {
        'strategy': strategy_name,
        'aggregate': result.aggregate_metrics,
        'folds': fold_summaries,
        'optimal_params': result.optimal_params_per_fold,
        'processing_time': result.total_processing_time,
    }


def main():
    parser = argparse.ArgumentParser(description='BTC Walk-Forward Optuna Optimization')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (1d, 4h, 1h)')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials per split')
    parser.add_argument('--splits', type=int, default=5, help='Walk-forward splits')
    parser.add_argument('--strategies', nargs='+', default=None, help='Strategies to test (default: all)')
    args = parser.parse_args()

    print("=" * 80)
    print("BTC WALK-FORWARD OPTUNA OPTIMIZATION")
    print("=" * 80)
    print(f"Timeframe: {args.timeframe}")
    print(f"Splits: {args.splits}, Trials/split: {args.trials}")

    # Load data
    data = load_btc_data(args.timeframe)
    print(f"Data: {len(data):,} bars ({data.index[0]} to {data.index[-1]})")

    # Select strategies
    if args.strategies:
        strategies_to_test = {k: v for k, v in STRATEGIES.items() if k in args.strategies}
    else:
        strategies_to_test = STRATEGIES

    print(f"Strategies: {list(strategies_to_test.keys())}")
    print("-" * 80)

    # Run optimization
    all_results = []

    for name, strategy in strategies_to_test.items():
        try:
            result = run_walkforward_optimization(
                strategy_name=name,
                strategy=strategy,
                data=data,
                n_splits=args.splits,
                n_trials=args.trials,
            )
            all_results.append(result)

            print(f"\n{name} Results:")
            print(f"  Avg Sharpe: {result['aggregate'].get('avg_sharpe', 0):.3f}")
            print(f"  Total Return: {result['aggregate'].get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {result['aggregate'].get('max_drawdown', 0):.2%}")
            print(f"  Total Trades: {result['aggregate'].get('total_trades', 0)}")
            print(f"  Processing Time: {result['processing_time']:.1f}s")

        except Exception as e:
            logger.error(f"Failed to optimize {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Strategy': r['strategy'],
            'Avg Sharpe': r['aggregate'].get('avg_sharpe', 0),
            'Total Return': r['aggregate'].get('total_return', 0),
            'Max DD': r['aggregate'].get('max_drawdown', 0),
            'Trades': r['aggregate'].get('total_trades', 0),
            'Time (s)': r['processing_time'],
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Avg Sharpe', ascending=False)
    print(summary_df.to_string(index=False))

    # Save results
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    results_path = f'{out_dir}/btc_walkforward_{args.timeframe}_{ts}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved: {results_path}")

    # Save summary
    summary_path = f'{out_dir}/btc_walkforward_summary_{args.timeframe}_{ts}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Print optimal parameters for best strategy
    if all_results:
        best = max(all_results, key=lambda x: x['aggregate'].get('avg_sharpe', 0))
        print(f"\n{'=' * 80}")
        print(f"BEST STRATEGY: {best['strategy']}")
        print(f"{'=' * 80}")
        print(f"Avg Sharpe: {best['aggregate'].get('avg_sharpe', 0):.3f}")
        print(f"\nOptimal Parameters per Fold:")
        for i, params in enumerate(best['optimal_params']):
            print(f"  Fold {i}: {params}")


if __name__ == '__main__':
    main()
