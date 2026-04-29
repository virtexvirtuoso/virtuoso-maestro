#!/usr/bin/env python3
"""
Multi-Timeframe Derivatives Signal Validation
Validates 18 derivatives signals across multiple timeframes with proper walk-forward testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple, Optional
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
DERIVATIVES_PATH = Path.home() / "Desktop/maestro/data/derivatives_5m/compiled"
SPOT_PATH = Path.home() / "Desktop/maestro/data/spot"
RESULTS_PATH = Path.home() / "Desktop/maestro/backend/research/results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

ASSETS = [
    'AAVEUSDT', 'ADAUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BNBUSDT', 
    'BTCUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT', 'ETHUSDT',
    'FETUSDT', 'FTMUSDT', 'INJUSDT', 'LINKUSDT', 'MATICUSDT', 'OPUSDT',
    'RENDERUSDT', 'SEIUSDT', 'SOLUSDT', 'SUIUSDT', 'TAOUSDT', 'TIAUSDT',
    'UNIUSDT', 'XRPUSDT'
]

# Timeframe configurations: (derivatives_tf, spot_tf, holding_periods, lookback)
# Ordered smallest first to reduce OOM risk
from collections import OrderedDict
TIMEFRAME_CONFIGS = OrderedDict([
    ('1h_1h', ('1h', '1h', ['4h', '12h', '1d'], 50)),
    ('1h_4h', ('1h', '4h', ['4h', '12h', '1d'], 50)),
    ('15m', ('15m', '15m', ['1h', '4h', '12h'], 20)),
    ('5m', ('5m', '5m', ['15m', '1h', '4h'], 20)),
])

# Convert holding periods to minutes
HOLDING_MINUTES = {
    '15m': 15, '1h': 60, '4h': 240, '12h': 720, '1d': 1440
}

TRANSACTION_COST_BPS = 7  # 7 bps realistic (limit open + market close)
MIN_TRADES = 30
ALPHA = 0.05


class SignalGenerator:
    """Generate trading signals from derivatives data."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std
    
    def _pct_change(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate percentage change."""
        return series.pct_change(periods)
    
    # OI-based signals (6 signals)
    
    def oi_momentum(self, df: pd.DataFrame) -> pd.Series:
        """OI increasing = long."""
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        return (oi_change > 0).astype(int)
    
    def oi_divergence_bullish(self, df: pd.DataFrame) -> pd.Series:
        """Price down but OI up = long."""
        price_change = self._pct_change(df['close'], self.lookback // 4)
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        return ((price_change < 0) & (oi_change > 0)).astype(int)
    
    def oi_divergence_bearish(self, df: pd.DataFrame) -> pd.Series:
        """Price up but OI down = flat."""
        price_change = self._pct_change(df['close'], self.lookback // 4)
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        return ((price_change > 0) & (oi_change < 0)).astype(int) * 0  # Always 0 (flat)
    
    def oi_breakout(self, df: pd.DataFrame) -> pd.Series:
        """OI z-score > 2 = long."""
        oi_z = self._zscore(df['oi_usd'], self.lookback)
        return (oi_z > 2).astype(int)
    
    def oi_flush(self, df: pd.DataFrame) -> pd.Series:
        """OI drops sharply (z-score < -2) = long."""
        oi_z = self._zscore(df['oi_usd'], self.lookback)
        return (oi_z < -2).astype(int)
    
    def oi_rate_of_change(self, df: pd.DataFrame) -> pd.Series:
        """OI ROC percentile > 80 = long. Vectorized version."""
        oi_roc = self._pct_change(df['oi_usd'], self.lookback // 4)
        window = self.lookback * 2
        oi_rank = oi_roc.rolling(window).rank(pct=True)
        return (oi_rank > 0.80).astype(int)
    
    # LSR-based signals (4 signals)
    
    def lsr_contrarian(self, df: pd.DataFrame) -> pd.Series:
        """LSR > 1.5 = flat; LSR < 0.7 = long."""
        lsr = df['global_accounts_lsr'].fillna(1.0)
        signal = pd.Series(0, index=df.index)
        signal[lsr < 0.7] = 1
        signal[lsr > 1.5] = 0
        return signal
    
    def lsr_extreme(self, df: pd.DataFrame) -> pd.Series:
        """LSR z-score > 2 = flat; z-score < -2 = long."""
        lsr = df['global_accounts_lsr'].fillna(1.0)
        lsr_z = self._zscore(lsr, self.lookback)
        signal = pd.Series(0, index=df.index)
        signal[lsr_z < -2] = 1
        signal[lsr_z > 2] = 0
        return signal
    
    def top_trader_follow(self, df: pd.DataFrame) -> pd.Series:
        """Top trader position LSR > 1.2 = long."""
        tt_lsr = df['top_trader_positions_lsr'].fillna(1.0)
        return (tt_lsr > 1.2).astype(int)
    
    def lsr_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Top trader LSR rising while global LSR falling = long."""
        tt_lsr_change = self._pct_change(df['top_trader_positions_lsr'].fillna(1.0), self.lookback // 4)
        global_lsr_change = self._pct_change(df['global_accounts_lsr'].fillna(1.0), self.lookback // 4)
        return ((tt_lsr_change > 0) & (global_lsr_change < 0)).astype(int)
    
    # Taker flow signals (4 signals)
    
    def taker_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Taker B/S ratio > 1.2 = long."""
        ratio = df['taker_buy_sell_ratio'].fillna(1.0)
        return (ratio > 1.2).astype(int)
    
    def taker_contrarian(self, df: pd.DataFrame) -> pd.Series:
        """Taker B/S ratio > 2.0 = flat; < 0.5 = long."""
        ratio = df['taker_buy_sell_ratio'].fillna(1.0)
        signal = pd.Series(0, index=df.index)
        signal[ratio < 0.5] = 1
        signal[ratio > 2.0] = 0
        return signal
    
    def taker_regime(self, df: pd.DataFrame) -> pd.Series:
        """Rolling mean taker ratio above/below threshold = trend signal."""
        ratio = df['taker_buy_sell_ratio'].fillna(1.0)
        ratio_ma = ratio.rolling(self.lookback).mean()
        return (ratio_ma > 1.0).astype(int)
    
    def taker_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Price falling but taker ratio rising = long."""
        price_change = self._pct_change(df['close'], self.lookback // 4)
        ratio_change = self._pct_change(df['taker_buy_sell_ratio'].fillna(1.0), self.lookback // 4)
        return ((price_change < 0) & (ratio_change > 0)).astype(int)
    
    # Combined signals (4 signals)
    
    def oi_lsr_combo(self, df: pd.DataFrame) -> pd.Series:
        """OI rising + LSR falling = long."""
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        lsr_change = self._pct_change(df['global_accounts_lsr'].fillna(1.0), self.lookback // 4)
        return ((oi_change > 0) & (lsr_change < 0)).astype(int)
    
    def oi_taker_combo(self, df: pd.DataFrame) -> pd.Series:
        """OI rising + taker ratio > 1 = long."""
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        ratio = df['taker_buy_sell_ratio'].fillna(1.0)
        return ((oi_change > 0) & (ratio > 1.0)).astype(int)
    
    def triple_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """OI rising + favorable LSR + taker > 1 = long."""
        oi_change = self._pct_change(df['oi_usd'], self.lookback // 4)
        lsr = df['global_accounts_lsr'].fillna(1.0)
        ratio = df['taker_buy_sell_ratio'].fillna(1.0)
        return ((oi_change > 0) & (lsr < 1.3) & (ratio > 1.0)).astype(int)
    
    def crowding_filter(self, df: pd.DataFrame) -> pd.Series:
        """Only long when LSR < 1.3 AND OI not extreme."""
        lsr = df['global_accounts_lsr'].fillna(1.0)
        oi_z = self._zscore(df['oi_usd'], self.lookback)
        return ((lsr < 1.3) & (oi_z.abs() < 2)).astype(int)
    
    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate all signals."""
        return {
            # OI-based
            'oi_momentum': self.oi_momentum(df),
            'oi_divergence_bullish': self.oi_divergence_bullish(df),
            'oi_divergence_bearish': self.oi_divergence_bearish(df),
            'oi_breakout': self.oi_breakout(df),
            'oi_flush': self.oi_flush(df),
            'oi_rate_of_change': self.oi_rate_of_change(df),
            # LSR-based
            'lsr_contrarian': self.lsr_contrarian(df),
            'lsr_extreme': self.lsr_extreme(df),
            'top_trader_follow': self.top_trader_follow(df),
            'lsr_divergence': self.lsr_divergence(df),
            # Taker flow
            'taker_momentum': self.taker_momentum(df),
            'taker_contrarian': self.taker_contrarian(df),
            'taker_regime': self.taker_regime(df),
            'taker_divergence': self.taker_divergence(df),
            # Combined
            'oi_lsr_combo': self.oi_lsr_combo(df),
            'oi_taker_combo': self.oi_taker_combo(df),
            'triple_confirmation': self.triple_confirmation(df),
            'crowding_filter': self.crowding_filter(df),
        }


class DataLoader:
    """Load and merge derivatives and spot data."""
    
    @staticmethod
    def symbol_to_ticker(symbol: str) -> str:
        """Convert BTCUSDT to BTC."""
        return symbol.replace('USDT', '')
    
    def load_derivatives(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load derivatives data."""
        file_path = DERIVATIVES_PATH / f"{symbol}_{timeframe}.csv"
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def load_spot(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load spot price data."""
        file_path = SPOT_PATH / timeframe / f"{ticker}_spot_{timeframe}.csv"
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        # Handle different column naming conventions
        col_map = {}
        for col in df.columns:
            col_map[col.lower()] = col
        ts_col = col_map.get('timestamp', col_map.get('date', df.columns[0]))
        close_col = col_map.get('close', df.columns[4])
        df = df.rename(columns={ts_col: 'timestamp', close_col: 'close'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df[['timestamp', 'close']]
    
    def merge_data(self, symbol: str, deriv_tf: str, spot_tf: str) -> Optional[pd.DataFrame]:
        """Merge derivatives and spot data."""
        # Load derivatives
        deriv_df = self.load_derivatives(symbol, deriv_tf)
        if deriv_df is None:
            return None
        
        # Load spot
        ticker = self.symbol_to_ticker(symbol)
        spot_df = self.load_spot(ticker, spot_tf)
        if spot_df is None:
            return None
        
        # Merge with forward-fill for spot prices
        merged = pd.merge_asof(
            deriv_df.sort_values('timestamp'),
            spot_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        return merged


class SignalValidator:
    """Validate signals with walk-forward testing."""
    
    def __init__(self, n_folds: int = 14, train_folds: int = 7):
        self.n_folds = n_folds
        self.train_folds = train_folds
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train (folds 1-7) and test (folds 8-14)."""
        fold_size = len(df) // self.n_folds
        train_end = fold_size * self.train_folds
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:].copy()
        
        return train_df, test_df
    
    def compute_non_overlapping_returns(
        self, 
        df: pd.DataFrame, 
        signals: pd.Series, 
        holding_minutes: int
    ) -> Tuple[np.ndarray, int]:
        """
        Compute non-overlapping forward returns.
        Only enter new position after previous holding period expires.
        """
        returns = []
        i = 0
        
        while i < len(df) - 1:
            if signals.iloc[i] == 1:
                # Find exit timestamp
                entry_time = df['timestamp'].iloc[i]
                exit_time = entry_time + pd.Timedelta(minutes=holding_minutes)
                
                # Find closest exit index
                exit_idx = df[df['timestamp'] >= exit_time].index
                if len(exit_idx) == 0:
                    break
                exit_idx = exit_idx[0]
                
                if exit_idx >= len(df):
                    break
                
                # Compute return
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[exit_idx]
                ret = (exit_price - entry_price) / entry_price
                
                # Subtract transaction costs
                ret -= TRANSACTION_COST_BPS / 10000
                
                returns.append(ret)
                
                # Jump to after exit
                i = exit_idx
            else:
                i += 1
        
        return np.array(returns), len(returns)
    
    def compute_statistics(self, returns: np.ndarray, trades_per_year: float) -> Dict:
        """Compute full performance statistics suite."""
        if len(returns) == 0:
            return {
                'n_trades': 0, 'mean_return': np.nan, 'sharpe': np.nan,
                'sortino': np.nan, 'calmar': np.nan, 'max_drawdown': np.nan,
                'profit_factor': np.nan, 'win_rate': np.nan, 'payoff_ratio': np.nan,
                'expectancy': np.nan, 'tail_ratio': np.nan, 'skewness': np.nan,
                'kurtosis': np.nan, 't_stat': np.nan, 'p_value': np.nan,
            }
        
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from metrics import compute_metrics
        return compute_metrics(returns, trades_per_year=trades_per_year)
    
    def estimate_trades_per_year(self, df: pd.DataFrame, n_trades: int) -> float:
        """Estimate annualization factor based on actual trade frequency."""
        if n_trades == 0 or len(df) < 2:
            return 1.0
        
        # Calculate time span in years
        time_span = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
        
        if time_span > 0:
            return n_trades / time_span
        return 1.0
    
    def validate_signal(
        self, 
        df: pd.DataFrame, 
        signal_name: str, 
        signals: pd.Series, 
        holding_period: str
    ) -> Dict:
        """Validate a single signal on out-of-sample data."""
        # Split data
        train_df, test_df = self.split_data(df)
        
        # Get OOS signals
        test_signals = signals.loc[test_df.index]
        
        # Compute non-overlapping returns
        holding_minutes = HOLDING_MINUTES[holding_period]
        returns, n_trades = self.compute_non_overlapping_returns(
            test_df.reset_index(drop=True), 
            test_signals.reset_index(drop=True), 
            holding_minutes
        )
        
        # Estimate trades per year
        trades_per_year = self.estimate_trades_per_year(test_df, n_trades)
        
        # Compute statistics
        stats_dict = self.compute_statistics(returns, trades_per_year)
        
        return stats_dict


def main():
    """Main validation loop."""
    print("=" * 80)
    print("DERIVATIVES SIGNAL VALIDATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAssets: {len(ASSETS)}")
    print(f"Signals: 18")
    print(f"Timeframes: {len(TIMEFRAME_CONFIGS)}")
    print(f"Transaction costs: {TRANSACTION_COST_BPS} bps per round trip")
    print(f"Minimum trades: {MIN_TRADES}")
    print(f"Alpha: {ALPHA}")
    print()
    
    # Calculate total tests for Bonferroni correction
    total_tests = 0
    for tf_name, (deriv_tf, spot_tf, holding_periods, _) in TIMEFRAME_CONFIGS.items():
        total_tests += 18 * len(holding_periods) * len(ASSETS)
    
    print(f"Total tests: {total_tests:,}")
    print(f"Bonferroni-corrected alpha: {ALPHA / total_tests:.2e}")
    print("=" * 80)
    print()
    
    loader = DataLoader()
    validator = SignalValidator()
    results = []
    
    total_iterations = sum(
        len(holding_periods) * len(ASSETS)
        for _, (_, _, holding_periods, _) in TIMEFRAME_CONFIGS.items()
    ) * 18
    
    current_iteration = 0
    
    # Loop over timeframes
    for tf_name, (deriv_tf, spot_tf, holding_periods, lookback) in TIMEFRAME_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"TIMEFRAME: {tf_name} (deriv={deriv_tf}, spot={spot_tf}, lookback={lookback})")
        print(f"{'=' * 80}")
        
        # Loop over assets
        for asset in ASSETS:
            ticker = DataLoader.symbol_to_ticker(asset)
            
            # Load and merge data
            df = loader.merge_data(asset, deriv_tf, spot_tf)
            if df is None or len(df) < 100:
                print(f"⚠️  Skipping {asset} (insufficient data)")
                current_iteration += 18 * len(holding_periods)
                continue
            
            # Generate signals
            signal_gen = SignalGenerator(lookback=lookback)
            signals_dict = signal_gen.get_all_signals(df)
            
            # Loop over signals
            for signal_name, signal_series in signals_dict.items():
                # Loop over holding periods
                for holding_period in holding_periods:
                    current_iteration += 1
                    progress = (current_iteration / total_iterations) * 100
                    
                    # Validate signal
                    stats = validator.validate_signal(df, signal_name, signal_series, holding_period)
                    
                    # Check significance
                    p_bonferroni = stats['p_value'] * total_tests if not np.isnan(stats['p_value']) else np.nan
                    pass_raw = (stats['n_trades'] >= MIN_TRADES and 
                               stats['p_value'] < ALPHA and 
                               not np.isnan(stats['sharpe']))
                    pass_bonferroni = (stats['n_trades'] >= MIN_TRADES and 
                                      p_bonferroni < ALPHA and 
                                      not np.isnan(stats['sharpe']))
                    
                    # Store result
                    result = {
                        'signal': signal_name,
                        'timeframe': tf_name,
                        'holding_period': holding_period,
                        'asset': asset,
                        'n_trades': stats['n_trades'],
                        'mean_return': stats['mean_return'],
                        'sharpe': stats['sharpe'],
                        't_stat': stats['t_stat'],
                        'p_value': stats['p_value'],
                        'p_bonferroni': p_bonferroni,
                        'win_rate': stats['win_rate'],
                        'pass_raw': pass_raw,
                        'pass_bonferroni': pass_bonferroni,
                    }
                    results.append(result)
                    
                    # Print progress
                    if stats['n_trades'] >= MIN_TRADES:
                        status = "✓" if pass_bonferroni else ("○" if pass_raw else "✗")
                        print(f"[{progress:5.1f}%] {status} {tf_name:6s} {holding_period:4s} {asset:10s} {signal_name:25s} "
                              f"N={int(stats['n_trades']) if not np.isnan(stats.get('n_trades',0)) else 0:4d} Sharpe={stats.get('sharpe',0) if not np.isnan(stats.get('sharpe',0)) else 0:6.2f} p={stats.get('p_value',1) if not np.isnan(stats.get('p_value',1)) else 1:.4f}")
                    else:
                        print(f"[{progress:5.1f}%] - {tf_name:6s} {holding_period:4s} {asset:10s} {signal_name:25s} "
                              f"N={int(stats['n_trades']) if not np.isnan(stats.get('n_trades', 0)) else 0:4d} (insufficient trades)")
            
            # Free memory after each asset
            del df, signals_dict
            gc.collect()
        
        # Incremental save after each TF config
        pd.DataFrame(results).to_csv(RESULTS_PATH / "derivatives_validation_results.csv", index=False)
        print(f"\n💾 Saved {len(results)} results after {tf_name}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = RESULTS_PATH / "derivatives_validation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Filter valid results (enough trades)
    valid_df = results_df[results_df['n_trades'] >= MIN_TRADES].copy()
    
    if len(valid_df) == 0:
        print("No valid results (insufficient trades)")
        return
    
    # By signal
    print("\n### BY SIGNAL ###")
    signal_summary = valid_df.groupby('signal').agg({
        'sharpe': 'mean',
        'pass_raw': 'mean',
        'pass_bonferroni': 'mean',
        'n_trades': 'sum'
    }).round(3)
    signal_summary.columns = ['Avg Sharpe', 'Pass Rate (Raw)', 'Pass Rate (Bonf)', 'Total Trades']
    print(signal_summary.sort_values('Avg Sharpe', ascending=False).to_string())
    
    # By timeframe
    print("\n### BY TIMEFRAME ###")
    tf_summary = valid_df.groupby('timeframe').agg({
        'sharpe': 'mean',
        'pass_raw': 'mean',
        'pass_bonferroni': 'mean',
        'n_trades': 'sum'
    }).round(3)
    tf_summary.columns = ['Avg Sharpe', 'Pass Rate (Raw)', 'Pass Rate (Bonf)', 'Total Trades']
    print(tf_summary.sort_values('Avg Sharpe', ascending=False).to_string())
    
    # By asset
    print("\n### BY ASSET ###")
    asset_summary = valid_df.groupby('asset').agg({
        'sharpe': 'mean',
        'pass_raw': 'mean',
        'pass_bonferroni': 'mean',
        'n_trades': 'sum'
    }).round(3)
    asset_summary.columns = ['Avg Sharpe', 'Pass Rate (Raw)', 'Pass Rate (Bonf)', 'Total Trades']
    print(asset_summary.sort_values('Avg Sharpe', ascending=False).head(20).to_string())
    
    # Top 20 individual results
    print("\n### TOP 20 RESULTS BY SHARPE ###")
    top_20 = valid_df.nlargest(20, 'sharpe')[
        ['signal', 'timeframe', 'holding_period', 'asset', 'sharpe', 't_stat', 'p_value', 'n_trades', 'pass_bonferroni']
    ]
    print(top_20.to_string(index=False))
    
    # Bonferroni survivors
    print("\n### BONFERRONI SURVIVORS ###")
    survivors = valid_df[valid_df['pass_bonferroni'] == True]
    print(f"Total survivors: {len(survivors)} / {len(valid_df)} ({len(survivors)/len(valid_df)*100:.1f}%)")
    
    if len(survivors) > 0:
        print("\nBonferroni-corrected significant results:")
        print(survivors[['signal', 'timeframe', 'holding_period', 'asset', 'sharpe', 't_stat', 'p_bonferroni', 'n_trades']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
