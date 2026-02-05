"""
Time Series Momentum Strategy (TSMOM)

Based on: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463

Key findings from the paper:
- Significant momentum in 58 liquid futures (1985-2009)
- 1-12 month persistence, partial reversal after
- Diversified portfolio Sharpe > 1.0
- Performs best during extreme markets

Implementation notes:
- Uses 12-month lookback for signal generation
- EWMA volatility for position sizing (target 40% vol)
- Monthly rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import vectorbt as vbt

# Strategy metadata
STRATEGY_NAME = "TSMOMStrategy"
STRATEGY_DESCRIPTION = "Time Series Momentum - trend following based on AQR research"
STRATEGY_VERSION = "1.0.0"


def get_params() -> Dict[str, Dict[str, Any]]:
    """
    Define optimizable parameters with Optuna-compatible ranges.
    """
    return {
        'lookback_days': {
            'type': 'int',
            'low': 60,
            'high': 365,
            'default': 252,  # ~12 months
            'description': 'Lookback period for momentum signal (days)'
        },
        'vol_target': {
            'type': 'float',
            'low': 0.10,
            'high': 0.60,
            'default': 0.40,
            'description': 'Target annualized volatility per position'
        },
        'vol_lookback_days': {
            'type': 'int',
            'low': 20,
            'high': 120,
            'default': 60,
            'description': 'EWMA volatility estimation window'
        },
        'rebalance_freq': {
            'type': 'categorical',
            'choices': ['daily', 'weekly', 'monthly'],
            'default': 'monthly',
            'description': 'Rebalancing frequency'
        },
        'use_shorts': {
            'type': 'bool',
            'default': True,
            'description': 'Allow short positions on negative momentum'
        }
    }


def calculate_ewma_volatility(returns: pd.Series, span: int = 60) -> pd.Series:
    """
    Calculate EWMA volatility as per the paper.
    
    Formula: σ²_t = 261 * Σ(1-δ)δⁱ(r_{t-1-i} - r̄)²
    Where δ = span / (span + 1)
    
    Args:
        returns: Daily returns series
        span: EWMA span (center of mass)
    
    Returns:
        Annualized volatility series
    """
    # Use pandas EWMA for efficiency
    ewma_var = returns.ewm(span=span, adjust=False).var()
    # Annualize (365 for crypto, 252 for traditional)
    annualized_vol = np.sqrt(ewma_var * 365)
    return annualized_vol


def generate_momentum_signal(prices: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Generate momentum signal based on past returns.
    
    Signal = sign(r_{t-lookback, t})
    +1 for positive momentum (go long)
    -1 for negative momentum (go short)
    
    Args:
        prices: Price series
        lookback: Lookback period in days
    
    Returns:
        Signal series (-1, 0, or +1)
    """
    # Calculate return over lookback period
    momentum_return = prices.pct_change(lookback)
    
    # Generate signal
    signal = np.sign(momentum_return)
    
    return signal


def calculate_position_size(volatility: pd.Series, 
                           vol_target: float = 0.40,
                           max_leverage: float = 3.0) -> pd.Series:
    """
    Calculate position size to target constant volatility.
    
    Size = vol_target / realized_vol
    Capped at max_leverage for risk management.
    
    Args:
        volatility: Annualized volatility series
        vol_target: Target portfolio volatility
        max_leverage: Maximum allowed leverage
    
    Returns:
        Position size series
    """
    # Avoid division by zero
    vol_floor = volatility.clip(lower=0.05)
    
    # Position size to achieve target vol
    raw_size = vol_target / vol_floor
    
    # Cap leverage
    size = raw_size.clip(upper=max_leverage)
    
    return size


def apply_rebalance_frequency(signal: pd.Series, 
                              freq: str = 'monthly') -> pd.Series:
    """
    Apply rebalancing frequency to reduce turnover.
    
    Args:
        signal: Raw daily signal
        freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
    
    Returns:
        Signal that only changes at rebalance dates
    """
    if freq == 'daily':
        return signal
    
    # Create rebalance mask
    if freq == 'weekly':
        # Rebalance on Mondays
        is_rebalance = signal.index.dayofweek == 0
    elif freq == 'monthly':
        # Rebalance on first day of month
        is_rebalance = signal.index.day == 1
    else:
        raise ValueError(f"Unknown frequency: {freq}")
    
    # Forward fill signal between rebalance dates
    rebalanced = signal.copy()
    rebalanced[~is_rebalance] = np.nan
    rebalanced = rebalanced.ffill()
    
    return rebalanced


def run_backtest(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run TSMOM backtest using VectorBT.
    
    Args:
        data: DataFrame with 'open', 'high', 'low', 'close', 'volume'
        params: Strategy parameters
    
    Returns:
        Dictionary with backtest results
    """
    close = data['close']
    
    # Extract parameters
    lookback = params.get('lookback_days', 252)
    vol_target = params.get('vol_target', 0.40)
    vol_lookback = params.get('vol_lookback_days', 60)
    rebalance_freq = params.get('rebalance_freq', 'monthly')
    use_shorts = params.get('use_shorts', True)
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Generate momentum signal
    raw_signal = generate_momentum_signal(close, lookback)
    
    # Calculate volatility
    volatility = calculate_ewma_volatility(returns, vol_lookback)
    
    # Calculate position size
    position_size = calculate_position_size(volatility, vol_target)
    
    # Apply rebalancing
    signal = apply_rebalance_frequency(raw_signal, rebalance_freq)
    
    # Final position: signal * size
    if use_shorts:
        position = signal * position_size
    else:
        # Long only
        position = signal.clip(lower=0) * position_size
    
    # Generate entry/exit signals for VectorBT
    entries_long = (position > 0) & (position.shift(1) <= 0)
    exits_long = (position <= 0) & (position.shift(1) > 0)
    entries_short = (position < 0) & (position.shift(1) >= 0) if use_shorts else pd.Series(False, index=close.index)
    exits_short = (position >= 0) & (position.shift(1) < 0) if use_shorts else pd.Series(False, index=close.index)
    
    # Run VectorBT backtest
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        size=position_size.abs(),
        size_type='percent',
        init_cash=100000,
        fees=0.001,  # 10 bps
        slippage=0.0005,  # 5 bps
        freq='1D'
    )
    
    # Extract metrics
    stats = pf.stats()
    
    return {
        'total_return': stats['Total Return [%]'],
        'sharpe_ratio': stats['Sharpe Ratio'],
        'sortino_ratio': stats['Sortino Ratio'],
        'max_drawdown': stats['Max Drawdown [%]'],
        'win_rate': stats['Win Rate [%]'],
        'profit_factor': stats.get('Profit Factor', np.nan),
        'total_trades': stats['Total Trades'],
        'exposure_time': stats['Exposure Time [%]'],
        'portfolio': pf,
        'positions': position,
        'signal': signal
    }


# VectorBT Strategy Adapter (for Maestro V2 integration)
class TSMOMStrategy:
    """
    Time Series Momentum Strategy for Maestro V2
    """
    
    name = STRATEGY_NAME
    description = STRATEGY_DESCRIPTION
    version = STRATEGY_VERSION
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {k: v['default'] for k, v in get_params().items()}
    
    @staticmethod
    def get_params() -> Dict[str, Dict[str, Any]]:
        return get_params()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data."""
        close = data['close']
        returns = close.pct_change()
        
        # Momentum signal
        signal = generate_momentum_signal(close, self.params['lookback_days'])
        
        # Volatility
        vol = calculate_ewma_volatility(returns, self.params['vol_lookback_days'])
        
        # Position size
        size = calculate_position_size(vol, self.params['vol_target'])
        
        # Rebalance
        signal = apply_rebalance_frequency(signal, self.params['rebalance_freq'])
        
        return pd.DataFrame({
            'signal': signal,
            'size': size,
            'position': signal * size if self.params['use_shorts'] else signal.clip(lower=0) * size
        }, index=data.index)
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest and return results."""
        return run_backtest(data, self.params)


if __name__ == "__main__":
    # Quick test
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Parameters: {get_params()}")
