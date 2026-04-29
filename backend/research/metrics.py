#!/usr/bin/env python3
"""
Multi-Metric Evaluation Module
===============================
Computes a full suite of risk-adjusted performance metrics beyond Sharpe.

Metrics:
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation only — rewards positive skew)
- Calmar Ratio (annualized return / max drawdown — survival metric)
- Max Drawdown + Max Drawdown Duration (bars)
- Profit Factor (gross wins / gross losses)
- Win Rate
- Payoff Ratio (avg win / avg loss)
- Expectancy (win_rate * avg_win - loss_rate * avg_loss)
- Tail Ratio (95th percentile / abs(5th percentile) — upside vs downside tails)
- Time-in-Market (fraction of bars with active position)
- Skewness + Kurtosis of return distribution
- Total Return (compounded)

Usage:
    from metrics import compute_metrics, METRIC_COLUMNS

    result = compute_metrics(trade_returns, bars_per_year=365, total_bars=None)
    # result is a dict with all metrics
"""

import numpy as np
from typing import Optional


METRIC_COLUMNS = [
    'sharpe', 'sortino', 'calmar', 'max_drawdown', 'max_dd_duration',
    'profit_factor', 'win_rate', 'payoff_ratio', 'expectancy',
    'tail_ratio', 'time_in_market', 'skewness', 'kurtosis',
    'total_return', 'n_trades', 'mean_return', 'std_return',
    't_stat', 'p_value'
]


def compute_metrics(
    returns: np.ndarray,
    bars_per_year: float = 365,
    total_bars: Optional[int] = None,
    trades_per_year: Optional[float] = None,
) -> dict:
    """
    Compute full metric suite from an array of non-overlapping trade returns.

    Args:
        returns: 1D array of per-trade returns (after costs)
        bars_per_year: for annualization (365 for daily, 365*6 for 4h, etc.)
        total_bars: total bars in test period (for time-in-market calc)
        trades_per_year: if known; otherwise estimated from bars_per_year

    Returns:
        dict with all metrics (NaN for undefined values)
    """
    from scipy import stats as sp_stats

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 2:
        return {k: np.nan for k in METRIC_COLUMNS}

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    # t-stat & p-value
    if std_ret > 1e-12:
        t_stat, p_value = sp_stats.ttest_1samp(returns, 0)
    else:
        t_stat, p_value = 0.0, 1.0

    # Trades per year estimate
    if trades_per_year is None:
        trades_per_year = max(n, 1)  # conservative: use actual count as 1-year estimate

    # --- Sharpe (annualized) ---
    sharpe = (mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 1e-12 else 0.0

    # --- Sortino (downside deviation) ---
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = np.std(downside, ddof=1)
        sortino = (mean_ret / downside_std * np.sqrt(trades_per_year)) if downside_std > 1e-12 else 0.0
    else:
        sortino = np.inf if mean_ret > 0 else 0.0

    # --- Cumulative returns & drawdown ---
    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = cum / rolling_max - 1
    max_dd = np.min(drawdowns)

    # Max drawdown duration (in trades)
    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(cum)):
        if cum[i] < rolling_max[i]:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # --- Calmar (annualized return / |max drawdown|) ---
    total_return = cum[-1] - 1
    # Estimate test period in years
    test_years = n / trades_per_year if trades_per_year > 0 else 1
    ann_return = (1 + total_return) ** (1 / max(test_years, 0.01)) - 1
    calmar = (ann_return / abs(max_dd)) if abs(max_dd) > 1e-12 else (np.inf if ann_return > 0 else 0.0)

    # --- Profit Factor ---
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (np.inf if gross_profit > 0 else 0.0)

    # --- Win Rate ---
    win_rate = np.mean(returns > 0)

    # --- Payoff Ratio (avg win / avg loss) ---
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 1e-12 else (np.inf if avg_win > 0 else 0.0)

    # --- Expectancy (per trade) ---
    loss_rate = 1 - win_rate
    expectancy = win_rate * avg_win - loss_rate * avg_loss

    # --- Tail Ratio (95th / |5th|) ---
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    tail_ratio = (p95 / abs(p5)) if abs(p5) > 1e-12 else (np.inf if p95 > 0 else 0.0)

    # --- Time in Market ---
    time_in_market = (n / total_bars) if total_bars and total_bars > 0 else np.nan

    # --- Skewness & Kurtosis ---
    skewness = sp_stats.skew(returns) if n >= 3 else np.nan
    kurtosis = sp_stats.kurtosis(returns) if n >= 4 else np.nan

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4) if not np.isinf(sortino) else 999.0,
        'calmar': round(calmar, 4) if not np.isinf(calmar) else 999.0,
        'max_drawdown': round(max_dd, 4),
        'max_dd_duration': max_dd_duration,
        'profit_factor': round(profit_factor, 4) if not np.isinf(profit_factor) else 999.0,
        'win_rate': round(win_rate, 4),
        'payoff_ratio': round(payoff_ratio, 4) if not np.isinf(payoff_ratio) else 999.0,
        'expectancy': round(expectancy, 6),
        'tail_ratio': round(tail_ratio, 4) if not np.isinf(tail_ratio) else 999.0,
        'time_in_market': round(time_in_market, 4) if not np.isnan(time_in_market) else np.nan,
        'skewness': round(skewness, 4) if not np.isnan(skewness) else np.nan,
        'kurtosis': round(kurtosis, 4) if not np.isnan(kurtosis) else np.nan,
        'total_return': round(total_return, 4),
        'n_trades': n,
        'mean_return': round(mean_ret, 6),
        'std_return': round(std_ret, 6),
        't_stat': round(t_stat, 4),
        'p_value': round(p_value, 8),
    }


def format_metrics_summary(m: dict) -> str:
    """One-line summary for console output."""
    return (
        f"Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} Calmar={m['calmar']:.2f} "
        f"MDD={m['max_drawdown']:.1%} PF={m['profit_factor']:.2f} "
        f"WR={m['win_rate']:.1%} Payoff={m['payoff_ratio']:.2f} "
        f"Tail={m['tail_ratio']:.2f} Skew={m['skewness']:.2f} "
        f"n={m['n_trades']}"
    )


def grade_strategy(m: dict) -> str:
    """
    Letter grade based on multi-metric assessment.
    A: Sharpe>1.5, Sortino>2, Calmar>1, MDD>-20%, PF>2
    B: Sharpe>1.0, Sortino>1.5, MDD>-30%, PF>1.5
    C: Sharpe>0.5, positive expectancy
    D: Sharpe>0, some positive metrics
    F: Sharpe<=0 or fatal flaws
    """
    s = m['sharpe']
    sort = m['sortino']
    cal = m['calmar']
    mdd = m['max_drawdown']
    pf = m['profit_factor']
    wr = m['win_rate']

    if s > 1.5 and sort > 2.0 and cal > 1.0 and mdd > -0.20 and pf > 2.0:
        return 'A'
    elif s > 1.0 and sort > 1.5 and mdd > -0.30 and pf > 1.5:
        return 'B'
    elif s > 0.5 and m['expectancy'] > 0:
        return 'C'
    elif s > 0 and (pf > 1.0 or wr > 0.5):
        return 'D'
    else:
        return 'F'


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    fake_returns = np.random.normal(0.002, 0.02, 200)
    m = compute_metrics(fake_returns, trades_per_year=200, total_bars=500)
    print("Test metrics:")
    print(format_metrics_summary(m))
    print(f"Grade: {grade_strategy(m)}")
    print(f"\nAll values: {m}")
