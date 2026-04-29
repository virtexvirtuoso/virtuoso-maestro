"""
Strategy 6: Earnings Tremor (Crypto Vol Around Big Tech Earnings)
Trade BTC based on big tech earnings calendar patterns.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datasource.yfinance_loader import StockDataLoader

def compute_metrics(equity):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (1 + total_ret) ** (365.25 / max(days, 1)) - 1
    daily_ret = equity.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
    return {'total_return': round(total_ret * 100, 2),
            'cagr': round(cagr * 100, 2),
            'sharpe': round(sharpe, 3),
            'max_drawdown': round(max_dd * 100, 2)}

def build_earnings_calendar(start_year=2019, end_year=2026):
    """Approximate FAANG+NVDA earnings: 4th week of Jan, Apr, Jul, Oct."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in [1, 4, 7, 10]:
            # 4th Monday of the month
            first = pd.Timestamp(year, month, 1)
            # Find first Monday
            offset = (7 - first.weekday()) % 7
            first_monday = first + pd.Timedelta(days=offset)
            fourth_monday = first_monday + pd.Timedelta(weeks=3)
            # Earnings week is Mon-Fri
            for d in range(5):
                dates.append(fourth_monday + pd.Timedelta(days=d))
    return sorted(set(dates))

def run():
    print("=" * 60)
    print("STRATEGY 6: Earnings Tremor")
    print("=" * 60)

    loader = StockDataLoader()
    btc = loader.get_ohlcv('BTC-USD', '1d', '2019-01-01', '2026-02-12')
    spy = loader.get_ohlcv('SPY', '1d', '2019-01-01', '2026-02-12')
    print(f"  BTC: {len(btc)} bars, SPY: {len(spy)} bars")

    btc_close = btc['close']
    spy_close = spy['close']
    btc_ret = btc_close.pct_change()
    spy_ret = spy_close.pct_change()

    # Align
    common = btc_ret.index.intersection(spy_ret.index)
    btc_ret = btc_ret.loc[common]
    spy_ret = spy_ret.loc[common]
    btc_close = btc_close.reindex(common, method='ffill')
    spy_close = spy_close.reindex(common, method='ffill')

    # Earnings calendar
    earnings_dates = build_earnings_calendar()
    earnings_week = pd.Series(False, index=common)
    for d in earnings_dates:
        if d in earnings_week.index:
            earnings_week[d] = True

    # Post earnings week: the 5 trading days after last earnings day of each block
    post_earnings = pd.Series(False, index=common)
    # Find end of each earnings week block
    ew_idx = earnings_week[earnings_week].index
    if len(ew_idx) > 0:
        blocks = []
        block_start = ew_idx[0]
        for i in range(1, len(ew_idx)):
            if (ew_idx[i] - ew_idx[i-1]).days > 3:
                blocks.append((block_start, ew_idx[i-1]))
                block_start = ew_idx[i]
        blocks.append((block_start, ew_idx[-1]))

        for _, block_end in blocks:
            pos = common.get_loc(block_end) if block_end in common else None
            if pos is not None:
                for j in range(1, 6):
                    if pos + j < len(common):
                        post_earnings.iloc[pos + j] = True

    # Pre-earnings: 2 days before earnings week
    pre_earnings = pd.Series(False, index=common)
    for block_start, _ in blocks if 'blocks' in dir() else []:
        pos = common.get_loc(block_start) if block_start in common else None
        if pos is not None:
            for j in range(1, 3):
                if pos - j >= 0:
                    pre_earnings.iloc[pos - j] = True

    # Analytics
    btc_vol_earn = btc_ret[earnings_week].std() * np.sqrt(252)
    btc_vol_non = btc_ret[~earnings_week].std() * np.sqrt(252)
    print(f"\n  BTC Vol during earnings weeks: {btc_vol_earn:.1%}")
    print(f"  BTC Vol outside earnings: {btc_vol_non:.1%}")
    print(f"  Vol ratio: {btc_vol_earn/btc_vol_non:.2f}x")

    # SPY weekly returns during earnings
    spy_weekly = spy_ret.rolling(5).sum()
    earnings_spy_up = (spy_weekly > 0.005) & earnings_week
    earnings_spy_down = (spy_weekly < -0.005) & earnings_week

    # BTC momentum signal
    sma50 = btc_close.rolling(50).mean()
    sma200 = btc_close.rolling(200).mean()
    btc_momentum = (btc_close > sma50).astype(float)

    results = {}

    # 1. Buy & Hold
    eq = (1 + btc_ret).cumprod().dropna()
    results['buy_hold'] = compute_metrics(eq)
    print(f"\n1. Buy & Hold: {results['buy_hold']['total_return']:.1f}%")

    # 2. Reduce Exposure during earnings
    signal = btc_momentum.copy()
    signal[earnings_week] *= 0.5
    port_ret = btc_ret * signal.shift(1).fillna(0)
    eq = (1 + port_ret).cumprod().dropna()
    results['reduce_exposure'] = compute_metrics(eq)
    print(f"2. Reduce Exposure: {results['reduce_exposure']['total_return']:.1f}%")

    # 3. Post-Earnings Momentum
    # Weekly SPY return during earnings week
    signal = pd.Series(0.0, index=common)
    for _, block_end in (blocks if 'blocks' in dir() else []):
        # Check SPY return during this earnings block
        block_mask = earnings_week.copy()
        pos_end = common.get_loc(block_end) if block_end in common else None
        if pos_end is None: continue
        # SPY return this week
        start_pos = max(0, pos_end - 4)
        spy_week_ret = spy_close.iloc[pos_end] / spy_close.iloc[start_pos] - 1

        # Set post-earnings signal
        for j in range(1, 6):
            if pos_end + j < len(common):
                if spy_week_ret > 0.01:
                    signal.iloc[pos_end + j] = 1.0
                # if down > 1%, stay flat (0)

    port_ret = btc_ret * signal.shift(1).fillna(0)
    eq = (1 + port_ret).cumprod().dropna()
    # Only meaningful if we have positions
    if eq.iloc[-1] != 1.0:
        results['post_earnings_momentum'] = compute_metrics(eq)
    else:
        results['post_earnings_momentum'] = {'total_return': 0, 'cagr': 0, 'sharpe': 0, 'max_drawdown': 0}
    print(f"3. Post-Earnings Momentum: {results['post_earnings_momentum']['total_return']:.1f}%")

    # 4. Earnings Vol Play: flat 2d before, re-enter 2d after
    signal = btc_momentum.copy()
    signal[pre_earnings] = 0
    signal[earnings_week] = 0
    # Re-enter 2d after: post_earnings handles this
    port_ret = btc_ret * signal.shift(1).fillna(0)
    eq = (1 + port_ret).cumprod().dropna()
    results['earnings_vol_play'] = compute_metrics(eq)
    print(f"4. Earnings Vol Play: {results['earnings_vol_play']['total_return']:.1f}%")

    # Earnings week stats
    n_earnings_weeks = len(blocks) if 'blocks' in dir() else 0
    results['analytics'] = {
        'btc_vol_earnings': round(btc_vol_earn * 100, 2),
        'btc_vol_non_earnings': round(btc_vol_non * 100, 2),
        'vol_ratio': round(btc_vol_earn / btc_vol_non, 3),
        'num_earnings_blocks': n_earnings_weeks,
    }

    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'earnings_tremor_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results

if __name__ == '__main__':
    run()
