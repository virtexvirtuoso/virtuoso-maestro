"""
Strategy 7: Saylor Signal (MSTR Premium as BTC Conviction Proxy)
Use MSTR/BTC relative performance as institutional conviction signal.
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

def run():
    print("=" * 60)
    print("STRATEGY 7: Saylor Signal (MSTR Premium)")
    print("=" * 60)

    loader = StockDataLoader()
    btc = loader.get_ohlcv('BTC-USD', '1d', '2020-01-01', '2026-02-12')
    mstr = loader.get_ohlcv('MSTR', '1d', '2020-01-01', '2026-02-12')
    print(f"  BTC: {len(btc)} bars, MSTR: {len(mstr)} bars")

    btc_close = btc['close']
    mstr_close = mstr['close']

    # Align on common dates
    common = btc_close.index.intersection(mstr_close.index)
    btc_close = btc_close.loc[common]
    mstr_close = mstr_close.loc[common]
    btc_ret = btc_close.pct_change()
    mstr_ret = mstr_close.pct_change()

    # MSTR Premium: rolling ratio of MSTR returns to BTC returns
    # Use rolling correlation of cumulative returns instead of raw ratio (more stable)
    mstr_cum = (1 + mstr_ret).cumprod()
    btc_cum = (1 + btc_ret).cumprod()

    # Relative performance: MSTR normalized by BTC
    relative_perf = mstr_cum / btc_cum
    mstr_premium = relative_perf.rolling(20).mean() / relative_perf.rolling(60).mean()

    # Signal: premium expanding = bullish
    mstr_signal = (mstr_premium > 1).astype(float)  # MSTR outperforming its own trend

    # MSTR-BTC correlation
    mstr_btc_corr = mstr_ret.rolling(30).corr(btc_ret)
    decorrelation = mstr_btc_corr < 0.5

    # BTC momentum
    sma200 = btc_close.rolling(200).mean()
    btc_above_200 = btc_close > sma200

    results = {}

    # Skip warmup
    start = common[252] if len(common) > 252 else common[60]

    # 1. Buy & Hold
    eq = (1 + btc_ret).cumprod()
    eq = eq.loc[eq.index >= start]
    results['buy_hold'] = compute_metrics(eq)
    print(f"\n1. Buy & Hold: {results['buy_hold']['total_return']:.1f}%")

    # 2. MSTR Premium: Long when premium expanding
    signal = mstr_signal.copy()
    port_ret = btc_ret * signal.shift(1).fillna(0)
    eq = (1 + port_ret).cumprod()
    eq = eq.loc[eq.index >= start]
    results['mstr_premium'] = compute_metrics(eq)
    pct_long = signal.loc[signal.index >= start].mean()
    results['mstr_premium']['pct_time_long'] = round(pct_long * 100, 1)
    print(f"2. MSTR Premium: {results['mstr_premium']['total_return']:.1f}% (long {pct_long:.0%} of time)")

    # 3. MSTR + Momentum: premium bullish AND BTC > 200 SMA
    signal = (mstr_signal * btc_above_200.astype(float))
    port_ret = btc_ret * signal.shift(1).fillna(0)
    eq = (1 + port_ret).cumprod()
    eq = eq.loc[eq.index >= start]
    results['mstr_momentum'] = compute_metrics(eq)
    pct_long = signal.loc[signal.index >= start].mean()
    results['mstr_momentum']['pct_time_long'] = round(pct_long * 100, 1)
    print(f"3. MSTR + Momentum: {results['mstr_momentum']['total_return']:.1f}% (long {pct_long:.0%} of time)")

    # 4. Decorrelation Alert: 50% when correlation < 0.5
    signal = pd.Series(1.0, index=common)
    signal[decorrelation] = 0.5
    port_ret = btc_ret * signal.shift(1).fillna(1)
    eq = (1 + port_ret).cumprod()
    eq = eq.loc[eq.index >= start]
    results['decorrelation_alert'] = compute_metrics(eq)
    pct_reduced = decorrelation.loc[decorrelation.index >= start].mean()
    results['decorrelation_alert']['pct_time_reduced'] = round(pct_reduced * 100, 1)
    print(f"4. Decorrelation Alert: {results['decorrelation_alert']['total_return']:.1f}% (reduced {pct_reduced:.0%} of time)")

    # Analytics
    avg_corr = mstr_btc_corr.loc[mstr_btc_corr.index >= start].mean()
    results['analytics'] = {
        'avg_mstr_btc_correlation': round(avg_corr, 3),
        'pct_decorrelated': round(pct_reduced * 100, 1),
        'premium_mean': round(mstr_premium.loc[mstr_premium.index >= start].mean(), 4),
    }
    print(f"\n  Avg MSTR-BTC correlation: {avg_corr:.3f}")
    print(f"  % time decorrelated: {pct_reduced:.1%}")

    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'saylor_signal_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results

if __name__ == '__main__':
    run()
