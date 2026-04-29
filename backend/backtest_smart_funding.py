"""
Strategy 2: "Smart Funding" - Funding Rate + Macro Divergence
Thesis: Funding rate extremes are better signals when macro confirms reversal direction.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))
from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

START = '2017-01-01'
END = '2026-02-12'

def load_data():
    stock = StockDataLoader()
    macro = MacroDataLoader()
    
    btc = stock.get_ohlcv('BTC-USD', '1d', START, END)
    
    # Real funding data (starts 2023-05)
    funding_path = os.path.expanduser('~/Desktop/maestro/data/derivatives/btc_funding.csv')
    real_funding = pd.read_csv(funding_path, parse_dates=['timestamp'])
    real_funding = real_funding.set_index('timestamp')['fundingRate']
    # Resample to daily (sum of 3 funding periods per day)
    real_funding_daily = real_funding.resample('1D').sum()
    
    # FRED macro
    t10y2y = macro.get_series('T10Y2Y', START, END)
    m2 = macro.get_series('M2SL', START, END)
    fedfunds = macro.get_series('FEDFUNDS', START, END)
    cpi = macro.get_series('CPIAUCSL', START, END)
    hy = macro.get_series('BAMLH0A0HYM2', START, END)
    
    return btc, real_funding_daily, t10y2y, m2, fedfunds, cpi, hy

def compute_signals(btc, real_funding, t10y2y, m2, fedfunds, cpi, hy):
    idx = btc.index
    btc_close = btc['close']
    
    # Funding proxy for pre-2023: (daily return - 24d MA of return)
    btc_ret = btc_close.pct_change()
    funding_proxy = btc_ret - btc_ret.rolling(24).mean()
    
    # Merge real funding where available
    real_aligned = real_funding.reindex(idx)
    funding = funding_proxy.copy()
    mask = real_aligned.notna()
    funding[mask] = real_aligned[mask]
    
    # Funding Z-score (168-period ~ 7 day equivalent on daily)
    funding_mean = funding.rolling(168, min_periods=30).mean()
    funding_std = funding.rolling(168, min_periods=30).std()
    funding_zscore = (funding - funding_mean) / funding_std.replace(0, np.nan)
    funding_zscore = funding_zscore.fillna(0)
    
    # Macro signals (forward-filled to daily)
    t10y2y_d = t10y2y.reindex(idx, method='ffill')
    m2_d = m2.reindex(idx, method='ffill')
    fedfunds_d = fedfunds.reindex(idx, method='ffill')
    cpi_d = cpi.reindex(idx, method='ffill')
    hy_d = hy.reindex(idx, method='ffill')
    
    # 6 macro signals
    s1 = (t10y2y_d > 0).astype(float)  # yield curve positive
    s2 = (m2_d.pct_change(252) > 0).astype(float)  # M2 expanding YoY
    s3 = (m2_d.pct_change(63) > m2_d.pct_change(126)).astype(float)  # M2 accelerating
    s4 = (cpi_d.pct_change(252) < cpi_d.pct_change(252).shift(63)).astype(float)  # CPI declining
    s5 = (fedfunds_d.diff(63) <= 0).astype(float)  # Fed not hiking
    hy_ma = hy_d.rolling(63).mean()
    s6 = (hy_d < hy_ma).astype(float)  # HY tightening
    
    macro_score = (s1 + s2 + s3 + s4 + s5 + s6).fillna(0)
    
    # BTC momentum
    btc_sma200 = btc_close.rolling(200).mean()
    btc_above_sma = btc_close > btc_sma200
    
    return funding_zscore, macro_score, btc_close, btc_above_sma

def compute_metrics(returns, name):
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    in_market = (returns != 0).mean()
    pos = (returns != 0).astype(int)
    trades = (pos.diff().abs() > 0).sum()
    
    return {
        'strategy': name,
        'total_return': round(float(total_ret * 100), 2),
        'cagr': round(float(cagr * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'max_drawdown': round(float(max_dd * 100), 2),
        'calmar': round(float(calmar), 3),
        'trades': int(trades),
        'time_in_market': round(float(in_market * 100), 1),
    }

def main():
    print("=" * 60)
    print("SMART FUNDING STRATEGY BACKTEST")
    print("=" * 60)
    
    print("\nLoading data...")
    btc, real_funding, t10y2y, m2, fedfunds, cpi, hy = load_data()
    print(f"BTC: {btc.index[0].date()} to {btc.index[-1].date()} ({len(btc)} bars)")
    print(f"Real funding data: {real_funding.index[0].date()} to {real_funding.index[-1].date()}")
    
    print("Computing signals...")
    funding_zscore, macro_score, btc_close, btc_above_sma = compute_signals(
        btc, real_funding, t10y2y, m2, fedfunds, cpi, hy
    )
    
    btc_returns = btc_close.pct_change().fillna(0)
    
    # Strategy 1: Buy & Hold
    bh = btc_returns.copy()
    
    # Strategy 2: Classic Funding
    # Long when zscore < -2, flat when > 2, hold otherwise
    classic_pos = pd.Series(0.0, index=btc_close.index)
    in_long = False
    for i in range(len(classic_pos)):
        z = funding_zscore.iloc[i]
        if z < -2:
            in_long = True
        elif z > 2:
            in_long = False
        classic_pos.iloc[i] = 1.0 if in_long else 0.0
    classic_ret = btc_returns * classic_pos.shift(1).fillna(0)
    
    # Strategy 3: Smart Funding
    smart_pos = pd.Series(0.0, index=btc_close.index)
    in_long = False
    for i in range(len(smart_pos)):
        z = funding_zscore.iloc[i]
        ms = macro_score.iloc[i]
        if z < -2 and ms >= 3:
            in_long = True
        elif z > 2 and ms <= 3:
            in_long = False
        smart_pos.iloc[i] = 1.0 if in_long else 0.0
    smart_ret = btc_returns * smart_pos.shift(1).fillna(0)
    
    # Strategy 4: Smart Funding Aggressive
    agg_pos = pd.Series(0.0, index=btc_close.index)
    in_long = False
    for i in range(len(agg_pos)):
        z = funding_zscore.iloc[i]
        ms = macro_score.iloc[i]
        if z < -1.5 and ms >= 4:
            in_long = True
        elif z > 2 and ms <= 3:
            in_long = False
        agg_pos.iloc[i] = 1.0 if in_long else 0.0
    agg_ret = btc_returns * agg_pos.shift(1).fillna(0)
    
    # Strategy 5: Smart Funding + Momentum
    mom_pos = pd.Series(0.0, index=btc_close.index)
    in_long = False
    for i in range(len(mom_pos)):
        z = funding_zscore.iloc[i]
        ms = macro_score.iloc[i]
        above = btc_above_sma.iloc[i] if not pd.isna(btc_above_sma.iloc[i]) else False
        if z < -2 and ms >= 3 and above:
            in_long = True
        elif z > 2 and ms <= 3:
            in_long = False
        elif not above:
            in_long = False
        mom_pos.iloc[i] = 1.0 if in_long else 0.0
    mom_ret = btc_returns * mom_pos.shift(1).fillna(0)
    
    results = []
    for ret, name in [
        (bh, "1. Buy & Hold BTC"),
        (classic_ret, "2. Classic Funding"),
        (smart_ret, "3. Smart Funding"),
        (agg_ret, "4. Smart Funding Aggressive"),
        (mom_ret, "5. Smart Funding + Momentum"),
    ]:
        m = compute_metrics(ret, name)
        results.append(m)
        print(f"\n{name}:")
        print(f"  Total Return: {m['total_return']}%  |  CAGR: {m['cagr']}%")
        print(f"  Sharpe: {m['sharpe']}  |  Sortino: {m['sortino']}")
        print(f"  Max DD: {m['max_drawdown']}%  |  Calmar: {m['calmar']}")
        print(f"  Trades: {m['trades']}  |  Time in Market: {m['time_in_market']}%")
    
    # Funding zscore distribution
    print(f"\nFunding Z-Score Stats:")
    print(f"  Mean: {funding_zscore.mean():.3f}, Std: {funding_zscore.std():.3f}")
    print(f"  < -2: {(funding_zscore < -2).sum()} days")
    print(f"  > 2: {(funding_zscore > 2).sum()} days")
    
    print(f"\nMacro Score Distribution:")
    for s in range(7):
        pct = (macro_score == s).mean() * 100
        print(f"  Score {s}: {pct:.1f}%")
    
    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'smart_funding_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/smart_funding_results.json")

if __name__ == '__main__':
    main()
