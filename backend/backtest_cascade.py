"""
Strategy 3: "Cascade" - Multi-Timeframe Momentum
Thesis: Signals confirmed across multiple timeframes are higher quality.
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
    
    # FRED macro for position sizing
    t10y2y = macro.get_series('T10Y2Y', START, END)
    m2 = macro.get_series('M2SL', START, END)
    fedfunds = macro.get_series('FEDFUNDS', START, END)
    cpi = macro.get_series('CPIAUCSL', START, END)
    hy = macro.get_series('BAMLH0A0HYM2', START, END)
    
    return btc, t10y2y, m2, fedfunds, cpi, hy

def compute_signals(btc, t10y2y, m2, fedfunds, cpi, hy):
    idx = btc.index
    btc_close = btc['close']
    
    # Weekly bars from daily
    btc_weekly = btc_close.resample('W').last().dropna()
    
    # Signal 1: Weekly TSMOM (vol-scaled momentum, 20-week lookback)
    weekly_ret = btc_weekly.pct_change(20)  # 20-week return
    weekly_vol = btc_weekly.pct_change().rolling(20).std()  # 20-week vol
    tsmom = (weekly_ret / weekly_vol.replace(0, np.nan)).fillna(0)
    tsmom_signal = (tsmom > 0).astype(float)
    # Reindex to daily (forward-fill)
    tsmom_daily = tsmom_signal.reindex(idx, method='ffill').fillna(0)
    
    # Signal 2: Daily golden cross (50 SMA > 200 SMA)
    sma50 = btc_close.rolling(50).mean()
    sma200 = btc_close.rolling(200).mean()
    golden_cross = (sma50 > sma200).astype(float)
    
    # Signal 3: Daily MACD bullish
    ema12 = btc_close.ewm(span=12).mean()
    ema26 = btc_close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_bullish = (macd > macd_signal).astype(float)
    
    # Signal 4: RSI between 30-70 (not overbought)
    delta = btc_close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_ok = ((rsi >= 30) & (rsi <= 70)).astype(float)
    
    cascade_score = tsmom_daily + golden_cross + macd_bullish + rsi_ok
    cascade_score = cascade_score.fillna(0)
    
    # Macro score for position sizing
    t10y2y_d = t10y2y.reindex(idx, method='ffill')
    m2_d = m2.reindex(idx, method='ffill')
    fedfunds_d = fedfunds.reindex(idx, method='ffill')
    cpi_d = cpi.reindex(idx, method='ffill')
    hy_d = hy.reindex(idx, method='ffill')
    
    ms1 = (t10y2y_d > 0).astype(float)
    ms2 = (m2_d.pct_change(252) > 0).astype(float)
    ms3 = (m2_d.pct_change(63) > m2_d.pct_change(126)).astype(float)
    ms4 = (cpi_d.pct_change(252) < cpi_d.pct_change(252).shift(63)).astype(float)
    ms5 = (fedfunds_d.diff(63) <= 0).astype(float)
    hy_ma = hy_d.rolling(63).mean()
    ms6 = (hy_d < hy_ma).astype(float)
    macro_score = (ms1 + ms2 + ms3 + ms4 + ms5 + ms6).fillna(0)
    
    return tsmom_daily, golden_cross, macd_bullish, rsi_ok, cascade_score, macro_score, btc_close

def trailing_stop_returns(btc_close, position_sizes, stop_pct=0.15):
    returns = btc_close.pct_change().fillna(0)
    strat_returns = pd.Series(0.0, index=btc_close.index)
    in_pos = False
    peak = 0
    size = 0
    for i in range(1, len(btc_close)):
        price = btc_close.iloc[i]
        sz = position_sizes.iloc[i]
        if in_pos:
            peak = max(peak, price)
            if price < peak * (1 - stop_pct):
                strat_returns.iloc[i] = returns.iloc[i] * size
                in_pos = False
                size = 0
            else:
                strat_returns.iloc[i] = returns.iloc[i] * size
                size = sz
        else:
            if sz > 0:
                in_pos = True
                peak = price
                size = sz
                strat_returns.iloc[i] = returns.iloc[i] * size
    return strat_returns

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
    print("CASCADE STRATEGY BACKTEST")
    print("=" * 60)
    
    print("\nLoading data...")
    btc, t10y2y, m2, fedfunds, cpi, hy = load_data()
    print(f"BTC: {btc.index[0].date()} to {btc.index[-1].date()} ({len(btc)} bars)")
    
    print("Computing signals...")
    tsmom, golden, macd_bull, rsi_ok, cascade_score, macro_score, btc_close = compute_signals(
        btc, t10y2y, m2, fedfunds, cpi, hy
    )
    
    btc_returns = btc_close.pct_change().fillna(0)
    
    # Strategy 1: Buy & Hold
    bh = btc_returns.copy()
    
    # Strategy 2: Single TF - golden cross only
    s2_sig = golden.shift(1).fillna(0)
    s2_ret = btc_returns * s2_sig
    
    # Strategy 3: Dual TF - TSMOM + golden cross
    s3_sig = ((tsmom > 0) & (golden > 0)).astype(float).shift(1).fillna(0)
    s3_ret = btc_returns * s3_sig
    
    # Strategy 4: Triple Cascade - TSMOM + golden + MACD
    s4_sig = ((tsmom > 0) & (golden > 0) & (macd_bull > 0)).astype(float).shift(1).fillna(0)
    s4_ret = btc_returns * s4_sig
    
    # Strategy 5: Full Cascade + Macro sizing + trailing stop
    triple = ((tsmom > 0) & (golden > 0) & (macd_bull > 0)).astype(float)
    # Position size based on macro score: 6=100%, 5=85%, 4=70%, 3=50%, 2=30%, 1=15%, 0=0%
    size_map = {6: 1.0, 5: 0.85, 4: 0.7, 3: 0.5, 2: 0.3, 1: 0.15, 0: 0.0}
    macro_size = macro_score.map(lambda x: size_map.get(int(min(x, 6)), 0))
    full_size = (triple * macro_size).shift(1).fillna(0)
    s5_ret = trailing_stop_returns(btc_close, full_size, 0.15)
    
    results = []
    for ret, name in [
        (bh, "1. Buy & Hold BTC"),
        (s2_ret, "2. Single TF: Golden Cross"),
        (s3_ret, "3. Dual TF: TSMOM + Golden Cross"),
        (s4_ret, "4. Triple Cascade"),
        (s5_ret, "5. Full Cascade + Macro + Trail Stop"),
    ]:
        m = compute_metrics(ret, name)
        results.append(m)
        print(f"\n{name}:")
        print(f"  Total Return: {m['total_return']}%  |  CAGR: {m['cagr']}%")
        print(f"  Sharpe: {m['sharpe']}  |  Sortino: {m['sortino']}")
        print(f"  Max DD: {m['max_drawdown']}%  |  Calmar: {m['calmar']}")
        print(f"  Trades: {m['trades']}  |  Time in Market: {m['time_in_market']}%")
    
    print(f"\nCascade Score Distribution:")
    for s in range(5):
        pct = (cascade_score == s).mean() * 100
        print(f"  Score {s}: {pct:.1f}%")
    
    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'cascade_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/cascade_results.json")

if __name__ == '__main__':
    main()
