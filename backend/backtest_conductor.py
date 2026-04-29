"""
Strategy 1: "Conductor" - Cross-Asset Regime Detection
Thesis: Copper, HY spreads, yield curve shift BEFORE crypto moves (1-3 week lag).
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
INITIAL_CASH = 100_000

def load_data():
    stock = StockDataLoader()
    macro = MacroDataLoader()
    
    # BTC & SPY daily
    btc = stock.get_ohlcv('BTC-USD', '1d', START, END)
    spy = stock.get_ohlcv('SPY', '1d', START, END)
    
    # FRED series
    oil = macro.get_series('DCOILWTICO', START, END)  # copper proxy via oil
    t10y2y = macro.get_series('T10Y2Y', START, END)
    hy = macro.get_series('BAMLH0A0HYM2', START, END)
    m2 = macro.get_series('M2SL', START, END)
    
    return btc, spy, oil, t10y2y, hy, m2

def compute_signals(btc, spy, oil, t10y2y, hy, m2):
    idx = btc.index
    
    # Forward-fill macro to daily
    oil_d = oil.reindex(idx, method='ffill')
    t10y2y_d = t10y2y.reindex(idx, method='ffill')
    hy_d = hy.reindex(idx, method='ffill')
    m2_d = m2.reindex(idx, method='ffill')
    spy_close = spy['close'].reindex(idx, method='ffill')
    
    # Signal 1: Oil/copper bullish - 3mo ROC > 0 AND > 6mo MA
    oil_roc = oil_d.pct_change(63)  # ~3 months
    oil_roc_ma = oil_roc.rolling(126).mean()  # 6mo MA of ROC
    copper_bullish = ((oil_roc > 0) & (oil_roc > oil_roc_ma)).astype(float)
    
    # Signal 2: HY tightening - spread < 3mo MA AND declining
    hy_ma = hy_d.rolling(63).mean()
    hy_declining = hy_d.diff(21) < 0  # declining over 1 month
    hy_tightening = ((hy_d < hy_ma) & hy_declining).astype(float)
    
    # Signal 3: Yield curve positive
    yield_positive = (t10y2y_d > 0).astype(float)
    
    # Signal 4: Liquidity expanding (M2 YoY > 0)
    m2_yoy = m2_d.pct_change(252)  # ~1 year daily
    liquidity_expanding = (m2_yoy > 0).astype(float)
    
    # Signal 5: SPY > 200 SMA
    spy_trend = (spy_close > spy_close.rolling(200).mean()).astype(float)
    
    conductor_score = copper_bullish + hy_tightening + yield_positive + liquidity_expanding + spy_trend
    conductor_score = conductor_score.fillna(0)
    
    # BTC momentum
    btc_close = btc['close']
    btc_sma200 = btc_close.rolling(200).mean()
    btc_roc30 = btc_close.pct_change(30)
    btc_momentum = ((btc_close > btc_sma200) & (btc_roc30 > 0)).astype(float)
    
    return conductor_score, btc_momentum, btc_close

def compute_metrics(returns, btc_returns, name):
    """Compute strategy metrics from daily returns series."""
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    in_market = (returns != 0).mean()
    
    # Count trades (signal changes)
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
        'n_years': round(n_years, 1),
    }

def run_trailing_stop(btc_close, position_sizes, stop_pct=0.15):
    """Apply trailing stop to position-sized strategy."""
    returns = btc_close.pct_change().fillna(0)
    strat_returns = pd.Series(0.0, index=btc_close.index)
    
    in_position = False
    entry_price = 0
    peak_price = 0
    current_size = 0
    
    for i in range(1, len(btc_close)):
        date = btc_close.index[i]
        price = btc_close.iloc[i]
        size = position_sizes.iloc[i]
        
        if in_position:
            peak_price = max(peak_price, price)
            if price < peak_price * (1 - stop_pct):
                # Stop hit
                strat_returns.iloc[i] = returns.iloc[i] * current_size
                in_position = False
                current_size = 0
            else:
                strat_returns.iloc[i] = returns.iloc[i] * current_size
                current_size = size  # update size
        else:
            if size > 0:
                in_position = True
                entry_price = price
                peak_price = price
                current_size = size
                strat_returns.iloc[i] = returns.iloc[i] * current_size
    
    return strat_returns

def main():
    print("=" * 60)
    print("CONDUCTOR STRATEGY BACKTEST")
    print("=" * 60)
    
    print("\nLoading data...")
    btc, spy, oil, t10y2y, hy, m2 = load_data()
    print(f"BTC: {btc.index[0].date()} to {btc.index[-1].date()} ({len(btc)} bars)")
    
    print("Computing signals...")
    conductor_score, btc_momentum, btc_close = compute_signals(btc, spy, oil, t10y2y, hy, m2)
    
    btc_returns = btc_close.pct_change().fillna(0)
    
    # Strategy 1: Buy & Hold
    bh_returns = btc_returns.copy()
    
    # Strategy 2: Simple Momentum
    mom_signal = btc_momentum.shift(1).fillna(0)
    mom_returns = btc_returns * mom_signal
    
    # Strategy 3: Conductor Entry (momentum AND score >= 3)
    cond_entry = ((btc_momentum > 0) & (conductor_score >= 3)).astype(float).shift(1).fillna(0)
    cond_entry_returns = btc_returns * cond_entry
    
    # Strategy 4: Conductor Position Sizing
    size_map = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.3, 1: 0.0, 0: 0.0}
    position_sizes = conductor_score.map(lambda x: size_map.get(int(min(x, 5)), 0))
    position_sizes = position_sizes * btc_momentum  # only when momentum is positive
    position_sizes = position_sizes.shift(1).fillna(0)
    cond_size_returns = btc_returns * position_sizes
    
    # Strategy 5: Conductor + Trailing Stop
    cond_trail_returns = run_trailing_stop(btc_close, position_sizes, 0.15)
    
    results = []
    for ret, name in [
        (bh_returns, "1. Buy & Hold BTC"),
        (mom_returns, "2. Simple BTC Momentum"),
        (cond_entry_returns, "3. Conductor Entry"),
        (cond_size_returns, "4. Conductor Position Sizing"),
        (cond_trail_returns, "5. Conductor + Trailing Stop"),
    ]:
        m = compute_metrics(ret, bh_returns, name)
        results.append(m)
        print(f"\n{name}:")
        print(f"  Total Return: {m['total_return']}%  |  CAGR: {m['cagr']}%")
        print(f"  Sharpe: {m['sharpe']}  |  Sortino: {m['sortino']}")
        print(f"  Max DD: {m['max_drawdown']}%  |  Calmar: {m['calmar']}")
        print(f"  Trades: {m['trades']}  |  Time in Market: {m['time_in_market']}%")
    
    # Conductor score distribution
    print(f"\nConductor Score Distribution:")
    for s in range(6):
        pct = (conductor_score == s).mean() * 100
        print(f"  Score {s}: {pct:.1f}%")
    
    # Save
    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'conductor_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/conductor_results.json")

if __name__ == '__main__':
    main()
