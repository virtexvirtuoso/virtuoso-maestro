"""
Strategy 4: Asset Allocator (BTC/SPY/GLD Rotation)
Multi-asset regime-based rotation with macro scoring.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_metrics(equity):
    """Compute return, CAGR, Sharpe, max DD from equity series."""
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (1 + total_ret) ** (365.25 / max(days, 1)) - 1
    daily_ret = equity.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = dd.min()
    return {'total_return': round(total_ret * 100, 2),
            'cagr': round(cagr * 100, 2),
            'sharpe': round(sharpe, 3),
            'max_drawdown': round(max_dd * 100, 2)}

def run():
    print("=" * 60)
    print("STRATEGY 4: Asset Allocator (BTC/SPY/GLD Rotation)")
    print("=" * 60)

    stock_loader = StockDataLoader()
    macro_loader = MacroDataLoader()

    # Load price data
    assets = {}
    for sym in ['BTC-USD', 'SPY', 'GLD']:
        assets[sym] = stock_loader.get_ohlcv(sym, '1d', '2017-01-01', '2026-02-12')
        print(f"  {sym}: {len(assets[sym])} bars")

    # Load macro data
    macro_series = {}
    for sid in ['T10Y2Y', 'M2SL', 'FEDFUNDS', 'CPIAUCSL', 'BAMLH0A0HYM2', 'DCOILWTICO']:
        try:
            macro_series[sid] = macro_loader.get_series(sid, '2016-01-01', '2026-02-12')
        except:
            print(f"  Warning: Could not load {sid}")

    # Build aligned close prices
    closes = pd.DataFrame({sym: df['close'] for sym, df in assets.items()})
    closes = closes.dropna()
    returns = closes.pct_change().dropna()
    closes = closes.loc[returns.index]

    # Macro score (0-5) for macro environment
    macro_df = pd.DataFrame(macro_series).resample('D').last().ffill()
    macro_df = macro_df.reindex(closes.index, method='ffill')

    macro_score = pd.Series(0, index=closes.index, dtype=float)
    if 'T10Y2Y' in macro_df: macro_score += (macro_df['T10Y2Y'] > 0).astype(float)
    if 'M2SL' in macro_df: macro_score += (macro_df['M2SL'].pct_change(252) > 0).astype(float)
    if 'FEDFUNDS' in macro_df: macro_score += (macro_df['FEDFUNDS'].diff(63) <= 0).astype(float)  # rates falling
    if 'CPIAUCSL' in macro_df:
        cpi_yoy = macro_df['CPIAUCSL'].pct_change(12)
        macro_score += (cpi_yoy < 0.04).astype(float)  # CPI < 4%
    if 'BAMLH0A0HYM2' in macro_df: macro_score += (macro_df['BAMLH0A0HYM2'] < 5).astype(float)

    # Regime scores per asset
    def asset_regime(sym, close_s):
        score = pd.Series(0, index=close_s.index, dtype=float)
        sma200 = close_s.rolling(200).mean()
        sma50 = close_s.rolling(50).mean()
        roc30 = close_s.pct_change(30)
        rsi = compute_rsi(close_s)

        score += (close_s > sma200).astype(float)
        score += (roc30 > 0).astype(float)
        score += (sma50 > sma200).astype(float)  # golden cross

        if 'BTC' in sym:
            score += ((rsi > 30) & (rsi < 70)).astype(float)
            score += (macro_score >= 3).astype(float)
        elif 'SPY' in sym:
            if 'DCOILWTICO' in macro_df:
                oil_roc = macro_df['DCOILWTICO'].pct_change(30)
                score += (oil_roc > 0).astype(float).reindex(close_s.index, method='ffill').fillna(0)
            if 'T10Y2Y' in macro_df:
                score += (macro_df['T10Y2Y'].reindex(close_s.index, method='ffill') > 0).astype(float).fillna(0)
        elif 'GLD' in sym:
            if 'CPIAUCSL' in macro_df and 'T10Y2Y' in macro_df:
                # real yields proxy declining
                score += (macro_df['T10Y2Y'].reindex(close_s.index, method='ffill').diff(30) < 0).astype(float).fillna(0)
            if 'BAMLH0A0HYM2' in macro_df:
                score += (macro_df['BAMLH0A0HYM2'].reindex(close_s.index, method='ffill').diff(30) > 0).astype(float).fillna(0)
            # SPY vol expanding as VIX proxy
            spy_vol = closes['SPY'].pct_change().rolling(30).std()
            score += (spy_vol > spy_vol.rolling(60).mean()).astype(float)

        return score.clip(0, 5)

    regimes = {}
    for sym in closes.columns:
        regimes[sym] = asset_regime(sym, closes[sym])

    regime_df = pd.DataFrame(regimes)

    # Monthly rebalance dates
    monthly = closes.resample('MS').first().index
    monthly = monthly[monthly >= closes.index[252]]  # skip warmup

    results = {}

    # Strategy 1: Buy & Hold 33/33/33
    eq = (returns * (1/3)).sum(axis=1)
    equity = (1 + eq).cumprod()
    equity = equity.loc[monthly[0]:]
    results['buy_hold_equal'] = compute_metrics(equity)
    results['buy_hold_equal']['monthly_turnover'] = 0
    print(f"\n1. Buy & Hold 33/33/33: {results['buy_hold_equal']['total_return']:.1f}%")

    # Strategy 2: Momentum Rotation
    equity = pd.Series(1.0, index=closes.index)
    current_alloc = None
    turnover_sum = 0
    for i, dt in enumerate(monthly):
        if i == 0: continue
        mom = closes.loc[:dt].iloc[-63:].iloc[-1] / closes.loc[:dt].iloc[-63:].iloc[0] - 1
        best = mom.idxmax()
        if current_alloc != best:
            turnover_sum += 2  # full switch
            current_alloc = best
        # Apply returns until next rebalance
        end = monthly[i+1] if i+1 < len(monthly) else closes.index[-1]
        mask = (returns.index > dt) & (returns.index <= end)
        period_ret = returns.loc[mask, best]
        if len(period_ret) > 0:
            equity.loc[period_ret.index] = equity.loc[period_ret.index[0]:period_ret.index[-1]]

    # Simpler approach: build weights then compute
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, dt in enumerate(monthly):
        if i == 0: continue
        mom = closes.loc[:dt].iloc[-63:].iloc[-1] / closes.loc[:dt].iloc[-63:].iloc[0] - 1
        best = mom.idxmax()
        end = monthly[i+1] if i+1 < len(monthly) else closes.index[-1]
        weights.loc[(weights.index > dt) & (weights.index <= end), best] = 1.0

    port_ret = (returns * weights.shift(1)).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    equity = equity.loc[monthly[0]:]
    w_changes = weights.diff().abs().sum(axis=1).resample('MS').sum()
    results['momentum_rotation'] = compute_metrics(equity)
    results['momentum_rotation']['monthly_turnover'] = round(w_changes.mean(), 3)
    print(f"2. Momentum Rotation: {results['momentum_rotation']['total_return']:.1f}%")

    # Strategy 3: Regime Rotation (60/30/10)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, dt in enumerate(monthly):
        if i == 0: continue
        scores = regime_df.loc[:dt].iloc[-1]
        ranked = scores.sort_values(ascending=False)
        end = monthly[i+1] if i+1 < len(monthly) else closes.index[-1]
        mask = (weights.index > dt) & (weights.index <= end)
        weights.loc[mask, ranked.index[0]] = 0.6
        weights.loc[mask, ranked.index[1]] = 0.3
        weights.loc[mask, ranked.index[2]] = 0.1

    port_ret = (returns * weights.shift(1)).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    equity = equity.loc[monthly[0]:]
    w_changes = weights.diff().abs().sum(axis=1).resample('MS').sum()
    results['regime_rotation'] = compute_metrics(equity)
    results['regime_rotation']['monthly_turnover'] = round(w_changes.mean(), 3)
    print(f"3. Regime Rotation: {results['regime_rotation']['total_return']:.1f}%")

    # Strategy 4: Dynamic Allocation (0% if score <= 1)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, dt in enumerate(monthly):
        if i == 0: continue
        scores = regime_df.loc[:dt].iloc[-1].copy()
        scores[scores <= 1] = 0  # go to cash
        total = scores.sum()
        end = monthly[i+1] if i+1 < len(monthly) else closes.index[-1]
        mask = (weights.index > dt) & (weights.index <= end)
        if total > 0:
            for sym in closes.columns:
                weights.loc[mask, sym] = scores[sym] / total

    port_ret = (returns * weights.shift(1)).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    equity = equity.loc[monthly[0]:]
    w_changes = weights.diff().abs().sum(axis=1).resample('MS').sum()
    results['dynamic_allocation'] = compute_metrics(equity)
    results['dynamic_allocation']['monthly_turnover'] = round(w_changes.mean(), 3)
    print(f"4. Dynamic Allocation: {results['dynamic_allocation']['total_return']:.1f}%")

    # Strategy 5: Risk Parity Regime
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, dt in enumerate(monthly):
        if i == 0: continue
        vol = returns.loc[:dt].iloc[-30:].std()
        inv_vol = 1.0 / vol.replace(0, np.nan).fillna(vol.max())
        scores = regime_df.loc[:dt].iloc[-1]
        combined = inv_vol * scores
        total = combined.sum()
        end = monthly[i+1] if i+1 < len(monthly) else closes.index[-1]
        mask = (weights.index > dt) & (weights.index <= end)
        if total > 0:
            for sym in closes.columns:
                weights.loc[mask, sym] = combined[sym] / total

    port_ret = (returns * weights.shift(1)).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    equity = equity.loc[monthly[0]:]
    w_changes = weights.diff().abs().sum(axis=1).resample('MS').sum()
    results['risk_parity_regime'] = compute_metrics(equity)
    results['risk_parity_regime']['monthly_turnover'] = round(w_changes.mean(), 3)
    print(f"5. Risk Parity Regime: {results['risk_parity_regime']['total_return']:.1f}%")

    # Save
    out_dir = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'asset_allocator_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results

if __name__ == '__main__':
    run()
