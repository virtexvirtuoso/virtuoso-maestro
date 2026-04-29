"""
Sanity checks for orderflow data and backtest methodology.

1. Data integrity: buy_vol + sell_vol ≈ volume, delta makes sense
2. Known signal: simple SMA crossover should produce known behavior
3. Perfect signal: future returns → should show massive Sharpe (proves WF works)
4. Random signal: should show ~0 Sharpe (proves we're not inflating)
5. Buy & hold: what's the actual market return over this period?
6. CVD as raw predictor: correlation of delta with forward returns
7. Cross-check: run HSAKA lean through same WF engine
"""
import pandas as pd
import numpy as np
from pathlib import Path

from backend.config.data_paths import BARS_1M_V1

DATA_DIR = BARS_1M_V1
OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")

def walk_forward(returns, signal, comm=0.0006, ann_factor=252*6, n_folds=7):
    """Same WF engine used in magus_orderflow.py"""
    sig = np.roll(signal, 1)
    sig[0] = 0
    n = len(returns)
    fold_size = n // (n_folds + 1)
    if fold_size < 50:
        return None
    oos = []
    for i in range(n_folds):
        ts = fold_size * (i + 2)
        te = min(ts + fold_size, n)
        if te <= ts: break
        fr = returns[ts:te] * sig[ts:te]
        sc = np.abs(np.diff(np.concatenate([[0], sig[ts:te]])))
        fr = fr - sc * comm
        oos.extend(fr.tolist())
    if len(oos) < 100:
        return None
    oos = np.array(oos)
    if np.std(oos) < 1e-10:
        return None
    sharpe = np.mean(oos) / np.std(oos) * np.sqrt(ann_factor)
    boot = np.array([np.mean(oos[np.random.randint(0, len(oos), len(oos))]) for _ in range(3000)])
    p = np.mean(boot <= 0)
    cum = np.cumprod(1 + oos)
    peak = np.maximum.accumulate(cum)
    max_dd = np.min((cum - peak) / peak)
    invested = np.mean(np.abs(sig)) * 100
    n_trades = int(np.sum(np.abs(np.diff(sig)) > 0))
    return {"sharpe": round(sharpe, 3), "p": round(p, 4), "dd": round(max_dd*100,1),
            "inv": round(invested,1), "trades": n_trades, "n_oos": len(oos)}

def main():
    print("=" * 70)
    print("ORDERFLOW DATA & METHODOLOGY SANITY CHECKS")
    print("=" * 70)
    
    # Load orderflow data
    df = pd.read_csv(DATA_DIR / "btcusdt_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
    print(f"\n1. DATA INTEGRITY (BTC 1m, {len(df):,} bars)")
    print(f"   Date range: {df.index[0]} → {df.index[-1]}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check buy_vol + sell_vol ≈ volume
    vol_check = df["buy_vol"] + df["sell_vol"]
    vol_diff = (vol_check - df["volume"]).abs()
    print(f"   buy_vol + sell_vol vs volume: max diff = {vol_diff.max():.6f}, mean diff = {vol_diff.mean():.6f}")
    print(f"   ✅ Volume split is consistent" if vol_diff.max() < 0.01 else "   ❌ Volume split BROKEN")
    
    # Check delta = buy_vol - sell_vol
    delta_check = df["buy_vol"] - df["sell_vol"]
    delta_diff = (delta_check - df["delta"]).abs()
    print(f"   delta vs buy-sell: max diff = {delta_diff.max():.6f}")
    print(f"   ✅ Delta is consistent" if delta_diff.max() < 0.01 else "   ❌ Delta BROKEN")
    
    # Buy percentage distribution
    print(f"   buy_pct mean: {df['buy_pct'].mean():.4f} (should be ~0.50)")
    print(f"   buy_pct std: {df['buy_pct'].std():.4f}")
    print(f"   buy_pct [25%,50%,75%]: [{df['buy_pct'].quantile(0.25):.3f}, {df['buy_pct'].median():.3f}, {df['buy_pct'].quantile(0.75):.3f}]")
    
    # Sample data
    print(f"\n   Sample rows (first 3):")
    print(df[["open","close","volume","buy_vol","sell_vol","delta","buy_pct"]].head(3).to_string())
    
    # Resample to 4h for tests
    print(f"\n{'='*70}")
    print("2. RESAMPLE TO 4h")
    df4h = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "delta": "sum", "trade_count": "sum",
    }).dropna(subset=["open"])
    df4h["return"] = df4h["close"].pct_change()
    returns = df4h["return"].values
    print(f"   4h bars: {len(df4h):,}")
    print(f"   Date range: {df4h.index[0]} → {df4h.index[-1]}")
    
    # Buy & hold
    total_ret = df4h["close"].iloc[-1] / df4h["close"].iloc[0] - 1
    ann_ret = (1 + total_ret) ** (365.25 / ((df4h.index[-1] - df4h.index[0]).days)) - 1
    print(f"   Buy & hold: {total_ret*100:.1f}% total, {ann_ret*100:.1f}% annualized")
    bh_sharpe = np.mean(returns[1:]) / np.std(returns[1:]) * np.sqrt(252*6)
    print(f"   Buy & hold Sharpe: {bh_sharpe:.3f}")
    
    # Test 3: Perfect signal (uses future returns — SHOULD show massive Sharpe)
    print(f"\n{'='*70}")
    print("3. PERFECT SIGNAL (future returns → proves WF engine works)")
    perfect_sig = np.where(returns > 0, 1, -1)
    perfect_sig[0] = 0
    # Note: WF rolls signal by 1, so this tests if the engine handles it right
    # The "perfect" signal here is actually 1-bar-lagged perfect, which is still good
    wf = walk_forward(returns, perfect_sig, comm=0, ann_factor=252*6)
    print(f"   Perfect (0 comm): Sharpe={wf['sharpe']}, p={wf['p']}, inv={wf['inv']}%")
    wf2 = walk_forward(returns, perfect_sig, comm=0.0006, ann_factor=252*6)
    print(f"   Perfect (6bps):   Sharpe={wf2['sharpe']}, p={wf2['p']}, inv={wf2['inv']}%")
    if wf['sharpe'] > 5:
        print(f"   ✅ WF engine CAN produce high Sharpe when signal has edge")
    else:
        print(f"   ❌ WF engine might be broken — perfect signal should dominate")
    
    # Test 4: Random signal (should be ~0 Sharpe)
    print(f"\n{'='*70}")
    print("4. RANDOM SIGNAL (should be ~0 Sharpe, p~0.50)")
    np.random.seed(42)
    sharpes = []
    pvals = []
    for _ in range(20):
        rand_sig = np.random.choice([-1, 0, 1], size=len(returns), p=[0.3, 0.4, 0.3])
        wf = walk_forward(returns, rand_sig, comm=0.0006, ann_factor=252*6)
        if wf:
            sharpes.append(wf['sharpe'])
            pvals.append(wf['p'])
    print(f"   20 random trials: Sharpe mean={np.mean(sharpes):.3f} std={np.std(sharpes):.3f}")
    print(f"   p-value mean={np.mean(pvals):.3f}")
    print(f"   Significant (p<0.05): {sum(1 for p in pvals if p < 0.05)}/20")
    if abs(np.mean(sharpes)) < 1.0 and sum(1 for p in pvals if p < 0.05) <= 2:
        print(f"   ✅ Random signals correctly show no edge")
    else:
        print(f"   ⚠ Random signals showing unexpected results")
    
    # Test 5: Simple SMA crossover (known working strategy)
    print(f"\n{'='*70}")
    print("5. SMA CROSSOVER (known trend-following signal)")
    close = df4h["close"].values
    for fast, slow in [(10, 50), (20, 100), (50, 200)]:
        sma_fast = pd.Series(close).rolling(fast).mean().values
        sma_slow = pd.Series(close).rolling(slow).mean().values
        sma_sig = np.where(sma_fast > sma_slow, 1, -1)
        sma_sig[:slow] = 0
        wf = walk_forward(returns, sma_sig, comm=0.0006, ann_factor=252*6)
        if wf:
            status = "✅" if wf['p'] < 0.05 else "❌"
            print(f"   {status} SMA({fast},{slow}): Sharpe={wf['sharpe']:.3f} p={wf['p']:.4f} inv={wf['inv']}% trades={wf['trades']}")
    
    # Test 6: Raw delta predictiveness
    print(f"\n{'='*70}")
    print("6. RAW ORDERFLOW PREDICTIVENESS")
    fwd_ret = np.roll(returns, -1)  # next bar return
    fwd_ret[-1] = 0
    
    # Correlation of delta with forward returns
    valid = ~np.isnan(fwd_ret) & ~np.isnan(df4h["delta"].values)
    corr = np.corrcoef(df4h["delta"].values[valid], fwd_ret[valid])[0, 1]
    print(f"   Delta vs next-bar return correlation: {corr:.6f}")
    
    # Buy/sell ratio vs forward returns
    buy_pct = df4h["buy_vol"].values / np.clip(df4h["volume"].values, 1, None)
    corr2 = np.corrcoef(buy_pct[valid], fwd_ret[valid])[0, 1]
    print(f"   Buy% vs next-bar return correlation: {corr2:.6f}")
    
    # Binned analysis
    delta_q = pd.qcut(df4h["delta"], 5, labels=False, duplicates="drop")
    print(f"\n   Delta quintile → mean forward return:")
    for q in range(5):
        mask = delta_q == q
        if mask.sum() > 0:
            mr = fwd_ret[mask].mean() * 100
            print(f"     Q{q} (n={mask.sum()}): {mr:+.4f}%")
    
    # Test 7: Does orderflow data match OHLCV data?
    print(f"\n{'='*70}")
    print("7. CROSS-CHECK: Orderflow OHLCV vs standalone OHLCV")
    ohlcv_path = OHLCV_DIR / "binance_btc_usdt_4h.csv"
    if ohlcv_path.exists():
        df_ohlcv = pd.read_csv(ohlcv_path, parse_dates=["timestamp"])
        # Find overlapping dates
        of_dates = set(df4h.index.date)
        oh_dates = set(df_ohlcv["timestamp"].dt.date)
        overlap = of_dates & oh_dates
        print(f"   Orderflow dates: {len(of_dates)}, OHLCV dates: {len(oh_dates)}, overlap: {len(overlap)}")
        
        # Compare close prices on a few dates
        df_ohlcv_idx = df_ohlcv.set_index("timestamp")
        common = df4h.index.intersection(df_ohlcv_idx.index)
        if len(common) > 0:
            of_close = df4h.loc[common[:5], "close"].values
            oh_close = df_ohlcv_idx.loc[common[:5], "close"].values
            print(f"   Sample close comparison (first 5 overlap):")
            for i in range(min(5, len(common))):
                diff_pct = abs(of_close[i] - oh_close[i]) / oh_close[i] * 100
                match = "✅" if diff_pct < 0.1 else "⚠"
                print(f"     {common[i]}: OF={of_close[i]:.2f} vs OHLCV={oh_close[i]:.2f} ({diff_pct:.3f}%) {match}")
    
    # Test 8: HSAKA lean through same data
    print(f"\n{'='*70}")
    print("8. HSAKA LEAN on orderflow 4h data (should match known results)")
    from scipy.signal import argrelextrema
    
    close = df4h["close"].values
    high = df4h["high"].values if "high" in df4h.columns else close
    low = df4h["low"].values if "low" in df4h.columns else close
    
    # Structure detection (argrelextrema, order=7)
    order = 7
    highs = argrelextrema(high, np.greater_equal, order=order)[0]
    lows = argrelextrema(low, np.less_equal, order=order)[0]
    
    structure = np.zeros(len(close))
    events = sorted([(i, 1) for i in highs] + [(i, -1) for i in lows])
    for idx, trend in events:
        structure[idx:] = trend
    
    # SFP detection
    sfp = np.zeros(len(close))
    lb = 20
    for i in range(lb, len(close)):
        prev_high = np.max(high[i-lb:i])
        prev_low = np.min(low[i-lb:i])
        if high[i] > prev_high and close[i] < prev_high:
            sfp[i] = -2
        elif low[i] < prev_low and close[i] > prev_low:
            sfp[i] = 2
    
    score = structure * 2 + sfp
    hsaka_sig = np.where(score >= 2, 1, np.where(score <= -2, -1, 0))
    
    wf = walk_forward(returns, hsaka_sig, comm=0.0006, ann_factor=252*6)
    if wf:
        status = "✅" if wf['p'] < 0.05 else "❌"
        print(f"   {status} HSAKA Lean (4h): Sharpe={wf['sharpe']:.3f} p={wf['p']:.4f} inv={wf['inv']}% trades={wf['trades']}")
        print(f"   (Expected: ~2.0 Sharpe from prior testing)")
    
    print(f"\n{'='*70}")
    print("SANITY CHECK COMPLETE")

if __name__ == "__main__":
    main()
