"""
BTC On-Chain Signal Backtester
==============================
Computes on-chain metric proxies from price data (using BTC Wiz estimation methods)
and backtests 4 strategies with walk-forward validation.

Strategies:
1. MVRV Z-Score Regime
2. Reserve Risk Trend  
3. Multi-Signal Composite
4. On-Chain + M2 Combo
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
OHLCV_PATH = DATA_DIR / "ohlcv" / "binance_btc_usdt_1d.csv"
M2_PATH = DATA_DIR / "macro" / "M2SL.parquet"
OUTPUT_PATH = DATA_DIR / "backtest_results" / "onchain_signals.json"

COMMISSION_BPS = 20
N_FOLDS = 10
N_PERMUTATIONS = 200


# ── On-Chain Metric Estimation (from BTC Wiz formulas) ──────────────────────

def estimate_realized_cap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Estimate realized cap via volume-weighted EWM price."""
    if volume is not None and len(volume) > 0:
        weighted = price * volume
        total_w = volume.expanding().sum()
        realized_price = weighted.expanding().sum() / total_w.replace(0, np.nan)
    else:
        realized_price = price.ewm(span=365, adjust=False).mean()
    days = np.arange(len(price))
    supply = 21_000_000 * (1 - np.exp(-days / 1000))
    supply = np.minimum(supply, 21_000_000)
    return realized_price * supply


def compute_onchain_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all on-chain proxy metrics from OHLCV data."""
    price = df["close"]
    volume = df["volume"]
    
    # Market cap estimate (price * estimated supply)
    days = np.arange(len(price))
    supply = pd.Series(21_000_000 * (1 - np.exp(-days / 1000)), index=price.index)
    supply = supply.clip(upper=21_000_000)
    market_cap = price * supply
    
    # Realized cap
    realized_cap = estimate_realized_cap(price, volume)
    
    metrics = pd.DataFrame(index=df.index)
    
    # 1. MVRV Z-Score
    diff = market_cap - realized_cap
    std = market_cap.rolling(365, min_periods=30).std()
    metrics["mvrv_zscore"] = diff / std.replace(0, np.nan)
    
    # 2. MVRV Ratio
    metrics["mvrv_ratio"] = market_cap / realized_cap.replace(0, np.nan)
    
    # 3. Reserve Risk (inverse volatility proxy)
    vol = price.pct_change().rolling(365, min_periods=30).std()
    confidence = 1 / vol.replace(0, np.nan)
    confidence = confidence.clip(lower=0)
    hodl_bank = confidence.cumsum()
    hodl_bank_norm = hodl_bank / hodl_bank.expanding().max()
    reserve_risk = price / (hodl_bank_norm * price.rolling(365, min_periods=30).mean()).replace(0, np.nan)
    metrics["reserve_risk"] = reserve_risk / reserve_risk.expanding().median().replace(0, np.nan) * 0.005
    
    # 4. SOPR estimate
    cost_basis = price.ewm(span=155, adjust=False).mean()
    metrics["sopr"] = price / cost_basis.replace(0, np.nan)
    
    # 5. Pi Cycle Top
    sma_111 = price.rolling(111, min_periods=80).mean()
    sma_350x2 = price.rolling(350, min_periods=280).mean() * 2
    metrics["pi_cycle_ratio"] = sma_111 / sma_350x2.replace(0, np.nan)
    metrics["pi_cycle_top"] = (sma_111 > sma_350x2).astype(int)
    
    # 6. Puell Multiple proxy (use daily return magnitude as revenue proxy)
    daily_rev_proxy = (price * volume).fillna(0)
    rev_ma = daily_rev_proxy.rolling(365, min_periods=30).mean()
    metrics["puell_multiple"] = daily_rev_proxy / rev_ma.replace(0, np.nan)
    
    # 7. 2-Year MA Multiplier position
    ma_730 = price.rolling(730, min_periods=365).mean()
    ma_730_upper = ma_730 * 5
    metrics["two_year_ma_pos"] = (price - ma_730) / (ma_730_upper - ma_730).replace(0, np.nan)
    metrics["two_year_ma_pos"] = metrics["two_year_ma_pos"].clip(0, 1)
    
    # 8. NUPL estimate = (market_cap - realized_cap) / market_cap
    metrics["nupl"] = (market_cap - realized_cap) / market_cap.replace(0, np.nan)
    
    return metrics


# ── Trading Strategies ───────────────────────────────────────────────────────

def strategy_mvrv_regime(metrics: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: MVRV Z-Score Regime
    Z < 0: full long, Z > 7: flat, linear scale between.
    """
    z = metrics["mvrv_zscore"].copy()
    # Position: 1 at z<=0, 0 at z>=7, linear between
    pos = 1.0 - z.clip(0, 7) / 7.0
    return pos.fillna(0)


def strategy_reserve_risk(metrics: pd.DataFrame) -> pd.Series:
    """
    Strategy 2: Reserve Risk Trend
    Low reserve risk = accumulate, high = distribute.
    Use SMA crossover on reserve risk.
    """
    rr = metrics["reserve_risk"].copy()
    rr_fast = rr.rolling(30, min_periods=10).mean()
    rr_slow = rr.rolling(90, min_periods=30).mean()
    # Falling RR (fast < slow) = accumulation = long
    # Rising RR (fast > slow) = distribution = flat
    pos = (rr_fast < rr_slow).astype(float)
    return pos.fillna(0)


def strategy_composite(metrics: pd.DataFrame) -> pd.Series:
    """
    Strategy 3: Multi-Signal Composite
    Combine MVRV, Reserve Risk, SOPR, NUPL, 2Y MA into [-1, 1] composite.
    """
    signals = pd.DataFrame(index=metrics.index)
    
    # MVRV Z-Score signal: Z<0 -> +1, Z>7 -> -1
    z = metrics["mvrv_zscore"]
    signals["mvrv"] = (1.0 - z.clip(-1, 8) / 3.5).clip(-1, 1)
    
    # SOPR signal: <1 = bullish (capitulation), >1.05 = bearish (profit taking)
    sopr = metrics["sopr"]
    signals["sopr"] = (1.025 - sopr).clip(-1, 1) * 10  # scale
    signals["sopr"] = signals["sopr"].clip(-1, 1)
    
    # NUPL signal: <0 = bullish, >0.75 = bearish
    nupl = metrics["nupl"]
    signals["nupl"] = (0.375 - nupl).clip(-0.5, 0.5) * 2
    
    # 2Y MA position: <0.2 = bullish, >0.8 = bearish
    pos2y = metrics["two_year_ma_pos"]
    signals["two_year"] = (0.5 - pos2y).clip(-1, 1)
    
    # Reserve Risk: low = bullish
    rr = metrics["reserve_risk"]
    rr_pctile = rr.rolling(365, min_periods=30).rank(pct=True)
    signals["rr"] = (0.5 - rr_pctile).clip(-1, 1) * 2
    
    # Equal weight composite
    composite = signals.mean(axis=1)
    # Map to position: composite > 0 = long, < 0 = flat (long-only)
    position = composite.clip(0, 1)
    return position.fillna(0)


def strategy_onchain_plus_m2(metrics: pd.DataFrame, m2_accel: pd.Series) -> pd.Series:
    """
    Strategy 4: On-Chain + M2 Combo
    Best on-chain signal (MVRV regime) combined with M2 acceleration.
    """
    # On-chain component (MVRV regime)
    onchain = strategy_mvrv_regime(metrics)
    
    # M2 acceleration signal: positive accel = bullish
    m2_sig = (m2_accel > 0).astype(float)
    
    # Combine: average of both, require at least one to be positive
    combined = (onchain + m2_sig) / 2.0
    return combined.fillna(0)


# ── Backtest Engine ──────────────────────────────────────────────────────────

def apply_commission(returns: pd.Series, positions: pd.Series, bps: int = COMMISSION_BPS) -> pd.Series:
    """Deduct commission on position changes."""
    trades = positions.diff().abs().fillna(0)
    cost = trades * (bps / 10_000)
    return returns - cost


def backtest_strategy(price: pd.Series, positions: pd.Series) -> dict:
    """Run backtest given price series and position series (0-1 scale, long only)."""
    # Shift positions by 1 to avoid lookahead
    pos = positions.shift(1).fillna(0)
    
    daily_ret = price.pct_change().fillna(0)
    strat_ret = daily_ret * pos
    strat_ret = apply_commission(strat_ret, pos)
    
    cum = (1 + strat_ret).cumprod()
    bnh = (1 + daily_ret).cumprod()
    
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    bnh_ret = bnh.iloc[-1] - 1 if len(bnh) > 0 else 0
    
    # Sharpe
    if strat_ret.std() > 0:
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(365)
    else:
        sharpe = 0
    
    # Max drawdown
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    # Win rate
    trading_days = strat_ret[pos.shift(1).fillna(0) > 0]
    win_rate = (trading_days > 0).mean() if len(trading_days) > 0 else 0
    
    # Per-year returns
    yearly = {}
    for year in sorted(strat_ret.index.year.unique()):
        yr_ret = strat_ret[strat_ret.index.year == year]
        yearly[str(year)] = round(float((1 + yr_ret).prod() - 1) * 100, 2)
    
    return {
        "total_return_pct": round(float(total_ret) * 100, 2),
        "buy_hold_pct": round(float(bnh_ret) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "win_rate": round(float(win_rate), 4),
        "num_days": len(strat_ret),
        "yearly_returns": yearly,
    }


# ── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward_test(price: pd.Series, strategy_fn, metrics: pd.DataFrame, 
                      m2_accel: pd.Series = None, n_folds: int = N_FOLDS) -> dict:
    """
    Expanding window walk-forward test.
    Each fold: train on expanding window, test on next chunk.
    """
    n = len(price)
    min_train = n // (n_folds + 1)  # Minimum training size
    fold_size = (n - min_train) // n_folds
    
    oos_results = []
    all_oos_returns = []
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        
        if test_end <= train_end:
            break
        
        test_metrics = metrics.iloc[train_end:test_end]
        test_price = price.iloc[train_end:test_end]
        
        if len(test_price) < 5:
            continue
        
        if m2_accel is not None:
            test_m2 = m2_accel.iloc[train_end:test_end]
            positions = strategy_fn(test_metrics, test_m2)
        else:
            positions = strategy_fn(test_metrics)
        
        result = backtest_strategy(test_price, positions)
        result["fold"] = fold
        result["test_start"] = str(test_price.index[0].date())
        result["test_end"] = str(test_price.index[-1].date())
        oos_results.append(result)
        
        # Collect OOS returns for permutation test
        pos = positions.shift(1).fillna(0)
        daily_ret = test_price.pct_change().fillna(0)
        strat_ret = daily_ret * pos
        strat_ret = apply_commission(strat_ret, pos)
        all_oos_returns.append(strat_ret)
    
    # Aggregate OOS metrics
    if not oos_results:
        return {"error": "No valid folds"}
    
    avg_ret = np.mean([r["total_return_pct"] for r in oos_results])
    avg_sharpe = np.mean([r["sharpe"] for r in oos_results])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in oos_results])
    
    return {
        "folds": oos_results,
        "avg_oos_return_pct": round(float(avg_ret), 2),
        "avg_oos_sharpe": round(float(avg_sharpe), 3),
        "avg_oos_max_dd_pct": round(float(avg_dd), 2),
        "all_oos_returns": pd.concat(all_oos_returns) if all_oos_returns else pd.Series(dtype=float),
    }


# ── Permutation Test ────────────────────────────────────────────────────────

def permutation_test(price: pd.Series, positions: pd.Series, 
                     observed_sharpe: float, n_perms: int = N_PERMUTATIONS) -> dict:
    """Random permutation test: shuffle signal timing, measure if observed Sharpe is significant."""
    daily_ret = price.pct_change().fillna(0)
    pos = positions.shift(1).fillna(0)
    
    perm_sharpes = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_perms):
        # Shuffle positions randomly
        shuffled = pos.copy()
        shuffled[:] = rng.permutation(pos.values)
        
        strat_ret = daily_ret * shuffled
        strat_ret = apply_commission(strat_ret, shuffled)
        
        if strat_ret.std() > 0:
            s = strat_ret.mean() / strat_ret.std() * np.sqrt(365)
        else:
            s = 0
        perm_sharpes.append(s)
    
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).mean()
    
    return {
        "observed_sharpe": round(float(observed_sharpe), 3),
        "p_value": round(float(p_value), 4),
        "significant_5pct": bool(p_value < 0.05),
        "significant_1pct": bool(p_value < 0.01),
        "perm_mean_sharpe": round(float(perm_sharpes.mean()), 3),
        "perm_95th": round(float(np.percentile(perm_sharpes, 95)), 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BTC On-Chain Signal Backtest")
    print("=" * 70)
    
    # Load BTC OHLCV
    df = pd.read_csv(OHLCV_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    print(f"BTC data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    
    # Load M2 data
    m2 = pd.read_parquet(M2_PATH)
    m2 = m2["M2SL"].resample("D").ffill()  # Forward-fill monthly to daily
    m2_yoy = m2.pct_change(365)  # YoY change
    m2_accel = m2_yoy.diff(30)   # 30-day acceleration
    # Align to BTC index
    m2_accel = m2_accel.reindex(df.index, method="ffill")
    print(f"M2 data aligned: {m2_accel.notna().sum()} valid days")
    
    # Compute on-chain metrics
    print("\nComputing on-chain metric proxies...")
    metrics = compute_onchain_metrics(df)
    print(f"Metrics computed: {list(metrics.columns)}")
    print(f"Valid MVRV Z-Score days: {metrics['mvrv_zscore'].notna().sum()}")
    
    price = df["close"]
    
    results = {}
    
    # ── Strategy 1: MVRV Z-Score Regime ──
    print("\n" + "-" * 50)
    print("Strategy 1: MVRV Z-Score Regime")
    print("-" * 50)
    pos1 = strategy_mvrv_regime(metrics)
    full1 = backtest_strategy(price, pos1)
    wf1 = walk_forward_test(price, strategy_mvrv_regime, metrics)
    
    # Permutation test on full sample
    perm1 = permutation_test(price, pos1, full1["sharpe"])
    
    print(f"  Full: {full1['total_return_pct']}% (B&H: {full1['buy_hold_pct']}%)")
    print(f"  Sharpe: {full1['sharpe']}, Max DD: {full1['max_drawdown_pct']}%")
    print(f"  WF Avg OOS: {wf1['avg_oos_return_pct']}%, Sharpe: {wf1['avg_oos_sharpe']}")
    print(f"  Permutation p={perm1['p_value']} (sig@5%: {perm1['significant_5pct']})")
    print(f"  Yearly: {full1['yearly_returns']}")
    
    results["mvrv_regime"] = {
        "full_sample": full1,
        "walk_forward": {k: v for k, v in wf1.items() if k != "all_oos_returns"},
        "permutation_test": perm1,
    }
    
    # ── Strategy 2: Reserve Risk Trend ──
    print("\n" + "-" * 50)
    print("Strategy 2: Reserve Risk Trend")
    print("-" * 50)
    pos2 = strategy_reserve_risk(metrics)
    full2 = backtest_strategy(price, pos2)
    wf2 = walk_forward_test(price, strategy_reserve_risk, metrics)
    perm2 = permutation_test(price, pos2, full2["sharpe"])
    
    print(f"  Full: {full2['total_return_pct']}% (B&H: {full2['buy_hold_pct']}%)")
    print(f"  Sharpe: {full2['sharpe']}, Max DD: {full2['max_drawdown_pct']}%")
    print(f"  WF Avg OOS: {wf2['avg_oos_return_pct']}%, Sharpe: {wf2['avg_oos_sharpe']}")
    print(f"  Permutation p={perm2['p_value']} (sig@5%: {perm2['significant_5pct']})")
    print(f"  Yearly: {full2['yearly_returns']}")
    
    results["reserve_risk_trend"] = {
        "full_sample": full2,
        "walk_forward": {k: v for k, v in wf2.items() if k != "all_oos_returns"},
        "permutation_test": perm2,
    }
    
    # ── Strategy 3: Multi-Signal Composite ──
    print("\n" + "-" * 50)
    print("Strategy 3: Multi-Signal Composite")
    print("-" * 50)
    pos3 = strategy_composite(metrics)
    full3 = backtest_strategy(price, pos3)
    wf3 = walk_forward_test(price, strategy_composite, metrics)
    perm3 = permutation_test(price, pos3, full3["sharpe"])
    
    print(f"  Full: {full3['total_return_pct']}% (B&H: {full3['buy_hold_pct']}%)")
    print(f"  Sharpe: {full3['sharpe']}, Max DD: {full3['max_drawdown_pct']}%")
    print(f"  WF Avg OOS: {wf3['avg_oos_return_pct']}%, Sharpe: {wf3['avg_oos_sharpe']}")
    print(f"  Permutation p={perm3['p_value']} (sig@5%: {perm3['significant_5pct']})")
    print(f"  Yearly: {full3['yearly_returns']}")
    
    results["composite"] = {
        "full_sample": full3,
        "walk_forward": {k: v for k, v in wf3.items() if k != "all_oos_returns"},
        "permutation_test": perm3,
    }
    
    # ── Strategy 4: On-Chain + M2 Combo ──
    print("\n" + "-" * 50)
    print("Strategy 4: On-Chain + M2 Combo")
    print("-" * 50)
    pos4 = strategy_onchain_plus_m2(metrics, m2_accel)
    full4 = backtest_strategy(price, pos4)
    
    def strat4_fn(m, m2a):
        return strategy_onchain_plus_m2(m, m2a)
    wf4 = walk_forward_test(price, strat4_fn, metrics, m2_accel=m2_accel)
    perm4 = permutation_test(price, pos4, full4["sharpe"])
    
    print(f"  Full: {full4['total_return_pct']}% (B&H: {full4['buy_hold_pct']}%)")
    print(f"  Sharpe: {full4['sharpe']}, Max DD: {full4['max_drawdown_pct']}%")
    print(f"  WF Avg OOS: {wf4['avg_oos_return_pct']}%, Sharpe: {wf4['avg_oos_sharpe']}")
    print(f"  Permutation p={perm4['p_value']} (sig@5%: {perm4['significant_5pct']})")
    print(f"  Yearly: {full4['yearly_returns']}")
    
    results["onchain_plus_m2"] = {
        "full_sample": full4,
        "walk_forward": {k: v for k, v in wf4.items() if k != "all_oos_returns"},
        "permutation_test": perm4,
    }
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        fs = res["full_sample"]
        wf = res["walk_forward"]
        pt = res["permutation_test"]
        sig = "✅" if pt["significant_5pct"] else "❌"
        print(f"  {name:25s} | Full: {fs['total_return_pct']:>8.1f}% | "
              f"WF: {wf['avg_oos_return_pct']:>7.1f}% | "
              f"Sharpe: {fs['sharpe']:>5.2f} | p={pt['p_value']:.3f} {sig}")
    
    # ── Save Results ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "btc_data_range": f"{df.index[0].date()} to {df.index[-1].date()}",
            "num_days": len(df),
            "commission_bps": COMMISSION_BPS,
            "walk_forward_folds": N_FOLDS,
            "permutation_tests": N_PERMUTATIONS,
            "note": "On-chain metrics are PROXIES estimated from price/volume data. "
                    "Real UTXO-based metrics (Glassnode, CryptoQuant) would be more accurate.",
        },
        "strategies": results,
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
