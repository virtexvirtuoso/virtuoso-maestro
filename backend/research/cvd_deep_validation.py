"""
CVD Deep Validation — Settling the Confirmation vs Contrarian Debate
====================================================================
Fixes from previous test:
1. NON-OVERLAPPING returns (no inflation of t-stats)
2. Transaction costs (10bps round-trip for perps)
3. Regime-aware analysis (bull vs bear vs chop markets)
4. Multiple assets via price-based proxy (where no orderflow data)
5. Proper strategy PnL simulation (not just mean returns)
6. Bootstrap confidence intervals on Sharpe
7. Comparison: CVD regime vs simple price momentum vs contrarian

Author: Maestro 🎼
Date: 2026-03-07
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

from backend.config.data_paths import BARS_1M_V1

OF_DIR = BARS_1M_V1
OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 10  # 10bps round-trip (5bps each way, typical for perps)


def load_orderflow_1h(asset="btcusdt"):
    """Load orderflow data resampled to 1h."""
    df = pd.read_csv(OF_DIR / f"{asset}_1m.csv",
                     parse_dates=["timestamp"], index_col="timestamp")
    df = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "delta": "sum", "trade_count": "sum",
        "dollar_volume": "sum",
    }).dropna(subset=["open"])
    df["return_1h"] = df["close"].pct_change()
    return df


def load_ohlcv_1h(filename):
    """Load OHLCV data (no orderflow)."""
    df = pd.read_csv(OHLCV_DIR / filename, parse_dates=["timestamp"] if "timestamp" in 
                     pd.read_csv(OHLCV_DIR / filename, nrows=1).columns else [0])
    # Normalize column names
    cols = df.columns.str.lower()
    df.columns = cols
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["return_1h"] = df["close"].pct_change()
    return df


def compute_cvd_regime(df):
    """Compute CVD regime signal (requires orderflow data)."""
    df["cvd"] = df["delta"].cumsum()
    df["cvd_sma20"] = df["cvd"].rolling(20, min_periods=10).mean()
    df["cvd_sma50"] = df["cvd"].rolling(50, min_periods=25).mean()
    df["cvd_regime"] = np.where(df["cvd_sma20"] > df["cvd_sma50"], 1, -1)
    return df


def compute_price_regime(df):
    """Compute price-based momentum regime (proxy for CVD regime)."""
    df["price_sma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["price_sma50"] = df["close"].rolling(50, min_periods=25).mean()
    df["price_regime"] = np.where(df["price_sma20"] > df["price_sma50"], 1, -1)
    return df


def compute_contrarian_delta(df):
    """Compute bar-level contrarian delta signal."""
    delta_mean = df["delta"].rolling(50, min_periods=20).mean()
    delta_std = df["delta"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    df["delta_z"] = (df["delta"] - delta_mean) / delta_std
    df["contrarian"] = -np.sign(df["delta_z"])  # Fade the delta
    return df


def simulate_strategy(df, signal_col, hold_bars=30, cost_bps=COST_BPS):
    """
    Non-overlapping strategy simulation.
    Enter at signal, hold for hold_bars, then re-evaluate.
    Includes transaction costs on position changes.
    """
    signals = df[signal_col].values
    closes = df["close"].values
    n = len(df)
    
    trades = []
    i = 50  # Skip warmup
    
    while i < n - hold_bars:
        sig = signals[i]
        if np.isnan(sig) or sig == 0:
            i += 1
            continue
        
        pos = np.sign(sig)
        entry_price = closes[i]
        exit_price = closes[min(i + hold_bars, n - 1)]
        
        raw_return = (exit_price / entry_price - 1) * pos
        net_return = raw_return - (cost_bps / 10000)  # Subtract costs
        
        trades.append({
            "entry_idx": i,
            "entry_date": str(df.index[i]),
            "exit_date": str(df.index[min(i + hold_bars, n - 1)]),
            "position": int(pos),
            "raw_return": float(raw_return),
            "net_return": float(net_return),
        })
        
        i += hold_bars  # Non-overlapping: jump forward
    
    return trades


def walk_forward_nonoverlap(df, signal_col, hold_bars=30, n_folds=14, cost_bps=COST_BPS):
    """
    Expanding-window walk-forward with NON-OVERLAPPING trades.
    """
    total_bars = len(df)
    date_range = df.index[-1] - df.index[0]
    total_months = date_range.days / 30.44
    bars_per_month = int(total_bars / total_months)
    
    min_train = bars_per_month * 3
    test_size = bars_per_month
    
    available = total_bars - min_train
    max_folds = available // test_size
    n_folds = min(n_folds, max_folds)
    
    if n_folds < 3:
        return {"error": "Not enough data", "n_folds": n_folds}
    
    all_oos_trades = []
    fold_results = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = min(test_start + test_size, total_bars)
        
        if test_end > total_bars:
            break
        
        test_df = df.iloc[test_start:test_end].copy()
        trades = simulate_strategy(test_df, signal_col, hold_bars, cost_bps)
        
        if len(trades) < 2:
            continue
        
        net_rets = [t["net_return"] for t in trades]
        fold_mean = np.mean(net_rets)
        fold_std = np.std(net_rets) if len(net_rets) > 1 else 1e-10
        
        # Annualize: trades_per_year = (365*24/hold_bars)
        trades_per_year = 365 * 24 / hold_bars
        fold_sharpe = (fold_mean / fold_std) * np.sqrt(trades_per_year) if fold_std > 0 else 0
        
        fold_results.append({
            "fold": fold + 1,
            "test_start": str(test_df.index[0]),
            "test_end": str(test_df.index[-1]),
            "n_trades": len(trades),
            "mean_return_pct": float(fold_mean * 100),
            "sharpe": float(fold_sharpe),
            "hit_rate": float(np.mean([r > 0 for r in net_rets])),
            "total_return_pct": float(sum(net_rets) * 100),
        })
        
        all_oos_trades.extend(trades)
    
    if len(all_oos_trades) < 10:
        return {"error": "Too few OOS trades", "n_trades": len(all_oos_trades)}
    
    net_rets = np.array([t["net_return"] for t in all_oos_trades])
    raw_rets = np.array([t["raw_return"] for t in all_oos_trades])
    
    t_stat, p_value = sp_stats.ttest_1samp(net_rets, 0)
    trades_per_year = 365 * 24 / hold_bars
    sharpe = (np.mean(net_rets) / np.std(net_rets)) * np.sqrt(trades_per_year) if np.std(net_rets) > 0 else 0
    
    # Bootstrap Sharpe CI (1000 resamples)
    bootstrap_sharpes = []
    for _ in range(1000):
        sample = np.random.choice(net_rets, size=len(net_rets), replace=True)
        bs = (np.mean(sample) / np.std(sample)) * np.sqrt(trades_per_year) if np.std(sample) > 0 else 0
        bootstrap_sharpes.append(bs)
    sharpe_ci_lo = np.percentile(bootstrap_sharpes, 2.5)
    sharpe_ci_hi = np.percentile(bootstrap_sharpes, 97.5)
    
    # Max drawdown on cumulative returns
    cum = np.cumsum(net_rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = np.min(dd)
    
    # Win/loss stats
    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets <= 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float("inf")
    
    # Long vs short breakdown
    long_rets = net_rets[[t["position"] == 1 for t in all_oos_trades]]
    short_rets = net_rets[[t["position"] == -1 for t in all_oos_trades]]
    
    return {
        "n_folds": len(fold_results),
        "n_trades": len(all_oos_trades),
        "n_long": int(len(long_rets)),
        "n_short": int(len(short_rets)),
        "mean_return_pct": float(np.mean(net_rets) * 100),
        "mean_raw_return_pct": float(np.mean(raw_rets) * 100),
        "cost_drag_pct": float((np.mean(raw_rets) - np.mean(net_rets)) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "sharpe_annualized": float(sharpe),
        "sharpe_ci_95": [float(sharpe_ci_lo), float(sharpe_ci_hi)],
        "hit_rate": float(np.mean(net_rets > 0)),
        "avg_win_pct": float(avg_win * 100),
        "avg_loss_pct": float(avg_loss * 100),
        "profit_factor": float(profit_factor),
        "total_return_pct": float(np.sum(net_rets) * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "long_mean_pct": float(np.mean(long_rets) * 100) if len(long_rets) > 0 else None,
        "short_mean_pct": float(np.mean(short_rets) * 100) if len(short_rets) > 0 else None,
        "folds": fold_results,
    }


def regime_split_analysis(df, signal_col, hold_bars=30):
    """Test signal in different market regimes (bull/bear/chop)."""
    # Define regimes by 60-bar rolling return
    df["regime_ret"] = df["close"].pct_change(60)
    
    results = {}
    for regime_name, condition in [
        ("bull", df["regime_ret"] > 0.05),
        ("bear", df["regime_ret"] < -0.05),
        ("chop", (df["regime_ret"] >= -0.05) & (df["regime_ret"] <= 0.05)),
    ]:
        regime_df = df[condition].copy()
        if len(regime_df) < 200:
            results[regime_name] = {"error": "Too few bars", "n_bars": len(regime_df)}
            continue
        
        trades = simulate_strategy(regime_df, signal_col, hold_bars)
        if len(trades) < 5:
            results[regime_name] = {"error": "Too few trades", "n_trades": len(trades)}
            continue
        
        net_rets = np.array([t["net_return"] for t in trades])
        t_stat, p_val = sp_stats.ttest_1samp(net_rets, 0)
        trades_per_year = 365 * 24 / hold_bars
        sharpe = (np.mean(net_rets) / np.std(net_rets)) * np.sqrt(trades_per_year) if np.std(net_rets) > 0 else 0
        
        results[regime_name] = {
            "n_bars": int(len(regime_df)),
            "n_trades": len(trades),
            "mean_return_pct": float(np.mean(net_rets) * 100),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "sharpe": float(sharpe),
            "hit_rate": float(np.mean(net_rets > 0)),
        }
    
    return results


def main():
    print("=" * 90)
    print("CVD DEEP VALIDATION — NON-OVERLAPPING, WITH COSTS, MULTI-ASSET")
    print(f"Transaction cost: {COST_BPS}bps round-trip")
    print("=" * 90)
    
    all_results = {}
    
    # ══════════════════════════════════════════════════════
    # PART 1: BTC & ETH with actual orderflow data
    # ══════════════════════════════════════════════════════
    print(f"\n{'█' * 90}")
    print("█ PART 1: ORDERFLOW ASSETS (BTC, ETH) — CVD Regime + Contrarian + Price Regime")
    print(f"{'█' * 90}")
    
    for asset in ["btcusdt", "ethusdt"]:
        print(f"\n{'#' * 80}")
        print(f"# {asset.upper()}")
        print(f"{'#' * 80}")
        
        df = load_orderflow_1h(asset)
        df = compute_cvd_regime(df)
        df = compute_price_regime(df)
        df = compute_contrarian_delta(df)
        
        print(f"  Bars: {len(df):,} | {df.index[0]} → {df.index[-1]}")
        
        asset_results = {}
        
        for signal_name, label in [
            ("cvd_regime", "CVD Regime (confirmation)"),
            ("price_regime", "Price Regime (SMA20>SMA50)"),
            ("contrarian", "Bar-Level Contrarian Delta"),
        ]:
            print(f"\n  ━━━ {label} ━━━")
            
            # Walk-forward
            wf = walk_forward_nonoverlap(df, signal_name, hold_bars=30, n_folds=14)
            asset_results[signal_name] = {"walk_forward": wf}
            
            if "error" in wf:
                print(f"  ERROR: {wf['error']}")
                continue
            
            print(f"  WF Results ({wf['n_folds']} folds, {wf['n_trades']} non-overlapping trades):")
            print(f"    Mean return (net):  {wf['mean_return_pct']:+.4f}%")
            print(f"    Mean return (raw):  {wf['mean_raw_return_pct']:+.4f}%")
            print(f"    Cost drag:          {wf['cost_drag_pct']:.4f}%")
            print(f"    t-stat:             {wf['t_stat']:.2f}")
            print(f"    p-value:            {wf['p_value']:.4f}")
            print(f"    Sharpe (ann.):      {wf['sharpe_annualized']:.2f}  [{wf['sharpe_ci_95'][0]:.2f}, {wf['sharpe_ci_95'][1]:.2f}] 95% CI")
            print(f"    Hit rate:           {wf['hit_rate']:.1%}")
            print(f"    Profit factor:      {wf['profit_factor']:.2f}")
            print(f"    Total OOS return:   {wf['total_return_pct']:+.2f}%")
            print(f"    Max drawdown:       {wf['max_drawdown_pct']:.2f}%")
            print(f"    Long trades:        {wf['n_long']} (mean {wf['long_mean_pct']:+.3f}%)" if wf['long_mean_pct'] else "")
            print(f"    Short trades:       {wf['n_short']} (mean {wf['short_mean_pct']:+.3f}%)" if wf['short_mean_pct'] else "")
            
            # Per-fold
            print(f"    Per-fold:")
            pos = 0
            for f in wf["folds"]:
                marker = "✓" if f["sharpe"] > 0 else "✗"
                print(f"      F{f['fold']:>2d}: Sharpe={f['sharpe']:+6.2f} ret={f['total_return_pct']:+6.2f}% "
                      f"hit={f['hit_rate']:.0%} n={f['n_trades']} {marker}")
                if f["sharpe"] > 0:
                    pos += 1
            print(f"    Positive folds: {pos}/{len(wf['folds'])}")
            
            # Regime analysis
            print(f"\n    Regime breakdown:")
            regimes = regime_split_analysis(df, signal_name, hold_bars=30)
            asset_results[signal_name]["regimes"] = regimes
            for rname, rdata in regimes.items():
                if "error" in rdata:
                    print(f"      {rname:>5s}: {rdata['error']}")
                else:
                    sig = "✓" if rdata["p_value"] < 0.05 else "✗"
                    print(f"      {rname:>5s}: Sharpe={rdata['sharpe']:+.2f} t={rdata['t_stat']:+.2f} "
                          f"p={rdata['p_value']:.3f} hit={rdata['hit_rate']:.0%} n={rdata['n_trades']} {sig}")
        
        all_results[asset] = asset_results
    
    # ══════════════════════════════════════════════════════
    # PART 2: Multi-asset with price regime proxy
    # ══════════════════════════════════════════════════════
    print(f"\n\n{'█' * 90}")
    print("█ PART 2: MULTI-ASSET PRICE REGIME (proxy for CVD regime)")
    print(f"{'█' * 90}")
    
    ohlcv_assets = [
        ("binance_btc_usdt_1h.csv", "BTC"),
        ("binance_eth_usdt_1h.csv", "ETH"),
        ("binance_sol_usdt_1h.csv", "SOL"),
        ("binance_sui_usdt_1h.csv", "SUI"),
        ("binance_link_usdt_1h.csv", "LINK"),
        ("binance_avax_usdt_1h.csv", "AVAX"),
        ("binance_inj_usdt_1h.csv", "INJ"),
        ("binance_op_usdt_1h.csv", "OP"),
        ("binance_render_usdt_1h.csv", "RENDER"),
        ("binance_arb_usdt_1h.csv", "ARB"),
        ("binance_tia_usdt_1h.csv", "TIA"),
        ("binance_fet_usdt_1h.csv", "FET"),
        ("binance_tao_usdt_1h.csv", "TAO"),
    ]
    
    multi_results = {}
    
    for filename, label in ohlcv_assets:
        fpath = OHLCV_DIR / filename
        if not fpath.exists():
            print(f"\n  {label}: file not found, skipping")
            continue
        
        try:
            df = load_ohlcv_1h(filename)
            if len(df) < 500:
                print(f"\n  {label}: only {len(df)} bars, skipping")
                continue
            df = compute_price_regime(df)
        except Exception as e:
            print(f"\n  {label}: load error: {e}")
            continue
        
        print(f"\n  ── {label} ({len(df):,} bars, {df.index[0].date()} → {df.index[-1].date()}) ──")
        
        wf = walk_forward_nonoverlap(df, "price_regime", hold_bars=30, n_folds=14)
        multi_results[label] = wf
        
        if "error" in wf:
            print(f"    ERROR: {wf['error']}")
            continue
        
        verdict = "✅" if wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5 else "❌"
        print(f"    Sharpe={wf['sharpe_annualized']:+.2f} [{wf['sharpe_ci_95'][0]:.2f},{wf['sharpe_ci_95'][1]:.2f}] "
              f"t={wf['t_stat']:.2f} p={wf['p_value']:.4f} "
              f"hit={wf['hit_rate']:.0%} n={wf['n_trades']} PF={wf['profit_factor']:.2f} {verdict}")
    
    all_results["multi_asset_price_regime"] = multi_results
    
    # ══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print("FINAL SUMMARY — CVD DEEP VALIDATION")
    print(f"{'=' * 90}")
    
    print(f"\n{'Asset':<10s} {'Signal':<28s} {'Sharpe':>8s} {'95% CI':<16s} {'t':>6s} {'p':>8s} {'Hit%':>6s} {'PF':>6s} {'n':>5s} {'Verdict'}")
    print("-" * 110)
    
    for asset in ["btcusdt", "ethusdt"]:
        for sig, label in [("cvd_regime", "CVD Regime"), ("price_regime", "Price Regime"), ("contrarian", "Contrarian Delta")]:
            wf = all_results[asset].get(sig, {}).get("walk_forward", {})
            if "error" in wf:
                print(f"{asset:<10s} {label:<28s} {'ERROR':>8s}")
                continue
            ci = f"[{wf['sharpe_ci_95'][0]:+.1f}, {wf['sharpe_ci_95'][1]:+.1f}]"
            verdict = "✅ PASS" if wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5 else "❌ FAIL"
            print(f"{asset:<10s} {label:<28s} {wf['sharpe_annualized']:>8.2f} {ci:<16s} {wf['t_stat']:>6.2f} "
                  f"{wf['p_value']:>8.4f} {wf['hit_rate']:>5.0%} {wf['profit_factor']:>6.2f} {wf['n_trades']:>5d} {verdict}")
    
    print(f"\n{'Multi-Asset Price Regime (SMA20>SMA50 on price, 1h)':}")
    print("-" * 110)
    passes = 0
    total = 0
    for label, wf in multi_results.items():
        if "error" in wf:
            continue
        total += 1
        ci = f"[{wf['sharpe_ci_95'][0]:+.1f}, {wf['sharpe_ci_95'][1]:+.1f}]"
        verdict = "✅" if wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5 else "❌"
        if wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5:
            passes += 1
        print(f"{label:<10s} {'Price Regime':<28s} {wf['sharpe_annualized']:>8.2f} {ci:<16s} {wf['t_stat']:>6.2f} "
              f"{wf['p_value']:>8.4f} {wf['hit_rate']:>5.0%} {wf['profit_factor']:>6.2f} {wf['n_trades']:>5d} {verdict}")
    
    print(f"\nMulti-asset pass rate: {passes}/{total}")
    
    # ══════════════════════════════════════════════════════
    # KEY QUESTION: Is confirmation or contrarian the answer?
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("THE VERDICT: CONFIRMATION vs CONTRARIAN")
    print(f"{'=' * 90}")
    
    for asset in ["btcusdt", "ethusdt"]:
        cvd_wf = all_results[asset].get("cvd_regime", {}).get("walk_forward", {})
        contra_wf = all_results[asset].get("contrarian", {}).get("walk_forward", {})
        price_wf = all_results[asset].get("price_regime", {}).get("walk_forward", {})
        
        print(f"\n  {asset.upper()}:")
        if "error" not in cvd_wf:
            print(f"    CVD Regime (confirmation): Sharpe={cvd_wf['sharpe_annualized']:+.2f}, p={cvd_wf['p_value']:.4f}")
        if "error" not in contra_wf:
            print(f"    Contrarian Delta:          Sharpe={contra_wf['sharpe_annualized']:+.2f}, p={contra_wf['p_value']:.4f}")
        if "error" not in price_wf:
            print(f"    Price Regime (proxy):       Sharpe={price_wf['sharpe_annualized']:+.2f}, p={price_wf['p_value']:.4f}")
    
    # Save
    out_json = RESULTS_DIR / "cvd_deep_validation_20260307.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved to {out_json}")
    
    # Save markdown
    out_md = RESULTS_DIR / "cvd_deep_validation_20260307.md"
    with open(out_md, "w") as f:
        f.write("# CVD Deep Validation Results\n\n")
        f.write(f"**Date:** 2026-03-07\n")
        f.write(f"**Methodology:** 14-fold expanding WF, non-overlapping trades, {COST_BPS}bps costs\n")
        f.write(f"**Data:** Jan 2024 – Feb 2026\n\n")
        f.write("## Key Finding\n\n")
        
        btc_cvd = all_results.get("btcusdt", {}).get("cvd_regime", {}).get("walk_forward", {})
        btc_contra = all_results.get("btcusdt", {}).get("contrarian", {}).get("walk_forward", {})
        
        if "error" not in btc_cvd and "error" not in btc_contra:
            if btc_cvd.get("sharpe_annualized", 0) > btc_contra.get("sharpe_annualized", 0):
                f.write("**CVD Regime (confirmation) wins over Contrarian Delta.**\n\n")
            else:
                f.write("**Contrarian Delta wins over CVD Regime.**\n\n")
        
        f.write("See JSON for full results.\n")
    
    print(f"Markdown saved to {out_md}")


if __name__ == "__main__":
    main()
