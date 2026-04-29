"""
CVD Multi-Asset Validation — All Orderflow Assets
===================================================
Extension of cvd_deep_validation.py to test CVD regime signals across
13 assets with actual orderflow data (not price proxy).

Methodology:
1. NON-OVERLAPPING trades (no return inflation)
2. 10bps round-trip transaction costs
3. Walk-forward expanding window, 14 folds
4. Bootstrap 95% CI on Sharpe
5. Regime-aware analysis (bull/bear/chop)
6. Tests: CVD Regime (confirmation), Contrarian Delta, Price Regime (benchmark)

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
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 10

# All assets with orderflow CSVs
ORDERFLOW_ASSETS = [
    "btcusdt", "ethusdt", "solusdt", "suiusdt", "linkusdt",
    "avaxusdt", "injusdt", "opusdt", "arbusdt", "tiausdt",
    "fetusdt", "taousdt", "rndrusdt",
]


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


def compute_cvd_regime(df):
    df["cvd"] = df["delta"].cumsum()
    df["cvd_sma20"] = df["cvd"].rolling(20, min_periods=10).mean()
    df["cvd_sma50"] = df["cvd"].rolling(50, min_periods=25).mean()
    df["cvd_regime"] = np.where(df["cvd_sma20"] > df["cvd_sma50"], 1, -1)
    return df


def compute_price_regime(df):
    df["price_sma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["price_sma50"] = df["close"].rolling(50, min_periods=25).mean()
    df["price_regime"] = np.where(df["price_sma20"] > df["price_sma50"], 1, -1)
    return df


def compute_contrarian_delta(df):
    delta_mean = df["delta"].rolling(50, min_periods=20).mean()
    delta_std = df["delta"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    df["delta_z"] = (df["delta"] - delta_mean) / delta_std
    df["contrarian"] = -np.sign(df["delta_z"])
    return df


def simulate_strategy(df, signal_col, hold_bars=30, cost_bps=COST_BPS):
    signals = df[signal_col].values
    closes = df["close"].values
    n = len(df)
    trades = []
    i = 50
    while i < n - hold_bars:
        sig = signals[i]
        if np.isnan(sig) or sig == 0:
            i += 1
            continue
        pos = np.sign(sig)
        entry_price = closes[i]
        exit_price = closes[min(i + hold_bars, n - 1)]
        raw_return = (exit_price / entry_price - 1) * pos
        net_return = raw_return - (cost_bps / 10000)
        trades.append({
            "entry_idx": i,
            "entry_date": str(df.index[i]),
            "exit_date": str(df.index[min(i + hold_bars, n - 1)]),
            "position": int(pos),
            "raw_return": float(raw_return),
            "net_return": float(net_return),
        })
        i += hold_bars
    return trades


def walk_forward_nonoverlap(df, signal_col, hold_bars=30, n_folds=14, cost_bps=COST_BPS):
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
    
    bootstrap_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        sample = rng.choice(net_rets, size=len(net_rets), replace=True)
        bs = (np.mean(sample) / np.std(sample)) * np.sqrt(trades_per_year) if np.std(sample) > 0 else 0
        bootstrap_sharpes.append(bs)
    sharpe_ci_lo = np.percentile(bootstrap_sharpes, 2.5)
    sharpe_ci_hi = np.percentile(bootstrap_sharpes, 97.5)
    
    cum = np.cumsum(net_rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = np.min(dd)
    
    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets <= 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float("inf")
    
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
    df = df.copy()
    df["regime_ret"] = df["close"].pct_change(60)
    results = {}
    for regime_name, condition in [
        ("bull", df["regime_ret"] > 0.05),
        ("bear", df["regime_ret"] < -0.05),
        ("chop", (df["regime_ret"] >= -0.05) & (df["regime_ret"] <= 0.05)),
    ]:
        regime_df = df[condition].copy()
        if len(regime_df) < 200:
            results[regime_name] = {"error": "Too few bars", "n_bars": int(len(regime_df))}
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


def generate_markdown(all_results):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# CVD Multi-Asset Validation Results")
    lines.append("")
    lines.append("**Date:** 2026-03-07")
    lines.append("**Methodology:** Walk-forward expanding window, non-overlapping trades, 10bps costs")
    lines.append("**Data:** Binance aggTrades → 1min orderflow → 1h bars, Jan 2024 – Feb 2026")
    lines.append(f"**Assets tested:** {len(ORDERFLOW_ASSETS)}")
    lines.append("")
    
    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Asset | Signal | Sharpe | 95% CI | t-stat | p-value | Hit% | PF | Trades | Verdict |")
    lines.append("|-------|--------|--------|--------|--------|---------|------|----|--------|---------|")
    
    passes = 0
    total_tests = 0
    asset_passes = {}
    
    for asset in ORDERFLOW_ASSETS:
        if asset not in all_results:
            continue
        asset_passes[asset] = False
        for sig_key, sig_label in [("cvd_regime", "CVD Regime"), ("contrarian", "Contrarian"), ("price_regime", "Price Regime")]:
            wf = all_results[asset].get(sig_key, {}).get("walk_forward", {})
            if "error" in wf:
                lines.append(f"| {asset.upper()} | {sig_label} | — | — | — | — | — | — | — | ⚠️ {wf.get('error','')} |")
                continue
            total_tests += 1
            ci = f"[{wf['sharpe_ci_95'][0]:+.1f}, {wf['sharpe_ci_95'][1]:+.1f}]"
            passed = wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5
            verdict = "✅ PASS" if passed else "❌ FAIL"
            if passed:
                passes += 1
                asset_passes[asset] = True
            lines.append(f"| {asset.upper()} | {sig_label} | {wf['sharpe_annualized']:+.2f} | {ci} | {wf['t_stat']:.2f} | {wf['p_value']:.4f} | {wf['hit_rate']:.0%} | {wf['profit_factor']:.2f} | {wf['n_trades']} | {verdict} |")
    
    lines.append("")
    lines.append(f"**Pass rate:** {passes}/{total_tests} tests ({passes/total_tests*100:.0f}%)" if total_tests > 0 else "")
    
    n_asset_pass = sum(1 for v in asset_passes.values() if v)
    lines.append(f"**Assets with at least one passing signal:** {n_asset_pass}/{len(asset_passes)}")
    lines.append("")
    
    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Check if CVD regime or contrarian is better
    cvd_sharpes = []
    contra_sharpes = []
    price_sharpes = []
    for asset in ORDERFLOW_ASSETS:
        if asset not in all_results:
            continue
        cvd_wf = all_results[asset].get("cvd_regime", {}).get("walk_forward", {})
        contra_wf = all_results[asset].get("contrarian", {}).get("walk_forward", {})
        price_wf = all_results[asset].get("price_regime", {}).get("walk_forward", {})
        if "error" not in cvd_wf:
            cvd_sharpes.append(cvd_wf["sharpe_annualized"])
        if "error" not in contra_wf:
            contra_sharpes.append(contra_wf["sharpe_annualized"])
        if "error" not in price_wf:
            price_sharpes.append(price_wf["sharpe_annualized"])
    
    if cvd_sharpes:
        lines.append(f"- **CVD Regime avg Sharpe:** {np.mean(cvd_sharpes):+.2f} (across {len(cvd_sharpes)} assets)")
    if contra_sharpes:
        lines.append(f"- **Contrarian Delta avg Sharpe:** {np.mean(contra_sharpes):+.2f} (across {len(contra_sharpes)} assets)")
    if price_sharpes:
        lines.append(f"- **Price Regime avg Sharpe:** {np.mean(price_sharpes):+.2f} (across {len(price_sharpes)} assets)")
    
    lines.append("")
    if cvd_sharpes and contra_sharpes:
        if np.mean(cvd_sharpes) > np.mean(contra_sharpes):
            lines.append("**Verdict:** CVD Regime (confirmation) outperforms Contrarian Delta on average.")
        else:
            lines.append("**Verdict:** Contrarian Delta outperforms CVD Regime on average.")
    
    if passes == 0:
        lines.append("")
        lines.append("### ⚠️ NULL RESULT CONFIRMED")
        lines.append("No signal passes statistical significance (p<0.05) with meaningful Sharpe (>0.5) after costs.")
        lines.append("CVD-based signals do not provide tradeable alpha with non-overlapping trades and realistic costs.")
    
    lines.append("")
    lines.append("## Regime Breakdown (select assets)")
    lines.append("")
    for asset in ["btcusdt", "ethusdt", "solusdt"]:
        if asset not in all_results:
            continue
        regimes = all_results[asset].get("cvd_regime", {}).get("regimes", {})
        if not regimes:
            continue
        lines.append(f"### {asset.upper()} — CVD Regime by Market Regime")
        lines.append("")
        lines.append("| Regime | Sharpe | t-stat | p-value | Hit% | Trades |")
        lines.append("|--------|--------|--------|---------|------|--------|")
        for rname in ["bull", "bear", "chop"]:
            rd = regimes.get(rname, {})
            if "error" in rd:
                lines.append(f"| {rname} | — | — | — | — | {rd.get('error','')} |")
            else:
                lines.append(f"| {rname} | {rd['sharpe']:+.2f} | {rd['t_stat']:.2f} | {rd['p_value']:.4f} | {rd['hit_rate']:.0%} | {rd['n_trades']} |")
        lines.append("")
    
    lines.append("---")
    lines.append("*Generated by cvd_multiasset_validation.py*")
    
    return "\n".join(lines)


def main():
    print("=" * 90)
    print("CVD MULTI-ASSET VALIDATION — ALL ORDERFLOW ASSETS")
    print(f"Transaction cost: {COST_BPS}bps | Hold: 30 bars (30h)")
    print("=" * 90)
    
    all_results = {}
    
    for asset in ORDERFLOW_ASSETS:
        csv_path = OF_DIR / f"{asset}_1m.csv"
        if not csv_path.exists():
            print(f"\n  ⚠️  {asset.upper()}: CSV not found, skipping")
            continue
        
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
            
            wf = walk_forward_nonoverlap(df, signal_name, hold_bars=30, n_folds=14)
            
            # Regime analysis
            regimes = regime_split_analysis(df, signal_name, hold_bars=30)
            
            asset_results[signal_name] = {"walk_forward": wf, "regimes": regimes}
            
            if "error" in wf:
                print(f"  ERROR: {wf['error']}")
                continue
            
            verdict = "✅ PASS" if wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5 else "❌ FAIL"
            print(f"  {wf['n_folds']} folds, {wf['n_trades']} trades")
            print(f"  Sharpe={wf['sharpe_annualized']:+.2f} [{wf['sharpe_ci_95'][0]:.2f},{wf['sharpe_ci_95'][1]:.2f}] "
                  f"t={wf['t_stat']:.2f} p={wf['p_value']:.4f} "
                  f"hit={wf['hit_rate']:.0%} PF={wf['profit_factor']:.2f} {verdict}")
            print(f"  Total OOS return: {wf['total_return_pct']:+.2f}% | MaxDD: {wf['max_drawdown_pct']:.2f}%")
        
        all_results[asset] = asset_results
    
    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    
    print(f"\n{'Asset':<12s} {'Signal':<26s} {'Sharpe':>7s} {'t':>6s} {'p':>8s} {'Hit%':>5s} {'PF':>5s} {'n':>4s} {'Verdict'}")
    print("-" * 90)
    
    passes = 0
    total = 0
    for asset in ORDERFLOW_ASSETS:
        if asset not in all_results:
            continue
        for sig, label in [("cvd_regime", "CVD Regime"), ("contrarian", "Contrarian"), ("price_regime", "Price Regime")]:
            wf = all_results[asset].get(sig, {}).get("walk_forward", {})
            if "error" in wf:
                continue
            total += 1
            passed = wf["p_value"] < 0.05 and wf["sharpe_annualized"] > 0.5
            if passed:
                passes += 1
            v = "✅" if passed else "❌"
            print(f"{asset:<12s} {label:<26s} {wf['sharpe_annualized']:>7.2f} {wf['t_stat']:>6.2f} {wf['p_value']:>8.4f} "
                  f"{wf['hit_rate']:>4.0%} {wf['profit_factor']:>5.2f} {wf['n_trades']:>4d} {v}")
    
    print(f"\nPass rate: {passes}/{total} ({passes/total*100:.0f}%)" if total > 0 else "")
    
    # Save JSON
    out_json = RESULTS_DIR / "cvd_multiasset_validation_20260307.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON: {out_json}")
    
    # Save markdown
    md_content = generate_markdown(all_results)
    out_md = RESULTS_DIR / "cvd_multiasset_validation_20260307.md"
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"Markdown: {out_md}")
    
    # Copy to Obsidian
    obsidian_path = Path.home() / "virtuoso-vault/04-Trading/Signals/cvd-multiasset-validation-2026-03-07.md"
    obsidian_path.parent.mkdir(parents=True, exist_ok=True)
    with open(obsidian_path, "w") as f:
        f.write(md_content)
    print(f"Obsidian: {obsidian_path}")


if __name__ == "__main__":
    main()
