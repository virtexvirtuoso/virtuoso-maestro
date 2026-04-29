#!/usr/bin/env python3
"""V4 Honest System — Expanded Portfolio Backtest"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Discover all USDT daily files + BONK/PEPE ──
def discover_assets():
    assets = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith("_1d.csv"):
            continue
        path = os.path.join(DATA_DIR, f)
        if "usdt_1d" in f and f.startswith("binance_"):
            sym = f.replace("binance_", "").replace("_usdt_1d.csv", "").upper()
            assets[sym] = path
        elif f in ("1000BONK_1d.csv", "1000PEPE_1d.csv"):
            sym = f.replace("_1d.csv", "")
            assets[sym] = path
    return assets

def load_asset(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    return df

# ── Trailing stop calculation ──
PRESET_STOPS = {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08}

def get_trailing_stop_pct(df, symbol):
    if symbol in PRESET_STOPS:
        return PRESET_STOPS[symbol]
    atr20 = df["close"].rolling(20).apply(
        lambda x: np.mean([df["high"].iloc[x.index[-20]:x.index[-1]+1].values - df["low"].iloc[x.index[-20]:x.index[-1]+1].values]) if len(x)==20 else np.nan
    )
    # Simpler: use true range
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                               abs(df["low"] - df["close"].shift(1))))
    atr20 = tr.rolling(20).mean()
    pct = (2 * atr20 / df["close"]).median()
    return np.clip(pct, 0.05, 0.20)

# ── V4d single-asset backtest ──
def v4d_backtest(df, symbol, trailing_stop_pct=None):
    """Returns daily returns series (0 when flat)."""
    close = df["close"].values
    high = df["high"].values
    n = len(close)
    
    if trailing_stop_pct is None:
        tr = np.maximum(high - df["low"].values,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(df["low"].values - np.roll(close, 1))))
        tr[0] = high[0] - df["low"].values[0]
        atr20 = pd.Series(tr).rolling(20).mean().values
        med_pct = np.nanmedian(2 * atr20 / close)
        trailing_stop_pct = np.clip(med_pct, 0.05, 0.20)
    
    # SMA50
    sma50 = pd.Series(close).rolling(50).mean().values
    
    # 30d realized vol (annualized)
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0], log_ret])
    vol30 = pd.Series(log_ret).rolling(30).std().values * np.sqrt(365)
    
    # Signal: long when close > sma50, shifted by 1
    signal = np.zeros(n)
    for i in range(50, n):
        signal[i] = 1.0 if close[i] > sma50[i] else 0.0
    
    # Shift signal by 1 (no look-ahead)
    signal = np.roll(signal, 1)
    signal[0] = 0.0
    
    # Daily returns
    daily_ret = np.zeros(n)
    daily_ret[1:] = close[1:] / close[:-1] - 1
    
    # Apply trailing stop
    position = np.zeros(n)
    peak = close[0]
    stopped = False
    
    for i in range(1, n):
        if signal[i] > 0:
            if stopped:
                # Re-enter only if signal is fresh (close > sma50 on prev bar)
                if close[i-1] > sma50[i-1] if i-1 >= 50 else False:
                    stopped = False
                    peak = close[i]
                    pos = 1.0
                else:
                    pos = 0.0
            else:
                pos = 1.0
                if close[i] > peak:
                    peak = close[i]
                # Check trailing stop
                drawdown_from_peak = (peak - close[i]) / peak
                if drawdown_from_peak >= trailing_stop_pct:
                    pos = 0.0
                    stopped = True
        else:
            pos = 0.0
            stopped = False
            peak = close[i]
        
        # Vol ceiling: halve if vol > 80%
        if vol30[i] > 0.80 and not np.isnan(vol30[i]):
            pos *= 0.5
        
        position[i] = pos
    
    strat_ret = position * daily_ret
    return pd.Series(strat_ret, index=df["timestamp"])

# ── Walk-forward (expanding, 14 folds) ──
def walk_forward_14(df, symbol, trailing_stop_pct=None):
    """Expanding walk-forward with 14 folds. Returns OOS returns concat."""
    n = len(df)
    min_train = max(100, n // 15)  # minimum training size
    fold_size = (n - min_train) // 14
    if fold_size < 10:
        return None, None
    
    oos_rets = []
    for fold in range(14):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        # We don't optimize params - just run V4d on expanding window
        # But we need full history for SMA50, so run on full df up to test_end
        sub_df = df.iloc[:test_end].copy().reset_index(drop=True)
        rets = v4d_backtest(sub_df, symbol, trailing_stop_pct)
        # Take only OOS portion
        oos_portion = rets.iloc[train_end:test_end]
        oos_rets.append(oos_portion)
    
    if not oos_rets:
        return None, None
    
    all_oos = pd.concat(oos_rets)
    sharpe = all_oos.mean() / all_oos.std() * np.sqrt(365) if all_oos.std() > 0 else 0
    return all_oos, sharpe

# ── Portfolio backtest ──
def portfolio_backtest(asset_returns_dict, apply_dd_breaker=True):
    """Equal-weight portfolio from dict of {symbol: daily_returns_series}."""
    if not asset_returns_dict:
        return {}
    
    # Align all to common dates
    combined = pd.DataFrame(asset_returns_dict)
    combined = combined.dropna(how='all')
    combined = combined.fillna(0)
    
    # Equal weight
    port_ret = combined.mean(axis=1)
    
    # DD breaker: go flat if portfolio DD > 25%
    if apply_dd_breaker:
        cum = (1 + port_ret).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        breaker_active = False
        adjusted = port_ret.copy()
        for i in range(len(port_ret)):
            if dd.iloc[i] < -0.25:
                breaker_active = True
            if breaker_active:
                adjusted.iloc[i] = 0.0
                # Reset when DD recovers above -15%
                if dd.iloc[i] > -0.15:
                    breaker_active = False
            # Recalc cum after adjustment
            if i < len(port_ret) - 1:
                cum.iloc[i+1] = cum.iloc[i] * (1 + adjusted.iloc[i+1]) if i+1 < len(cum) else cum.iloc[i]
        port_ret = adjusted
    
    cum = (1 + port_ret).cumprod()
    total_ret = cum.iloc[-1] - 1
    days = len(port_ret)
    years = days / 365
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(365) if port_ret.std() > 0 else 0
    peak = cum.expanding().max()
    maxdd = ((cum - peak) / peak).min()
    
    # Year-by-year
    yearly = {}
    if hasattr(port_ret.index, 'year'):
        for yr in sorted(port_ret.index.year.unique()):
            yr_ret = port_ret[port_ret.index.year == yr]
            yr_cum = (1 + yr_ret).cumprod().iloc[-1] - 1
            yearly[int(yr)] = round(float(yr_cum) * 100, 2)
    
    return {
        "total_return": round(float(total_ret) * 100, 2),
        "cagr": round(float(cagr) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(maxdd) * 100, 2),
        "n_days": days,
        "yearly_returns": yearly,
        "daily_returns": port_ret
    }

# ── Bootstrap CI ──
def bootstrap_ci(daily_returns, n_boot=2000, ci=0.95):
    returns = daily_returns.values
    n = len(returns)
    sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)
        s = sample.mean() / sample.std() * np.sqrt(365) if sample.std() > 0 else 0
        sharpes.append(s)
    sharpes = sorted(sharpes)
    lo = sharpes[int((1-ci)/2 * n_boot)]
    hi = sharpes[int((1+ci)/2 * n_boot)]
    return round(lo, 3), round(hi, 3)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
print("=" * 80)
print("V4 HONEST SYSTEM — EXPANDED PORTFOLIO")
print("=" * 80)

# 1. Discover and load
assets = discover_assets()
print(f"\nDiscovered {len(assets)} assets: {', '.join(sorted(assets.keys()))}")

# Filter by min 730 days
asset_data = {}
excluded_short = []
for sym, path in sorted(assets.items()):
    df = load_asset(path)
    if len(df) >= 730:
        asset_data[sym] = df
    else:
        excluded_short.append((sym, len(df)))

print(f"Assets with ≥730 days: {len(asset_data)}")
if excluded_short:
    print(f"Excluded (too short): {', '.join(f'{s}({d}d)' for s,d in excluded_short)}")

# 2. Screen each asset with SMA50 walk-forward
print("\n" + "=" * 80)
print("ASSET SCREENING — SMA50 Individual Performance (14-fold WF)")
print("=" * 80)

screening = {}
full_returns = {}  # Full-sample V4d returns for portfolio construction

for sym in sorted(asset_data.keys()):
    df = asset_data[sym]
    
    # Get trailing stop
    tr = np.maximum(df["high"].values - df["low"].values,
                    np.maximum(np.abs(df["high"].values - np.roll(df["close"].values, 1)),
                               np.abs(df["low"].values - np.roll(df["close"].values, 1))))
    tr[0] = df["high"].values[0] - df["low"].values[0]
    atr20 = pd.Series(tr).rolling(20).mean().values
    ts_pct = PRESET_STOPS.get(sym, np.clip(np.nanmedian(2 * atr20 / df["close"].values), 0.05, 0.20))
    
    # Full sample backtest
    rets = v4d_backtest(df, sym, ts_pct)
    full_returns[sym] = rets
    
    # Walk-forward
    oos_rets, oos_sharpe = walk_forward_14(df, sym, ts_pct)
    
    # Full sample stats
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / 365
    cagr = (1 + total) ** (1/years) - 1 if years > 0 else 0
    sharpe_full = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    peak = cum.expanding().max()
    maxdd = ((cum - peak) / peak).min()
    
    screening[sym] = {
        "days": len(df),
        "trailing_stop": round(ts_pct * 100, 1),
        "full_sharpe": round(float(sharpe_full), 3),
        "full_cagr": round(float(cagr) * 100, 2),
        "full_maxdd": round(float(maxdd) * 100, 2),
        "oos_sharpe": round(float(oos_sharpe), 3) if oos_sharpe is not None else None,
        "survivor": oos_sharpe is not None and oos_sharpe > 0
    }

# Print screening table
print(f"\n{'Symbol':<12} {'Days':>5} {'TS%':>5} {'Sharpe':>7} {'CAGR%':>8} {'MaxDD%':>8} {'OOS_Sh':>7} {'Status':<8}")
print("-" * 70)
for sym in sorted(screening.keys(), key=lambda s: screening[s].get('oos_sharpe') or -99, reverse=True):
    s = screening[sym]
    status = "✅ IN" if s["survivor"] else "❌ OUT"
    oos = f"{s['oos_sharpe']:>7.3f}" if s['oos_sharpe'] is not None else "    N/A"
    print(f"{sym:<12} {s['days']:>5} {s['trailing_stop']:>5.1f} {s['full_sharpe']:>7.3f} {s['full_cagr']:>8.2f} {s['full_maxdd']:>8.2f} {oos} {status}")

survivors = [s for s in screening if screening[s]["survivor"]]
print(f"\nSurvivors: {len(survivors)} / {len(screening)}")

# 3. Rank survivors by OOS Sharpe
ranked = sorted(survivors, key=lambda s: screening[s]["oos_sharpe"], reverse=True)
print(f"Ranked: {', '.join(ranked)}")

# 4. Build portfolio variants
# Align returns to common timeline
def build_portfolio_returns(symbols):
    rets = {}
    for sym in symbols:
        r = full_returns[sym]
        r.index = pd.to_datetime(r.index)
        rets[sym] = r
    return rets

variants = {}
for name, syms in [
    ("Top 5", ranked[:5]),
    ("Top 10", ranked[:10]),
    ("Top 15", ranked[:15]),
    ("All Survivors", ranked),
]:
    if len(syms) == 0:
        continue
    actual_syms = syms[:len(syms)]  # trim if fewer available
    if not actual_syms:
        continue
    port = portfolio_backtest(build_portfolio_returns(actual_syms))
    variants[name] = {
        "symbols": actual_syms,
        "n_assets": len(actual_syms),
        **{k: v for k, v in port.items() if k != "daily_returns"}
    }
    variants[name]["_daily"] = port.get("daily_returns")

# 5. Print portfolio comparison
print("\n" + "=" * 80)
print("PORTFOLIO COMPARISON")
print("=" * 80)
print(f"\n{'Portfolio':<18} {'#Assets':>7} {'Sharpe':>7} {'CAGR%':>8} {'MaxDD%':>8} {'Total%':>9}")
print("-" * 60)
best_name = None
best_sharpe = -999
for name in ["Top 5", "Top 10", "Top 15", "All Survivors"]:
    if name not in variants:
        continue
    v = variants[name]
    print(f"{name:<18} {v['n_assets']:>7} {v['sharpe']:>7.3f} {v['cagr']:>8.2f} {v['max_dd']:>8.2f} {v['total_return']:>9.2f}")
    if v['sharpe'] > best_sharpe:
        best_sharpe = v['sharpe']
        best_name = name

print(f"\n🏆 Best portfolio: {best_name} (Sharpe {best_sharpe:.3f})")

# 6. Year-by-year for best
if best_name:
    print(f"\n{'Year':<8} {'Return%':>10}")
    print("-" * 20)
    for yr, ret in sorted(variants[best_name]["yearly_returns"].items()):
        print(f"{yr:<8} {ret:>10.2f}")

# 7. Walk-forward on best portfolio
print("\n" + "=" * 80)
print(f"14-FOLD WALK-FORWARD — {best_name}")
print("=" * 80)

if best_name:
    best_syms = variants[best_name]["symbols"]
    # Build combined daily returns df
    port_rets = build_portfolio_returns(best_syms)
    combined = pd.DataFrame(port_rets).fillna(0)
    port_daily = combined.mean(axis=1)
    port_daily.index = pd.to_datetime(port_daily.index)
    
    n = len(port_daily)
    min_train = max(100, n // 15)
    fold_size = (n - min_train) // 14
    
    print(f"\n{'Fold':>4} {'Train':>6} {'Test':>6} {'OOS_Sharpe':>11} {'OOS_Ret%':>10}")
    print("-" * 45)
    
    oos_sharpes = []
    for fold in range(14):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        oos = port_daily.iloc[train_end:test_end]
        sh = oos.mean() / oos.std() * np.sqrt(365) if oos.std() > 0 else 0
        ret = (1 + oos).prod() - 1
        oos_sharpes.append(sh)
        print(f"{fold+1:>4} {train_end:>6} {test_end-train_end:>6} {sh:>11.3f} {ret*100:>10.2f}")
    
    avg_oos = np.mean(oos_sharpes)
    pct_positive = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) * 100
    print(f"\nAvg OOS Sharpe: {avg_oos:.3f}")
    print(f"Positive folds: {pct_positive:.0f}%")

# 8. Bootstrap CI on best
print("\n" + "=" * 80)
print(f"BOOTSTRAP 95% CI — {best_name}")
print("=" * 80)

if best_name and variants[best_name].get("_daily") is not None:
    lo, hi = bootstrap_ci(variants[best_name]["_daily"])
    print(f"\nSharpe 95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Point estimate: {variants[best_name]['sharpe']:.3f}")

# 9. IN vs OUT
print("\n" + "=" * 80)
print("ASSET STATUS — IN vs OUT")
print("=" * 80)
print(f"\n{'Symbol':<12} {'Status':<6} {'Reason'}")
print("-" * 50)
for sym in sorted(screening.keys()):
    s = screening[sym]
    if s["survivor"]:
        in_best = sym in variants.get(best_name, {}).get("symbols", [])
        extra = f" (in {best_name})" if in_best else ""
        print(f"{sym:<12} {'IN':<6} OOS Sharpe {s['oos_sharpe']:.3f}{extra}")
    else:
        reason = f"OOS Sharpe {s['oos_sharpe']:.3f} ≤ 0" if s['oos_sharpe'] is not None else "WF failed"
        print(f"{sym:<12} {'OUT':<6} {reason}")

# 10. Save results
output = {
    "screening": screening,
    "survivors": ranked,
    "portfolios": {k: {kk: vv for kk, vv in v.items() if kk != "_daily"} for k, v in variants.items()},
    "best_portfolio": best_name,
    "best_sharpe": best_sharpe,
}
if best_name:
    output["best_yearly"] = variants[best_name]["yearly_returns"]
    if variants[best_name].get("_daily") is not None:
        lo, hi = bootstrap_ci(variants[best_name]["_daily"])
        output["bootstrap_95ci"] = [lo, hi]

with open(os.path.join(OUT_DIR, "v4_portfolio_expanded.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to {OUT_DIR}/v4_portfolio_expanded.json")
