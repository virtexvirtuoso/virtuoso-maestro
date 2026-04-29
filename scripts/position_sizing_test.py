#!/usr/bin/env python3
"""Position Sizing Optimization Test for V4 System."""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

DATA_DIR = Path.home() / "Desktop/maestro/data/spot"
OUT_DIR = Path.home() / "Desktop/maestro/data/backtest_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
TRAIL_STOPS = {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08}
SMA_PERIOD = 50
VOL_CEIL_THRESHOLD = 0.80
DD_BREAKER = 0.25
REBAL_DAYS = 21
N_FOLDS = 14

PORTFOLIOS = {
    "Core3": ["BTC", "ETH", "SOL"],
    "Top5": ["SOL", "FTM", "AVAX", "BNB"],
}

SIZING_METHODS = [
    "EqualWeight", "InverseVol", "RiskParity", "Kelly",
    "MomentumWeighted", "VolAdjMomentum", "FixedFractional"
]

def load_data(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_spot_daily.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]

def compute_atr(df, period=20):
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_trail_stop(symbol, close_series):
    if symbol in TRAIL_STOPS:
        return pd.Series(TRAIL_STOPS[symbol], index=close_series.index)
    atr = compute_atr(pd.DataFrame({"High": close_series, "Low": close_series, "Close": close_series}), 20)
    trail = (2 * atr / close_series).clip(0.05, 0.20)
    return trail

def generate_signals(dfs):
    """Generate V4d signals: long when close > SMA50, with trailing stops, vol ceiling, DD breaker."""
    signals = {}
    for sym, df in dfs.items():
        sma = df["Close"].rolling(SMA_PERIOD).mean()
        raw_signal = (df["Close"] > sma).astype(int)
        # Signal bar N -> trade bar N+1
        signal = raw_signal.shift(1).fillna(0)
        
        # Vol ceiling: halve when 30d vol > 80% annualized
        ret = df["Close"].pct_change()
        vol_30d = ret.rolling(30).std() * np.sqrt(365)
        vol_halve = (vol_30d > VOL_CEIL_THRESHOLD).astype(int)
        
        signals[sym] = pd.DataFrame({
            "signal": signal,
            "vol_halve": vol_halve,
            "close": df["Close"],
            "ret": ret,
            "vol_30d": vol_30d,
            "sma50": sma,
        }, index=df.index)
    return signals

def compute_weights(method, signals_dict, date, assets, portfolio_value, hist_len=90):
    """Compute portfolio weights for given sizing method at given date."""
    n = len(assets)
    weights = {}
    
    for sym in assets:
        s = signals_dict[sym]
        mask = s.index <= date
        s_hist = s.loc[mask]
        if len(s_hist) < SMA_PERIOD + 10:
            weights[sym] = 0.0
            continue
        
        # Must be in signal (close > SMA50 on prior bar)
        if s_hist["signal"].iloc[-1] == 0:
            weights[sym] = 0.0
            continue
        
        # Vol ceiling
        vol_mult = 0.5 if s_hist["vol_halve"].iloc[-1] == 1 else 1.0
        
        vol_30d = s_hist["vol_30d"].iloc[-1] if not np.isnan(s_hist["vol_30d"].iloc[-1]) else 0.5
        ret_series = s_hist["ret"].dropna()
        
        if method == "EqualWeight":
            weights[sym] = vol_mult / n
        elif method == "InverseVol":
            weights[sym] = (1.0 / max(vol_30d, 0.01)) * vol_mult
        elif method == "RiskParity":
            weights[sym] = (1.0 / max(vol_30d, 0.01)) * vol_mult
        elif method == "Kelly":
            if len(ret_series) >= 90:
                mu = ret_series.iloc[-90:].mean() * 365
                sigma = ret_series.iloc[-90:].std() * np.sqrt(365)
                f_star = mu / max(sigma**2, 0.01) if sigma > 0 else 0
                f_star = max(0, min(f_star, 0.25))
            else:
                f_star = 1.0 / n
            weights[sym] = f_star * vol_mult
        elif method == "MomentumWeighted":
            if len(s_hist) >= 60:
                mom = s_hist["close"].iloc[-1] / s_hist["close"].iloc[-60] - 1
                weights[sym] = max(mom, 0) * vol_mult
            else:
                weights[sym] = vol_mult / n
        elif method == "VolAdjMomentum":
            if len(s_hist) >= 60:
                mom = s_hist["close"].iloc[-1] / s_hist["close"].iloc[-60] - 1
                weights[sym] = max(mom, 0) / max(vol_30d, 0.01) * vol_mult
            else:
                weights[sym] = vol_mult / n
        elif method == "FixedFractional":
            trail = TRAIL_STOPS.get(sym, None)
            if trail is None:
                atr = compute_atr(signals_dict[sym][["close"]].rename(columns={"close":"Close"}).assign(High=lambda x: x["Close"], Low=lambda x: x["Close"]), 20)
                atr_val = atr.loc[atr.index <= date].iloc[-1] if len(atr.loc[atr.index <= date]) > 0 else 0
                trail = np.clip(2 * atr_val / s_hist["close"].iloc[-1], 0.05, 0.20)
            weights[sym] = (0.02 / max(trail, 0.01)) * vol_mult
    
    # Normalize
    total = sum(weights.values())
    if total > 0:
        # Cap total at 1.0
        if total > 1.0:
            for sym in weights:
                weights[sym] /= total
    else:
        for sym in assets:
            weights[sym] = 0.0
    
    return weights

def run_backtest(method, signals_dict, assets, start_date, end_date):
    """Run backtest for a given sizing method over date range."""
    # Get common dates
    all_dates = None
    for sym in assets:
        idx = signals_dict[sym].index
        mask = (idx >= start_date) & (idx <= end_date)
        dates = idx[mask]
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    if len(all_dates) < 60:
        return None
    
    all_dates = all_dates.sort_values()
    
    portfolio_value = 1.0
    peak = 1.0
    values = []
    weight_history = []
    
    current_weights = {sym: 0.0 for sym in assets}
    trailing_highs = {sym: 0.0 for sym in assets}
    position_active = {sym: False for sym in assets}
    dd_breaker_active = False
    last_rebal = 0
    
    for i, date in enumerate(all_dates):
        # DD breaker check
        if portfolio_value / peak - 1 < -DD_BREAKER:
            dd_breaker_active = True
        if dd_breaker_active:
            # Stay flat until DD recovers to -15%
            if portfolio_value / peak - 1 > -0.15:
                dd_breaker_active = False
            else:
                daily_ret = 0.0
                values.append(portfolio_value)
                continue
        
        # Rebalance every 21 days
        if i % REBAL_DAYS == 0 or i == 0:
            current_weights = compute_weights(method, signals_dict, date, assets, portfolio_value)
            weight_history.append({"date": str(date.date()), **current_weights})
            # Reset trailing highs on rebalance
            for sym in assets:
                if current_weights[sym] > 0:
                    trailing_highs[sym] = signals_dict[sym].loc[date, "close"]
                    position_active[sym] = True
                else:
                    position_active[sym] = False
        
        # Compute daily portfolio return
        daily_pnl = 0.0
        for sym in assets:
            w = current_weights.get(sym, 0)
            if w <= 0 or not position_active[sym]:
                continue
            
            s = signals_dict[sym]
            if date not in s.index:
                continue
            
            price = s.loc[date, "close"]
            ret = s.loc[date, "ret"]
            if np.isnan(ret):
                ret = 0.0
            
            # Update trailing high
            trailing_highs[sym] = max(trailing_highs[sym], price)
            
            # Check trailing stop
            trail_pct = TRAIL_STOPS.get(sym, None)
            if trail_pct is None:
                trail_pct = 0.10  # default
            
            drawdown_from_high = 1 - price / trailing_highs[sym] if trailing_highs[sym] > 0 else 0
            if drawdown_from_high > trail_pct:
                position_active[sym] = False
                current_weights[sym] = 0.0
                continue
            
            daily_pnl += w * ret
        
        portfolio_value *= (1 + daily_pnl)
        peak = max(peak, portfolio_value)
        values.append(portfolio_value)
    
    if len(values) < 30:
        return None
    
    values = np.array(values)
    total_days = len(values)
    total_years = total_days / 365.25
    
    cagr = (values[-1] / values[0]) ** (1 / max(total_years, 0.01)) - 1
    daily_rets = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_rets) / max(np.std(daily_rets), 1e-8) * np.sqrt(365)
    
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1
    max_dd = drawdowns.min()
    calmar = cagr / max(abs(max_dd), 0.01)
    
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "final_value": round(values[-1], 4),
        "n_days": total_days,
        "weight_history": weight_history,
    }

def walk_forward(method, signals_dict, assets, n_folds=14):
    """Expanding walk-forward with n_folds."""
    # Get common date range
    start_dates = []
    end_dates = []
    for sym in assets:
        s = signals_dict[sym]
        start_dates.append(s.index.min())
        end_dates.append(s.index.max())
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Need at least SMA_PERIOD + 90 days warmup
    warmup = pd.Timedelta(days=SMA_PERIOD + 100)
    effective_start = common_start + warmup
    
    total_days = (common_end - effective_start).days
    fold_size = total_days // (n_folds + 1)  # +1 for initial training
    
    fold_results = []
    for fold in range(n_folds):
        # Expanding window: train from start, test on next fold
        test_start = effective_start + pd.Timedelta(days=fold_size * (fold + 1))
        test_end = test_start + pd.Timedelta(days=fold_size)
        if test_end > common_end:
            test_end = common_end
        
        result = run_backtest(method, signals_dict, assets, test_start, test_end)
        if result is not None:
            fold_results.append(result)
    
    if not fold_results:
        return None
    
    # Aggregate
    avg = lambda key: np.mean([r[key] for r in fold_results])
    std = lambda key: np.std([r[key] for r in fold_results])
    
    return {
        "cagr_mean": round(avg("cagr"), 2),
        "cagr_std": round(std("cagr"), 2),
        "sharpe_mean": round(avg("sharpe"), 3),
        "sharpe_std": round(std("sharpe"), 3),
        "max_dd_mean": round(avg("max_dd"), 2),
        "max_dd_std": round(std("max_dd"), 2),
        "calmar_mean": round(avg("calmar"), 3),
        "calmar_std": round(std("calmar"), 3),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }

def bootstrap_ci(fold_results, metric, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval for a metric."""
    vals = [r[metric] for r in fold_results]
    if len(vals) < 3:
        return None, None
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return round(lower, 3), round(upper, 3)

# === MAIN ===
print("Loading data...")
all_data = {}
for sym in ["BTC", "ETH", "SOL", "BNB", "FTM", "AVAX", "SUI"]:
    all_data[sym] = load_data(sym)
    print(f"  {sym}: {all_data[sym].index[0].date()} to {all_data[sym].index[-1].date()}, {len(all_data[sym])} rows")

print("\nGenerating signals...")
signals = generate_signals(all_data)

results = {}
best_variant = {"sharpe": -999, "name": None, "portfolio": None}

for port_name, assets in PORTFOLIOS.items():
    print(f"\n{'='*60}")
    print(f"Portfolio: {port_name} — {assets}")
    print(f"{'='*60}")
    
    results[port_name] = {}
    
    for method in SIZING_METHODS:
        print(f"  Testing {method}...", end=" ", flush=True)
        wf = walk_forward(method, signals, assets, N_FOLDS)
        if wf is None:
            print("SKIP (insufficient data)")
            continue
        
        results[port_name][method] = wf
        print(f"Sharpe={wf['sharpe_mean']:.3f}  CAGR={wf['cagr_mean']:.1f}%  MaxDD={wf['max_dd_mean']:.1f}%  Calmar={wf['calmar_mean']:.3f}  (n={wf['n_folds']})")
        
        # Track best by Sharpe
        if wf['sharpe_mean'] > best_variant['sharpe']:
            best_variant = {"sharpe": wf['sharpe_mean'], "name": method, "portfolio": port_name, "data": wf}

# === COMPARISON TABLES ===
print(f"\n{'='*80}")
print("FULL COMPARISON TABLE")
print(f"{'='*80}")

for port_name in PORTFOLIOS:
    print(f"\n--- {port_name} ---")
    print(f"{'Method':<22} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Calmar':>8} {'Folds':>6}")
    print("-" * 62)
    for method in SIZING_METHODS:
        if method in results.get(port_name, {}):
            r = results[port_name][method]
            print(f"{method:<22} {r['sharpe_mean']:>8.3f} {r['cagr_mean']:>8.1f} {r['max_dd_mean']:>8.1f} {r['calmar_mean']:>8.3f} {r['n_folds']:>6}")

# Bootstrap CIs on best
print(f"\n{'='*80}")
print(f"BOOTSTRAP 95% CIs — Best Variant: {best_variant['name']} ({best_variant['portfolio']})")
print(f"{'='*80}")

if best_variant['data']:
    folds = best_variant['data']['fold_results']
    for metric in ["sharpe", "cagr", "max_dd", "calmar"]:
        lo, hi = bootstrap_ci(folds, metric)
        print(f"  {metric}: [{lo}, {hi}]")

# Weight history for best
print(f"\n{'='*80}")
print(f"WEIGHT HISTORY — {best_variant['name']} ({best_variant['portfolio']})")
print(f"{'='*80}")

if best_variant['data']:
    all_wh = []
    for fold in best_variant['data']['fold_results']:
        all_wh.extend(fold.get('weight_history', []))
    
    if all_wh:
        wh_df = pd.DataFrame(all_wh).sort_values("date")
        # Show first 10 and last 10
        assets_cols = [c for c in wh_df.columns if c != "date"]
        print(f"\n{'Date':<12}", end="")
        for c in assets_cols:
            print(f"{c:>8}", end="")
        print()
        
        display_rows = pd.concat([wh_df.head(10), wh_df.tail(10)]).drop_duplicates()
        for _, row in display_rows.iterrows():
            print(f"{row['date']:<12}", end="")
            for c in assets_cols:
                print(f"{row[c]:>8.3f}", end="")
            print()

# === VERDICT ===
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

# Find best Sharpe and best Calmar across all
best_sharpe = {"val": -999, "method": None, "port": None}
best_calmar = {"val": -999, "method": None, "port": None}

for port_name in PORTFOLIOS:
    for method in SIZING_METHODS:
        if method in results.get(port_name, {}):
            r = results[port_name][method]
            if r['sharpe_mean'] > best_sharpe['val']:
                best_sharpe = {"val": r['sharpe_mean'], "method": method, "port": port_name}
            if r['calmar_mean'] > best_calmar['val']:
                best_calmar = {"val": r['calmar_mean'], "method": method, "port": port_name}

print(f"\n  Best Sharpe:  {best_sharpe['method']} ({best_sharpe['port']}) = {best_sharpe['val']:.3f}")
print(f"  Best Calmar:  {best_calmar['method']} ({best_calmar['port']}) = {best_calmar['val']:.3f}")

if best_sharpe['method'] == best_calmar['method']:
    print(f"\n  ✅ WINNER: {best_sharpe['method']} dominates both Sharpe and Calmar!")
else:
    print(f"\n  ⚠️  Split verdict — Sharpe favors {best_sharpe['method']}, Calmar favors {best_calmar['method']}")
    print(f"  Recommendation: Use {best_calmar['method']} for risk-adjusted returns (Calmar weights DD more)")

# Save results (strip weight_history from fold_results for JSON size)
save_results = {}
for port_name in results:
    save_results[port_name] = {}
    for method, r in results[port_name].items():
        save_r = {k: v for k, v in r.items() if k != 'fold_results'}
        # Keep summary of fold results
        save_r['fold_sharpes'] = [f['sharpe'] for f in r['fold_results']]
        save_r['fold_cagrs'] = [f['cagr'] for f in r['fold_results']]
        save_r['fold_max_dds'] = [f['max_dd'] for f in r['fold_results']]
        save_r['fold_calmars'] = [f['calmar'] for f in r['fold_results']]
        save_results[port_name][method] = save_r

save_results["_meta"] = {
    "best_sharpe": best_sharpe,
    "best_calmar": best_calmar,
    "n_folds": N_FOLDS,
    "rebal_days": REBAL_DAYS,
}

with open(OUT_DIR / "position_sizing_test.json", "w") as f:
    json.dump(save_results, f, indent=2)

print(f"\n✅ Results saved to {OUT_DIR / 'position_sizing_test.json'}")
