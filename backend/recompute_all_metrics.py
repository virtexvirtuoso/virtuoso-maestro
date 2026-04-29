"""
Recompute ALL backtests with comprehensive metrics including Omega, Profit Factor,
Tail Ratio, Ulcer Index, and UPI.
"""
import sys, os, json, warnings, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from datasource.factor_loader import FactorDataLoader
from strategies.composite.mega_strategy_v3 import (
    run_full_strategy, ASSET_CONFIGS, DEFAULT_LEVERAGE_MAP,
    compute_confluence, detect_regime, _rsi, _realized_vol,
)
from strategies.composite.mega_strategy_v4 import run_mega_v4

START = "2020-01-01"
END = "2026-02-12"
OUTPUT = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results/all_strategies_comprehensive.json"))

# ── New Metrics ──

def omega_ratio(returns, threshold=0.0):
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    return round(gains / losses, 3) if losses > 0 else float('inf')

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else float('inf')

def tail_ratio_metric(returns):
    p95 = np.percentile(returns.dropna(), 95)
    p5 = abs(np.percentile(returns.dropna(), 5))
    return round(p95 / p5, 3) if p5 > 0 else float('inf')

def ulcer_index(equity_curve):
    peak = equity_curve.cummax()
    dd_pct = ((equity_curve - peak) / peak) * 100
    return round(np.sqrt((dd_pct ** 2).mean()), 3)

def ulcer_performance_index(returns, equity_curve):
    n_years = len(returns) / 252
    if n_years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/n_years) - 1
    ui = ulcer_index(equity_curve)
    return round(cagr / ui * 100, 3) if ui > 0 else float('inf')

def compute_all_metrics(daily_returns, equity=None, label=""):
    """Compute all metrics from daily returns series."""
    dr = daily_returns.dropna()
    if len(dr) < 30:
        return {"error": f"Too few data points ({len(dr)})"}
    
    if equity is None:
        equity = (1 + dr).cumprod()
    
    n_years = len(dr) / 252
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = round((dr.mean() / dr.std()) * np.sqrt(252), 3) if dr.std() > 0 else 0
    
    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = round((dr.mean() * 252) / downside, 3) if downside > 0 else 0
    
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = round(dd.min() * 100, 2)
    
    calmar = round(cagr / abs(max_dd), 3) if max_dd != 0 else 0
    
    # Monthly returns for win rate
    monthly = dr.resample('M').sum() if hasattr(dr.index, 'freq') or True else dr
    try:
        monthly = dr.resample('ME').sum()
    except:
        monthly = dr.resample('M').sum()
    win_rate_monthly = round((monthly > 0).sum() / len(monthly) * 100, 1) if len(monthly) > 0 else 0
    
    # Time in market
    time_in_market = round((dr != 0).sum() / len(dr) * 100, 1)
    
    return {
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "omega": omega_ratio(dr),
        "profit_factor": profit_factor(dr),
        "tail_ratio": tail_ratio_metric(dr),
        "ulcer_index": ulcer_index(equity),
        "upi": ulcer_performance_index(dr, equity),
        "win_rate_monthly": win_rate_monthly,
        "time_in_market": time_in_market,
    }


# ── Load Data ──

print("=" * 70)
print("COMPREHENSIVE METRICS RECOMPUTATION")
print("=" * 70)

stock_loader = StockDataLoader()
macro_loader = MacroDataLoader()
factor_loader = FactorDataLoader()

# Load assets
tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD", "SPY": "SPY"}
asset_data = {}
for name, ticker in tickers.items():
    try:
        df = stock_loader.get_ohlcv(ticker, "1d", START, END)
        asset_data[name] = df
        print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

# Load macro
macro_series = {
    'T10Y2Y': 'T10Y2Y', 'M2SL': 'M2SL', 'FEDFUNDS': 'FEDFUNDS',
    'CPIAUCSL': 'CPIAUCSL', 'DCOILWTICO': 'DCOILWTICO',
    'BAMLH0A0HYM2': 'BAMLH0A0HYM2',
}
macro_df = macro_loader.get_multiple(macro_series, start_date="2015-01-01", end_date=END)
print(f"  Macro: {len(macro_df)} observations")
macro_daily = macro_df.resample('D').last().ffill()

# Build macro_data for V3
m2_yoy = macro_daily['M2SL'].pct_change(365)
m2_yoy_6m = m2_yoy.rolling(180).mean()
macro_data = pd.DataFrame({
    'yield_curve': macro_daily.get('T10Y2Y', pd.Series(dtype=float)),
    'm2_yoy': m2_yoy,
    'm2_accel': (m2_yoy > m2_yoy_6m).astype(float),
}, index=macro_daily.index)

# Cross-asset data
cross_tickers = {"dxy": "DX-Y.NYB", "gold": "GC=F", "bonds": "TLT", "hyg": "HYG", "copper": "HG=F"}
cross_dfs = {}
for name, ticker in cross_tickers.items():
    try:
        df = stock_loader.get_ohlcv(ticker, "1d", "2015-01-01", END)
        cross_dfs[name] = df["close"]
    except:
        pass
cross_asset = pd.DataFrame(cross_dfs)

# Fama-French
try:
    ff3 = factor_loader.get_ff3()
    print(f"  FF3: {len(ff3)} months")
except:
    ff3 = None
    print("  FF3: unavailable")

# ── Strategy Runners ──

results = {}
crypto_assets = {k: v for k, v in asset_data.items() if k in ("BTC", "ETH", "SOL", "LINK")}

# 1. Buy & Hold BTC
print("\n[1/10] Buy & Hold BTC...")
btc_ret = asset_data["BTC"]["close"].pct_change().dropna()
results["S1_BH_BTC"] = compute_all_metrics(btc_ret, label="B&H BTC")

# 2. Buy & Hold Equal-Weight
print("[2/10] Buy & Hold Equal-Weight...")
ew_rets = []
for name in ["BTC", "ETH", "SOL", "LINK"]:
    if name in asset_data:
        r = asset_data[name]["close"].pct_change()
        ew_rets.append(r)
ew_combined = pd.concat(ew_rets, axis=1).dropna()
ew_daily = ew_combined.mean(axis=1)
results["S2_BH_EW"] = compute_all_metrics(ew_daily, label="B&H EW")

# 3. V1 (Macro Momentum - simplified S5 from backtest_macro_crypto)
print("[3/10] V1 Macro Momentum...")
try:
    btc_close = asset_data["BTC"]["close"]
    sma_200 = btc_close.rolling(200).mean()
    sma_50 = btc_close.rolling(50).mean()
    rsi_14 = _rsi(btc_close, 14)
    
    # V1 signals: trend + momentum + golden cross + macro
    trend_up = btc_close > sma_200
    golden_cross = sma_50 > sma_200
    rsi_ok = rsi_14 < 70
    macro_score = macro_data['m2_accel'].reindex(btc_close.index, method='ffill').fillna(0)
    yc = macro_data['yield_curve'].reindex(btc_close.index, method='ffill').fillna(0)
    macro_bull = (macro_score > 0) | (yc > 0)
    
    v1_signal = (trend_up & golden_cross & rsi_ok & macro_bull).shift(1).fillna(False)
    btc_daily = btc_close.pct_change()
    v1_ret = btc_daily * v1_signal.astype(float)
    
    # 15% trailing stop
    eq = (1 + v1_ret).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    stopped = dd < -0.15
    for i in range(1, len(stopped)):
        if stopped.iloc[i]:
            v1_ret.iloc[i] = 0
            
    results["S3_V1_MacroMom"] = compute_all_metrics(v1_ret, label="V1 Macro Mom")
except Exception as e:
    print(f"  V1 error: {e}")
    traceback.print_exc()

# 4. V2 BTC Only (dip-buying, sma=90, trail=18%, rsi=44)
print("[4/10] V2 BTC Only...")
try:
    btc_df = asset_data["BTC"]
    btc_close = btc_df["close"]
    sma_90 = btc_close.rolling(90).mean()
    rsi = _rsi(btc_close, 14)
    
    v2_signal = ((btc_close > sma_90) & (rsi.shift(1) < 44)).shift(1).fillna(False)
    btc_daily = btc_close.pct_change()
    
    # Track position with trailing stop
    pos = pd.Series(0.0, index=btc_close.index)
    eq = 1.0
    peak_eq = 1.0
    in_pos = False
    for i in range(91, len(btc_close)):
        if in_pos:
            eq *= (1 + btc_daily.iloc[i])
            peak_eq = max(peak_eq, eq)
            if (eq / peak_eq - 1) < -0.18:
                in_pos = False
                pos.iloc[i] = 0
                continue
            pos.iloc[i] = 1.0
        else:
            if v2_signal.iloc[i]:
                in_pos = True
                pos.iloc[i] = 1.0
                peak_eq = eq
            else:
                pos.iloc[i] = 0.0
    
    v2_ret = btc_daily * pos
    results["S4_V2_BTC"] = compute_all_metrics(v2_ret, label="V2 BTC Only")
except Exception as e:
    print(f"  V2 BTC error: {e}")
    traceback.print_exc()

# 5. V2 Portfolio (equal-weight, momentum-weighted, 1.34x leverage)
print("[5/10] V2 Portfolio...")
try:
    port_rets = []
    for name in ["BTC", "ETH", "SOL", "LINK"]:
        if name not in asset_data:
            continue
        c = asset_data[name]["close"]
        sma = c.rolling(90).mean()
        mom = c.pct_change(30)
        sig = ((c > sma) & (mom > 0)).shift(1).fillna(False).astype(float)
        r = c.pct_change() * sig
        port_rets.append(r)
    
    combined = pd.concat(port_rets, axis=1).dropna()
    v2_port_ret = combined.mean(axis=1) * 1.34
    results["S5_V2_Portfolio"] = compute_all_metrics(v2_port_ret, label="V2 Portfolio")
except Exception as e:
    print(f"  V2 Portfolio error: {e}")

# 6. V3 Long+Adaptive (no shorts)
print("[6/10] V3 Long+Adaptive...")
try:
    portfolio_df, per_asset = run_full_strategy(
        crypto_assets, macro_data, cross_asset,
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
    )
    v3_long_ret = portfolio_df["daily_pnl"]
    results["S6_V3_LongAdaptive"] = compute_all_metrics(v3_long_ret, label="V3 Long+Adaptive")
except Exception as e:
    print(f"  V3 Long error: {e}")
    traceback.print_exc()

# 7. V3 Full (long + short + adaptive)
print("[7/10] V3 Full...")
try:
    portfolio_df_v3, per_asset_v3 = run_full_strategy(
        crypto_assets, macro_data, cross_asset,
        enable_long=True, enable_short=True, enable_adaptive_leverage=True,
    )
    v3_full_ret = portfolio_df_v3["daily_pnl"]
    results["S7_V3_Full"] = compute_all_metrics(v3_full_ret, label="V3 Full")
    
    # Per-regime metrics for V3 Full
    v3_regime = portfolio_df_v3["regime"]
    v3_regime_metrics = {}
    for regime_name in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]:
        mask = v3_regime == regime_name
        if mask.sum() > 10:
            regime_ret = v3_full_ret[mask]
            regime_eq = (1 + regime_ret).cumprod()
            v3_regime_metrics[regime_name] = compute_all_metrics(regime_ret, regime_eq, f"V3 {regime_name}")
            v3_regime_metrics[regime_name]["days"] = int(mask.sum())
        else:
            v3_regime_metrics[regime_name] = {"days": int(mask.sum()), "note": "insufficient data"}
    
except Exception as e:
    print(f"  V3 Full error: {e}")
    traceback.print_exc()
    v3_regime_metrics = {}

# 8. V3 Optimized (Optuna best params)
print("[8/10] V3 Optimized...")
try:
    optimized_leverage = {5: 1.68, 4: 1.3, 3: 1.0, 2: 0.5, 1: 0.2, 0: 0.0}
    portfolio_df_opt, _ = run_full_strategy(
        crypto_assets, macro_data, cross_asset,
        enable_long=True, enable_short=True, enable_adaptive_leverage=True,
        leverage_map=optimized_leverage,
    )
    v3_opt_ret = portfolio_df_opt["daily_pnl"]
    results["S8_V3_Optimized"] = compute_all_metrics(v3_opt_ret, label="V3 Optimized")
except Exception as e:
    print(f"  V3 Optimized error: {e}")
    traceback.print_exc()

# 9. V4 Full
print("[9/10] V4 Full...")
try:
    v4_result = run_mega_v4(
        crypto_assets, macro_data, cross_asset, ff_data=ff3,
    )
    v4_full_ret = v4_result["portfolio_returns"]
    v4_equity = v4_result["equity"]
    results["S9_V4_Full"] = compute_all_metrics(v4_full_ret, v4_equity, "V4 Full")
    
    # Per-regime for V4
    v4_regime = v4_result["regime"].reindex(v4_full_ret.index, method='ffill')
    v4_regime_metrics = {}
    for regime_name in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]:
        mask = v4_regime == regime_name
        if mask.sum() > 10:
            regime_ret = v4_full_ret[mask]
            regime_eq = (1 + regime_ret).cumprod()
            v4_regime_metrics[regime_name] = compute_all_metrics(regime_ret, regime_eq, f"V4 {regime_name}")
            v4_regime_metrics[regime_name]["days"] = int(mask.sum())
        else:
            v4_regime_metrics[regime_name] = {"days": int(mask.sum()), "note": "insufficient data"}
    
    # Module ablation
    ablation_results = {}
    for disabled_mod in ["vol_breakout", "ema_ribbon", "ff_bridge", "multitf"]:
        try:
            ab_result = run_mega_v4(
                crypto_assets, macro_data, cross_asset, ff_data=ff3,
                disabled_modules=[disabled_mod],
            )
            ab_ret = ab_result["portfolio_returns"]
            ab_eq = ab_result["equity"]
            ablation_results[f"No_{disabled_mod}"] = compute_all_metrics(ab_ret, ab_eq, f"No {disabled_mod}")
        except Exception as e:
            print(f"    Ablation {disabled_mod} error: {e}")
    
except Exception as e:
    print(f"  V4 Full error: {e}")
    traceback.print_exc()
    v4_regime_metrics = {}
    ablation_results = {}

# 10. V4 Conservative (max leverage 1.5x)
print("[10/10] V4 Conservative...")
try:
    v4c_result = run_mega_v4(
        crypto_assets, macro_data, cross_asset, ff_data=ff3,
        max_total_leverage=1.5,
    )
    v4c_ret = v4c_result["portfolio_returns"]
    v4c_equity = v4c_result["equity"]
    results["S10_V4_Conservative"] = compute_all_metrics(v4c_ret, v4c_equity, "V4 Conservative")
    
    # Monthly heatmap for V4 Conservative
    v4c_monthly = {}
    for year in range(2020, 2027):
        yearly = {}
        for month in range(1, 13):
            mask = (v4c_ret.index.year == year) & (v4c_ret.index.month == month)
            if mask.sum() > 0:
                yearly[str(month)] = round(v4c_ret[mask].sum() * 100, 2)
        if yearly:
            v4c_monthly[str(year)] = yearly
            
except Exception as e:
    print(f"  V4 Conservative error: {e}")
    traceback.print_exc()
    v4c_monthly = {}

# ── Save Results ──

output_data = {
    "timestamp": datetime.now().isoformat(),
    "date_range": f"{START} to {END}",
    "metrics": results,
    "regime_breakdown_v3": v3_regime_metrics if 'v3_regime_metrics' in dir() else {},
    "regime_breakdown_v4": v4_regime_metrics if 'v4_regime_metrics' in dir() else {},
    "ablation_v4": ablation_results if 'ablation_results' in dir() else {},
    "monthly_heatmap_v4c": v4c_monthly if 'v4c_monthly' in dir() else {},
}

# Handle inf/nan for JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return 9999.0
        return obj
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return 9999.0 if np.isinf(v) or np.isnan(v) else v
    return obj

output_data = clean_for_json(output_data)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT}")
print(f"\nStrategies computed: {len(results)}")
for name, m in results.items():
    if "error" not in m:
        print(f"  {name}: Return={m['total_return']:.1f}% Sharpe={m['sharpe']} Omega={m['omega']} PF={m['profit_factor']} Tail={m['tail_ratio']} UPI={m['upi']}")
    else:
        print(f"  {name}: {m['error']}")
