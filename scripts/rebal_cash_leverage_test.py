#!/usr/bin/env python3
"""V4d Rebalancing, Cash Yield, and Regime-Based Leverage Test"""

import pandas as pd
import numpy as np
import duckdb
import json
import os
from pathlib import Path

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data")
SPOT_DIR = f"{DATA_DIR}/spot"
DB_PATH = f"{DATA_DIR}/maestro.duckdb"
OUT_PATH = f"{DATA_DIR}/backtest_results/rebal_cash_leverage_test.json"

CORE_ASSETS = ["BTC", "ETH", "SOL"]
ALL_ASSETS = ["BTC", "ETH", "SOL", "BNB", "FTM", "AVAX", "SUI"]
TRAILING_STOPS = {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08}
N_BOOTSTRAP = 1000
np.random.seed(42)

# ── Load spot data ──
def load_spot(symbol):
    df = pd.read_csv(f"{SPOT_DIR}/{symbol}_spot_daily.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    return df

# ── V4d signal generation ──
def v4d_signals(df, symbol):
    """Long when Close > SMA(50), flat otherwise. Trailing stop."""
    close = df["close"].copy()
    sma50 = close.rolling(50).mean()
    
    # Raw signal: 1=long, 0=flat. Shift by 1 for next-bar execution.
    raw = (close > sma50).astype(int)
    signal = raw.shift(1).fillna(0).astype(int)
    
    # Trailing stop
    if symbol in TRAILING_STOPS:
        stop_pct = TRAILING_STOPS[symbol]
    else:
        atr20 = (df["high"] - df["low"]).rolling(20).mean()
        stop_pct_series = (2 * atr20 / close).clip(0.05, 0.20)
        stop_pct = None  # dynamic
    
    # Apply trailing stop
    position = np.zeros(len(close))
    peak = close.iloc[0]
    in_trade = False
    
    for i in range(1, len(close)):
        if signal.iloc[i] == 1:
            if not in_trade:
                in_trade = True
                peak = close.iloc[i]
            else:
                peak = max(peak, close.iloc[i])
                sp = stop_pct if stop_pct is not None else stop_pct_series.iloc[i]
                if close.iloc[i] < peak * (1 - sp):
                    in_trade = False
                    position[i] = 0
                    continue
            position[i] = 1
        else:
            in_trade = False
            peak = close.iloc[i]
            position[i] = 0
    
    # Vol ceiling: reduce position if 20d vol > 2x median vol
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    vol_median = vol20.rolling(252).median()
    vol_ceiling = np.where(vol20 > 2 * vol_median, 0.5, 1.0)
    
    # DD breaker: go flat if drawdown > 25%
    cum_ret = (1 + ret * position).cumprod()
    running_max = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - running_max) / running_max
    dd_breaker = np.where(dd < -0.25, 0, 1)
    
    final_position = position * vol_ceiling * dd_breaker
    return pd.Series(final_position, index=df.index)

# ── Per-asset daily returns ──
def asset_returns(df, positions):
    """Returns with position sizing applied."""
    daily_ret = df["close"].pct_change().fillna(0)
    return daily_ret * positions

# ── Portfolio metrics ──
def calc_metrics(equity_curve):
    """Sharpe, CAGR, MaxDD from equity curve (cumulative value starting at 1)."""
    daily_ret = equity_curve.pct_change().dropna()
    n_years = len(daily_ret) / 252
    if n_years < 0.5 or daily_ret.std() == 0:
        return {"sharpe": 0, "cagr": 0, "maxdd": 0}
    
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/n_years) - 1
    running_max = equity_curve.cummax()
    maxdd = ((equity_curve - running_max) / running_max).min()
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr, 4), "maxdd": round(maxdd, 4)}

# ── Walk-forward expanding window (14 folds) ──
def walk_forward_folds(dates, n_folds=14):
    """Expanding window: min 252 days IS, ~equal OOS chunks."""
    n = len(dates)
    min_is = 252
    oos_total = n - min_is
    oos_per_fold = max(21, oos_total // n_folds)
    
    folds = []
    for i in range(n_folds):
        oos_start = min_is + i * oos_per_fold
        oos_end = min(oos_start + oos_per_fold, n)
        if oos_start >= n:
            break
        folds.append((0, oos_start, oos_end))  # is_start, oos_start, oos_end
    return folds

# ══════════════════════════════════════════════════
# Load all data
# ══════════════════════════════════════════════════
print("Loading data...")
spot_data = {}
asset_sigs = {}
asset_rets = {}

for sym in ALL_ASSETS:
    df = load_spot(sym)
    spot_data[sym] = df
    pos = v4d_signals(df, sym)
    asset_sigs[sym] = pos
    asset_rets[sym] = df["close"].pct_change().fillna(0)

# Align dates across core assets
common_dates = spot_data["BTC"].index
for sym in CORE_ASSETS:
    common_dates = common_dates.intersection(spot_data[sym].index)
common_dates = common_dates.sort_values()

print(f"Common dates: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} days)")

# Build aligned returns and positions for core assets
core_pos = pd.DataFrame({sym: asset_sigs[sym].reindex(common_dates).fillna(0) for sym in CORE_ASSETS})
core_raw_ret = pd.DataFrame({sym: asset_rets[sym].reindex(common_dates).fillna(0) for sym in CORE_ASSETS})
core_strat_ret = core_raw_ret * core_pos  # strategy returns per asset

# ══════════════════════════════════════════════════
# PART 1: REBALANCING FREQUENCY
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: REBALANCING FREQUENCY")
print("="*60)

def run_rebalance_portfolio(strat_ret_df, method, param=None):
    """
    Simulate equal-weight portfolio with rebalancing.
    Returns equity curve, number of rebalance events.
    """
    n_assets = len(strat_ret_df.columns)
    target_w = 1.0 / n_assets
    
    # Track weights and portfolio value
    n = len(strat_ret_df)
    weights = np.full(n_assets, target_w)
    portfolio_val = 1.0
    equity = np.zeros(n)
    equity[0] = 1.0
    rebal_count = 0
    turnover_total = 0.0
    
    for i in range(1, n):
        # Apply returns to each asset's weight
        asset_rets_today = strat_ret_df.iloc[i].values
        new_vals = weights * (1 + asset_rets_today)
        portfolio_val_new = new_vals.sum()
        
        if portfolio_val_new <= 0:
            portfolio_val_new = 1e-10
        
        weights = new_vals / portfolio_val_new
        portfolio_val *= portfolio_val_new
        
        # Check if rebalance needed
        do_rebal = False
        if method == "daily":
            do_rebal = True
        elif method == "weekly" and i % 5 == 0:
            do_rebal = True
        elif method == "monthly" and i % 21 == 0:
            do_rebal = True
        elif method == "quarterly" and i % 63 == 0:
            do_rebal = True
        elif method == "threshold":
            drift = np.abs(weights - target_w).max()
            if drift > param:
                do_rebal = True
        elif method == "none":
            do_rebal = False
        
        if do_rebal:
            turnover = np.abs(weights - target_w).sum()
            turnover_total += turnover
            weights = np.full(n_assets, target_w)
            rebal_count += 1
        
        equity[i] = portfolio_val
    
    eq_series = pd.Series(equity, index=strat_ret_df.index)
    return eq_series, rebal_count, turnover_total

rebal_methods = [
    ("Daily", "daily", None),
    ("Weekly", "weekly", None),
    ("Monthly", "monthly", None),
    ("Quarterly", "quarterly", None),
    ("Threshold 5%", "threshold", 0.05),
    ("Threshold 10%", "threshold", 0.10),
    ("No Rebalance", "none", None),
]

rebal_results = {}
for name, method, param in rebal_methods:
    eq, count, turnover = run_rebalance_portfolio(core_strat_ret, method, param)
    m = calc_metrics(eq)
    
    # Transaction cost impact
    cost_10bps = turnover * 0.001  # total cost as fraction of portfolio
    cost_30bps = turnover * 0.003
    
    rebal_results[name] = {
        **m,
        "rebal_events": count,
        "total_turnover": round(turnover, 2),
        "cost_10bps": round(cost_10bps, 4),
        "cost_30bps": round(cost_30bps, 4),
        "cagr_net_10bps": round(m["cagr"] - cost_10bps / (len(core_strat_ret)/252), 4),
        "cagr_net_30bps": round(m["cagr"] - cost_30bps / (len(core_strat_ret)/252), 4),
    }

print(f"\n{'Method':<18} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Events':>7} {'Turnover':>9} {'CAGR-10bp':>10} {'CAGR-30bp':>10}")
print("-" * 90)
for name, r in rebal_results.items():
    print(f"{name:<18} {r['sharpe']:>7.3f} {r['cagr']:>7.2%} {r['maxdd']:>7.2%} {r['rebal_events']:>7} {r['total_turnover']:>9.1f} {r['cagr_net_10bps']:>9.2%} {r['cagr_net_30bps']:>9.2%}")

# ══════════════════════════════════════════════════
# PART 2: CASH YIELD
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: CASH YIELD")
print("="*60)

# Calculate % time flat per asset
flat_pcts = {}
for sym in CORE_ASSETS:
    pos = core_pos[sym].reindex(common_dates)
    flat_pcts[sym] = (pos == 0).mean()

avg_flat = np.mean(list(flat_pcts.values()))
print(f"\nFlat time: BTC={flat_pcts['BTC']:.1%}, ETH={flat_pcts['ETH']:.1%}, SOL={flat_pcts['SOL']:.1%}, Avg={avg_flat:.1%}")

cash_yield_results = {}
for yield_name, apy in [("No Yield", 0.0), ("3% APY", 0.03), ("5% APY", 0.05), ("8% APY", 0.08)]:
    daily_yield = apy / 365
    
    # For each asset, when flat, earn daily_yield on that portion
    # Equal weight = 1/3 per asset
    n_assets = len(CORE_ASSETS)
    target_w = 1.0 / n_assets
    
    # Portfolio daily return = sum of (weight * strat_return) + sum of (weight * yield * is_flat)
    port_ret = pd.Series(0.0, index=common_dates)
    for sym in CORE_ASSETS:
        strat_r = core_strat_ret[sym].reindex(common_dates).fillna(0)
        is_flat = (core_pos[sym].reindex(common_dates) == 0).astype(float)
        port_ret += target_w * (strat_r + is_flat * daily_yield)
    
    eq = (1 + port_ret).cumprod()
    m = calc_metrics(eq)
    cash_yield_results[yield_name] = m

baseline_cagr = cash_yield_results["No Yield"]["cagr"]
baseline_sharpe = cash_yield_results["No Yield"]["sharpe"]

print(f"\n{'Yield':<12} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'ΔCAGR':>8} {'ΔSharpe':>8}")
print("-" * 55)
for name, r in cash_yield_results.items():
    dcagr = r["cagr"] - baseline_cagr
    dsharpe = r["sharpe"] - baseline_sharpe
    print(f"{name:<12} {r['sharpe']:>7.3f} {r['cagr']:>7.2%} {r['maxdd']:>7.2%} {dcagr:>+7.2%} {dsharpe:>+7.3f}")

# ══════════════════════════════════════════════════
# PART 3: REGIME-BASED LEVERAGE
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 3: REGIME-BASED LEVERAGE")
print("="*60)

# Load regime data from DuckDB
con = duckdb.connect(DB_PATH, read_only=True)

lsr = con.execute("SELECT date, global_account_long_short_ratio as lsr FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date").fetchdf()
lsr["date"] = pd.to_datetime(lsr["date"])
lsr = lsr.set_index("date")

funding = con.execute("SELECT date, close as funding FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date").fetchdf()
funding["date"] = pd.to_datetime(funding["date"])
funding = funding.set_index("date")

liqs = con.execute("SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as total_liq FROM cg_liquidations WHERE symbol='BTC' ORDER BY date").fetchdf()
liqs["date"] = pd.to_datetime(liqs["date"])
liqs = liqs.set_index("date")

taker = con.execute("SELECT date, taker_buy_volume_usd, taker_sell_volume_usd FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date").fetchdf()
taker["date"] = pd.to_datetime(taker["date"])
taker = taker.set_index("date")
taker["buy_sell_ratio"] = taker["taker_buy_volume_usd"] / taker["taker_sell_volume_usd"].replace(0, np.nan)

con.close()

# Compute regime score aligned to common_dates
regime_df = pd.DataFrame(index=common_dates)

# LSR score: +1 if LSR < 50th percentile (30d rolling)
regime_df["lsr"] = lsr["lsr"].reindex(common_dates, method="ffill")
lsr_pctl50 = regime_df["lsr"].rolling(30, min_periods=10).median()
regime_df["lsr_score"] = (regime_df["lsr"] < lsr_pctl50).astype(float)

# Funding score: +1 if funding < 0.03%
regime_df["funding"] = funding["funding"].reindex(common_dates, method="ffill")
regime_df["funding_score"] = (regime_df["funding"] < 0.03).astype(float)

# Liq score: +1 if liq < 80th percentile
regime_df["liq"] = liqs["total_liq"].reindex(common_dates, method="ffill")
liq_pctl80 = regime_df["liq"].rolling(30, min_periods=10).quantile(0.8)
regime_df["liq_score"] = (regime_df["liq"] < liq_pctl80).astype(float)

# Taker score: +1 if buy/sell > 1.0
regime_df["taker_ratio"] = taker["buy_sell_ratio"].reindex(common_dates, method="ffill")
regime_df["taker_score"] = (regime_df["taker_ratio"] > 1.0).astype(float)

# Weighted regime score
regime_df["regime_score"] = (
    0.35 * regime_df["lsr_score"].fillna(0.5) +
    0.35 * regime_df["funding_score"].fillna(0.5) +
    0.15 * regime_df["liq_score"].fillna(0.5) +
    0.15 * regime_df["taker_score"].fillna(0.5)
)

print(f"Regime score: mean={regime_df['regime_score'].mean():.3f}, "
      f"≥0.7: {(regime_df['regime_score']>=0.7).mean():.1%}, "
      f"<0.3: {(regime_df['regime_score']<0.3).mean():.1%}")

# Leverage variants
MARGIN_COST_DAILY = 0.08 / 365  # 8% APY

def apply_leverage(strat_ret_df, pos_df, regime_scores, variant):
    """Apply regime-based leverage and return portfolio equity curve."""
    n_assets = len(strat_ret_df.columns)
    target_w = 1.0 / n_assets
    
    port_ret = pd.Series(0.0, index=strat_ret_df.index)
    
    for sym in strat_ret_df.columns:
        raw_ret = strat_ret_df[sym]
        pos = pos_df[sym]
        
        # Determine leverage multiplier
        if variant == "none":
            lev = pd.Series(1.0, index=strat_ret_df.index)
        elif variant == "mild":
            lev = pd.Series(1.0, index=strat_ret_df.index)
            lev[regime_scores >= 0.7] = 1.3
            lev[regime_scores < 0.3] = 0.5
        elif variant == "moderate":
            lev = pd.Series(1.0, index=strat_ret_df.index)
            lev[regime_scores >= 0.7] = 1.5
            lev[regime_scores < 0.3] = 0.3
        elif variant == "aggressive":
            lev = pd.Series(1.0, index=strat_ret_df.index)
            lev[regime_scores >= 0.7] = 2.0
            lev[regime_scores < 0.3] = 0.0
        elif variant == "binary":
            lev = pd.Series(0.7, index=strat_ret_df.index)
            lev[regime_scores >= 0.5] = 1.5
        
        # Only apply leverage when in position
        effective_lev = np.where(pos > 0, lev, 1.0)
        
        # Margin cost on leveraged portion (lev > 1)
        margin_portion = np.maximum(lev.values - 1.0, 0) * (pos > 0).astype(float).values
        margin_cost = margin_portion * MARGIN_COST_DAILY
        
        asset_ret = raw_ret * effective_lev - margin_cost * target_w
        port_ret += target_w * asset_ret
    
    eq = (1 + port_ret).cumprod()
    return eq

leverage_variants = ["none", "mild", "moderate", "aggressive", "binary"]
leverage_names = {"none": "No Leverage (1.0x)", "mild": "Regime Mild", "moderate": "Regime Moderate",
                  "aggressive": "Regime Aggressive", "binary": "Binary"}

leverage_results = {}
for var in leverage_variants:
    eq = apply_leverage(core_strat_ret, core_pos, regime_df["regime_score"], var)
    m = calc_metrics(eq)
    leverage_results[leverage_names[var]] = m

lev_baseline = leverage_results["No Leverage (1.0x)"]
print(f"\n{'Variant':<22} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'ΔSharpe':>8} {'ΔCAGR':>8}")
print("-" * 60)
for name, r in leverage_results.items():
    ds = r["sharpe"] - lev_baseline["sharpe"]
    dc = r["cagr"] - lev_baseline["cagr"]
    print(f"{name:<22} {r['sharpe']:>7.3f} {r['cagr']:>7.2%} {r['maxdd']:>7.2%} {ds:>+7.3f} {dc:>+7.2%}")

# ══════════════════════════════════════════════════
# PART 4: WALK-FORWARD + COMBINED
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 4: WALK-FORWARD VALIDATION & COMBINED")
print("="*60)

# Pick best from each part
# Best rebalance: highest net Sharpe
best_rebal_name = max(rebal_results.keys(), key=lambda k: rebal_results[k]["sharpe"])
print(f"Best rebalance: {best_rebal_name}")

# Best cash yield: highest Sharpe
best_yield_name = max(cash_yield_results.keys(), key=lambda k: cash_yield_results[k]["sharpe"])
best_yield_apy = {"No Yield": 0, "3% APY": 0.03, "5% APY": 0.05, "8% APY": 0.08}[best_yield_name]
print(f"Best cash yield: {best_yield_name}")

# Best leverage: highest Sharpe
best_lev_name = max(leverage_results.keys(), key=lambda k: leverage_results[k]["sharpe"])
best_lev_var = {v: k for k, v in leverage_names.items()}[best_lev_name]
print(f"Best leverage: {best_lev_name}")

# Combined: apply best rebalancing + cash yield + leverage
def run_combined(strat_ret_df, pos_df, regime_scores, rebal_method, rebal_param, 
                 cash_apy, lev_variant, dates_slice=None):
    """Full combined backtest."""
    if dates_slice is not None:
        strat_ret_df = strat_ret_df.loc[dates_slice]
        pos_df = pos_df.loc[dates_slice]
        regime_scores = regime_scores.loc[dates_slice]
    
    n_assets = len(strat_ret_df.columns)
    target_w = 1.0 / n_assets
    daily_yield = cash_apy / 365
    
    n = len(strat_ret_df)
    weights = np.full(n_assets, target_w)
    portfolio_val = 1.0
    equity = np.zeros(n)
    equity[0] = 1.0
    
    for i in range(1, n):
        rs = regime_scores.iloc[i]
        
        # Determine leverage
        if lev_variant == "none":
            lev = 1.0
        elif lev_variant == "mild":
            lev = 1.3 if rs >= 0.7 else (0.5 if rs < 0.3 else 1.0)
        elif lev_variant == "moderate":
            lev = 1.5 if rs >= 0.7 else (0.3 if rs < 0.3 else 1.0)
        elif lev_variant == "aggressive":
            lev = 2.0 if rs >= 0.7 else (0.0 if rs < 0.3 else 1.0)
        elif lev_variant == "binary":
            lev = 1.5 if rs >= 0.5 else 0.7
        
        new_vals = np.zeros(n_assets)
        for j, sym in enumerate(strat_ret_df.columns):
            r = strat_ret_df.iloc[i, j]
            p = pos_df.iloc[i, j]
            
            eff_lev = lev if p > 0 else 1.0
            margin_cost = max(lev - 1.0, 0) * MARGIN_COST_DAILY if p > 0 else 0
            cash_earn = daily_yield if p == 0 else 0
            
            asset_ret = r * eff_lev - margin_cost + cash_earn
            new_vals[j] = weights[j] * (1 + asset_ret)
        
        pv_new = new_vals.sum()
        if pv_new <= 0:
            pv_new = 1e-10
        weights = new_vals / pv_new
        portfolio_val *= pv_new
        equity[i] = portfolio_val
        
        # Rebalance check
        do_rebal = False
        if rebal_method == "daily":
            do_rebal = True
        elif rebal_method == "weekly" and i % 5 == 0:
            do_rebal = True
        elif rebal_method == "monthly" and i % 21 == 0:
            do_rebal = True
        elif rebal_method == "quarterly" and i % 63 == 0:
            do_rebal = True
        elif rebal_method == "threshold":
            if np.abs(weights - target_w).max() > rebal_param:
                do_rebal = True
        
        if do_rebal:
            weights = np.full(n_assets, target_w)
    
    return pd.Series(equity, index=strat_ret_df.index)

# Map best rebal name to method/param
rebal_map = {
    "Daily": ("daily", None), "Weekly": ("weekly", None),
    "Monthly": ("monthly", None), "Quarterly": ("quarterly", None),
    "Threshold 5%": ("threshold", 0.05), "Threshold 10%": ("threshold", 0.10),
    "No Rebalance": ("none", None),
}
best_rebal_method, best_rebal_param = rebal_map[best_rebal_name]

# Walk-forward on combined
folds = walk_forward_folds(common_dates, n_folds=14)
print(f"\nWalk-forward: {len(folds)} folds")

oos_rets_baseline = []
oos_rets_combined = []

for fold_i, (is_start, oos_start, oos_end) in enumerate(folds):
    oos_dates = common_dates[oos_start:oos_end]
    
    # Baseline (no rebal, no yield, no leverage)
    eq_base = run_combined(core_strat_ret, core_pos, regime_df["regime_score"],
                           "none", None, 0.0, "none", oos_dates)
    
    # Combined best
    eq_comb = run_combined(core_strat_ret, core_pos, regime_df["regime_score"],
                           best_rebal_method, best_rebal_param, best_yield_apy, best_lev_var, oos_dates)
    
    base_ret = eq_base.iloc[-1] / eq_base.iloc[0] - 1 if len(eq_base) > 1 else 0
    comb_ret = eq_comb.iloc[-1] / eq_comb.iloc[0] - 1 if len(eq_comb) > 1 else 0
    oos_rets_baseline.append(base_ret)
    oos_rets_combined.append(comb_ret)

print(f"WF Baseline avg OOS return: {np.mean(oos_rets_baseline):.2%}")
print(f"WF Combined avg OOS return: {np.mean(oos_rets_combined):.2%}")

# Full-period combined vs baseline
eq_baseline_full = run_combined(core_strat_ret, core_pos, regime_df["regime_score"],
                                "none", None, 0.0, "none")
eq_combined_full = run_combined(core_strat_ret, core_pos, regime_df["regime_score"],
                                best_rebal_method, best_rebal_param, best_yield_apy, best_lev_var)

m_baseline = calc_metrics(eq_baseline_full)
m_combined = calc_metrics(eq_combined_full)

print(f"\n{'Metric':<12} {'V4d Baseline':>14} {'Combined Best':>14} {'Delta':>10}")
print("-" * 55)
for metric in ["sharpe", "cagr", "maxdd"]:
    b = m_baseline[metric]
    c = m_combined[metric]
    d = c - b
    if metric in ["cagr", "maxdd"]:
        print(f"{metric:<12} {b:>13.2%} {c:>13.2%} {d:>+9.2%}")
    else:
        print(f"{metric:<12} {b:>14.3f} {c:>14.3f} {d:>+10.3f}")

# Bootstrap CIs on combined
print("\nBootstrap CIs (combined, 1000 iterations)...")
combined_daily_ret = eq_combined_full.pct_change().dropna().values
boot_sharpes = []
boot_cagrs = []
n_days = len(combined_daily_ret)

for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n_days, size=n_days, replace=True)
    sample = combined_daily_ret[idx]
    s = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
    c = (1 + sample).prod() ** (252/n_days) - 1
    boot_sharpes.append(s)
    boot_cagrs.append(c)

print(f"Sharpe 95% CI: [{np.percentile(boot_sharpes, 2.5):.3f}, {np.percentile(boot_sharpes, 97.5):.3f}]")
print(f"CAGR 95% CI:   [{np.percentile(boot_cagrs, 2.5):.2%}, {np.percentile(boot_cagrs, 97.5):.2%}]")

# ── Top 5 assets test ──
print("\n" + "="*60)
print("TOP 5 ASSETS TEST (BTC, ETH, SOL, BNB, AVAX)")
print("="*60)

top5 = ["BTC", "ETH", "SOL", "BNB", "AVAX"]
top5_dates = common_dates.copy()
for sym in top5:
    top5_dates = top5_dates.intersection(spot_data[sym].index)
top5_dates = top5_dates.sort_values()

top5_pos = pd.DataFrame({sym: asset_sigs[sym].reindex(top5_dates).fillna(0) for sym in top5})
top5_raw_ret = pd.DataFrame({sym: asset_rets[sym].reindex(top5_dates).fillna(0) for sym in top5})
top5_strat_ret = top5_raw_ret * top5_pos

regime_top5 = regime_df["regime_score"].reindex(top5_dates, method="ffill").fillna(0.5)

eq_top5_base = run_combined(top5_strat_ret, top5_pos, regime_top5, "none", None, 0.0, "none")
eq_top5_comb = run_combined(top5_strat_ret, top5_pos, regime_top5,
                            best_rebal_method, best_rebal_param, best_yield_apy, best_lev_var)

m_top5_base = calc_metrics(eq_top5_base)
m_top5_comb = calc_metrics(eq_top5_comb)

print(f"\n{'Metric':<12} {'Baseline':>14} {'Combined':>14} {'Delta':>10}")
print("-" * 55)
for metric in ["sharpe", "cagr", "maxdd"]:
    b = m_top5_base[metric]
    c = m_top5_comb[metric]
    d = c - b
    if metric in ["cagr", "maxdd"]:
        print(f"{metric:<12} {b:>13.2%} {c:>13.2%} {d:>+9.2%}")
    else:
        print(f"{metric:<12} {b:>14.3f} {c:>14.3f} {d:>+10.3f}")

# ══════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("VERDICT")
print("="*60)

cagr_add = m_combined["cagr"] - m_baseline["cagr"]
sharpe_add = m_combined["sharpe"] - m_baseline["sharpe"]
print(f"\nExecution optimization adds:")
print(f"  CAGR:   {cagr_add:+.2%} ({m_baseline['cagr']:.2%} → {m_combined['cagr']:.2%})")
print(f"  Sharpe: {sharpe_add:+.3f} ({m_baseline['sharpe']:.3f} → {m_combined['sharpe']:.3f})")
print(f"  MaxDD:  {m_combined['maxdd'] - m_baseline['maxdd']:+.2%}")
print(f"\nBest combo: {best_rebal_name} rebal + {best_yield_name} + {best_lev_name}")
print(f"Cash idle: {avg_flat:.1%} of time")

# ── Save results ──
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
output = {
    "rebalancing": rebal_results,
    "cash_yield": cash_yield_results,
    "leverage": leverage_results,
    "combined_vs_baseline": {
        "baseline": m_baseline,
        "combined": m_combined,
        "best_rebalance": best_rebal_name,
        "best_yield": best_yield_name,
        "best_leverage": best_lev_name,
        "wf_baseline_avg_oos": round(float(np.mean(oos_rets_baseline)), 4),
        "wf_combined_avg_oos": round(float(np.mean(oos_rets_combined)), 4),
        "bootstrap_sharpe_ci": [round(float(np.percentile(boot_sharpes, 2.5)), 3),
                                 round(float(np.percentile(boot_sharpes, 97.5)), 3)],
        "bootstrap_cagr_ci": [round(float(np.percentile(boot_cagrs, 2.5)), 4),
                               round(float(np.percentile(boot_cagrs, 97.5)), 4)],
    },
    "top5": {
        "baseline": m_top5_base,
        "combined": m_top5_comb,
    },
    "flat_pct": {k: round(v, 4) for k, v in flat_pcts.items()},
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT_PATH}")
