#!/usr/bin/env python3
"""
Pyramid & Profit-Taking Test for V4d Framework
12 variants × 14-fold expanding walk-forward × BTC/ETH/SOL portfolio
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Data Loading ──
DATA_DIR = Path.home() / "Desktop/maestro/data/spot"
SYMBOLS = ["BTC", "ETH", "SOL"]
TRAIL_STOPS = {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08}
VOL_CEILING = 0.04  # daily vol ceiling
DD_BREAKER = -0.25  # portfolio DD breaker

def load_data(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_spot_daily.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    # Pre-compute indicators
    df["sma50"] = df["close"].rolling(50).mean()
    df["atr20"] = compute_atr(df, 20)
    df["high20"] = df["high"].rolling(20).max()
    df["rsi14"] = compute_rsi(df["close"], 14)
    df["mom14"] = df["close"] / df["close"].shift(14) - 1
    df["daily_ret"] = df["close"].pct_change()
    df["vol20"] = df["daily_ret"].rolling(20).std()
    return df

def compute_atr(df, period):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

# ── Walk-Forward Splits ──
def get_wf_splits(n_rows, n_folds=14, min_train=252):
    """Expanding window: train grows, test is fixed ~chunk size"""
    chunk = (n_rows - min_train) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * chunk
        test_start = train_end
        test_end = min(test_start + chunk, n_rows)
        if test_start >= n_rows:
            break
        splits.append((0, train_end, test_start, test_end))
    return splits

# ── Simulation Engine ──
@dataclass
class Position:
    entry_price: float
    entry_bar: int
    size: float  # fraction of allocation (e.g., 0.7)
    trail_high: float = 0.0
    partial_taken: bool = False
    pyramid_adds: int = 0
    days_held: int = 0
    total_size: float = 0.0  # track total including adds

    def __post_init__(self):
        self.trail_high = self.entry_price
        self.total_size = self.size

def simulate_variant(df, variant_id, trail_pct, start_idx, end_idx):
    """
    Simulate one variant on one symbol's data slice.
    Returns daily returns array and trade stats.
    """
    cash = 1.0
    pos: Optional[Position] = None
    daily_returns = []
    trades = []
    pyramid_triggers = 0
    pyramid_fires = 0
    profit_take_events = 0
    profit_left_on_table = 0.0
    profit_locked = 0.0
    
    dd_breaker_active = False
    peak_equity = 1.0
    equity = 1.0

    for i in range(start_idx, end_idx):
        if i < 1:
            daily_returns.append(0.0)
            continue
            
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        # Vol ceiling check
        vol = df.iloc[i]["vol20"] if not pd.isna(df.iloc[i]["vol20"]) else 0
        vol_ok = vol < VOL_CEILING
        
        # DD breaker
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity / peak_equity) - 1
        if dd < DD_BREAKER:
            dd_breaker_active = True
        if dd > DD_BREAKER * 0.5:
            dd_breaker_active = False

        # Signal: prev close > prev sma50 (signal bar N, trade bar N+1)
        signal_long = (not pd.isna(prev["sma50"])) and (prev["close"] > prev["sma50"])
        
        day_ret = 0.0
        
        if pos is not None:
            pos.days_held += 1
            price_change = (row["close"] - prev["close"]) / prev["close"]
            day_ret = price_change * pos.total_size
            
            # Update trail high
            if row["high"] > pos.trail_high:
                pos.trail_high = row["high"]
            
            # Determine effective trail stop
            eff_trail = trail_pct
            
            # Variant 9: Progressive tightening
            if variant_id in [9, 11]:
                gain = (row["close"] / pos.entry_price) - 1
                if gain > 0.5:
                    eff_trail = trail_pct * 0.5
            
            # Variant 10: Time-based tightening
            if variant_id == 10:
                if pos.days_held > 120:
                    eff_trail = trail_pct * 0.5
                elif pos.days_held > 60:
                    eff_trail = trail_pct * 0.7
            
            # Check trail stop
            trail_stop = pos.trail_high * (1 - eff_trail)
            stopped = row["low"] <= trail_stop
            
            # Check SMA exit
            sma_exit = not signal_long  # prev close < sma50
            
            # Pyramiding logic
            if variant_id in [2, 11]:  # 20d breakout pyramid
                if pos.pyramid_adds == 0 and pos.days_held >= 2:
                    pyramid_triggers += 1
                    if row["high"] > prev["high20"] and not pd.isna(prev["high20"]):
                        pos.total_size = min(pos.total_size + 0.30, 1.0)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
            
            elif variant_id == 3:  # ATR breakout pyramid
                if not pd.isna(prev["atr20"]):
                    atr = prev["atr20"]
                    if pos.pyramid_adds == 0 and row["high"] > pos.entry_price + atr:
                        pyramid_triggers += 1
                        pos.total_size = min(pos.total_size + 0.15, 1.0)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
                    if pos.pyramid_adds == 1 and row["high"] > pos.entry_price + 2 * atr:
                        pyramid_triggers += 1
                        pos.total_size = min(pos.total_size + 0.15, 1.0)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
            
            elif variant_id == 4:  # Momentum pyramid
                if not pd.isna(prev["mom14"]):
                    if pos.pyramid_adds == 0 and pos.days_held >= 20 and prev["mom14"] > 0:
                        pyramid_triggers += 1
                        pos.total_size = min(pos.total_size + 0.20, 1.0)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
                    if pos.pyramid_adds == 1 and pos.days_held >= 40 and prev["mom14"] > 0:
                        pyramid_triggers += 1
                        pos.total_size = min(pos.total_size + 0.20, 1.0)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
            
            elif variant_id in [5, 12]:  # RSI dip add
                if not pd.isna(prev["rsi14"]) and prev["rsi14"] < 30:
                    pyramid_triggers += 1
                    if pos.total_size < 1.3:
                        pos.total_size = min(pos.total_size + 0.30, 1.3)
                        pos.pyramid_adds += 1
                        pyramid_fires += 1
            
            # Profit-taking logic
            if not pd.isna(prev["atr20"]):
                atr = prev["atr20"]
                gain_abs = row["close"] - pos.entry_price
                
                if variant_id in [7, 12]:  # Partial at 2x ATR
                    if not pos.partial_taken and gain_abs > 2 * atr:
                        take_frac = 0.25
                        realized = take_frac * pos.total_size * (gain_abs / pos.entry_price)
                        profit_locked += realized
                        pos.total_size *= (1 - take_frac)
                        pos.partial_taken = True
                        profit_take_events += 1
                
                elif variant_id == 8:  # Partial at 3x ATR
                    if not pos.partial_taken and gain_abs > 3 * atr:
                        take_frac = 0.33
                        realized = take_frac * pos.total_size * (gain_abs / pos.entry_price)
                        profit_locked += realized
                        pos.total_size *= (1 - take_frac)
                        pos.partial_taken = True
                        profit_take_events += 1
            
            # Exit logic
            if stopped or sma_exit:
                exit_price = trail_stop if stopped else row["close"]
                trade_ret = (exit_price / pos.entry_price) - 1
                trades.append({
                    "ret": trade_ret,
                    "days": pos.days_held,
                    "pyramid_adds": pos.pyramid_adds,
                    "partial_taken": pos.partial_taken,
                    "max_size": pos.total_size
                })
                pos = None
        
        else:
            # Entry logic
            if signal_long and vol_ok and not dd_breaker_active:
                init_size = 1.0
                if variant_id in [2, 3, 11]:
                    init_size = 0.70
                elif variant_id == 4:
                    init_size = 0.60
                # variants 5, 12: full size, can add later
                
                pos = Position(
                    entry_price=row["open"],  # enter on open of trade bar
                    entry_bar=i,
                    size=init_size
                )
        
        daily_returns.append(day_ret)
        equity *= (1 + day_ret)
    
    # Close any open position
    if pos is not None:
        exit_price = df.iloc[end_idx - 1]["close"]
        trade_ret = (exit_price / pos.entry_price) - 1
        trades.append({
            "ret": trade_ret,
            "days": pos.days_held,
            "pyramid_adds": pos.pyramid_adds,
            "partial_taken": pos.partial_taken,
            "max_size": pos.total_size
        })
    
    return {
        "daily_returns": daily_returns,
        "trades": trades,
        "pyramid_triggers": pyramid_triggers,
        "pyramid_fires": pyramid_fires,
        "profit_take_events": profit_take_events,
        "profit_locked": profit_locked,
    }

# ── Portfolio Simulation ──
VARIANT_NAMES = {
    1: "No Pyramid (baseline)",
    2: "20d Breakout Pyramid",
    3: "ATR Breakout Pyramid", 
    4: "Momentum Pyramid",
    5: "RSI Dip Add",
    6: "No Take-Profit (baseline)",
    7: "Partial at 2x ATR",
    8: "Partial at 3x ATR",
    9: "Progressive Tightening",
    10: "Time-Based Trail Tighten",
    11: "Pyramid + Progressive Tighten",
    12: "RSI Dip Add + Partial Take",
}

def run_all():
    # Load data
    data = {}
    for sym in SYMBOLS:
        data[sym] = load_data(sym)
        print(f"Loaded {sym}: {len(data[sym])} rows, {data[sym]['date'].min()} to {data[sym]['date'].max()}")
    
    # Find common date range
    common_start = max(d["date"].min() for d in data.values())
    common_end = min(d["date"].max() for d in data.values())
    print(f"Common range: {common_start} to {common_end}")
    
    for sym in SYMBOLS:
        mask = (data[sym]["date"] >= common_start) & (data[sym]["date"] <= common_end)
        data[sym] = data[sym][mask].reset_index(drop=True)
    
    n_rows = len(data["BTC"])
    splits = get_wf_splits(n_rows, n_folds=14, min_train=252)
    print(f"Walk-forward: {len(splits)} folds, {n_rows} common rows")
    
    results = {}
    
    for vid in range(1, 13):
        print(f"\n── Variant {vid}: {VARIANT_NAMES[vid]} ──")
        
        all_oos_returns = []
        all_trades = []
        total_pyramid_triggers = 0
        total_pyramid_fires = 0
        total_profit_takes = 0
        total_profit_locked = 0.0
        
        for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
            # Portfolio: equal-weight 1/3 each
            fold_daily = np.zeros(te_e - te_s)
            
            for sym in SYMBOLS:
                trail = TRAIL_STOPS[sym]
                res = simulate_variant(data[sym], vid, trail, te_s, te_e)
                rets = np.array(res["daily_returns"])
                fold_daily += rets / 3.0
                all_trades.extend(res["trades"])
                total_pyramid_triggers += res["pyramid_triggers"]
                total_pyramid_fires += res["pyramid_fires"]
                total_profit_takes += res["profit_take_events"]
                total_profit_locked += res["profit_locked"]
            
            all_oos_returns.extend(fold_daily.tolist())
        
        # Compute metrics
        rets = np.array(all_oos_returns)
        cum = np.cumprod(1 + rets)
        total_ret = cum[-1] - 1
        n_years = len(rets) / 252
        cagr = (cum[-1]) ** (1 / max(n_years, 0.01)) - 1
        
        # Sharpe
        sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(252)
        
        # Max DD
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Trade stats
        n_trades = len(all_trades)
        avg_dur = np.mean([t["days"] for t in all_trades]) if all_trades else 0
        avg_profit = np.mean([t["ret"] for t in all_trades]) if all_trades else 0
        win_rate = np.mean([1 if t["ret"] > 0 else 0 for t in all_trades]) if all_trades else 0
        
        pyramid_util = total_pyramid_fires / max(total_pyramid_triggers, 1)
        
        results[vid] = {
            "name": VARIANT_NAMES[vid],
            "sharpe": round(sharpe, 3),
            "cagr": round(cagr * 100, 2),
            "max_dd": round(max_dd * 100, 2),
            "calmar": round(calmar, 3),
            "total_ret": round(total_ret * 100, 2),
            "n_trades": n_trades,
            "avg_duration_days": round(avg_dur, 1),
            "avg_profit_pct": round(avg_profit * 100, 2),
            "win_rate": round(win_rate * 100, 1),
            "pyramid_triggers": total_pyramid_triggers,
            "pyramid_fires": total_pyramid_fires,
            "pyramid_util_pct": round(pyramid_util * 100, 1),
            "profit_take_events": total_profit_takes,
            "profit_locked_pct": round(total_profit_locked * 100, 2),
        }
        
        print(f"  Sharpe={sharpe:.3f}  CAGR={cagr*100:.1f}%  MaxDD={max_dd*100:.1f}%  Calmar={calmar:.3f}  Trades={n_trades}  WR={win_rate*100:.0f}%")
    
    # ── Save results ──
    out_dir = Path.home() / "Desktop/maestro/data/backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pyramid_profittake_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'pyramid_profittake_test.json'}")
    
    # ── Print Comparison Table ──
    print("\n" + "="*130)
    print("FULL COMPARISON TABLE — 12 Variants (14-fold Expanding WF, BTC/ETH/SOL Portfolio)")
    print("="*130)
    header = f"{'#':>2} {'Variant':<35} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} {'Calmar':>7} {'Trades':>6} {'AvgDur':>6} {'AvgP%':>6} {'WR%':>5}"
    print(header)
    print("-"*130)
    for vid in range(1, 13):
        r = results[vid]
        print(f"{vid:>2} {r['name']:<35} {r['sharpe']:>7.3f} {r['cagr']:>7.2f} {r['max_dd']:>7.2f} {r['calmar']:>7.3f} {r['n_trades']:>6} {r['avg_duration_days']:>6.1f} {r['avg_profit_pct']:>6.2f} {r['win_rate']:>5.1f}")
    
    # ── Pyramid Stats ──
    print("\n" + "="*90)
    print("PYRAMID TRIGGER FREQUENCY & IMPACT")
    print("="*90)
    print(f"{'#':>2} {'Variant':<35} {'Triggers':>8} {'Fires':>6} {'Util%':>6} {'Sharpe Δ':>9}")
    print("-"*90)
    baseline_sharpe = results[1]["sharpe"]
    for vid in [2, 3, 4, 5, 11, 12]:
        r = results[vid]
        delta = r["sharpe"] - baseline_sharpe
        print(f"{vid:>2} {r['name']:<35} {r['pyramid_triggers']:>8} {r['pyramid_fires']:>6} {r['pyramid_util_pct']:>6.1f} {delta:>+9.3f}")
    
    # ── Profit-Taking Stats ──
    print("\n" + "="*90)
    print("PROFIT-TAKING STATS")
    print("="*90)
    print(f"{'#':>2} {'Variant':<35} {'PT Events':>9} {'Locked%':>8} {'CAGR Δ':>8} {'MaxDD Δ':>8}")
    print("-"*90)
    baseline_cagr = results[6]["cagr"]
    baseline_dd = results[6]["max_dd"]
    for vid in [7, 8, 9, 10, 11, 12]:
        r = results[vid]
        cagr_d = r["cagr"] - baseline_cagr
        dd_d = r["max_dd"] - baseline_dd  # less negative = improvement
        print(f"{vid:>2} {r['name']:<35} {r['profit_take_events']:>9} {r['profit_locked_pct']:>8.2f} {cagr_d:>+8.2f} {dd_d:>+8.2f}")
    
    # ── VERDICT ──
    print("\n" + "="*90)
    print("VERDICT")
    print("="*90)
    
    # Rank by Sharpe, then Calmar
    ranked = sorted(results.items(), key=lambda x: (x[1]["sharpe"], x[1]["calmar"]), reverse=True)
    best_id, best = ranked[0]
    
    print(f"\n🏆 BEST VARIANT: #{best_id} — {best['name']}")
    print(f"   Sharpe: {best['sharpe']:.3f}  |  CAGR: {best['cagr']:.2f}%  |  MaxDD: {best['max_dd']:.2f}%  |  Calmar: {best['calmar']:.3f}")
    
    # Top 3
    print(f"\n   Top 3 by Sharpe:")
    for rank, (vid, r) in enumerate(ranked[:3], 1):
        print(f"   {rank}. #{vid} {r['name']} — Sharpe {r['sharpe']:.3f}, Calmar {r['calmar']:.3f}")
    
    # Compare vs baseline
    print(f"\n   vs Baseline (#1): Sharpe Δ = {best['sharpe'] - results[1]['sharpe']:+.3f}, CAGR Δ = {best['cagr'] - results[1]['cagr']:+.2f}%")
    
    # Recommendation
    print(f"\n   RECOMMENDATION: Use Variant #{best_id} for V4d production.")
    if best_id in [2, 3, 4, 5, 11, 12]:
        print(f"   Pyramid utilization: {best['pyramid_fires']}/{best['pyramid_triggers']} triggers fired ({best['pyramid_util_pct']:.0f}%)")
    if best_id in [7, 8, 9, 10, 11, 12]:
        print(f"   Profit-taking events: {best['profit_take_events']}, locked {best['profit_locked_pct']:.2f}% incremental")

if __name__ == "__main__":
    run_all()
