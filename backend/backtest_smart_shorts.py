#!/usr/bin/env python3
"""
Smart Shorts Only Framework
Tests whether combining macro regime + derivatives extremes produces reliable short-side alpha.
Key question: Do smart shorts beat simply going flat during bear regimes?
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ── Paths ──
DATA_DIR = Path.home() / "Desktop/maestro/data"
OHLCV_DIR = DATA_DIR / "ohlcv"
DERIV_DIR = DATA_DIR / "derivatives"
RESULTS_DIR = DATA_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

TOKENS = ["arb", "avax", "btc", "eth", "fet", "inj", "link", "op", "sol", "sui"]
FOCUS_TOKENS = ["btc", "eth", "sol"]
COMMISSION = 0.002  # 20bps per trade

# ── Data Loading ──

def load_ohlcv(token: str) -> pd.DataFrame:
    df = pd.read_csv(OHLCV_DIR / f"binance_{token}_usdt_1d.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[~df.index.duplicated(keep='first')]
    return df

def load_funding(token: str) -> pd.DataFrame:
    """Load 8-hourly funding and resample to daily average rate."""
    path = DERIV_DIR / f"{token}_funding.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["fundingRate"].mean().reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("timestamp")[["fundingRate"]]
    return daily

def load_funding_full(token: str) -> pd.DataFrame:
    path = DERIV_DIR / f"{token}_funding_full.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[~df.index.duplicated(keep='first')]
    # funding_rate column
    if "funding_rate" in df.columns:
        df = df.rename(columns={"funding_rate": "fundingRate"})
    return df

def load_oi_full(token: str) -> pd.DataFrame:
    path = DERIV_DIR / f"{token}_oi_daily_full.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[~df.index.duplicated(keep='first')]
    # Use 'c' (close) as OI value
    if "c" in df.columns:
        df["oi"] = df["c"]
    return df

def load_lsr_full(token: str) -> pd.DataFrame:
    path = DERIV_DIR / f"{token}_lsr_daily_full.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[~df.index.duplicated(keep='first')]
    # long_ratio / short_ratio → LSR
    if "long_ratio" in df.columns and "short_ratio" in df.columns:
        df["lsr"] = df["long_ratio"] / df["short_ratio"].replace(0, np.nan)
    return df

def build_dataset(token: str) -> pd.DataFrame:
    """Merge OHLCV with derivatives data."""
    ohlcv = load_ohlcv(token)
    
    # Try full derivatives first (744 days, 2024-2026), fall back to shorter
    funding = load_funding_full(token)
    if funding.empty:
        funding = load_funding(token)
    
    oi = load_oi_full(token)
    lsr = load_lsr_full(token)
    
    df = ohlcv.copy()
    
    if not funding.empty and "fundingRate" in funding.columns:
        df = df.join(funding[["fundingRate"]], how="left")
    
    if not oi.empty and "oi" in oi.columns:
        df = df.join(oi[["oi"]], how="left")
    
    if not lsr.empty and "lsr" in lsr.columns:
        df = df.join(lsr[["lsr"]], how="left")
    
    return df


# ── Indicator Calculation ──

def add_indicators(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add all indicators needed for the framework."""
    d = df.copy()
    c = d["close"]
    
    # SMAs
    d["sma50"] = c.rolling(50).mean()
    d["sma60"] = c.rolling(60).mean()
    d["sma100"] = c.rolling(100).mean()
    d["sma200"] = c.rolling(200).mean()
    
    # SMA slopes (10d change)
    d["sma50_slope"] = d["sma50"].diff(10)
    d["sma200_slope"] = d["sma200"].diff(10)
    
    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))
    
    # Bollinger-like: std above 60d SMA
    d["std60"] = c.rolling(60).std()
    d["zscore_price"] = (c - d["sma60"]) / d["std60"].replace(0, np.nan)
    
    # 14d return
    d["ret14d"] = c.pct_change(14)
    
    # 20d high
    d["high20d"] = c.rolling(20).max()
    d["is_20d_high"] = (c >= d["high20d"])
    
    # Daily return
    d["ret"] = c.pct_change()
    
    # ── Macro regime (uses BTC data) ──
    if btc_df is not None:
        btc = btc_df.copy()
        btc["btc_sma200"] = btc["close"].rolling(200).mean()
        btc["btc_sma50"] = btc["close"].rolling(50).mean()
        btc["btc_sma50_slope"] = btc["btc_sma50"].diff(10)
        btc["btc_below_200"] = btc["close"] < btc["btc_sma200"]
        btc["btc_slope_neg"] = btc["btc_sma50_slope"] < 0
        btc["bear_regime"] = btc["btc_below_200"] & btc["btc_slope_neg"]
        d = d.join(btc[["bear_regime"]], how="left")
    else:
        # Token IS BTC
        d["btc_sma200"] = c.rolling(200).mean()
        d["btc_sma50"] = c.rolling(50).mean()
        d["btc_sma50_slope"] = d["btc_sma50"].diff(10)
        d["bear_regime"] = (c < d["btc_sma200"]) & (d["btc_sma50_slope"] < 0)
    
    d["bear_regime"] = d["bear_regime"].fillna(False).astype(bool)
    
    # ── Derivatives indicators ──
    if "oi" in d.columns:
        d["oi_change"] = d["oi"].pct_change()
        d["oi_z"] = (d["oi_change"] - d["oi_change"].rolling(30).mean()) / d["oi_change"].rolling(30).std().replace(0, np.nan)
        d["oi_declining_20d"] = d["oi"].diff(20) < 0
    
    if "fundingRate" in d.columns:
        d["fr_z"] = (d["fundingRate"] - d["fundingRate"].rolling(30).mean()) / d["fundingRate"].rolling(30).std().replace(0, np.nan)
    
    if "lsr" in d.columns:
        d["lsr_z"] = (d["lsr"] - d["lsr"].rolling(30).mean()) / d["lsr"].rolling(30).std().replace(0, np.nan)
    
    return d


# ── Trigger Functions ──

def trigger_ctus(df: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
    """Crowded Trade Unwinding Signal. ALL three z-scores > threshold."""
    has_all = all(c in df.columns for c in ["oi_z", "fr_z", "lsr_z"])
    if not has_all:
        return pd.Series(False, index=df.index)
    return (df["oi_z"] > threshold) & (df["fr_z"] > threshold) & (df["lsr_z"] > threshold)

def trigger_funding_euphoria(df: pd.DataFrame, threshold: float = 0.0005, sustained_days: int = 3) -> pd.Series:
    """Funding rate > threshold sustained for N+ days."""
    if "fundingRate" not in df.columns:
        return pd.Series(False, index=df.index)
    high_fr = df["fundingRate"] > threshold
    # Rolling sum of consecutive True days
    sustained = high_fr.rolling(sustained_days).sum() >= sustained_days
    return sustained.fillna(False)

def trigger_momentum_exhaustion(df: pd.DataFrame) -> pd.Series:
    """Price > 2 std above 60d SMA, RSI > 80, 14d return > 30%."""
    return (
        (df["zscore_price"] > 2) & 
        (df["rsi14"] > 80) & 
        (df["ret14d"] > 0.30)
    ).fillna(False)

def trigger_oi_divergence(df: pd.DataFrame) -> pd.Series:
    """Price making 20d highs but OI declining over 20d."""
    if "oi_declining_20d" not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["is_20d_high"] & df["oi_declining_20d"]).fillna(False)


# ── Backtesting Engine ──

def simulate_strategy(df: pd.DataFrame, signals: pd.Series, mode: str = "short_only",
                      tp: float = 0.15, sl: float = 0.10, time_stop: int = 30,
                      leverage_func=None) -> pd.DataFrame:
    """
    Simulate trading based on signals.
    signals: -1 = short, 0 = flat, 1 = long
    mode: 'short_only' (flat when no signal) or 'long_biased' (long when no short)
    Returns DataFrame with daily returns.
    """
    n = len(df)
    position = np.zeros(n)  # -1, 0, 1
    leverage = np.ones(n) * 0.5
    entry_price = np.zeros(n)
    hold_days = np.zeros(n)
    trade_returns = []
    
    if leverage_func is None:
        leverage_func = lambda row_idx: 0.5
    
    in_trade = False
    trade_dir = 0
    trade_entry = 0.0
    trade_hold = 0
    
    closes = df["close"].values
    sig = signals.values
    bear = df["bear_regime"].values if "bear_regime" in df.columns else np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if in_trade:
            trade_hold += 1
            pnl_pct = (closes[i] / trade_entry - 1) * trade_dir
            
            # Exit conditions
            exit_trade = False
            if trade_dir == -1:
                if pnl_pct >= tp:  # TP hit (price dropped enough)
                    exit_trade = True
                elif pnl_pct <= -sl:  # SL hit (price rose)
                    exit_trade = True
                elif trade_hold >= time_stop:
                    exit_trade = True
                elif not bear[i] and trade_dir == -1:  # Macro flipped bull
                    exit_trade = True
                elif sig[i] == 0 and trade_dir == -1:  # Signal reversed
                    exit_trade = True
            
            if exit_trade:
                # Close trade
                ret = pnl_pct * leverage[i-1] - COMMISSION
                trade_returns.append({
                    "entry_idx": i - trade_hold,
                    "exit_idx": i,
                    "direction": trade_dir,
                    "hold_days": trade_hold,
                    "pnl_pct": pnl_pct,
                    "ret": ret
                })
                in_trade = False
                trade_dir = 0
                position[i] = 0
                
                # Check if we should enter new position
                if mode == "long_biased" and sig[i] >= 0:
                    position[i] = 1
                    leverage[i] = 1.0
            else:
                position[i] = trade_dir
                leverage[i] = leverage[i-1]
        
        if not in_trade:
            if sig[i] == -1:
                # Enter short
                in_trade = True
                trade_dir = -1
                trade_entry = closes[i]
                trade_hold = 0
                position[i] = -1
                leverage[i] = leverage_func(i)
            elif mode == "long_biased":
                position[i] = 1
                leverage[i] = 1.0
    
    # Calculate daily returns
    daily_ret = df["ret"].values
    strat_ret = np.zeros(n)
    for i in range(1, n):
        if position[i] != 0:
            strat_ret[i] = daily_ret[i] * position[i] * leverage[i]
            # Deduct commission on position changes
            if position[i] != position[i-1]:
                strat_ret[i] -= COMMISSION
    
    result = pd.DataFrame({
        "date": df.index,
        "close": closes,
        "position": position,
        "strat_ret": strat_ret,
        "bear_regime": bear
    }).set_index("date")
    
    result["equity"] = (1 + result["strat_ret"]).cumprod()
    result["bnh_ret"] = df["ret"].values
    result["bnh_equity"] = (1 + result["bnh_ret"].fillna(0)).cumprod()
    
    return result, trade_returns


def compute_metrics(result: pd.DataFrame, trades: list) -> dict:
    """Compute strategy metrics."""
    r = result["strat_ret"].dropna()
    eq = result["equity"]
    
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 0 else 0
    ann_ret = (1 + total_ret) ** (365 / max(len(r), 1)) - 1
    
    vol = r.std() * np.sqrt(365) if r.std() > 0 else 0
    sharpe = ann_ret / vol if vol > 0 else 0
    
    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        short_trades = [t for t in trades if t["direction"] == -1]
        n_shorts = len(short_trades)
        win_rate = sum(1 for t in short_trades if t["pnl_pct"] > 0) / max(n_shorts, 1)
        avg_hold = np.mean([t["hold_days"] for t in short_trades]) if n_shorts > 0 else 0
        
        wins = [t["ret"] for t in short_trades if t["ret"] > 0]
        losses = [t["ret"] for t in short_trades if t["ret"] <= 0]
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf') if wins else 0
    else:
        n_shorts = 0
        win_rate = 0
        avg_hold = 0
        profit_factor = 0
    
    # BnH comparison
    bnh_ret = result["bnh_equity"].iloc[-1] / result["bnh_equity"].iloc[0] - 1 if len(result) > 0 else 0
    
    # Per-year breakdown
    yearly = {}
    for year in result.index.year.unique():
        yr = result[result.index.year == year]
        yr_ret = (1 + yr["strat_ret"]).prod() - 1
        yr_bnh = (1 + yr["bnh_ret"].fillna(0)).prod() - 1
        yr_shorts = len([t for t in trades if t["direction"] == -1 and result.index[t["entry_idx"]].year == year]) if trades else 0
        yearly[int(year)] = {
            "strat_return": round(float(yr_ret), 4),
            "bnh_return": round(float(yr_bnh), 4),
            "n_shorts": yr_shorts
        }
    
    # Max DD during shorts only
    short_mask = result["position"] == -1
    if short_mask.any():
        short_eq = (1 + result.loc[short_mask, "strat_ret"]).cumprod()
        short_peak = short_eq.cummax()
        short_dd = ((short_eq - short_peak) / short_peak).min()
    else:
        short_dd = 0
    
    return {
        "total_return": round(float(total_ret), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(max_dd), 4),
        "max_dd_shorts": round(float(short_dd), 4),
        "n_trades": n_trades,
        "n_shorts": n_shorts if n_trades > 0 else 0,
        "win_rate": round(float(win_rate), 3),
        "avg_hold_days": round(float(avg_hold), 1),
        "profit_factor": round(float(profit_factor), 3) if profit_factor != float('inf') else 999,
        "bnh_return": round(float(bnh_ret), 4),
        "yearly": yearly
    }


# ── Strategy Variants ──

def generate_signals(df: pd.DataFrame, variant: str, ctus_thresh: float = 1.5,
                     funding_thresh: float = 0.0005) -> pd.Series:
    """Generate signals for each variant. Returns Series of -1/0."""
    n = len(df)
    signals = pd.Series(0, index=df.index)
    
    bear = df["bear_regime"].fillna(False)
    
    # Compute triggers
    tA = trigger_ctus(df, ctus_thresh)
    tB = trigger_funding_euphoria(df, funding_thresh)
    tC = trigger_momentum_exhaustion(df)
    tD = trigger_oi_divergence(df)
    
    trigger_count = tA.astype(int) + tB.astype(int) + tC.astype(int) + tD.astype(int)
    any_trigger = trigger_count >= 1
    two_triggers = trigger_count >= 2
    
    if variant == "V0":
        # Always short
        signals[:] = -1
    elif variant == "V1":
        # Macro only
        signals[bear] = -1
    elif variant == "V2":
        # Macro + any 1 trigger
        signals[bear & any_trigger] = -1
    elif variant == "V3":
        # Macro + 2+ triggers
        signals[bear & two_triggers] = -1
    elif variant == "V4":
        # CTUS only (no macro gate)
        signals[tA] = -1
    elif variant == "V5":
        # Funding only (no macro gate)
        signals[tB] = -1
    elif variant == "V6":
        # Full framework: Macro + any trigger (best combo)
        signals[bear & any_trigger] = -1
    elif variant == "FLAT_BEAR":
        # Flat during bear (comparison)
        signals[:] = 0  # Will be used as long-biased with flat during bear
    
    return signals


def get_leverage_func(df: pd.DataFrame, ctus_thresh: float = 1.5, funding_thresh: float = 0.0005):
    """Returns leverage function based on trigger count."""
    tA = trigger_ctus(df, ctus_thresh)
    tB = trigger_funding_euphoria(df, funding_thresh)
    tC = trigger_momentum_exhaustion(df)
    tD = trigger_oi_divergence(df)
    trigger_count = (tA.astype(int) + tB.astype(int) + tC.astype(int) + tD.astype(int)).values
    
    def lev(i):
        tc = trigger_count[i] if i < len(trigger_count) else 1
        if tc >= 3:
            return 1.5
        elif tc >= 2:
            return 1.0
        else:
            return 0.5
    return lev


# ── Walk-Forward ──

def walk_forward_test(df: pd.DataFrame, variant: str, n_folds: int = 10,
                      mode: str = "short_only") -> dict:
    """Expanding window walk-forward test."""
    n = len(df)
    min_train = max(250, n // (n_folds + 1))  # At least 250 days training
    fold_size = (n - min_train) // n_folds
    
    if fold_size < 20:
        # Not enough data for walk-forward
        return {"error": "insufficient_data", "n_rows": n}
    
    all_oos_returns = []
    fold_results = []
    all_trades = []
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        
        if test_end <= train_end:
            break
        
        test_df = df.iloc[train_end:test_end].copy()
        if len(test_df) < 10:
            continue
        
        signals = generate_signals(test_df, variant)
        lev_func = get_leverage_func(test_df)
        
        result, trades = simulate_strategy(test_df, signals, mode=mode, leverage_func=lev_func)
        metrics = compute_metrics(result, trades)
        
        fold_results.append({
            "fold": fold,
            "train_end": str(df.index[train_end]),
            "test_end": str(df.index[min(test_end - 1, n - 1)]),
            "test_days": len(test_df),
            **metrics
        })
        
        all_oos_returns.extend(result["strat_ret"].tolist())
        all_trades.extend(trades)
    
    if not fold_results:
        return {"error": "no_folds_completed"}
    
    # Aggregate OOS metrics
    oos_returns = np.array(all_oos_returns)
    total_eq = np.cumprod(1 + oos_returns)
    total_ret = total_eq[-1] - 1 if len(total_eq) > 0 else 0
    
    ann_ret = (1 + total_ret) ** (365 / max(len(oos_returns), 1)) - 1
    vol = oos_returns.std() * np.sqrt(365)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    peak = np.maximum.accumulate(total_eq)
    max_dd = ((total_eq - peak) / peak).min()
    
    # P-value via t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(oos_returns[oos_returns != 0], 0) if (oos_returns != 0).sum() > 5 else (0, 1)
    
    n_shorts = len([t for t in all_trades if t["direction"] == -1])
    
    return {
        "variant": variant,
        "mode": mode,
        "oos_total_return": round(float(total_ret), 4),
        "oos_ann_return": round(float(ann_ret), 4),
        "oos_sharpe": round(float(sharpe), 3),
        "oos_max_dd": round(float(max_dd), 4),
        "oos_p_value": round(float(p_value), 4),
        "total_short_trades": n_shorts,
        "avg_fold_metrics": {
            k: round(float(np.mean([f[k] for f in fold_results if k in f])), 4)
            for k in ["total_return", "sharpe", "win_rate", "profit_factor", "n_shorts"]
        },
        "fold_details": fold_results
    }


def permutation_test(df: pd.DataFrame, variant: str, mode: str, n_perms: int = 200) -> float:
    """Randomized permutation test for significance."""
    # Get actual strategy return
    signals = generate_signals(df, variant)
    lev_func = get_leverage_func(df)
    result, _ = simulate_strategy(df, signals, mode=mode, leverage_func=lev_func)
    actual_ret = result["equity"].iloc[-1] / result["equity"].iloc[0] - 1
    
    # Random permutations
    better_count = 0
    for _ in range(n_perms):
        # Shuffle the signal dates
        perm_vals = np.random.permutation(signals.values)
        perm_signals = pd.Series(perm_vals, index=signals.index)
        result_p, _ = simulate_strategy(df, perm_signals, mode=mode, leverage_func=lev_func)
        perm_ret = result_p["equity"].iloc[-1] / result_p["equity"].iloc[0] - 1
        if perm_ret >= actual_ret:
            better_count += 1
    
    return better_count / n_perms


# ── Main Execution ──

def run_full_analysis():
    print("=" * 80)
    print("SMART SHORTS FRAMEWORK — Full Analysis")
    print("=" * 80)
    
    # Load BTC for macro regime
    btc_ohlcv = load_ohlcv("btc")
    
    results = {
        "metadata": {
            "tokens_tested": FOCUS_TOKENS,
            "commission": COMMISSION,
            "walk_forward_folds": 10,
            "date_range": f"{btc_ohlcv.index[0]} to {btc_ohlcv.index[-1]}"
        },
        "by_token": {},
        "portfolio": {},
        "key_findings": {}
    }
    
    variants = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
    modes = ["short_only", "long_biased"]
    
    for token in FOCUS_TOKENS:
        print(f"\n{'='*60}")
        print(f"  Processing {token.upper()}")
        print(f"{'='*60}")
        
        df = build_dataset(token)
        btc_ref = btc_ohlcv if token != "btc" else None
        df = add_indicators(df, btc_ref)
        
        # Print data availability
        has_oi = "oi" in df.columns and df["oi"].notna().sum() > 30
        has_fr = "fundingRate" in df.columns and df["fundingRate"].notna().sum() > 30
        has_lsr = "lsr" in df.columns and df["lsr"].notna().sum() > 30
        print(f"  Data: OI={'✓' if has_oi else '✗'} Funding={'✓' if has_fr else '✗'} LSR={'✓' if has_lsr else '✗'}")
        print(f"  Rows: {len(df)}, Bear regime days: {df['bear_regime'].sum()}")
        
        # Check triggers firing
        tA = trigger_ctus(df)
        tB = trigger_funding_euphoria(df)
        tC = trigger_momentum_exhaustion(df)
        tD = trigger_oi_divergence(df)
        print(f"  Trigger counts: CTUS={tA.sum()}, FundEuph={tB.sum()}, MomExh={tC.sum()}, OIDiv={tD.sum()}")
        
        token_results = {}
        
        for variant in variants:
            for mode in modes:
                key = f"{variant}_{mode[0].upper()}"  # e.g. V0_S, V0_L
                print(f"  Testing {key}...", end=" ")
                
                wf = walk_forward_test(df, variant, n_folds=10, mode=mode)
                
                if "error" in wf:
                    print(f"SKIP ({wf['error']})")
                    token_results[key] = wf
                    continue
                
                print(f"Return={wf['oos_total_return']:.1%} Sharpe={wf['oos_sharpe']:.2f} "
                      f"Shorts={wf['total_short_trades']} p={wf['oos_p_value']:.3f}")
                
                # Permutation test if significant
                if wf['oos_p_value'] < 0.05 and wf['total_short_trades'] > 3:
                    print(f"    → Running 200 permutation tests...")
                    perm_p = permutation_test(df, variant, mode, n_perms=200)
                    wf["permutation_p"] = round(perm_p, 4)
                    print(f"    → Permutation p-value: {perm_p:.3f}")
                
                token_results[key] = wf
        
        # Also test "flat during bear" baseline
        for mode in ["long_biased"]:
            print(f"  Testing FLAT_BEAR_{mode[0].upper()}...", end=" ")
            # Long always, flat during bear
            flat_signals = pd.Series(0, index=df.index)
            # In long_biased mode with all-zero signals = always long
            # We want: long during bull, flat during bear
            result_fb, trades_fb = simulate_strategy(df, flat_signals, mode="long_biased")
            # Override: set position to 0 during bear
            bear_mask = df["bear_regime"].values
            daily_ret = df["ret"].values
            strat_ret = np.zeros(len(df))
            pos = np.zeros(len(df))
            for i in range(1, len(df)):
                if not bear_mask[i]:
                    pos[i] = 1
                    strat_ret[i] = daily_ret[i]
                    if pos[i] != pos[i-1]:
                        strat_ret[i] -= COMMISSION
            
            fb_result = pd.DataFrame({
                "date": df.index,
                "close": df["close"].values,
                "position": pos,
                "strat_ret": strat_ret,
                "bnh_ret": daily_ret,
                "bear_regime": bear_mask
            }).set_index("date")
            fb_result["equity"] = (1 + fb_result["strat_ret"]).cumprod()
            fb_result["bnh_equity"] = (1 + fb_result["bnh_ret"].fillna(0)).cumprod()
            fb_metrics = compute_metrics(fb_result, [])
            print(f"Return={fb_metrics['total_return']:.1%} Sharpe={fb_metrics['sharpe']:.2f}")
            token_results["FLAT_BEAR_L"] = {"full_period": fb_metrics}
        
        # Buy and hold baseline
        bnh_eq = (1 + df["ret"].fillna(0)).cumprod()
        bnh_ret = bnh_eq.iloc[-1] - 1
        bnh_vol = df["ret"].std() * np.sqrt(365)
        bnh_ann = (1 + bnh_ret) ** (365 / len(df)) - 1
        bnh_sharpe = bnh_ann / bnh_vol if bnh_vol > 0 else 0
        bnh_peak = bnh_eq.cummax()
        bnh_dd = ((bnh_eq - bnh_peak) / bnh_peak).min()
        
        token_results["BUY_AND_HOLD"] = {
            "total_return": round(float(bnh_ret), 4),
            "ann_return": round(float(bnh_ann), 4),
            "sharpe": round(float(bnh_sharpe), 3),
            "max_dd": round(float(bnh_dd), 4)
        }
        
        results["by_token"][token] = token_results
    
    # ── Portfolio Analysis (equal-weight BTC+ETH+SOL) ──
    print(f"\n{'='*60}")
    print(f"  Portfolio Analysis (Equal-weight BTC+ETH+SOL)")
    print(f"{'='*60}")
    
    for variant in variants:
        for mode in modes:
            key = f"{variant}_{mode[0].upper()}"
            port_rets = []
            
            for token in FOCUS_TOKENS:
                tk = results["by_token"].get(token, {}).get(key, {})
                if "fold_details" in tk:
                    for fold in tk["fold_details"]:
                        port_rets.append(fold.get("total_return", 0))
            
            if port_rets:
                avg_ret = np.mean(port_rets)
                results["portfolio"][key] = {
                    "avg_fold_return": round(float(avg_ret), 4),
                    "n_folds": len(port_rets)
                }
    
    # ── Key Findings ──
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    # Compare: smart shorts vs flat during bear
    for token in FOCUS_TOKENS:
        tr = results["by_token"][token]
        bnh = tr.get("BUY_AND_HOLD", {})
        flat = tr.get("FLAT_BEAR_L", {}).get("full_period", {})
        
        # Best short variant (long-biased mode)
        best_short_key = None
        best_short_sharpe = -999
        for v in variants:
            k = f"{v}_L"
            vr = tr.get(k, {})
            if isinstance(vr, dict) and "oos_sharpe" in vr:
                if vr["oos_sharpe"] > best_short_sharpe:
                    best_short_sharpe = vr["oos_sharpe"]
                    best_short_key = k
        
        finding = {
            "bnh_return": bnh.get("total_return"),
            "bnh_sharpe": bnh.get("sharpe"),
            "flat_bear_return": flat.get("total_return"),
            "flat_bear_sharpe": flat.get("sharpe"),
            "best_short_variant": best_short_key,
            "best_short_sharpe": round(best_short_sharpe, 3) if best_short_sharpe > -999 else None,
        }
        
        # Key comparison
        flat_sharpe = flat.get("sharpe", 0)
        if best_short_sharpe > flat_sharpe:
            finding["verdict"] = "SHORTS_ADD_VALUE"
        else:
            finding["verdict"] = "FLAT_IS_BETTER"
        
        results["key_findings"][token] = finding
        
        print(f"\n{token.upper()}:")
        print(f"  Buy&Hold:    Return={bnh.get('total_return', 'N/A'):.1%}  Sharpe={bnh.get('sharpe', 'N/A')}")
        print(f"  Flat Bear:   Return={flat.get('total_return', 'N/A'):.1%}  Sharpe={flat.get('sharpe', 'N/A')}")
        print(f"  Best Short:  {best_short_key} Sharpe={best_short_sharpe:.3f}")
        print(f"  → Verdict:   {finding['verdict']}")
    
    # ── Save Results ──
    output_path = RESULTS_DIR / "smart_shorts_results.json"
    
    # Clean NaN/inf for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (float, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return round(float(obj), 6)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
    
    with open(output_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_path}")
    return results


if __name__ == "__main__":
    results = run_full_analysis()
