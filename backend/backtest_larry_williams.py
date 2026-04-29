#!/usr/bin/env python3
"""
Larry Williams Market Structure Backtest
From "Long-Term Secrets to Short-Term Trading"

Mechanical fractal swing point hierarchy:
- Short-term low: daily low with HIGHER lows on BOTH sides (N days each)
- Short-term high: daily high with LOWER highs on BOTH sides (N days each)
- Intermediate: nested short-term points
- Long-term: nested intermediate points

4 Strategies tested with 10-fold expanding window walk-forward.
"""

import pandas as pd
import numpy as np
import json
import os
from itertools import product
from datetime import datetime

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

TOKENS = ["BTC", "ETH", "SOL", "LINK", "AVAX", "INJ", "ARB", "OP", "SUI", "FET"]
COMMISSION = 0.002  # 20bps per trade (applied on entry and exit)
N_FOLDS = 10
N_PERMUTATIONS = 200


def load_data(token):
    path = os.path.join(DATA_DIR, f"binance_{token.lower()}_usdt_1d.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Swing Point Detection ──────────────────────────────────────────────

def find_short_term_lows(lows, n):
    """Short-term low at index i: low[i] < low[j] for all j in [i-n,i-1] and [i+1,i+n]"""
    st_lows = []  # list of (index, value)
    for i in range(n, len(lows) - n):
        val = lows[i]
        left = all(lows[i - k] > val for k in range(1, n + 1))
        right = all(lows[i + k] > val for k in range(1, n + 1))
        if left and right:
            st_lows.append((i, val))
    return st_lows


def find_short_term_highs(highs, n):
    """Short-term high at index i: high[i] > high[j] for all j in [i-n,i-1] and [i+1,i+n]"""
    st_highs = []
    for i in range(n, len(highs) - n):
        val = highs[i]
        left = all(highs[i - k] < val for k in range(1, n + 1))
        right = all(highs[i + k] < val for k in range(1, n + 1))
        if left and right:
            st_highs.append((i, val))
    return st_highs


def find_intermediate_lows(st_lows):
    """Intermediate-term low: a short-term low with higher short-term lows on both sides."""
    if len(st_lows) < 3:
        return []
    int_lows = []
    for i in range(1, len(st_lows) - 1):
        idx, val = st_lows[i]
        if st_lows[i - 1][1] > val and st_lows[i + 1][1] > val:
            int_lows.append((idx, val))
    return int_lows


def find_intermediate_highs(st_highs):
    """Intermediate-term high: a short-term high with lower short-term highs on both sides."""
    if len(st_highs) < 3:
        return []
    int_highs = []
    for i in range(1, len(st_highs) - 1):
        idx, val = st_highs[i]
        if st_highs[i - 1][1] < val and st_highs[i + 1][1] < val:
            int_highs.append((idx, val))
    return int_highs


# ── Signal Generation ──────────────────────────────────────────────────

def strategy1_signals(df, n):
    """Short-Term Structure Follow: trend from consecutive swing points."""
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)
    
    st_lows = find_short_term_lows(lows, n)
    st_highs = find_short_term_highs(highs, n)
    
    signals = np.zeros(length)  # 0=flat, 1=long, -1=short
    
    # We need the confirmation day (n days after the swing point)
    # A swing low at i is confirmed at i+n, swing high at i+n
    events = []
    for idx, val in st_lows:
        confirm = idx + n  # confirmed when right side is verified
        if confirm < length:
            events.append((confirm, "low", val))
    for idx, val in st_highs:
        confirm = idx + n
        if confirm < length:
            events.append((confirm, "high", val))
    
    events.sort(key=lambda x: x[0])
    
    pos = 0
    prev_low = None
    prev_high = None
    
    for conf_idx, typ, val in events:
        if typ == "low":
            if prev_low is not None and val > prev_low:
                pos = 1  # higher low = uptrend
            elif prev_low is not None and val < prev_low:
                pass  # lower low, but don't go short on lows alone
            prev_low = val
        else:  # high
            if prev_high is not None and val < prev_high:
                pos = -1  # lower high = downtrend
            elif prev_high is not None and val > prev_high:
                pass
            prev_high = val
        signals[conf_idx] = pos
    
    # Forward fill signals
    for i in range(1, length):
        if signals[i] == 0 and i > 0:
            signals[i] = signals[i - 1]
    
    return signals


def strategy2_signals(df, n):
    """Intermediate Structure Follow: trend from intermediate swing points."""
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)
    
    st_lows = find_short_term_lows(lows, n)
    st_highs = find_short_term_highs(highs, n)
    int_lows = find_intermediate_lows(st_lows)
    int_highs = find_intermediate_highs(st_highs)
    
    signals = np.zeros(length)
    
    # For intermediate points, confirmation is when the 3rd short-term point confirms
    # Use the index of the right-side confirming short-term point
    # Approximate: intermediate low at index i is confirmed when next st_low after it is known
    # Simplification: use the index of the intermediate point + 2*n as rough confirmation
    events = []
    for idx, val in int_lows:
        # Find the confirming st_low (the one after this intermediate low)
        for si, sv in st_lows:
            if si > idx:
                confirm = si + n
                break
        else:
            continue
        if confirm < length:
            events.append((confirm, "low", val))
    
    for idx, val in int_highs:
        for si, sv in st_highs:
            if si > idx:
                confirm = si + n
                break
        else:
            continue
        if confirm < length:
            events.append((confirm, "high", val))
    
    events.sort(key=lambda x: x[0])
    
    pos = 0
    prev_low = None
    prev_high = None
    
    for conf_idx, typ, val in events:
        if typ == "low":
            if prev_low is not None and val > prev_low:
                pos = 1
            prev_low = val
        else:
            if prev_high is not None and val < prev_high:
                pos = -1
            prev_high = val
        signals[conf_idx] = pos
    
    for i in range(1, length):
        if signals[i] == 0:
            signals[i] = signals[i - 1]
    
    return signals


def strategy3_signals(df, n):
    """Multi-Level Confluence: long only when both ST and IT agree."""
    s1 = strategy1_signals(df, n)
    s2 = strategy2_signals(df, n)
    signals = np.zeros(len(df))
    for i in range(len(df)):
        if s1[i] == 1 and s2[i] == 1:
            signals[i] = 1
        elif s1[i] == -1 and s2[i] == -1:
            signals[i] = -1
        else:
            signals[i] = 0  # flat when disagreement
    return signals


def strategy4_signals(df, n, recovery_days):
    """Structure Break Fade: failed breakdowns/breakouts."""
    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values
    length = len(df)
    
    st_lows = find_short_term_lows(lows, n)
    st_highs = find_short_term_highs(highs, n)
    
    signals = np.zeros(length)
    
    # Failed breakdown: new st_low below prev st_low, but close recovers above prev st_low within recovery_days
    for i in range(1, len(st_lows)):
        idx, val = st_lows[i]
        prev_val = st_lows[i - 1][1]
        if val < prev_val:  # breakdown
            confirm = idx + n  # swing confirmed at idx+n
            for d in range(1, recovery_days + 1):
                check = confirm + d
                if check < length and close[check] > prev_val:
                    signals[check] = 1  # buy on failed breakdown
                    break
    
    # Failed breakout: new st_high above prev st_high, but close drops below prev st_high
    for i in range(1, len(st_highs)):
        idx, val = st_highs[i]
        prev_val = st_highs[i - 1][1]
        if val > prev_val:  # breakout
            confirm = idx + n
            for d in range(1, recovery_days + 1):
                check = confirm + d
                if check < length and close[check] < prev_val:
                    signals[check] = -1  # sell on failed breakout
                    break
    
    # Hold position until opposite signal, forward fill
    pos = 0
    for i in range(length):
        if signals[i] != 0:
            pos = signals[i]
        signals[i] = pos
    
    return signals


# ── Backtest Engine ────────────────────────────────────────────────────

def backtest_signals(close, signals, commission=COMMISSION):
    """Returns equity curve and stats from signal array."""
    returns = np.diff(close) / close[:-1]
    # Signal at day i determines position for day i→i+1
    pos = signals[:-1]
    
    # Detect trades (position changes) for commission
    trades = np.diff(np.concatenate([[0], pos]))
    trade_costs = np.abs(trades) * commission
    
    strategy_returns = pos * returns - trade_costs
    equity = np.cumprod(1 + strategy_returns)
    
    total_return = equity[-1] - 1 if len(equity) > 0 else 0
    n_trades = int(np.sum(np.abs(trades) > 0))
    
    # Sharpe (annualized)
    if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(365)
    else:
        sharpe = 0
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0
    
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_trades": n_trades,
        "equity": equity
    }


# ── Walk-Forward ───────────────────────────────────────────────────────

def walk_forward(df, strategy_fn, param_combos, n_folds=N_FOLDS):
    """Expanding window walk-forward. Returns OOS results per param combo."""
    length = len(df)
    min_train = length // (n_folds + 1)  # minimum training size
    
    results = {}
    
    for params in param_combos:
        param_key = str(params)
        oos_returns = []
        oos_sharpes = []
        
        for fold in range(n_folds):
            # Expanding window: train on first (min_train + fold * step), test on next step
            step = (length - min_train) // n_folds
            train_end = min_train + fold * step
            test_end = min(train_end + step, length)
            
            if train_end >= length or test_end <= train_end:
                continue
            
            test_df = df.iloc[:test_end].copy()
            
            # Generate signals on full data up to test_end
            if len(params) == 1:
                signals = strategy_fn(test_df, params[0])
            else:
                signals = strategy_fn(test_df, *params)
            
            # Only evaluate OOS portion
            oos_signals = signals[train_end:test_end]
            oos_close = df["close"].values[train_end:test_end]
            
            if len(oos_close) < 10:
                continue
            
            stats = backtest_signals(oos_close, oos_signals)
            oos_returns.append(stats["total_return"])
            oos_sharpes.append(stats["sharpe"])
        
        if oos_returns:
            avg_return = np.mean(oos_returns)
            avg_sharpe = np.mean(oos_sharpes)
            # t-test for significance
            if len(oos_returns) > 1 and np.std(oos_returns) > 0:
                t_stat = avg_return / (np.std(oos_returns) / np.sqrt(len(oos_returns)))
                from scipy import stats as scipy_stats
                p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(oos_returns) - 1)))
            else:
                t_stat = 0
                p_value = 1.0
            
            results[param_key] = {
                "params": params,
                "avg_oos_return": float(avg_return),
                "avg_oos_sharpe": float(avg_sharpe),
                "fold_returns": [float(r) for r in oos_returns],
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "n_folds": len(oos_returns)
            }
    
    return results


def permutation_test(df, strategy_fn, params, n_perms=N_PERMUTATIONS):
    """Shuffle returns to test if strategy alpha is real."""
    if len(params) == 1:
        signals = strategy_fn(df, params[0])
    else:
        signals = strategy_fn(df, *params)
    
    close = df["close"].values
    actual = backtest_signals(close, signals)
    actual_ret = actual["total_return"]
    
    returns = np.diff(close) / close[:-1]
    count_better = 0
    
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(returns)
        shuffled_close = np.cumprod(np.concatenate([[close[0]], 1 + shuffled]))
        shuffled_df = df.copy()
        shuffled_df["close"] = shuffled_close[:len(df)]
        
        if len(params) == 1:
            shuf_signals = strategy_fn(shuffled_df, params[0])
        else:
            shuf_signals = strategy_fn(shuffled_df, *params)
        
        shuf_stats = backtest_signals(shuffled_close, shuf_signals)
        if shuf_stats["total_return"] >= actual_ret:
            count_better += 1
    
    perm_p = (count_better + 1) / (n_perms + 1)
    return float(perm_p)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LARRY WILLIAMS MARKET STRUCTURE BACKTEST")
    print("=" * 70)
    
    all_results = {}
    
    strategies = {
        "S1_ShortTermFollow": {
            "fn": strategy1_signals,
            "params": [(n,) for n in [1, 2, 3]]
        },
        "S2_IntermediateFollow": {
            "fn": strategy2_signals,
            "params": [(n,) for n in [1, 2, 3]]
        },
        "S3_MultiLevelConfluence": {
            "fn": strategy3_signals,
            "params": [(n,) for n in [1, 2, 3]]
        },
        "S4_StructureBreakFade": {
            "fn": strategy4_signals,
            "params": [(n, r) for n in [1, 2, 3] for r in [1, 2, 3]]
        }
    }
    
    for token in TOKENS:
        print(f"\n{'─' * 50}")
        print(f"  {token}")
        print(f"{'─' * 50}")
        
        df = load_data(token)
        print(f"  Data: {len(df)} days ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
        
        token_results = {}
        
        for strat_name, strat_info in strategies.items():
            print(f"\n  {strat_name}:")
            
            wf = walk_forward(df, strat_info["fn"], strat_info["params"])
            
            # Find best params by OOS Sharpe
            best_key = max(wf, key=lambda k: wf[k]["avg_oos_sharpe"]) if wf else None
            
            strat_results = {"all_params": {}}
            
            for pk, pv in wf.items():
                strat_results["all_params"][pk] = pv
                ret = pv["avg_oos_return"]
                sharpe = pv["avg_oos_sharpe"]
                p = pv["p_value"]
                marker = " ★" if p < 0.05 else ""
                print(f"    params={pv['params']}: ret={ret:+.1%} sharpe={sharpe:.2f} p={p:.3f}{marker}")
                
                # Permutation test if p < 0.05
                if p < 0.05:
                    print(f"      → Running {N_PERMUTATIONS} permutations...")
                    perm_p = permutation_test(df, strat_info["fn"], pv["params"])
                    pv["perm_p_value"] = perm_p
                    print(f"      → Permutation p={perm_p:.3f} {'✓ SIGNIFICANT' if perm_p < 0.05 else '✗ NOT significant'}")
            
            if best_key:
                strat_results["best"] = wf[best_key]
                strat_results["best"]["param_key"] = best_key
            
            token_results[strat_name] = strat_results
        
        # Also run full-sample backtest for best params of each strategy
        print(f"\n  Full-sample results (best params):")
        for strat_name, strat_info in strategies.items():
            if strat_name not in token_results or "best" not in token_results[strat_name]:
                continue
            best = token_results[strat_name]["best"]
            params = best["params"]
            if len(params) == 1:
                signals = strat_info["fn"](df, params[0])
            else:
                signals = strat_info["fn"](df, *params)
            stats = backtest_signals(df["close"].values, signals)
            bh = df["close"].values[-1] / df["close"].values[0] - 1
            print(f"    {strat_name} {params}: ret={stats['total_return']:+.1%} sharpe={stats['sharpe']:.2f} "
                  f"dd={stats['max_dd']:.1%} trades={stats['n_trades']} (B&H={bh:+.1%})")
            token_results[strat_name]["full_sample"] = {
                "total_return": stats["total_return"],
                "sharpe": stats["sharpe"],
                "max_dd": stats["max_dd"],
                "n_trades": stats["n_trades"],
                "buy_hold": float(bh)
            }
        
        all_results[token] = token_results
    
    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: LARRY WILLIAMS vs SIMPLE HH/HL/LH/LL")
    print("=" * 70)
    
    summary = {"by_strategy": {}, "significant_results": [], "verdict": ""}
    
    for strat_name in strategies:
        sharpes = []
        returns = []
        sig_count = 0
        perm_sig_count = 0
        
        for token in TOKENS:
            if token in all_results and strat_name in all_results[token]:
                tr = all_results[token][strat_name]
                if "best" in tr:
                    sharpes.append(tr["best"]["avg_oos_sharpe"])
                    returns.append(tr["best"]["avg_oos_return"])
                    if tr["best"]["p_value"] < 0.05:
                        sig_count += 1
                    if tr["best"].get("perm_p_value", 1) < 0.05:
                        perm_sig_count += 1
        
        avg_s = np.mean(sharpes) if sharpes else 0
        avg_r = np.mean(returns) if returns else 0
        
        summary["by_strategy"][strat_name] = {
            "avg_oos_sharpe": float(avg_s),
            "avg_oos_return": float(avg_r),
            "significant_tokens": sig_count,
            "perm_significant_tokens": perm_sig_count,
            "n_tokens": len(sharpes)
        }
        
        print(f"\n  {strat_name}:")
        print(f"    Avg OOS Sharpe: {avg_s:.3f}")
        print(f"    Avg OOS Return: {avg_r:+.1%}")
        print(f"    Significant (p<0.05): {sig_count}/{len(sharpes)} tokens")
        print(f"    Perm-significant: {perm_sig_count}/{len(sharpes)} tokens")
    
    # Collect all significant results
    for token in TOKENS:
        for strat_name in strategies:
            if token in all_results and strat_name in all_results[token]:
                tr = all_results[token][strat_name]
                for pk, pv in tr.get("all_params", {}).items():
                    if pv.get("perm_p_value", 1) < 0.05:
                        summary["significant_results"].append({
                            "token": token,
                            "strategy": strat_name,
                            "params": pv["params"],
                            "avg_oos_return": pv["avg_oos_return"],
                            "avg_oos_sharpe": pv["avg_oos_sharpe"],
                            "p_value": pv["p_value"],
                            "perm_p_value": pv["perm_p_value"]
                        })
    
    n_sig = len(summary["significant_results"])
    total_tests = sum(len(s["params"]) for s in strategies.values()) * len(TOKENS)
    
    if n_sig == 0:
        verdict = ("NEGATIVE: Larry Williams' nested hierarchy shows NO statistically significant "
                   "edge after permutation testing. Same conclusion as simple HH/HL/LH/LL — "
                   "market structure alone does not generate reliable alpha in crypto.")
    elif n_sig <= total_tests * 0.05:
        verdict = ("INCONCLUSIVE: A few significant results exist but count is within "
                   "false-discovery rate (~5% of tests). Likely noise, not alpha.")
    else:
        verdict = (f"POSITIVE: {n_sig} param/token combos survive permutation testing. "
                   "Larry Williams' hierarchy may offer marginal improvement over simple structure.")
    
    summary["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    
    all_results["_summary"] = summary
    all_results["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "tokens": TOKENS,
        "commission": COMMISSION,
        "n_folds": N_FOLDS,
        "n_permutations": N_PERMUTATIONS,
        "total_param_token_tests": total_tests
    }
    
    # Save (remove non-serializable equity arrays)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "equity"}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    
    out_path = os.path.join(OUT_DIR, "larry_williams_results.json")
    with open(out_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    
    print(f"\n  Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
