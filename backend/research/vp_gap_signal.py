"""
Volume Profile Gap Detection — Entry Accelerator Signal
Test: Price entering a low-volume area predicts fast directional moves.

Signals:
1. VP Gap Entry: Price crosses into LVN zone → expect continuation in entry direction
2. VP Magnet: Price near LVN boundary → expect pull toward other side
3. VP+HSAKA: Only take HSAKA signals when in/near VP gap
4. VP POC Bounce: Price touches POC → expect bounce (mean reversion)
5. VP Breakout: Price breaks through HVN → expect trend continuation

Also tests different VP lookback periods to find optimal auction window.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))
from strategies.composite.hsaka_lean import generate_signal
from strategies.composite.tilopa import calculate_atr, calculate_volume_profile

COMMISSION = 0.0006
N_FOLDS = 14
PPY_4H = 2190
ASSETS = ["btc", "eth", "sol", "link", "avax", "sui", "inj", "arb", "op", "render", "tia", "tao", "fet"]

def load_data(asset, tf):
    path = os.path.expanduser(f"~/Desktop/maestro/data/ohlcv/binance_{asset}_usdt_{tf}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df

def walk_forward_sharpe(returns, n_folds=N_FOLDS):
    n = len(returns)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos_sharpes = []
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, n)
        oos_ret = returns[oos_start:oos_end]
        if len(oos_ret) < 10 or oos_ret.std() == 0:
            oos_sharpes.append(0.0)
        else:
            oos_sharpes.append(float(oos_ret.mean() / oos_ret.std() * np.sqrt(PPY_4H)))
    return oos_sharpes

def compute_returns(signal, df):
    pos = pd.Series(signal, index=df.index)
    ret = df["close"].pct_change().fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat_ret = pos.shift(1).fillna(0) * ret - trades * COMMISSION
    return strat_ret.values

def vp_gap_signals(df, vp_lb=50):
    """Generate VP-based signals."""
    n = len(df)
    close = df["close"].values
    atr = calculate_atr(df).values
    poc, lvn = calculate_volume_profile(df, lookback=vp_lb)
    poc_v = poc.values
    lvn_v = lvn.values
    
    # Signal 1: VP Gap Entry — direction of entry into LVN
    sig_gap_entry = np.zeros(n)
    for i in range(1, n):
        if lvn_v[i] and not lvn_v[i-1]:  # Just entered LVN
            if close[i] > close[i-1]:
                sig_gap_entry[i:min(i+6, n)] = 1  # Hold for 6 bars (1 day on 4h)
            else:
                sig_gap_entry[i:min(i+6, n)] = -1
    
    # Signal 2: VP POC Bounce (mean reversion at POC)
    sig_poc_bounce = np.zeros(n)
    for i in range(1, n):
        if np.isnan(poc_v[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue
        dist_to_poc = (close[i] - poc_v[i]) / atr[i]
        if dist_to_poc > 1.5:  # Far above POC → short toward it
            sig_poc_bounce[i] = -1
        elif dist_to_poc < -1.5:  # Far below POC → long toward it
            sig_poc_bounce[i] = 1
    
    # Signal 3: VP Breakout (price breaking through HVN = high volume cluster)
    sig_breakout = np.zeros(n)
    for i in range(1, n):
        if np.isnan(poc_v[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue
        # If NOT in LVN (meaning in HVN) and moving strongly
        if not lvn_v[i]:
            move = (close[i] - close[i-1]) / atr[i]
            if move > 1.0:
                sig_breakout[i:min(i+6, n)] = 1
            elif move < -1.0:
                sig_breakout[i:min(i+6, n)] = -1
    
    return sig_gap_entry, sig_poc_bounce, sig_breakout

def test_asset(asset):
    results = {}
    try:
        df_4h = load_data(asset, "4h")
    except FileNotFoundError:
        return {"error": f"No 4h data for {asset}"}
    
    # Baseline: HSAKA Lean
    sig_base = generate_signal(df_4h, sw=10, sfp_lb=20, ct=2)
    
    for vp_lb in [30, 50, 70, 100]:
        try:
            sig_gap, sig_poc, sig_break = vp_gap_signals(df_4h, vp_lb=vp_lb)
        except Exception as e:
            results[f"vp{vp_lb}_error"] = str(e)
            continue
        
        # VP+HSAKA: Only take HSAKA when in LVN
        _, lvn = calculate_volume_profile(df_4h, lookback=vp_lb)
        lvn_v = lvn.values
        sig_hsaka_vp = sig_base.copy()
        sig_hsaka_vp[~lvn_v] = 0  # Only trade in LVN zones
        
        for name, sig in [("gap_entry", sig_gap), ("poc_bounce", sig_poc), 
                           ("breakout", sig_break), ("hsaka_vp_filter", sig_hsaka_vp)]:
            ret = compute_returns(sig, df_4h)
            oos = walk_forward_sharpe(ret)
            mean_s = np.mean(oos)
            t_stat, p_val = stats.ttest_1samp(oos, 0) if len(oos) > 1 else (0, 1)
            invested = np.mean(np.array(sig) != 0) * 100
            results[f"{name}_lb{vp_lb}"] = {
                "oos_sharpe": round(mean_s, 3),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_val), 4),
                "significant": bool(p_val < 0.05),
                "invested_pct": round(invested, 1),
            }
    
    # Baseline for comparison
    ret_base = compute_returns(sig_base, df_4h)
    oos_base = walk_forward_sharpe(ret_base)
    results["baseline_hsaka"] = {
        "oos_sharpe": round(np.mean(oos_base), 3),
        "t_stat": round(float(stats.ttest_1samp(oos_base, 0)[0]), 3),
        "p_value": round(float(stats.ttest_1samp(oos_base, 0)[1]), 4),
        "significant": bool(stats.ttest_1samp(oos_base, 0)[1] < 0.05),
        "invested_pct": round(np.mean(sig_base != 0) * 100, 1),
    }
    
    return results

if __name__ == "__main__":
    all_results = {}
    for asset in ASSETS:
        print(f"Testing {asset.upper()}...")
        all_results[asset] = test_asset(asset)
    
    print("\n" + "="*80)
    print("VOLUME PROFILE GAP SIGNALS (4h)")
    print("="*80)
    
    # Show best VP signal per asset
    print(f"\n{'Asset':<8} {'Best VP Signal':<25} {'OOS Sharpe':>10} {'p-value':>8} {'Inv%':>6} {'Base':>6}")
    print("-"*66)
    for asset, res in all_results.items():
        if "error" in res:
            print(f"{asset.upper():<8} ERROR")
            continue
        base = res.get("baseline_hsaka", {}).get("oos_sharpe", 0)
        best_name, best_sharpe, best_p = "", -99, 1
        for k, v in res.items():
            if k == "baseline_hsaka" or "error" in k:
                continue
            if isinstance(v, dict) and v.get("oos_sharpe", -99) > best_sharpe:
                best_name = k
                best_sharpe = v["oos_sharpe"]
                best_p = v["p_value"]
        inv = res.get(best_name, {}).get("invested_pct", 0)
        sig = "✓" if best_p < 0.05 else ""
        print(f"{asset.upper():<8} {best_name:<25} {best_sharpe:>10.3f} {best_p:>8.4f} {inv:>5.1f}% {base:>6.3f} {sig}")
    
    # Count significant VP signals
    total = 0
    sig_count = 0
    for asset, res in all_results.items():
        for k, v in res.items():
            if isinstance(v, dict) and "oos_sharpe" in v and k != "baseline_hsaka":
                total += 1
                if v.get("significant"):
                    sig_count += 1
    print(f"\nTotal VP tests: {total}, Significant: {sig_count} ({sig_count/total*100:.1f}%)")
    bonf = sum(1 for a, res in all_results.items() for k, v in res.items() 
               if isinstance(v, dict) and k != "baseline_hsaka" and v.get("p_value", 1) < 0.05/total)
    print(f"Bonferroni survivors (p < {0.05/total:.5f}): {bonf}")
    
    out = os.path.expanduser("~/Desktop/maestro/data/research/vp_gap_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
