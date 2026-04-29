"""
Multi-TF Stacking: 1d structure bias + 4h HSAKA entry
Test: Does adding daily structure confirmation improve 4h HSAKA Lean?

Modes:
1. Baseline: HSAKA Lean 4h alone (ct=2)
2. Stacked: Only take 4h signal when 1d structure agrees
3. Contra-filter: Only take 4h signal when 1d structure DISAGREES (sanity check — should be worse)
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))
from strategies.composite.hsaka_lean import generate_signal, market_structure

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
    """Expanding window WF — returns list of OOS fold Sharpes."""
    n = len(returns)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos_sharpes = []
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = oos_start + fold_size
        if oos_end > n:
            oos_end = n
        oos_ret = returns[oos_start:oos_end]
        if len(oos_ret) < 10 or oos_ret.std() == 0:
            oos_sharpes.append(0.0)
        else:
            oos_sharpes.append(float(oos_ret.mean() / oos_ret.std() * np.sqrt(PPY_4H)))
    return oos_sharpes

def compute_returns(signal_4h, df_4h):
    pos = pd.Series(signal_4h, index=df_4h.index)
    ret = df_4h["close"].pct_change().fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat_ret = pos.shift(1).fillna(0) * ret - trades * COMMISSION
    return strat_ret.values

def test_asset(asset):
    results = {}
    
    # Load data
    try:
        df_4h = load_data(asset, "4h")
        df_1d = load_data(asset, "1d")
    except FileNotFoundError as e:
        return {"error": str(e)}
    
    # --- Mode 1: Baseline HSAKA 4h ---
    sig_4h = generate_signal(df_4h, sw=10, sfp_lb=20, ct=2)
    ret_base = compute_returns(sig_4h, df_4h)
    oos_base = walk_forward_sharpe(ret_base)
    
    # --- 1d structure bias ---
    struct_1d = market_structure(df_1d, sw=10)
    # Resample 1d structure to 4h by forward-filling date alignment
    struct_1d_series = pd.Series(struct_1d, index=df_1d.index)
    struct_1d_4h = struct_1d_series.reindex(df_4h.index, method="ffill").fillna(0).values
    
    # --- Mode 2: Stacked (4h signal only when 1d agrees) ---
    sig_stacked = np.where(np.sign(sig_4h) == np.sign(struct_1d_4h), sig_4h, 0)
    ret_stacked = compute_returns(sig_stacked, df_4h)
    oos_stacked = walk_forward_sharpe(ret_stacked)
    
    # --- Mode 3: Contra-filter (4h signal only when 1d disagrees — sanity) ---
    sig_contra = np.where(np.sign(sig_4h) != np.sign(struct_1d_4h), sig_4h, 0)
    ret_contra = compute_returns(sig_contra, df_4h)
    oos_contra = walk_forward_sharpe(ret_contra)
    
    # --- Mode 4: 1d bias overrides direction (take 4h magnitude, 1d direction) ---
    sig_override = np.where(sig_4h != 0, np.sign(struct_1d_4h) * np.abs(sig_4h), 0)
    ret_override = compute_returns(sig_override, df_4h)
    oos_override = walk_forward_sharpe(ret_override)
    
    # Stats
    for name, oos in [("baseline", oos_base), ("stacked", oos_stacked), 
                       ("contra", oos_contra), ("override", oos_override)]:
        mean_s = np.mean(oos)
        t_stat, p_val = stats.ttest_1samp(oos, 0) if len(oos) > 1 else (0, 1)
        sig_arr = sig_4h if name == "baseline" else (sig_stacked if name == "stacked" else (sig_contra if name == "contra" else sig_override))
        invested = np.mean(sig_arr != 0) * 100
        results[name] = {
            "oos_sharpe": round(mean_s, 3),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
            "invested_pct": round(invested, 1),
            "pos_folds": int(sum(1 for s in oos if s > 0)),
            "n_folds": len(oos),
        }
    
    return results

if __name__ == "__main__":
    all_results = {}
    for asset in ASSETS:
        print(f"Testing {asset.upper()}...")
        all_results[asset] = test_asset(asset)
    
    # Summary
    print("\n" + "="*80)
    print("MULTI-TF STACKING: 1d Structure + 4h HSAKA Lean")
    print("="*80)
    print(f"{'Asset':<8} {'Mode':<12} {'OOS Sharpe':>10} {'p-value':>8} {'Inv%':>6} {'Sig':>4}")
    print("-"*52)
    for asset, res in all_results.items():
        if "error" in res:
            print(f"{asset.upper():<8} ERROR: {res['error']}")
            continue
        for mode in ["baseline", "stacked", "contra", "override"]:
            r = res[mode]
            sig = "✓" if r["significant"] else ""
            print(f"{asset.upper():<8} {mode:<12} {r['oos_sharpe']:>10.3f} {r['p_value']:>8.4f} {r['invested_pct']:>5.1f}% {sig:>4}")
        print()
    
    # Save
    out = os.path.expanduser("~/Desktop/maestro/data/research/multi_tf_stacking_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
    
    # Verdict
    print("\n--- VERDICT ---")
    base_wins = 0
    stack_wins = 0
    for asset, res in all_results.items():
        if "error" in res:
            continue
        if res["stacked"]["oos_sharpe"] > res["baseline"]["oos_sharpe"]:
            stack_wins += 1
        else:
            base_wins += 1
    print(f"Stacking wins: {stack_wins}/{stack_wins+base_wins}")
    print(f"Baseline wins: {base_wins}/{stack_wins+base_wins}")
