"""
Fear & Greed Index as regime filter for HSAKA Lean 4h
Test: Does filtering out extreme sentiment improve risk-adjusted returns?

Modes:
1. Baseline: HSAKA Lean 4h alone
2. Neutral-only: No trades when F&G > 75 (extreme greed) or < 25 (extreme fear)
3. Contrarian: Only trade when F&G is extreme (buy fear, sell greed)
4. Adaptive: Use F&G to scale position size (neutral=full, extreme=half)
5. Fear-buy-only: Only take longs when F&G < 30
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import requests
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))
from strategies.composite.hsaka_lean import generate_signal

COMMISSION = 0.0006
N_FOLDS = 14
PPY_4H = 2190
ASSETS = ["btc", "eth", "sol", "link", "avax", "sui", "inj", "arb", "op", "render", "tia", "tao", "fet"]

def load_fear_greed():
    """Fetch F&G from CoinGlass and return daily Series."""
    headers = {"CG-API-KEY": os.environ["COINGLASS_API_KEY"]}
    r = requests.get("https://open-api-v4.coinglass.com/api/index/fear-greed-history",
                     headers=headers, params={"limit": 3000})
    data = r.json()["data"]
    dates = pd.to_datetime(data["time_list"], unit="ms")
    fg = pd.Series(data["data_list"], index=dates, name="fear_greed", dtype=float)
    fg = fg.sort_index()
    return fg

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

def test_asset(asset, fg_daily):
    results = {}
    try:
        df_4h = load_data(asset, "4h")
    except FileNotFoundError as e:
        return {"error": str(e)}
    
    # Align F&G to 4h index (forward-fill daily value)
    fg_4h = fg_daily.reindex(df_4h.index, method="ffill").fillna(50).values
    
    # Baseline
    sig_base = generate_signal(df_4h, sw=10, sfp_lb=20, ct=2)
    
    # Mode 2: Neutral-only (no trades at extremes)
    sig_neutral = sig_base.copy()
    sig_neutral[(fg_4h > 75) | (fg_4h < 25)] = 0
    
    # Mode 3: Contrarian (only trade at extremes, flip greed→short, fear→long)
    sig_contrarian = np.zeros_like(sig_base)
    sig_contrarian[fg_4h < 25] = 1  # buy fear
    sig_contrarian[fg_4h > 75] = -1  # sell greed
    
    # Mode 4: Adaptive sizing
    sig_adaptive = sig_base.copy().astype(float)
    sig_adaptive[(fg_4h > 70) | (fg_4h < 30)] *= 0.5
    
    # Mode 5: Fear-buy-only (only longs when F&G < 30)
    sig_fear_buy = sig_base.copy()
    sig_fear_buy[(sig_base == 1) & (fg_4h >= 30)] = 0  # Remove non-fear longs
    sig_fear_buy[sig_base == -1] = sig_base[sig_base == -1]  # Keep all shorts
    
    for name, sig in [("baseline", sig_base), ("neutral_filter", sig_neutral),
                       ("contrarian", sig_contrarian), ("adaptive_size", sig_adaptive),
                       ("fear_buy", sig_fear_buy)]:
        ret = compute_returns(sig, df_4h)
        oos = walk_forward_sharpe(ret)
        mean_s = np.mean(oos)
        t_stat, p_val = stats.ttest_1samp(oos, 0) if len(oos) > 1 else (0, 1)
        invested = np.mean(np.array(sig) != 0) * 100
        results[name] = {
            "oos_sharpe": round(mean_s, 3),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
            "invested_pct": round(invested, 1),
            "pos_folds": int(sum(1 for s in oos if s > 0)),
        }
    
    return results

if __name__ == "__main__":
    print("Loading Fear & Greed index...")
    fg = load_fear_greed()
    print(f"  {len(fg)} days: {fg.index[0].date()} → {fg.index[-1].date()}")
    print(f"  Current: {fg.iloc[-1]:.0f}")
    
    all_results = {}
    for asset in ASSETS:
        print(f"Testing {asset.upper()}...")
        all_results[asset] = test_asset(asset, fg)
    
    print("\n" + "="*80)
    print("FEAR & GREED FILTER on HSAKA Lean 4h")
    print("="*80)
    print(f"{'Asset':<8} {'Mode':<16} {'OOS Sharpe':>10} {'p-value':>8} {'Inv%':>6} {'Sig':>4}")
    print("-"*56)
    for asset, res in all_results.items():
        if "error" in res:
            print(f"{asset.upper():<8} ERROR")
            continue
        for mode in ["baseline", "neutral_filter", "contrarian", "adaptive_size", "fear_buy"]:
            r = res[mode]
            sig = "✓" if r["significant"] else ""
            print(f"{asset.upper():<8} {mode:<16} {r['oos_sharpe']:>10.3f} {r['p_value']:>8.4f} {r['invested_pct']:>5.1f}% {sig:>4}")
        print()
    
    out = os.path.expanduser("~/Desktop/maestro/data/research/fear_greed_filter_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out}")
    
    # Delta analysis
    print("\n--- DELTA vs BASELINE ---")
    for mode in ["neutral_filter", "contrarian", "adaptive_size", "fear_buy"]:
        deltas = []
        for asset, res in all_results.items():
            if "error" in res:
                continue
            deltas.append(res[mode]["oos_sharpe"] - res["baseline"]["oos_sharpe"])
        print(f"{mode:<16}: mean Δ = {np.mean(deltas):+.3f}, wins = {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
