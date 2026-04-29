"""
Walk-forward backtest of Cycle Index strategies (CPS, TPI, BPI).

Strategies:
1. CPS Regime Trading
2. TPI/BPI Gating
3. CPS + M2 Combo
4. Multi-Asset CPS
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from strategies.filters.cycle_indices import compute_indices

DATA_DIR = Path.home() / "Desktop" / "maestro" / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
RESULTS_DIR = DATA_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

COMMISSION = 0.002  # 20bps round-trip
TOKENS_MAIN = ["btc", "eth", "sol", "link"]
TOKENS_ALL = ["btc", "eth", "sol", "link", "doge", "avax", "bnb", "xrp", "ada", "dot"]
N_PERMS = 200


def load_ohlcv(token: str) -> pd.DataFrame:
    p = OHLCV_DIR / f"binance_{token}_usdt_1d.csv"
    df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df.sort_index()


def apply_commission(returns: pd.Series, positions: pd.Series) -> pd.Series:
    """Subtract commission on position changes."""
    trades = positions.diff().abs().fillna(0)
    cost = trades * COMMISSION
    return returns - cost


def sharpe(rets: pd.Series) -> float:
    if rets.std() == 0:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(365)


def max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def per_year_returns(rets: pd.Series) -> dict:
    yearly = {}
    for yr, g in rets.groupby(rets.index.year):
        yearly[str(yr)] = round(float((1 + g).prod() - 1), 4)
    return yearly


def permutation_test(strategy_rets: pd.Series, benchmark_rets: pd.Series, n_perms: int = N_PERMS) -> float:
    """Test if strategy Sharpe is significantly better than random via permutation."""
    observed = sharpe(strategy_rets) - sharpe(benchmark_rets)
    combined = pd.concat([strategy_rets, benchmark_rets])
    count = 0
    n = len(strategy_rets)
    for _ in range(n_perms):
        perm = np.random.permutation(len(combined))
        perm_strat = combined.iloc[perm[:n]]
        perm_bench = combined.iloc[perm[n:]]
        if sharpe(perm_strat) - sharpe(perm_bench) >= observed:
            count += 1
    return count / n_perms


# ============ Strategy Signals ============

def strategy_cps_regime(indices: pd.DataFrame, close: pd.Series,
                        bull_thresh=70, bear_thresh=40, short_bear=False):
    """Strategy 1: CPS regime trading."""
    pos = pd.Series(0.0, index=indices.index)
    pos[indices["cps"] > bull_thresh] = 1.0
    pos[(indices["cps"] >= bear_thresh) & (indices["cps"] <= bull_thresh)] = 0.5
    if short_bear:
        pos[indices["cps"] < bear_thresh] = -1.0
    return pos


def strategy_tpi_bpi(indices: pd.DataFrame, close: pd.Series,
                     tpi_thresh=5, bpi_thresh=5):
    """Strategy 2: TPI/BPI gating with momentum fallback."""
    mom = (close / close.shift(20) - 1)
    pos = pd.Series(0.0, index=indices.index)
    # Default: follow momentum
    pos[mom > 0] = 1.0
    pos[mom <= 0] = 0.0
    # Override with TPI/BPI
    pos[indices["bpi"] >= bpi_thresh] = 1.0  # accumulate
    pos[indices["tpi"] >= tpi_thresh] = -1.0  # distribute
    return pos


def strategy_cps_m2(indices: pd.DataFrame, close: pd.Series,
                    cps_thresh=55, m2_window=100):
    """Strategy 3: CPS + M2 proxy (BTC 100d SMA slope as liquidity proxy)."""
    sma = close.rolling(m2_window).mean()
    m2_bullish = sma.diff(20) > 0  # rising SMA = liquidity expansion

    cps_bullish = indices["cps"] > cps_thresh

    pos = pd.Series(0.0, index=indices.index)
    pos[cps_bullish & m2_bullish] = 1.0
    pos[cps_bullish ^ m2_bullish] = 0.5  # disagreement
    pos[~cps_bullish & ~m2_bullish] = 0.0
    return pos


def strategy_multi_asset_cps(long_thresh=60, short_thresh=30):
    """Strategy 4: Multi-asset CPS rotation. Returns equity curve directly."""
    all_indices = {}
    all_close = {}
    for token in TOKENS_ALL:
        try:
            ohlcv = load_ohlcv(token)
            idx = compute_indices(ohlcv, token)
            all_indices[token] = idx
            all_close[token] = ohlcv["close"]
        except Exception:
            continue

    # Align to common dates
    common = None
    for token in all_indices:
        dates = all_indices[token].index
        common = dates if common is None else common.intersection(dates)
    common = common.sort_values()

    daily_rets = pd.Series(0.0, index=common)
    positions_prev = {t: 0.0 for t in all_indices}

    for i in range(1, len(common)):
        dt = common[i]
        dt_prev = common[i - 1]
        n_assets = len(all_indices)
        weight = 1.0 / max(n_assets, 1)

        total_ret = 0.0
        total_cost = 0.0
        for token in all_indices:
            cps_val = all_indices[token].loc[dt_prev, "cps"] if dt_prev in all_indices[token].index else 50
            if np.isnan(cps_val):
                cps_val = 50

            if cps_val > long_thresh:
                target = 1.0
            elif cps_val < short_thresh:
                target = -1.0
            else:
                target = 0.0

            # Weekly rebalance
            if i % 7 != 0 and positions_prev[token] != 0:
                target = positions_prev[token]

            price_ret = (all_close[token].loc[dt] / all_close[token].loc[dt_prev] - 1) if dt in all_close[token].index and dt_prev in all_close[token].index else 0
            trade_cost = abs(target - positions_prev[token]) * COMMISSION
            total_ret += target * price_ret * weight
            total_cost += trade_cost * weight
            positions_prev[token] = target

        daily_rets.iloc[i] = total_ret - total_cost

    return daily_rets


# ============ Walk-Forward Engine ============

def walk_forward(ohlcv: pd.DataFrame, token: str, strategy_fn, param_grid: dict,
                 n_folds: int = 10) -> list:
    """
    Expanding-window walk-forward.
    Returns list of fold results.
    """
    indices = compute_indices(ohlcv, token)
    close = ohlcv["close"]
    daily_ret = close.pct_change().fillna(0)
    n = len(ohlcv)
    min_train = n // 4  # minimum training size

    results = []
    fold_size = (n - min_train) // n_folds

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break

        train_idx = indices.iloc[:train_end]
        train_close = close.iloc[:train_end]
        train_ret = daily_ret.iloc[:train_end]

        # Optimize on training set
        best_sharpe = -999
        best_params = None
        keys = list(param_grid.keys())
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            pos = strategy_fn(train_idx, train_close, **params)
            strat_ret = apply_commission(pos.shift(1).fillna(0) * train_ret, pos)
            s = sharpe(strat_ret.dropna())
            if s > best_sharpe:
                best_sharpe = s
                best_params = params

        # Test on OOS
        test_idx = indices.iloc[:test_end]
        test_close = close.iloc[:test_end]
        test_ret = daily_ret.iloc[:test_end]

        pos_full = strategy_fn(test_idx, test_close, **best_params)
        pos_oos = pos_full.iloc[train_end:test_end]
        ret_oos = (pos_oos.shift(1).fillna(0) * test_ret.iloc[train_end:test_end])
        ret_oos = apply_commission(ret_oos, pos_oos)

        # Buy & hold OOS
        bh_oos = test_ret.iloc[train_end:test_end]

        oos_sharpe = sharpe(ret_oos.dropna())
        oos_total = float((1 + ret_oos.dropna()).prod() - 1)
        bh_total = float((1 + bh_oos.dropna()).prod() - 1)

        # Permutation test if Sharpe looks good
        p_val = None
        if oos_sharpe > 0.5 and len(ret_oos.dropna()) > 30:
            p_val = permutation_test(ret_oos.dropna(), bh_oos.dropna())

        eq = (1 + ret_oos.dropna()).cumprod()

        results.append({
            "fold": fold,
            "train_end": str(ohlcv.index[train_end].date()) if hasattr(ohlcv.index[train_end], 'date') else str(ohlcv.index[train_end]),
            "test_end": str(ohlcv.index[min(test_end - 1, n - 1)].date()) if hasattr(ohlcv.index[min(test_end - 1, n - 1)], 'date') else str(ohlcv.index[min(test_end - 1, n - 1)]),
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in best_params.items()},
            "oos_return": round(oos_total, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "oos_max_dd": round(float(max_dd(eq)) if len(eq) > 0 else 0, 4),
            "bh_return": round(bh_total, 4),
            "p_value": round(p_val, 4) if p_val is not None else None,
            "per_year": per_year_returns(ret_oos.dropna()),
        })

    return results


def run_all():
    all_results = {}

    # Strategy 1: CPS Regime
    print("=== Strategy 1: CPS Regime Trading ===")
    param_grid_1 = {
        "bull_thresh": [60, 65, 70, 75],
        "bear_thresh": [30, 35, 40, 45],
        "short_bear": [False, True],
    }
    for token in TOKENS_MAIN:
        print(f"  {token.upper()}...")
        ohlcv = load_ohlcv(token)
        folds = walk_forward(ohlcv, token, strategy_cps_regime, param_grid_1)
        all_results[f"cps_regime_{token}"] = folds

    # Strategy 2: TPI/BPI Gating
    print("=== Strategy 2: TPI/BPI Gating ===")
    param_grid_2 = {
        "tpi_thresh": [3, 4, 5, 6],
        "bpi_thresh": [3, 4, 5],
    }
    for token in TOKENS_MAIN:
        print(f"  {token.upper()}...")
        ohlcv = load_ohlcv(token)
        folds = walk_forward(ohlcv, token, strategy_tpi_bpi, param_grid_2)
        all_results[f"tpi_bpi_{token}"] = folds

    # Strategy 3: CPS + M2 Combo
    print("=== Strategy 3: CPS + M2 Combo ===")
    param_grid_3 = {
        "cps_thresh": [45, 50, 55, 60, 65],
        "m2_window": [50, 100, 150],
    }
    for token in TOKENS_MAIN:
        print(f"  {token.upper()}...")
        ohlcv = load_ohlcv(token)
        folds = walk_forward(ohlcv, token, strategy_cps_m2, param_grid_3)
        all_results[f"cps_m2_{token}"] = folds

    # Strategy 4: Multi-Asset CPS
    print("=== Strategy 4: Multi-Asset CPS ===")
    multi_results = []
    for lt, st in [(55, 30), (60, 30), (60, 35), (65, 25)]:
        print(f"  long>{lt}, short<{st}...")
        rets = strategy_multi_asset_cps(long_thresh=lt, short_thresh=st)
        eq = (1 + rets).cumprod()
        s = sharpe(rets.dropna())
        total = float(eq.iloc[-1] - 1) if len(eq) > 0 else 0
        dd = float(max_dd(eq)) if len(eq) > 0 else 0
        multi_results.append({
            "long_thresh": lt, "short_thresh": st,
            "total_return": round(total, 4),
            "sharpe": round(s, 4),
            "max_dd": round(dd, 4),
            "per_year": per_year_returns(rets.dropna()),
        })
    all_results["multi_asset_cps"] = multi_results

    # ============ Summary ============
    print("\n" + "=" * 70)
    print("CYCLE INDICES BACKTEST RESULTS SUMMARY")
    print("=" * 70)

    for key, folds in all_results.items():
        if key == "multi_asset_cps":
            print(f"\n--- {key} ---")
            for r in folds:
                print(f"  L>{r['long_thresh']} S<{r['short_thresh']}: "
                      f"Ret={r['total_return']:.1%} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1%}")
        else:
            oos_rets = [f["oos_return"] for f in folds]
            oos_sharpes = [f["oos_sharpe"] for f in folds]
            bh_rets = [f["bh_return"] for f in folds]
            pvals = [f["p_value"] for f in folds if f["p_value"] is not None]
            print(f"\n--- {key} ({len(folds)} folds) ---")
            print(f"  Avg OOS Return: {np.mean(oos_rets):.1%} (vs B&H {np.mean(bh_rets):.1%})")
            print(f"  Avg OOS Sharpe: {np.mean(oos_sharpes):.2f}")
            if pvals:
                print(f"  Avg p-value: {np.mean(pvals):.3f} (sig folds: {sum(p < 0.05 for p in pvals)}/{len(pvals)})")
            # Show best params across folds
            from collections import Counter
            param_counts = Counter(str(f["best_params"]) for f in folds)
            most_common = param_counts.most_common(1)[0]
            print(f"  Most chosen params: {most_common[0]} ({most_common[1]}/{len(folds)} folds)")

    # Save
    out_path = RESULTS_DIR / "cycle_indices_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    run_all()
