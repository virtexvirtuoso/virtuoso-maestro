"""
CREW Filter Backtest - Walk-forward test of CREW as overlay on 3 strategies.

Strategies tested:
  A) 14-day momentum cross-sectional L/S
  B) Buy-and-hold BTC
  C) Equal-weight portfolio

Walk-forward: 10-fold expanding window, 20bps commission.
Parameter grid: corr_window x zscore_lookback x threshold.
"""

import sys, os, json, itertools, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategies", "filters"))
from crew_filter import CREWFilter, CREWRegime

# ── Config ──────────────────────────────────────────────────────────────
TOKENS = ["arb", "avax", "btc", "eth", "fet", "inj", "link", "op", "sol", "sui"]
DATA_DIR = Path.home() / "Desktop" / "maestro" / "data" / "ohlcv"
RESULTS_PATH = Path.home() / "Desktop" / "maestro" / "data" / "backtest_results" / "crew_filter_results.json"
COMMISSION_BPS = 20
N_FOLDS = 10

PARAM_GRID = {
    "corr_window": [20, 30, 60],
    "zscore_lookback": [60, 90, 120],
    "threshold": [1.0, 1.5, 2.0],
}

# CREW adjustment factors for Strategy A
DECORR_BOOSTS = [1.5, 2.0]
CORR_REDUCTIONS = [0.5, 0.25]

# ── Data Loading ────────────────────────────────────────────────────────
def load_returns():
    prices = {}
    for token in TOKENS:
        df = pd.read_csv(DATA_DIR / f"binance_{token}_usdt_1d.csv", parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        prices[token] = df["close"]
    prices_df = pd.DataFrame(prices).dropna()
    returns_df = prices_df.pct_change().dropna()
    return prices_df, returns_df


# ── Strategy Implementations ───────────────────────────────────────────
def calc_turnover(weights: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes per rebalance."""
    return weights.diff().abs().sum(axis=1)


def apply_commission(pnl: pd.Series, turnover: pd.Series) -> pd.Series:
    return pnl - turnover * COMMISSION_BPS / 10000


def sharpe(returns: pd.Series) -> float:
    if returns.std() == 0 or len(returns) < 30:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(365)


# Strategy A: Cross-sectional momentum L/S
def strategy_momentum(returns_df: pd.DataFrame, crew_regime: pd.Series = None,
                       decorr_boost: float = 1.0, corr_reduce: float = 1.0):
    """14-day momentum, weekly rebalance, long top quintile / short bottom."""
    alts = [c for c in returns_df.columns if c != "btc"]
    mom = returns_df[alts].rolling(14).sum().shift(1)
    
    weights = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    
    rebal_mask = pd.Series(False, index=returns_df.index)
    rebal_mask.iloc[::7] = True
    
    n_q = max(1, len(alts) // 5)
    w = 1.0 / n_q
    
    # Vectorized ranking
    ranks = mom[alts].rank(axis=1, method="first", na_option="keep")
    n_valid = mom[alts].notna().sum(axis=1)
    
    for dt in returns_df.index[rebal_mask]:
        if mom.loc[dt, alts].isna().all():
            continue
        nv = int(n_valid.loc[dt])
        if nv < 2:
            continue
        r = ranks.loc[dt].dropna()
        shorts = r[r <= n_q].index
        longs = r[r > nv - n_q].index
        weights.loc[dt, shorts] = -w
        weights.loc[dt, longs] = w
    
    # Forward-fill weights on non-rebalance days
    # Mark non-rebalance rows as NaN, then ffill
    rebal_idx = returns_df.index[rebal_mask]
    mask_non_rebal = ~returns_df.index.isin(rebal_idx)
    weights.loc[mask_non_rebal] = np.nan
    weights = weights.ffill().fillna(0)
    
    # CREW adjustment (vectorized)
    if crew_regime is not None:
        regime = crew_regime.reindex(returns_df.index).fillna(0)
        decorr_mask = regime == CREWRegime.DECORRELATING
        corr_mask = regime == CREWRegime.CORRELATING
        for alt in alts:
            weights.loc[decorr_mask, alt] *= decorr_boost
            weights.loc[corr_mask, alt] *= corr_reduce
    
    pnl = (weights * returns_df).sum(axis=1)
    turnover = calc_turnover(weights)
    return apply_commission(pnl, turnover), weights


# Strategy B: Buy-and-hold BTC
def strategy_btc_hold(returns_df: pd.DataFrame, crew_regime: pd.Series = None,
                       corr_reduce: float = 1.0):
    weights = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    weights["btc"] = 1.0
    
    if crew_regime is not None:
        regime = crew_regime.reindex(returns_df.index).fillna(0)
        weights.loc[regime == CREWRegime.CORRELATING, "btc"] = corr_reduce
    
    pnl = (weights * returns_df).sum(axis=1)
    turnover = calc_turnover(weights)
    return apply_commission(pnl, turnover), weights


# Strategy C: Equal-weight
def strategy_equal_weight(returns_df: pd.DataFrame, crew_regime: pd.Series = None,
                           crew_detail: pd.DataFrame = None):
    n = len(returns_df.columns)
    weights = pd.DataFrame(1.0 / n, index=returns_df.index, columns=returns_df.columns)
    
    if crew_regime is not None and crew_detail is not None:
        alts = [c for c in returns_df.columns if c != "btc"]
        for alt in alts:
            col = f"{alt}_regime"
            if col in crew_detail.columns:
                r = crew_detail[col].reindex(returns_df.index).fillna(0)
                weights.loc[r == CREWRegime.DECORRELATING, alt] *= 1.5
                weights.loc[r == CREWRegime.CORRELATING, alt] *= 0.5
        # Renormalize
        row_sum = weights.abs().sum(axis=1).replace(0, 1)
        weights = weights.div(row_sum, axis=0)
    
    pnl = (weights * returns_df).sum(axis=1)
    turnover = calc_turnover(weights)
    return apply_commission(pnl, turnover), weights


# ── Walk-Forward Engine ────────────────────────────────────────────────
def walk_forward_test(returns_df: pd.DataFrame, prices_df: pd.DataFrame):
    """10-fold expanding window walk-forward."""
    n = len(returns_df)
    min_train = n // 5  # minimum training size
    fold_size = (n - min_train) // N_FOLDS
    
    results = []
    param_combos = list(itertools.product(
        PARAM_GRID["corr_window"],
        PARAM_GRID["zscore_lookback"],
        PARAM_GRID["threshold"],
    ))
    
    print(f"Data: {n} days, {len(param_combos)} param combos, {N_FOLDS} folds")
    print(f"Tokens: {TOKENS}")
    print()
    
    for pi, (cw, zl, th) in enumerate(param_combos):
        print(f"  [{pi+1}/{len(param_combos)}] corr_window={cw}, zscore_lb={zl}, threshold={th}")
        
        # Collect OOS returns per fold for each strategy variant
        oos_collections = {
            "mom_base": [], "mom_decorr1.5_corr0.5": [], "mom_decorr1.5_corr0.25": [],
            "mom_decorr2.0_corr0.5": [], "mom_decorr2.0_corr0.25": [],
            "btc_base": [], "btc_corr0.5": [], "btc_corr0.0": [],
            "ew_base": [], "ew_crew": [],
        }
        
        for fold in range(N_FOLDS):
            train_end = min_train + fold * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n)
            if test_start >= n:
                break
            
            full_slice = returns_df.iloc[:test_end]
            oos_slice = slice(test_start, test_end)
            
            # Compute CREW on full history up to test_end (no lookahead - uses rolling)
            crew = CREWFilter(corr_window=cw, zscore_lookback=zl, threshold=th)
            crew_result = crew.compute(full_slice)
            regime = crew_result["aggregate_regime"]
            
            # Strategy A: Momentum variants
            pnl_base, _ = strategy_momentum(full_slice)
            oos_collections["mom_base"].append(pnl_base.iloc[oos_slice])
            
            for db, cr in itertools.product(DECORR_BOOSTS, CORR_REDUCTIONS):
                key = f"mom_decorr{db}_corr{cr}"
                pnl_filtered, _ = strategy_momentum(full_slice, regime, db, cr)
                oos_collections[key].append(pnl_filtered.iloc[oos_slice])
            
            # Strategy B: BTC hold variants
            pnl_btc_base, _ = strategy_btc_hold(full_slice)
            oos_collections["btc_base"].append(pnl_btc_base.iloc[oos_slice])
            
            for cr in [0.5, 0.0]:
                key = f"btc_corr{cr}"
                pnl_btc_f, _ = strategy_btc_hold(full_slice, regime, cr)
                oos_collections[key].append(pnl_btc_f.iloc[oos_slice])
            
            # Strategy C: Equal-weight variants
            pnl_ew_base, _ = strategy_equal_weight(full_slice)
            oos_collections["ew_base"].append(pnl_ew_base.iloc[oos_slice])
            
            pnl_ew_crew, _ = strategy_equal_weight(full_slice, regime, crew_result)
            oos_collections["ew_crew"].append(pnl_ew_crew.iloc[oos_slice])
        
        # Concatenate OOS returns and compute Sharpes
        oos_sharpes = {}
        oos_returns_full = {}
        for key, chunks in oos_collections.items():
            if chunks:
                combined = pd.concat(chunks)
                oos_sharpes[key] = sharpe(combined)
                oos_returns_full[key] = combined
            else:
                oos_sharpes[key] = 0.0
        
        # Compute improvements and p-values
        def improvement(base_key, filtered_key):
            base_s = oos_sharpes[base_key]
            filt_s = oos_sharpes[filtered_key]
            if base_s == 0:
                imp_pct = 0.0
            else:
                imp_pct = (filt_s - base_s) / abs(base_s) * 100
            
            # p-value: paired t-test on daily returns
            if base_key in oos_returns_full and filtered_key in oos_returns_full:
                b = oos_returns_full[base_key]
                f = oos_returns_full[filtered_key]
                common = b.index.intersection(f.index)
                if len(common) > 30:
                    diff = f.loc[common] - b.loc[common]
                    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
                    p_val = float(p_val)
                else:
                    p_val = 1.0
            else:
                p_val = 1.0
            
            return {
                "base_sharpe": round(base_s, 4),
                "filtered_sharpe": round(filt_s, 4),
                "improvement_pct": round(imp_pct, 2),
                "p_value": round(p_val, 4),
            }
        
        entry = {
            "params": {"corr_window": cw, "zscore_lookback": zl, "threshold": th},
            "momentum": {},
            "btc_hold": {},
            "equal_weight": {},
        }
        
        # Momentum comparisons
        for db, cr in itertools.product(DECORR_BOOSTS, CORR_REDUCTIONS):
            key = f"decorr{db}_corr{cr}"
            entry["momentum"][key] = improvement("mom_base", f"mom_{key}")
        
        # BTC hold comparisons
        for cr in [0.5, 0.0]:
            entry["btc_hold"][f"corr_{cr}"] = improvement("btc_base", f"btc_corr{cr}")
        
        # Equal-weight comparison
        entry["equal_weight"]["crew_overlay"] = improvement("ew_base", "ew_crew")
        
        results.append(entry)
    
    return results


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CREW Filter Walk-Forward Backtest")
    print("=" * 70)
    
    prices_df, returns_df = load_returns()
    print(f"Loaded {len(returns_df)} days of returns for {len(returns_df.columns)} tokens")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print()
    
    results = walk_forward_test(returns_df, prices_df)
    
    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY - Best CREW configurations per strategy")
    print("=" * 70)
    
    # Find best improvement for each strategy
    for strat_name in ["momentum", "btc_hold", "equal_weight"]:
        print(f"\n{'─' * 50}")
        print(f"Strategy: {strat_name.upper()}")
        print(f"{'─' * 50}")
        
        best_imp = -999
        best_entry = None
        best_variant = None
        
        for entry in results:
            for variant, metrics in entry[strat_name].items():
                if metrics["improvement_pct"] > best_imp:
                    best_imp = metrics["improvement_pct"]
                    best_entry = entry
                    best_variant = variant
        
        if best_entry:
            m = best_entry[strat_name][best_variant]
            p = best_entry["params"]
            print(f"  Best variant: {best_variant}")
            print(f"  Params: corr_window={p['corr_window']}, zscore_lb={p['zscore_lookback']}, threshold={p['threshold']}")
            print(f"  Base Sharpe:     {m['base_sharpe']:.4f}")
            print(f"  Filtered Sharpe: {m['filtered_sharpe']:.4f}")
            print(f"  Improvement:     {m['improvement_pct']:+.2f}%")
            print(f"  p-value:         {m['p_value']:.4f}")
            sig = "✓" if m["p_value"] < 0.05 else "✗"
            print(f"  Significant (p<0.05): {sig}")
    
    # Overall verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    
    any_good = False
    for entry in results:
        for strat_name in ["momentum", "btc_hold", "equal_weight"]:
            for variant, metrics in entry[strat_name].items():
                if metrics["improvement_pct"] >= 10 and metrics["p_value"] < 0.05:
                    any_good = True
    
    if any_good:
        print("✓ CREW filter shows statistically significant 10%+ Sharpe improvement")
        print("  in at least one strategy/parameter combination.")
    else:
        print("✗ CREW filter does NOT reliably deliver 10-15% Sharpe improvement")
        print("  with statistical significance across tested configurations.")


if __name__ == "__main__":
    main()
