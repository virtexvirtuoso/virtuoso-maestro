"""
3-Way Proxy Comparison inside V3 full system.
Tests: Original (20d, 3-of-4) vs Tiered (20d+120d) vs Validated (120d, 2-of-4)
"""
import sys, os, json, warnings, copy
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite import mega_strategy_v3 as v3

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}


def load_data():
    stock = StockDataLoader()
    fred = MacroDataLoader()

    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
        except:
            pass

    cross_asset_data = pd.DataFrame()
    for col_name, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col_name] = df["close"]
        except:
            pass

    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    return crypto_data, cross_asset_data, macro_data


def monkey_patch_proxy(variant):
    """
    Temporarily replace compute_confluence's Signal 2 logic.
    We do this by wrapping the original function.
    """
    original_compute = v3.compute_confluence

    def patched_confluence(crypto_close, macro_data, cross_asset_data,
                          sma_slow=100, momentum_period=35):
        # Call original — it uses the current code (Fixed-120d validated)
        # For other variants, we need to override sig2

        idx = crypto_close.index

        # Compute all signals except sig2 using original logic
        # We'll call the original but then override sig2
        confluence, breakdown = original_compute(
            crypto_close, macro_data, cross_asset_data,
            sma_slow=sma_slow, momentum_period=momentum_period
        )

        if variant == "validated":
            # Already the current code — no change needed
            return confluence, breakdown

        # Need to recompute sig2 for other variants
        sig2_new = pd.Series(0, index=idx, dtype=int)

        if cross_asset_data is not None:
            if variant == "original":
                # Original: 20d lookback, 3-of-4 consensus
                lb, thr = 20, 3
                liq_score = pd.Series(0.0, index=idx)
                for col, direction in [("dxy", "down"), ("gold", "up"), ("bonds", "up"), ("hyg", "up")]:
                    if col in cross_asset_data.columns:
                        s = cross_asset_data[col].reindex(idx, method="ffill").ffill()
                        if direction == "down":
                            liq_score += (s.pct_change(lb) < 0).astype(float).fillna(0)
                        else:
                            liq_score += (s.pct_change(lb) > 0).astype(float).fillna(0)
                sig2_new = (liq_score >= thr).astype(int)

            elif variant == "tiered":
                # Tiered: 120d macro + 20d crash, partial scoring
                def _liq(data, idx, lb):
                    score = pd.Series(0.0, index=idx)
                    for col, direction in [("dxy", "down"), ("gold", "up"), ("bonds", "up"), ("hyg", "up")]:
                        if col in data.columns:
                            s = data[col].reindex(idx, method="ffill").ffill()
                            if direction == "down":
                                score += (s.pct_change(lb) < 0).astype(float).fillna(0)
                            else:
                                score += (s.pct_change(lb) > 0).astype(float).fillna(0)
                    return score

                long_s = _liq(cross_asset_data, idx, 120)
                short_s = _liq(cross_asset_data, idx, 20)
                long_bull = (long_s >= 2).astype(float)
                short_bull = (short_s >= 2).astype(float)
                sig2_raw = long_bull * 0.5 + (long_bull * short_bull) * 0.5
                sig2_new = (sig2_raw >= 0.5).astype(int)

        # Shift sig2 by 1 day
        sig2_new = sig2_new.shift(1).fillna(0).astype(int)

        # Recompute confluence with new sig2
        old_sig2 = breakdown["liquidity_proxy"]
        diff = sig2_new - old_sig2
        new_confluence = confluence + diff
        new_confluence = new_confluence.clip(0, 5)

        new_breakdown = breakdown.copy()
        new_breakdown["liquidity_proxy"] = sig2_new
        new_breakdown["confluence"] = new_confluence

        return new_confluence, new_breakdown

    return patched_confluence


def compute_metrics(returns):
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["sharpe", "cagr", "max_dd", "sortino", "calmar"]}
    eq = (1 + returns).cumprod()
    n_yr = len(returns) / 252
    cagr = float(eq.iloc[-1] ** (1 / max(n_yr, 0.1)) - 1)
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    dd = float((eq / eq.cummax() - 1).min())
    down_std = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(returns.mean() * 252 / down_std) if down_std > 0 else 0
    calmar = float(cagr / abs(dd)) if dd != 0 else 0
    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 1),
        "max_dd": round(dd * 100, 1),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
    }


def crash_return(portfolio_df, start, end):
    mask = (portfolio_df.index >= pd.Timestamp(start)) & (portfolio_df.index <= pd.Timestamp(end))
    if mask.sum() < 5:
        return 0
    eq = (1 + portfolio_df.loc[mask, "daily_pnl"]).cumprod()
    return round(float(eq.iloc[-1] - 1) * 100, 1)


def walk_forward(crypto_data, macro_data, cross_asset_data, variant,
                 n_folds=7, train_days=730, test_days=180):
    """Walk-forward OOS Sharpe for a proxy variant."""
    # Patch the proxy
    original_fn = v3.compute_confluence
    v3.compute_confluence = monkey_patch_proxy(variant)

    try:
        portfolio_df, _ = v3.run_full_strategy(
            crypto_data, macro_data, cross_asset_data,
            enable_long=True, enable_short=True,
            enable_adaptive_leverage=True,
        )
    finally:
        v3.compute_confluence = original_fn

    ret = portfolio_df["daily_pnl"]

    # Collect OOS
    oos_returns = []
    start_idx = 400
    for fold in range(n_folds):
        train_end_idx = start_idx + fold * test_days + train_days
        test_end_idx = train_end_idx + test_days
        if test_end_idx > len(ret):
            break
        train_end = ret.index[min(train_end_idx, len(ret)-1)]
        test_end = ret.index[min(test_end_idx, len(ret)-1)]
        oos = ret.iloc[train_end_idx:test_end_idx]
        oos_returns.append(oos)

    if not oos_returns:
        return -1
    all_oos = pd.concat(oos_returns).dropna()
    if len(all_oos) < 100 or all_oos.std() == 0:
        return -1
    return round(float(all_oos.mean() / all_oos.std() * np.sqrt(252)), 3)


def main():
    print("=" * 80)
    print("3-WAY PROXY COMPARISON INSIDE V3 FULL SYSTEM")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\nLoading data...")
    crypto_data, cross_asset_data, macro_data = load_data()
    print(f"  Crypto: {list(crypto_data.keys())}")
    print(f"  Cross-asset: {list(cross_asset_data.columns)}")

    variants = {
        "Original (20d, 3-of-4)": "original",
        "Tiered (20d+120d)": "tiered",
        "Validated (120d, 2-of-4)": "validated",
    }

    results = {}
    original_fn = v3.compute_confluence

    for name, variant in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Testing: {name}")
        print(f"{'─' * 60}")

        # Patch proxy
        v3.compute_confluence = monkey_patch_proxy(variant)

        try:
            # Run S4 (Long+Adaptive) and S7 (Full)
            s4_df, _ = v3.run_full_strategy(
                crypto_data, macro_data, cross_asset_data,
                enable_long=True, enable_short=False,
                enable_adaptive_leverage=True,
            )
            s4_metrics = compute_metrics(s4_df["daily_pnl"])

            s7_df, _ = v3.run_full_strategy(
                crypto_data, macro_data, cross_asset_data,
                enable_long=True, enable_short=True,
                enable_adaptive_leverage=True,
            )
            s7_metrics = compute_metrics(s7_df["daily_pnl"])

            # Crash analysis
            covid = crash_return(s7_df, "2020-02-15", "2020-04-15")
            may21 = crash_return(s7_df, "2021-04-15", "2021-07-20")
            bear22 = crash_return(s7_df, "2022-01-01", "2022-12-31")

            results[name] = {
                "s4_sharpe": s4_metrics["sharpe"],
                "s4_cagr": s4_metrics["cagr"],
                "s4_maxdd": s4_metrics["max_dd"],
                "s7_sharpe": s7_metrics["sharpe"],
                "s7_cagr": s7_metrics["cagr"],
                "s7_maxdd": s7_metrics["max_dd"],
                "s7_sortino": s7_metrics["sortino"],
                "covid": covid,
                "may21": may21,
                "bear22": bear22,
            }

            print(f"  S4 Long+Adapt: Sharpe {s4_metrics['sharpe']:.3f}, CAGR {s4_metrics['cagr']:.1f}%, MaxDD {s4_metrics['max_dd']:.1f}%")
            print(f"  S7 Full:       Sharpe {s7_metrics['sharpe']:.3f}, CAGR {s7_metrics['cagr']:.1f}%, MaxDD {s7_metrics['max_dd']:.1f}%")
            print(f"  Crashes:       COVID {covid:+.1f}%, May21 {may21:+.1f}%, 2022 {bear22:+.1f}%")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
        finally:
            v3.compute_confluence = original_fn

    # Walk-forward for each
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD OOS SHARPE")
    print(f"{'=' * 80}")
    for name, variant in variants.items():
        print(f"  Computing WF for {name}...")
        oos = walk_forward(crypto_data, macro_data, cross_asset_data, variant)
        if name in results and "error" not in results[name]:
            results[name]["wf_oos_sharpe"] = oos
        print(f"  {name}: OOS Sharpe = {oos:.3f}")

    # Final comparison table
    print(f"\n{'=' * 100}")
    print("FINAL COMPARISON")
    print(f"{'=' * 100}")
    header = f"{'Proxy Variant':<25} {'S4 Sharpe':>10} {'S7 Sharpe':>10} {'S7 CAGR':>8} {'S7 MaxDD':>9} {'COVID':>7} {'May21':>7} {'2022':>7} {'OOS':>7}"
    print(header)
    print("-" * 100)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} ERROR: {r['error']}")
            continue
        print(f"{name:<25} {r['s4_sharpe']:>10.3f} {r['s7_sharpe']:>10.3f} {r['s7_cagr']:>8.1f} {r['s7_maxdd']:>9.1f} {r['covid']:>+7.1f} {r['may21']:>+7.1f} {r['bear22']:>+7.1f} {r.get('wf_oos_sharpe', 0):>7.3f}")

    # Save
    outpath = RESULTS_DIR / "proxy_3way_comparison.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Winner
    best = max([(n, r.get("wf_oos_sharpe", -99)) for n, r in results.items() if "error" not in r],
               key=lambda x: x[1])
    print(f"\n{'=' * 60}")
    print(f"WINNER: {best[0]} (OOS Sharpe {best[1]:.3f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
