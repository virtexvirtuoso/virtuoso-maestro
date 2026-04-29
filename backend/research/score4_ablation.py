"""
Score 4 Ablation — which signal is the dissenter when confluence = 4?
Then build and test V3.1 fixes.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    compute_confluence, ASSET_CONFIGS, run_full_strategy
)

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
            if len(df) > 100: crypto_data[name] = df
        except: pass
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col] = df["close"]
        except: pass
    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    return crypto_data, cross_asset_data, macro_data


def main():
    print("=" * 80)
    print("SCORE 4 ABLATION + V3.1 DEVELOPMENT")
    print("=" * 80)

    crypto_data, cross_asset_data, macro_data = load_data()
    btc = crypto_data["BTC"]
    btc_ret = btc["close"].pct_change()

    confluence, breakdown = compute_confluence(
        btc["close"], macro_data, cross_asset_data,
        sma_slow=100, momentum_period=35
    )

    signals = ["m2_accel", "liquidity_proxy", "yield_curve", "cross_asset_mom", "crypto_momentum"]
    signal_labels = ["M2", "Proxy", "YieldCurve", "CrossAsset", "CryptoMom"]

    # ── PART 1: Score 4 dissenter analysis ──
    print("\n── SCORE 4 DISSENTER ANALYSIS ──")
    score4 = confluence == 4
    print(f"Total score-4 days: {score4.sum()} ({score4.mean()*100:.1f}%)")

    print(f"\n{'Signal OFF at Score 4':<25} {'Days':>6} {'% of Score4':>12} {'BTC Ann%':>10} {'Sharpe':>8}")
    print("-" * 65)

    for sig, label in zip(signals, signal_labels):
        # Score 4 means exactly one signal is off
        mask = score4 & (breakdown[sig] == 0)
        days = mask.sum()
        if days < 5:
            print(f"{label + ' OFF':<25} {days:>6} {'N/A':>12}")
            continue
        pct = days / score4.sum() * 100
        ret = btc_ret[mask]
        ann = ret.mean() * 252 * 100
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        print(f"{label + ' OFF':<25} {days:>6} {pct:>11.1f}% {ann:>+10.1f} {sharpe:>8.3f}")

    # By year
    print(f"\n── SCORE 4 DISSENTER BY YEAR ──")
    breakdown["year"] = breakdown.index.year
    breakdown["score4"] = score4

    print(f"{'Year':<6}", end="")
    for label in signal_labels:
        print(f"{label:>12}", end="")
    print()
    print("-" * 66)

    for yr in sorted(breakdown["year"].unique()):
        yr_s4 = breakdown[(breakdown["year"] == yr) & (breakdown["score4"])]
        if len(yr_s4) == 0:
            continue
        print(f"{yr:<6}", end="")
        for sig in signals:
            off_pct = (yr_s4[sig] == 0).mean() * 100
            print(f"{off_pct:>11.1f}%", end="")
        print()

    # ── PART 2: Build and test V3.1 variants ──
    print(f"\n{'=' * 80}")
    print("V3.1 VARIANT TESTING")
    print(f"{'=' * 80}")

    def compute_metrics(returns, label=""):
        if len(returns) < 10 or returns.std() == 0:
            return {"sharpe": 0, "cagr": 0, "maxdd": 0, "mean_lev": 0}
        eq = (1 + returns).cumprod()
        n_yr = len(returns) / 252
        cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1) * 100
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
        dd = float((eq / eq.cummax() - 1).min()) * 100
        return {"sharpe": round(sharpe, 3), "cagr": round(cagr, 1), "maxdd": round(dd, 1)}

    def run_variant(crypto_data, macro_data, cross_asset_data, label, **kwargs):
        try:
            pdf, per_asset = run_full_strategy(
                crypto_data, macro_data, cross_asset_data, **kwargs
            )
            m = compute_metrics(pdf["daily_pnl"])
            m["mean_lev"] = round(float(pdf["total_leverage"].mean()), 3)
            m["max_lev"] = round(float(pdf["total_leverage"].max()), 3)
            m["pct_flat"] = round(float((pdf["total_leverage"] == 0).mean()) * 100, 1)

            # Crash analysis
            for period, s, e in [("covid", "2020-02-15", "2020-04-15"),
                                  ("may21", "2021-04-15", "2021-07-20"),
                                  ("bear22", "2022-01-01", "2022-12-31")]:
                mask = (pdf.index >= pd.Timestamp(s)) & (pdf.index <= pd.Timestamp(e))
                if mask.sum() > 5:
                    ceq = (1 + pdf.loc[mask, "daily_pnl"]).cumprod()
                    m[period] = round(float(ceq.iloc[-1] - 1) * 100, 1)
                else:
                    m[period] = 0

            # Per-year returns
            yearly = pdf["daily_pnl"].resample("YE").sum() * 100
            m["yearly"] = {str(dt.year): round(float(v), 1) for dt, v in yearly.items()}

            print(f"  {label}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%, MaxDD {m['maxdd']:.1f}%, "
                  f"MeanLev {m['mean_lev']:.3f}, Flat {m['pct_flat']:.0f}%")
            return m
        except Exception as e:
            print(f"  {label}: FAILED — {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e)}

    results = {}

    # V3 Baseline (S4: Long+Adaptive, no shorts)
    print("\n[V3 Baseline] S4: Long+Adaptive, no shorts")
    results["V3_S4_baseline"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3 S4 Baseline",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True
    )

    # V3 Baseline S7: Full
    print("\n[V3 Baseline] S7: Full")
    results["V3_S7_baseline"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3 S7 Baseline",
        enable_long=True, enable_short=True, enable_adaptive_leverage=True
    )

    # V3.1a: Score 4 → 0.6x leverage (demote)
    print("\n[V3.1a] Score 4 demoted to 0.6x")
    lev_a = {5: 2.0, 4: 0.6, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
    results["V3.1a_score4_demoted"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1a Score4→0.6x",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_a
    )

    # V3.1b: 3-tier collapse (0-1: 0.3x, 2-3: 1.0x, 4-5: 1.5x)
    print("\n[V3.1b] 3-tier leverage")
    lev_b = {5: 1.5, 4: 1.5, 3: 1.0, 2: 1.0, 1: 0.3, 0: 0.0}
    results["V3.1b_3tier"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1b 3-tier",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_b
    )

    # V3.1c: Aggressive (0: 0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 1.0, 5: 2.0)
    print("\n[V3.1c] Score4 at 1.0x, others bumped")
    lev_c = {5: 2.0, 4: 1.0, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.0}
    results["V3.1c_bump_others"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1c Bump others, Score4→1x",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_c
    )

    # V3.1d: Extreme — always in, leverage = score/3 (min 0.3)
    print("\n[V3.1d] Always-in: leverage = max(score/3, 0.33)")
    lev_d = {5: 1.67, 4: 1.33, 3: 1.0, 2: 0.67, 1: 0.33, 0: 0.33}
    results["V3.1d_always_in"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1d Always-in",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_d
    )

    # V3.1e: Conservative bump — same map but raise minimum
    print("\n[V3.1e] Conservative bump (floor at 0.5x)")
    lev_e = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.5, 0: 0.5}
    results["V3.1e_floor_0.5"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1e Floor 0.5x",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_e
    )

    # V3.1f: Score4 demoted + aggressive sizing
    print("\n[V3.1f] Score4 demoted + aggressive base")
    lev_f = {5: 2.5, 4: 0.8, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.0}
    results["V3.1f_demote4_aggressive"] = run_variant(
        crypto_data, macro_data, cross_asset_data,
        "V3.1f Demote4+Aggressive",
        enable_long=True, enable_short=False, enable_adaptive_leverage=True,
        leverage_map=lev_f
    )

    # Final comparison
    print(f"\n{'=' * 120}")
    print("FINAL COMPARISON")
    print(f"{'=' * 120}")
    print(f"{'Variant':<35} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'MeanLev':>8} {'Flat%':>6} {'COVID':>7} {'May21':>7} {'2022':>7}")
    print("-" * 120)

    for name, r in results.items():
        if "error" in r:
            print(f"{name:<35} ERROR")
            continue
        print(f"{name:<35} {r['sharpe']:>7.3f} {r['cagr']:>7.1f} {r['maxdd']:>8.1f} "
              f"{r['mean_lev']:>8.3f} {r['pct_flat']:>6.1f} "
              f"{r.get('covid', 0):>+7.1f} {r.get('may21', 0):>+7.1f} {r.get('bear22', 0):>+7.1f}")

    # Per-year table
    print(f"\n{'=' * 100}")
    print("PER-YEAR RETURNS (%)")
    print(f"{'=' * 100}")
    years = sorted(set(y for r in results.values() if "yearly" in r for y in r["yearly"]))
    print(f"{'Variant':<35}", end="")
    for yr in years:
        print(f"{yr:>8}", end="")
    print()
    print("-" * 100)
    for name, r in results.items():
        if "yearly" not in r: continue
        print(f"{name:<35}", end="")
        for yr in years:
            v = r["yearly"].get(yr, 0)
            print(f"{v:>+8.1f}", end="")
        print()

    # Save
    outpath = os.path.expanduser("~/Desktop/maestro/data/research/v31_variant_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
