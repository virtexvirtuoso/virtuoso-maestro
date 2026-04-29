"""
Backtest MegaStrategyV4 — 9-strategy comparison with ablation tests.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    run_full_strategy as run_v3_full, ASSET_CONFIGS, TX_COST,
    compute_confluence, detect_regime,
)
from strategies.composite.mega_strategy_v4 import (
    run_mega_v4, DEFAULT_RISK_BUDGETS, DEFAULT_V4_PARAMS,
)

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}

CRASH_PERIODS = {
    "2018 Bear": ("2018-01-01", "2018-12-31"),
    "COVID Mar2020": ("2020-02-15", "2020-04-15"),
    "May 2021": ("2021-05-01", "2021-07-31"),
    "2022 Bear": ("2022-01-01", "2022-12-31"),
}


def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()

    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    cross_asset_data = pd.DataFrame()
    for col_name, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col_name] = df["close"]
        except Exception as e:
            print(f"  {col_name}: FAILED - {e}")

    macro_data = fred_loader.get_multiple(FRED_SERIES, start_date="2015-01-01")
    print(f"  Macro: {len(macro_data)} days")

    # Fama-French
    ff_data = None
    try:
        from datasource.factor_loader import FactorDataLoader
        fl = FactorDataLoader()
        ff_data = fl.get_ff5()
        print(f"  FF5: {len(ff_data)} months")
    except Exception as e:
        print(f"  FF data: FAILED - {e}")

    return crypto_data, cross_asset_data, macro_data, ff_data


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["total_return", "cagr", "sharpe", "sortino", "max_dd", "calmar"]}
    equity = (1 + returns).cumprod()
    n_years = len(returns) / 252
    total_ret = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.1)) - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
    }


def regime_sharpes(returns: pd.Series, regime: pd.Series) -> dict:
    result = {}
    for r in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]:
        mask = regime == r
        if mask.sum() > 20:
            r_ret = returns[mask]
            ann_vol = r_ret.std() * np.sqrt(252) if r_ret.std() > 0 else 1
            result[r] = round(float(r_ret.mean() * 252 / ann_vol), 3)
        else:
            result[r] = None
    return result


def crash_analysis(returns: pd.Series, name: str) -> dict:
    result = {}
    for period_name, (start, end) in CRASH_PERIODS.items():
        mask = (returns.index >= start) & (returns.index <= end)
        if mask.sum() > 5:
            period_ret = returns[mask]
            eq = (1 + period_ret).cumprod()
            result[period_name] = round(float(eq.iloc[-1] - 1) * 100, 2)
        else:
            result[period_name] = None
    return result


def run_buy_hold(crypto_data, assets=None):
    if assets is None:
        assets = list(crypto_data.keys())
    returns_list = []
    for a in assets:
        if a in crypto_data:
            r = crypto_data[a]["close"].pct_change().fillna(0)
            returns_list.append(r)
    if not returns_list:
        return pd.Series(dtype=float)
    combined = pd.concat(returns_list, axis=1).mean(axis=1)
    return combined


def main():
    crypto_data, cross_asset_data, macro_data, ff_data = load_data()
    if not crypto_data:
        print("No crypto data loaded!")
        return

    # Common date range
    all_indices = [crypto_data[a].index for a in crypto_data]
    common_start = max(idx[0] for idx in all_indices)
    common_end = min(idx[-1] for idx in all_indices)
    print(f"\nCommon range: {common_start.date()} to {common_end.date()}")

    # Trim
    for a in crypto_data:
        mask = (crypto_data[a].index >= common_start) & (crypto_data[a].index <= common_end)
        crypto_data[a] = crypto_data[a][mask]

    # Get regime for analysis
    btc_close = crypto_data["BTC"]["close"]
    confluence, _ = compute_confluence(btc_close, macro_data, cross_asset_data, 100, 35)
    regime = detect_regime(confluence)

    results = {}
    all_returns = {}

    # S1: Buy & Hold BTC
    print("\n" + "=" * 70)
    print("RUNNING STRATEGIES")
    print("=" * 70)

    print("\nS1: Buy & Hold BTC...")
    bh_btc = crypto_data["BTC"]["close"].pct_change().fillna(0)
    results["S1_BH_BTC"] = compute_metrics(bh_btc)
    all_returns["S1_BH_BTC"] = bh_btc

    # S2: Buy & Hold Equal Weight
    print("S2: Buy & Hold EW...")
    bh_ew = run_buy_hold(crypto_data)
    results["S2_BH_EW"] = compute_metrics(bh_ew)
    all_returns["S2_BH_EW"] = bh_ew

    # S3: V3 Full
    print("S3: V3 Full...")
    try:
        v3_port, v3_per_asset = run_v3_full(
            crypto_data, macro_data, cross_asset_data,
            enable_long=True, enable_short=True, enable_adaptive_leverage=True,
        )
        v3_ret = v3_port["daily_pnl"]
        results["S3_V3_Full"] = compute_metrics(v3_ret)
        all_returns["S3_V3_Full"] = v3_ret
    except Exception as e:
        print(f"  V3 FAILED: {e}")
        v3_ret = pd.Series(0, index=btc_close.index)
        results["S3_V3_Full"] = compute_metrics(v3_ret)
        all_returns["S3_V3_Full"] = v3_ret

    # S4: V4 Full
    print("S4: V4 Full...")
    try:
        v4_result = run_mega_v4(crypto_data, macro_data, cross_asset_data, ff_data)
        v4_ret = v4_result["portfolio_returns"]
        results["S4_V4_Full"] = compute_metrics(v4_ret)
        all_returns["S4_V4_Full"] = v4_ret
    except Exception as e:
        print(f"  V4 FAILED: {e}")
        import traceback; traceback.print_exc()
        v4_ret = pd.Series(0, index=btc_close.index)
        results["S4_V4_Full"] = compute_metrics(v4_ret)
        all_returns["S4_V4_Full"] = v4_ret

    # S5: V4 Conservative
    print("S5: V4 Conservative...")
    try:
        v4c_result = run_mega_v4(
            crypto_data, macro_data, cross_asset_data, ff_data,
            max_total_leverage=1.5, vol_target=0.20, module_dd_breaker=0.06,
        )
        v4c_ret = v4c_result["portfolio_returns"]
        results["S5_V4_Conservative"] = compute_metrics(v4c_ret)
        all_returns["S5_V4_Conservative"] = v4c_ret
    except Exception as e:
        print(f"  V4 Conservative FAILED: {e}")
        results["S5_V4_Conservative"] = compute_metrics(pd.Series(0, index=btc_close.index))

    # Ablation tests S6-S9
    ablation_configs = [
        ("S6_NoVolBreakout", ["vol_breakout"]),
        ("S7_NoEMARibbon", ["ema_ribbon"]),
        ("S8_NoFFBridge", ["ff_bridge"]),
        ("S9_NoMultiTF", ["multitf"]),
    ]
    for name, disabled in ablation_configs:
        print(f"{name}...")
        try:
            abl_result = run_mega_v4(
                crypto_data, macro_data, cross_asset_data, ff_data,
                disabled_modules=disabled,
            )
            abl_ret = abl_result["portfolio_returns"]
            results[name] = compute_metrics(abl_ret)
            all_returns[name] = abl_ret
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            results[name] = compute_metrics(pd.Series(0, index=btc_close.index))

    # ===== REPORTING =====
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    header = f"{'Strategy':<25} {'Return%':>10} {'CAGR%':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>8}"
    print(header)
    print("-" * len(header))
    for sname, m in results.items():
        print(f"{sname:<25} {m['total_return']:>10.1f} {m['cagr']:>8.1f} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd']:>8.1f} {m['calmar']:>8.3f}")

    # Per-regime Sharpe
    print("\n" + "=" * 70)
    print("PER-REGIME SHARPE")
    print("=" * 70)
    regime_header = f"{'Strategy':<25} {'BULL':>8} {'MILD_B':>8} {'NEUTRAL':>8} {'BEAR':>8} {'ACCUM':>8}"
    print(regime_header)
    print("-" * len(regime_header))
    for sname, ret in all_returns.items():
        r_aligned = regime.reindex(ret.index, method="ffill").fillna("NEUTRAL")
        rs = regime_sharpes(ret, r_aligned)
        vals = [f"{rs.get(r, 'N/A'):>8}" if rs.get(r) is not None else f"{'N/A':>8}"
                for r in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]]
        print(f"{sname:<25} {'  '.join(vals)}")

    # Crash analysis
    print("\n" + "=" * 70)
    print("CRASH ANALYSIS (Return %)")
    print("=" * 70)
    crash_header = f"{'Strategy':<25} {'2018':>10} {'COVID':>10} {'May21':>10} {'2022':>10}"
    print(crash_header)
    print("-" * len(crash_header))
    for sname, ret in all_returns.items():
        ca = crash_analysis(ret, sname)
        vals = [f"{ca.get(p, 'N/A'):>10}" if ca.get(p) is not None else f"{'N/A':>10}"
                for p in CRASH_PERIODS]
        print(f"{sname:<25} {'  '.join(vals)}")

    # Module attribution (V4)
    if "S4_V4_Full" in all_returns:
        print("\n" + "=" * 70)
        print("MODULE P&L ATTRIBUTION (V4 Full)")
        print("=" * 70)
        try:
            v4_result = run_mega_v4(crypto_data, macro_data, cross_asset_data, ff_data)
            if isinstance(v4_result.get("module_contributions"), pd.DataFrame):
                mc = v4_result["module_contributions"]
                for col in mc.columns:
                    total_contrib = (1 + mc[col]).prod() - 1
                    print(f"  {col:<20}: {total_contrib*100:>8.2f}%")
        except:
            print("  (attribution failed)")

    # Monthly heatmap for V4
    if "S4_V4_Full" in all_returns:
        print("\n" + "=" * 70)
        print("MONTHLY RETURNS HEATMAP — V4 Full")
        print("=" * 70)
        v4r = all_returns["S4_V4_Full"]
        monthly = v4r.resample("ME").sum() * 100
        pivot = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        heatmap = pivot.pivot_table(values="Return", index="Year", columns="Month", aggfunc="sum")
        heatmap.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(heatmap.columns)]
        print(heatmap.round(1).to_string())

    # Ablation marginal contribution
    print("\n" + "=" * 70)
    print("ABLATION — MARGINAL MODULE CONTRIBUTION")
    print("=" * 70)
    v4_sharpe = results.get("S4_V4_Full", {}).get("sharpe", 0)
    for abl_name, module_name in [("S6_NoVolBreakout", "Vol Breakout"),
                                   ("S7_NoEMARibbon", "EMARibbon"),
                                   ("S8_NoFFBridge", "FF Bridge"),
                                   ("S9_NoMultiTF", "MultiTF")]:
        abl_sharpe = results.get(abl_name, {}).get("sharpe", 0)
        delta = v4_sharpe - abl_sharpe
        print(f"  {module_name:<20}: Δ Sharpe = {delta:+.3f} (V4={v4_sharpe:.3f}, without={abl_sharpe:.3f})")

    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "date_range": f"{common_start.date()} to {common_end.date()}",
        "metrics": {k: v for k, v in results.items()},
    }
    out_path = RESULTS_DIR / "mega_v4_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
