"""
Backtest MacroMomentum V2 — 7 strategy variants with pyramiding simulation.
"""
import sys, os, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score, FRED_SERIES
from strategies.composite.macro_momentum import generate_signals as v1_signals, position_size as v1_position_size
from strategies.composite.macro_momentum_v2 import generate_signals, generate_signals_simple, DEFAULT_PARAMS

RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results/macro_momentum_v2_results.json"))
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

START = "2017-01-01"
END = "2026-02-01"
TX_COST = 0.001  # 0.1% per trade


def load_data():
    print("Loading data...")
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()

    btc = stock_loader.get_ohlcv("BTC-USD", "1d", start_date=START, end_date=END)
    
    try:
        mstr = stock_loader.get_ohlcv("MSTR", "1d", start_date=START, end_date=END)
    except Exception:
        mstr = pd.DataFrame()
    
    # Macro data
    macro_score = compute_macro_score(fred_loader, start_date="2015-01-01", end_date=END)
    
    # M2 acceleration
    m2 = fred_loader.get_series("M2SL", start_date="2015-01-01", end_date=END)
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6)
    # Forward fill to daily
    m2_acc = m2_acc.resample("D").ffill().fillna(False)
    
    # MSTR excess return
    mstr_excess = None
    if not mstr.empty:
        btc_ret5 = btc["close"].pct_change(5)
        mstr_ret5 = mstr["close"].pct_change(5)
        mstr_excess = (mstr_ret5 - btc_ret5.reindex(mstr_ret5.index, method="ffill")).reindex(btc.index, method="ffill")

    print(f"BTC: {len(btc)} days, MSTR: {len(mstr)} days")
    return btc, mstr, macro_score, m2_acc, mstr_excess


def simulate_pnl(btc: pd.DataFrame, result_df: pd.DataFrame):
    """Simulate P&L from position_size series with transaction costs."""
    close = btc["close"]
    daily_ret = close.pct_change().fillna(0)
    pos = result_df["position_size"].fillna(0)
    
    # Detect position changes for tx costs
    pos_change = pos.diff().fillna(0).abs()
    tx_costs = pos_change * TX_COST
    
    strat_ret = daily_ret * pos - tx_costs
    equity = (1 + strat_ret).cumprod()
    
    # Trade stats
    signals = result_df["signal"]
    entries = result_df["entry_type"]
    n_entries = ((entries.str.startswith("entry_")) | (entries == "")).sum() - (entries == "").sum()
    n_entries = entries.str.startswith("entry_").sum()
    n_pyramids = (entries == "pyramid").sum()
    n_trims = entries.str.startswith("trim_").sum()
    n_exits = entries.isin(["trailing_stop", "macro_exit"]).sum()
    
    return equity, strat_ret, {
        "entries": int(n_entries),
        "pyramids": int(n_pyramids),
        "trims": int(n_trims),
        "exits": int(n_exits),
        "total_trades": int(n_entries + n_pyramids + n_trims + n_exits),
    }


def simulate_v1(btc, macro_score):
    """Simulate original V1 strategy."""
    sig = v1_signals(btc)
    pos = v1_position_size(sig, macro_score)
    daily_ret = btc["close"].pct_change().fillna(0)
    
    pos_change = pos.diff().fillna(0).abs()
    tx = pos_change * TX_COST
    strat_ret = daily_ret * pos - tx
    equity = (1 + strat_ret).cumprod()
    return equity, strat_ret, pos


def calc_metrics(equity: pd.Series, strat_ret: pd.Series, pos=None):
    """Calculate performance metrics."""
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate (daily)
    win_rate = (strat_ret[strat_ret != 0] > 0).mean() if (strat_ret != 0).any() else 0
    
    # Time in market
    if pos is not None:
        time_in = (pos > 0).mean()
    else:
        time_in = (strat_ret != 0).mean()
    
    return {
        "total_return": round(total_ret * 100, 1),
        "cagr": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "win_rate": round(win_rate * 100, 1),
        "time_in_market": round(time_in * 100, 1),
    }


def crash_analysis(equity: pd.Series, name: str):
    """Analyze performance during crash periods."""
    crashes = {
        "2018 Bear": ("2018-01-01", "2018-12-31"),
        "COVID Mar 2020": ("2020-02-15", "2020-04-15"),
        "May 2021": ("2021-05-01", "2021-07-31"),
        "2022 Bear": ("2022-01-01", "2022-12-31"),
    }
    results = {}
    for label, (s, e) in crashes.items():
        subset = equity.loc[s:e]
        if len(subset) < 2:
            results[label] = "N/A"
            continue
        ret = subset.iloc[-1] / subset.iloc[0] - 1
        dd = (subset / subset.cummax() - 1).min()
        results[label] = f"Return: {ret*100:.1f}%, MaxDD: {dd*100:.1f}%"
    return results


def monthly_heatmap(strat_ret: pd.Series, name: str):
    """Print monthly return heatmap."""
    monthly = strat_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    heatmap = pivot.pivot(index="year", columns="month", values="return")
    heatmap.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"\n{'='*80}")
    print(f"Monthly Returns Heatmap — {name}")
    print(f"{'='*80}")
    print(heatmap.round(1).to_string())
    annual = strat_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    print(f"\nAnnual: {dict(zip(annual.index.year, annual.round(1)))}")


def main():
    btc, mstr, macro_score, m2_acc, mstr_excess = load_data()
    
    print("\n" + "="*100)
    print("RUNNING 7 STRATEGY VARIANTS")
    print("="*100)
    
    all_results = {}
    equities = {}
    
    # S1: Buy & Hold
    daily_ret_bh = btc["close"].pct_change().fillna(0)
    eq_bh = (1 + daily_ret_bh).cumprod()
    metrics_bh = calc_metrics(eq_bh, daily_ret_bh, pd.Series(1.0, index=btc.index))
    metrics_bh["entries"] = 1
    metrics_bh["pyramids"] = 0
    metrics_bh["total_trades"] = 1
    all_results["S1: Buy & Hold"] = metrics_bh
    equities["S1"] = eq_bh
    print("S1: Buy & Hold — done")
    
    # S2: Original V1 (S5)
    eq_v1, ret_v1, pos_v1 = simulate_v1(btc, macro_score)
    metrics_v1 = calc_metrics(eq_v1, ret_v1, pos_v1)
    all_results["S2: Original S5"] = metrics_v1
    equities["S2"] = eq_v1
    print("S2: Original S5 — done")
    
    # S3: V2 Simple (price only)
    res3 = generate_signals_simple(btc)
    eq3, ret3, trades3 = simulate_pnl(btc, res3)
    m3 = calc_metrics(eq3, ret3, res3["position_size"])
    m3.update(trades3)
    all_results["S3: V2 Simple"] = m3
    equities["S3"] = eq3
    print("S3: V2 Simple — done")
    
    # S4: V2 + M2 (no funding, no MSTR)
    res4 = generate_signals(btc, macro_score=macro_score, m2_accelerating=m2_acc)
    eq4, ret4, trades4 = simulate_pnl(btc, res4)
    m4 = calc_metrics(eq4, ret4, res4["position_size"])
    m4.update(trades4)
    all_results["S4: V2 + Macro"] = m4
    equities["S4"] = eq4
    print("S4: V2 + Macro — done")
    
    # S5: V2 + MSTR flow
    res5 = generate_signals(btc, macro_score=macro_score, m2_accelerating=m2_acc, mstr_excess=mstr_excess)
    eq5, ret5, trades5 = simulate_pnl(btc, res5)
    m5 = calc_metrics(eq5, ret5, res5["position_size"])
    m5.update(trades5)
    all_results["S5: V2 + MSTR"] = m5
    equities["S5"] = eq5
    print("S5: V2 + MSTR — done")
    
    # S6: V2 Full (all signals, no funding data available so same as S5 for now)
    res6 = generate_signals(btc, macro_score=macro_score, m2_accelerating=m2_acc, 
                           mstr_excess=mstr_excess)
    eq6, ret6, trades6 = simulate_pnl(btc, res6)
    m6 = calc_metrics(eq6, ret6, res6["position_size"])
    m6.update(trades6)
    all_results["S6: V2 Full"] = m6
    equities["S6"] = eq6
    print("S6: V2 Full — done")
    
    # S7: V2 Aggressive pyramiding
    res7 = generate_signals(btc, macro_score=macro_score, m2_accelerating=m2_acc,
                           mstr_excess=mstr_excess,
                           pyramid_size=0.3, max_position=1.5)
    eq7, ret7, trades7 = simulate_pnl(btc, res7)
    m7 = calc_metrics(eq7, ret7, res7["position_size"])
    m7.update(trades7)
    all_results["S7: V2 Aggressive"] = m7
    equities["S7"] = eq7
    print("S7: V2 Aggressive — done")
    
    # Print comparison table
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON")
    print("="*120)
    
    headers = ["Strategy", "TotalRet%", "CAGR%", "Sharpe", "Sortino", "MaxDD%", "Calmar", 
               "WinRate%", "InMkt%", "Entries", "Pyramids", "Trades"]
    header_fmt = f"{'Strategy':<22} {'TotRet%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'WR%':>6} {'InMkt%':>7} {'Ent':>5} {'Pyr':>5} {'Trd':>5}"
    print(header_fmt)
    print("-" * 120)
    
    for name, m in all_results.items():
        print(f"{name:<22} {m['total_return']:>8.1f} {m['cagr']:>7.1f} {m['sharpe']:>7.2f} "
              f"{m['sortino']:>8.2f} {m['max_dd']:>7.1f} {m['calmar']:>7.2f} "
              f"{m['win_rate']:>6.1f} {m['time_in_market']:>7.1f} "
              f"{m.get('entries', '-'):>5} {m.get('pyramids', '-'):>5} {m.get('total_trades', '-'):>5}")
    
    # Crash analysis for best strategy
    best_name = max([k for k in all_results if "V2" in k], 
                    key=lambda k: all_results[k]["sharpe"])
    best_key = best_name.split(":")[0].strip()
    
    print(f"\n{'='*80}")
    print(f"CRASH ANALYSIS — {best_name}")
    print(f"{'='*80}")
    
    crash_bh = crash_analysis(equities["S1"], "Buy & Hold")
    crash_best = crash_analysis(equities[best_key], best_name)
    
    for period in crash_bh:
        print(f"  {period}:")
        print(f"    Buy & Hold: {crash_bh[period]}")
        print(f"    {best_name}: {crash_best[period]}")
    
    # Monthly heatmap for best
    if best_key in equities:
        best_ret = btc["close"].pct_change().fillna(0)
        # Recompute strat returns for best
        if best_key == "S3":
            monthly_heatmap(ret3, best_name)
        elif best_key == "S4":
            monthly_heatmap(ret4, best_name)
        elif best_key == "S5":
            monthly_heatmap(ret5, best_name)
        elif best_key == "S6":
            monthly_heatmap(ret6, best_name)
        elif best_key == "S7":
            monthly_heatmap(ret7, best_name)
    
    # Save results
    save_data = {k: v for k, v in all_results.items()}
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
