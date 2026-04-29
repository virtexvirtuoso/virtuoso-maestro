"""
Macro Overlay Backtest on SPY
Tests 4 macro-driven strategies against buy-and-hold using VectorBT.
"""

import sys
import os
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

import vectorbt as vbt

START = "2016-01-01"
END = "2026-02-12"

def load_data():
    """Load all required data."""
    stock = StockDataLoader()
    macro = MacroDataLoader()
    
    logger.info("Loading SPY...")
    spy = stock.get_ohlcv("SPY", "1d", START, END)
    logger.info(f"SPY: {len(spy)} rows, {spy.index[0]} to {spy.index[-1]}")
    
    logger.info("Loading TLT...")
    tlt = stock.get_ohlcv("TLT", "1d", START, END)
    logger.info(f"TLT: {len(tlt)} rows")
    
    logger.info("Loading FRED series...")
    t10y2y = macro.get_series("T10Y2Y", START, END)
    m2 = macro.get_series("M2SL", START, END)
    fedfunds = macro.get_series("FEDFUNDS", START, END)
    cpi = macro.get_series("CPIAUCSL", START, END)
    
    logger.info(f"T10Y2Y: {len(t10y2y)}, M2SL: {len(m2)}, FEDFUNDS: {len(fedfunds)}, CPI: {len(cpi)}")
    
    return spy, tlt, t10y2y, m2, fedfunds, cpi


def compute_signals(spy, t10y2y, m2, fedfunds, cpi):
    """Compute macro signals aligned to SPY's daily index."""
    idx = spy.index
    
    # Forward-fill all macro series to daily
    t10y2y_daily = t10y2y.reindex(idx, method="ffill")
    m2_daily = m2.reindex(idx, method="ffill")
    fedfunds_daily = fedfunds.reindex(idx, method="ffill")
    cpi_daily = cpi.reindex(idx, method="ffill")
    
    # 1. Yield curve positive
    yield_curve_positive = (t10y2y_daily > 0).astype(int)
    
    # 2. M2 expanding (YoY > 0) — M2 is monthly, so 12-period = 12 months
    m2_yoy = m2_daily.pct_change(252)  # ~252 trading days = 1 year on daily index
    # But M2 is monthly and forward-filled, so many consecutive days have same value.
    # Better: compute YoY on original monthly, then forward-fill
    m2_yoy_monthly = m2.pct_change(12)
    m2_yoy_daily = m2_yoy_monthly.reindex(idx, method="ffill")
    m2_expanding = (m2_yoy_daily > 0).astype(int)
    
    # 3. M2 accelerating: YoY > 6-month MA of YoY
    m2_yoy_ma6 = m2_yoy_monthly.rolling(6).mean()
    m2_yoy_ma6_daily = m2_yoy_ma6.reindex(idx, method="ffill")
    m2_accelerating = (m2_yoy_daily > m2_yoy_ma6_daily).astype(int)
    
    # 4. CPI declining: YoY < 3-month MA of YoY
    cpi_yoy_monthly = cpi.pct_change(12)
    cpi_yoy_ma3 = cpi_yoy_monthly.rolling(3).mean()
    cpi_yoy_daily = cpi_yoy_monthly.reindex(idx, method="ffill")
    cpi_yoy_ma3_daily = cpi_yoy_ma3.reindex(idx, method="ffill")
    cpi_declining = (cpi_yoy_daily < cpi_yoy_ma3_daily).astype(int)
    
    # 5. Fed not hiking: 3-month change in fed funds <= 0
    ff_change_3m = fedfunds_daily - fedfunds_daily.shift(63)  # ~63 trading days = 3 months
    fed_not_hiking = (ff_change_3m <= 0).astype(int)
    
    signals = pd.DataFrame({
        "yield_curve_positive": yield_curve_positive,
        "m2_expanding": m2_expanding,
        "m2_accelerating": m2_accelerating,
        "cpi_declining": cpi_declining,
        "fed_not_hiking": fed_not_hiking,
    }, index=idx).fillna(0).astype(int)
    
    signals["score"] = signals.sum(axis=1)
    
    logger.info(f"Signal coverage: {signals.notna().all(axis=1).sum()}/{len(signals)} days")
    logger.info(f"Score distribution:\n{signals['score'].value_counts().sort_index()}")
    
    return signals


def run_backtests(spy, tlt, signals):
    """Run all 4 strategies + benchmark."""
    close = spy["close"]
    tlt_close = tlt["close"].reindex(close.index, method="ffill")
    
    # Align signals
    sig = signals.reindex(close.index).fillna(0)
    
    # Strategy entries/exits for VBT
    results = {}
    
    # Buy and hold
    bh_pf = vbt.Portfolio.from_holding(close, init_cash=100000)
    results["Buy & Hold SPY"] = bh_pf
    
    # Strategy 1: Yield Curve Filter
    s1_long = sig["yield_curve_positive"].astype(bool)
    s1_entries = s1_long & ~s1_long.shift(1, fill_value=False)
    s1_exits = ~s1_long & s1_long.shift(1, fill_value=False)
    pf1 = vbt.Portfolio.from_signals(close, entries=s1_entries, exits=s1_exits, init_cash=100000)
    results["1: Yield Curve"] = pf1
    
    # Strategy 2: Liquidity Regime
    s2_long = (sig["m2_expanding"].astype(bool)) & (sig["yield_curve_positive"].astype(bool))
    s2_entries = s2_long & ~s2_long.shift(1, fill_value=False)
    s2_exits = ~s2_long & s2_long.shift(1, fill_value=False)
    pf2 = vbt.Portfolio.from_signals(close, entries=s2_entries, exits=s2_exits, init_cash=100000)
    results["2: Liquidity Regime"] = pf2
    
    # Strategy 3: Full Macro Overlay (score >= 3)
    s3_long = sig["score"] >= 3
    s3_entries = s3_long & ~s3_long.shift(1, fill_value=False)
    s3_exits = ~s3_long & s3_long.shift(1, fill_value=False)
    pf3 = vbt.Portfolio.from_signals(close, entries=s3_entries, exits=s3_exits, init_cash=100000)
    results["3: Full Macro"] = pf3
    
    # Strategy 4: Risk-Off Macro (SPY when score>=3, TLT when score<3)
    # Simulate by computing weighted returns manually
    s4_spy = (sig["score"] >= 3).astype(float)
    s4_tlt = (sig["score"] < 3).astype(float)
    spy_ret = close.pct_change().fillna(0)
    tlt_ret = tlt_close.pct_change().fillna(0)
    s4_daily_ret = s4_spy * spy_ret + s4_tlt * tlt_ret
    s4_equity = (1 + s4_daily_ret).cumprod() * 100000
    # Create a pseudo price series for VBT
    s4_price = s4_equity / s4_equity.iloc[0] * close.iloc[0]
    pf4 = vbt.Portfolio.from_holding(s4_price, init_cash=100000)
    results["4: Risk-Off Macro"] = pf4
    
    return results, sig


def extract_stats(results):
    """Extract key metrics from each portfolio."""
    rows = []
    for name, pf in results.items():
        stats = pf.stats()
        total_ret = pf.total_return() * 100
        
        # CAGR
        days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
        years = days / 365.25
        cagr = ((1 + pf.total_return()) ** (1/years) - 1) * 100 if years > 0 else 0
        
        sharpe = stats.get("Sharpe Ratio", 0)
        max_dd = stats.get("Max Drawdown [%]", 0)
        
        # Trade count and win rate
        n_trades = stats.get("Total Trades", 0)
        win_rate = stats.get("Win Rate [%]", 0)
        
        rows.append({
            "Strategy": name,
            "Total Return %": round(total_ret, 2),
            "CAGR %": round(cagr, 2),
            "Sharpe": round(sharpe, 3) if not pd.isna(sharpe) else "N/A",
            "Max DD %": round(max_dd, 2) if not pd.isna(max_dd) else "N/A",
            "Win Rate %": round(win_rate, 2) if not pd.isna(win_rate) else "N/A",
            "Trades": int(n_trades) if not pd.isna(n_trades) else 0,
        })
    
    return pd.DataFrame(rows)


def compute_ff_alpha(spy, results_dict, signals):
    """Compute Fama-French alpha for each strategy."""
    try:
        # Try to get FF3 data from Kenneth French data library
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=START, end=END)[0]
        ff = ff / 100  # Convert from percent to decimal
        ff.index = pd.DatetimeIndex(ff.index)
        logger.info(f"Loaded Fama-French data: {len(ff)} rows")
    except Exception as e:
        logger.warning(f"Could not load Fama-French data: {e}")
        logger.info("Attempting alternative FF download...")
        try:
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
            ff = pd.read_csv(url, skiprows=3, index_col=0)
            ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
            ff = ff[["Mkt-RF", "SMB", "HML", "RF"]]
            ff = ff.apply(pd.to_numeric, errors="coerce") / 100
            ff = ff.loc[START:END]
            logger.info(f"Loaded FF data from web: {len(ff)} rows")
        except Exception as e2:
            logger.warning(f"FF data unavailable: {e2}. Skipping alpha analysis.")
            return None
    
    close = spy["close"]
    spy_ret = close.pct_change().dropna()
    
    from numpy.linalg import lstsq
    
    alphas = {}
    sig = signals.reindex(close.index).fillna(0)
    
    strategy_returns = {}
    # Strategy 1
    s1 = sig["yield_curve_positive"].astype(bool)
    strategy_returns["1: Yield Curve"] = spy_ret * s1.reindex(spy_ret.index, fill_value=False)
    
    # Strategy 2
    s2 = s1 & sig["m2_expanding"].astype(bool)
    strategy_returns["2: Liquidity Regime"] = spy_ret * s2.reindex(spy_ret.index, fill_value=False)
    
    # Strategy 3
    s3 = sig["score"] >= 3
    strategy_returns["3: Full Macro"] = spy_ret * s3.reindex(spy_ret.index, fill_value=False)
    
    for name, strat_ret in strategy_returns.items():
        try:
            common = ff.index.intersection(strat_ret.index)
            if len(common) < 100:
                alphas[name] = {"alpha_annualized": "N/A", "note": "insufficient data"}
                continue
            
            y = strat_ret.loc[common] - ff.loc[common, "RF"]
            X = ff.loc[common, ["Mkt-RF", "SMB", "HML"]].values
            X = np.column_stack([np.ones(len(X)), X])
            y_vals = y.values
            
            mask = ~(np.isnan(y_vals) | np.isnan(X).any(axis=1))
            coeffs, _, _, _ = lstsq(X[mask], y_vals[mask], rcond=None)
            
            alpha_daily = coeffs[0]
            alpha_annual = alpha_daily * 252 * 100
            
            alphas[name] = {
                "alpha_annualized_%": round(alpha_annual, 3),
                "mkt_beta": round(coeffs[1], 3),
                "smb_beta": round(coeffs[2], 3),
                "hml_beta": round(coeffs[3], 3),
            }
        except Exception as e:
            alphas[name] = {"error": str(e)}
    
    return alphas


def find_regime_periods(spy, signals):
    """Find best/worst periods for each strategy."""
    close = spy["close"]
    spy_ret = close.pct_change().fillna(0)
    sig = signals.reindex(close.index).fillna(0)
    
    regimes = {}
    
    # For score-based strategy, identify regime periods
    score = sig["score"]
    in_market = score >= 3
    
    # Find continuous regime blocks
    regime_changes = in_market.astype(int).diff().fillna(0) != 0
    regime_id = regime_changes.cumsum()
    
    periods = []
    for rid in regime_id.unique():
        mask = regime_id == rid
        block = spy_ret[mask]
        if len(block) < 5:
            continue
        cumret = (1 + block).prod() - 1
        periods.append({
            "start": str(block.index[0].date()),
            "end": str(block.index[-1].date()),
            "days": len(block),
            "return_%": round(cumret * 100, 2),
            "in_market": bool(in_market[mask].iloc[0]),
        })
    
    periods.sort(key=lambda x: x["return_%"])
    
    in_periods = [p for p in periods if p["in_market"]]
    out_periods = [p for p in periods if not p["in_market"]]
    
    regimes["best_in_market"] = in_periods[-3:] if in_periods else []
    regimes["worst_in_market"] = in_periods[:3] if in_periods else []
    regimes["best_avoided"] = out_periods[:3] if out_periods else []  # worst market periods we avoided
    
    return regimes


def main():
    logger.info("=" * 60)
    logger.info("MACRO OVERLAY BACKTEST ON SPY")
    logger.info("=" * 60)
    
    # 1. Load data
    spy, tlt, t10y2y, m2, fedfunds, cpi = load_data()
    
    # 2. Compute signals
    signals = compute_signals(spy, t10y2y, m2, fedfunds, cpi)
    
    # 3. Run backtests
    results, sig = run_backtests(spy, tlt, signals)
    
    # 4. Extract stats
    stats_df = extract_stats(results)
    
    # Compute time in market for each strategy
    close = spy["close"]
    sig_aligned = signals.reindex(close.index).fillna(0)
    total_days = len(close)
    
    time_in_market = {
        "Buy & Hold SPY": 100.0,
        "1: Yield Curve": round(sig_aligned["yield_curve_positive"].mean() * 100, 1),
        "2: Liquidity Regime": round((sig_aligned["yield_curve_positive"].astype(bool) & sig_aligned["m2_expanding"].astype(bool)).mean() * 100, 1),
        "3: Full Macro": round((sig_aligned["score"] >= 3).mean() * 100, 1),
        "4: Risk-Off Macro": 100.0,  # Always invested (SPY or TLT)
    }
    stats_df["Time in Mkt %"] = stats_df["Strategy"].map(time_in_market)
    
    # 5. Print summary
    print("\n" + "=" * 80)
    print("MACRO OVERLAY BACKTEST RESULTS (2016-01-01 to 2026-02-12)")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80)
    
    # 6. Fama-French alpha
    alphas = compute_ff_alpha(spy, results, signals)
    if alphas:
        print("\nFAMA-FRENCH 3-FACTOR ALPHA:")
        print("-" * 60)
        for name, a in alphas.items():
            print(f"  {name}: {a}")
    
    # 7. Regime periods
    regimes = find_regime_periods(spy, signals)
    print("\nREGIME PERIODS (Full Macro Overlay):")
    print("-" * 60)
    print("Best in-market periods:")
    for p in regimes.get("best_in_market", []):
        print(f"  {p['start']} to {p['end']} ({p['days']}d): {p['return_%']:+.1f}%")
    print("Worst in-market periods:")
    for p in regimes.get("worst_in_market", []):
        print(f"  {p['start']} to {p['end']} ({p['days']}d): {p['return_%']:+.1f}%")
    print("Best periods avoided (out of market):")
    for p in regimes.get("best_avoided", []):
        print(f"  {p['start']} to {p['end']} ({p['days']}d): {p['return_%']:+.1f}%")
    
    # 8. Save results
    output_dir = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "run_date": datetime.now().isoformat(),
        "period": f"{START} to {END}",
        "strategies": stats_df.to_dict(orient="records"),
        "fama_french_alpha": alphas if alphas else "unavailable",
        "regime_periods": regimes,
        "signal_stats": {
            "score_distribution": signals["score"].value_counts().sort_index().to_dict(),
            "signal_means": {col: round(signals[col].mean(), 3) for col in signals.columns if col != "score"},
        },
    }
    
    # Convert any numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")
    
    out_path = output_dir / "macro_overlay_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=convert)
    
    logger.info(f"\nResults saved to {out_path}")
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
