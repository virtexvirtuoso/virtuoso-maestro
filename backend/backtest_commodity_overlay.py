"""
Commodity-Macro Overlay Backtest on SPY (2005-2026)
Tests whether adding commodity signals improves macro timing.
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

sys.path.insert(0, str(Path(__file__).parent))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

START = "2005-01-01"
END = "2026-02-12"
RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    stock = StockDataLoader()
    macro = MacroDataLoader()

    logger.info("Loading SPY, TLT, GLD from yfinance...")
    spy = stock.get_ohlcv("SPY", "1d", START, END)
    tlt = stock.get_ohlcv("TLT", "1d", START, END)
    gld = stock.get_ohlcv("GLD", "1d", START, END)

    logger.info(f"SPY: {len(spy)} rows ({spy.index[0].date()} to {spy.index[-1].date()})")
    logger.info(f"TLT: {len(tlt)} rows")
    logger.info(f"GLD: {len(gld)} rows")

    # FRED data (unlimited calls)
    logger.info("Loading FRED series...")
    fred_series = {}
    for s in ['T10Y2Y', 'FEDFUNDS', 'CPIAUCSL', 'M2SL', 'DCOILWTICO']:
        fred_series[s] = macro.get_series(s, START, END)
        logger.info(f"  {s}: {len(fred_series[s])} rows")

    return spy, tlt, gld, fred_series


def compute_macro_signals(fred: dict, spy_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 4 macro signals, forward-filled to daily."""
    signals = pd.DataFrame(index=spy_index)

    # 1. Yield curve positive (T10Y2Y > 0)
    t10y2y = fred['T10Y2Y'].reindex(spy_index).ffill()
    signals['yield_curve_positive'] = (t10y2y > 0).astype(int)

    # 2. M2 expanding (3-month ROC > 0)
    m2 = fred['M2SL'].reindex(spy_index).ffill()
    m2_roc = m2.pct_change(63)  # ~3 months trading days
    signals['m2_expanding'] = (m2_roc > 0).astype(int)

    # 3. CPI declining (3-month ROC < 0 means inflation decelerating)
    cpi = fred['CPIAUCSL'].reindex(spy_index).ffill()
    # YoY CPI change declining over 3 months
    cpi_yoy = cpi.pct_change(252)
    cpi_yoy_roc = cpi_yoy - cpi_yoy.shift(63)
    signals['cpi_declining'] = (cpi_yoy_roc < 0).astype(int)

    # 4. Fed not hiking (fed funds rate 3-month change <= 0)
    ff = fred['FEDFUNDS'].reindex(spy_index).ffill()
    ff_chg = ff - ff.shift(63)
    signals['fed_not_hiking'] = (ff_chg <= 0).astype(int)

    signals['macro_score'] = signals.sum(axis=1)
    return signals


def compute_commodity_signals(fred: dict, spy_index: pd.DatetimeIndex,
                               gld_close: pd.Series = None) -> pd.DataFrame:
    """Compute 3 commodity signals using FRED data (saves AV calls)."""
    signals = pd.DataFrame(index=spy_index)

    # WTI Oil from FRED
    oil = fred['DCOILWTICO'].reindex(spy_index).ffill()
    oil_roc_3m = oil.pct_change(63)
    signals['oil_declining'] = (oil_roc_3m < 0).astype(int)

    # Gold from GLD ETF (avoids FRED series issues)
    gold = gld_close.reindex(spy_index).ffill() if gld_close is not None else pd.Series(1.0, index=spy_index)
    gold_roc_3m = gold.pct_change(63)
    signals['gold_rising'] = (gold_roc_3m > 0).astype(int)

    # Copper - use FRED PCOPPUSDM (monthly copper price) if available, else try AV
    try:
        from datasource.fred_loader import MacroDataLoader
        macro = MacroDataLoader()
        copper = macro.get_series('PCOPPUSDM', START, END)
        logger.info(f"Copper from FRED: {len(copper)} rows")
    except Exception:
        copper = pd.Series(dtype=float)

    if copper.empty or len(copper) < 10:
        # Fallback: try Alpha Vantage
        try:
            from datasource.alphavantage_loader import AlphaVantageLoader
            av = AlphaVantageLoader()
            copper_df = av.get_commodity('COPPER', interval='monthly')
            copper = copper_df.iloc[:, 0] if not copper_df.empty else pd.Series(dtype=float)
            logger.info(f"Copper from AV: {len(copper)} rows")
        except Exception as e:
            logger.warning(f"Copper load failed: {e}")
            copper = pd.Series(dtype=float)

    if not copper.empty:
        copper = copper.reindex(spy_index).ffill()
        copper_roc_6m = copper.pct_change(126)
        signals['copper_momentum'] = (copper_roc_6m > 0).astype(int)
    else:
        signals['copper_momentum'] = 1  # Default bullish if no data

    signals['commodity_breadth'] = (
        (signals['oil_declining'] + signals['gold_rising'] + signals['copper_momentum']) >= 2
    ).astype(int)
    signals['commodity_score'] = signals[['oil_declining', 'gold_rising', 'copper_momentum']].sum(axis=1)

    return signals


def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0)


def strategy_metrics(equity: pd.Series, name: str, time_in_market: float = 1.0) -> dict:
    """Compute strategy performance metrics from equity curve."""
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1

    daily_ret = equity.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    return {
        'strategy': name,
        'total_return': round(total_return * 100, 1),
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd * 100, 1),
        'time_in_market': round(time_in_market * 100, 1),
        'start': str(equity.index[0].date()),
        'end': str(equity.index[-1].date()),
        'years': round(n_years, 1),
    }


def run_backtest():
    spy, tlt, gld, fred = load_data()

    spy_close = spy['close']
    tlt_close = tlt['close']
    gld_close = gld['close']

    # Align everything to SPY index
    common_start = max(spy.index[0], tlt.index[0], gld.index[0])
    idx = spy.index[spy.index >= common_start]

    spy_close = spy_close.reindex(idx).ffill()
    tlt_close = tlt_close.reindex(idx).ffill()
    gld_close = gld_close.reindex(idx).ffill()

    spy_ret = compute_returns(spy_close)
    tlt_ret = compute_returns(tlt_close)
    gld_ret = compute_returns(gld_close)

    logger.info("Computing macro signals...")
    macro_sig = compute_macro_signals(fred, idx)

    logger.info("Computing commodity signals...")
    comm_sig = compute_commodity_signals(fred, idx, gld_close)

    combined_score = macro_sig['macro_score'] + comm_sig['commodity_score']

    # Skip warm-up period (need 252 days for YoY calcs)
    warmup = 252
    idx = idx[warmup:]
    spy_ret = spy_ret.loc[idx]
    tlt_ret = tlt_ret.loc[idx]
    gld_ret = gld_ret.loc[idx]
    macro_sig = macro_sig.loc[idx]
    comm_sig = comm_sig.loc[idx]
    combined_score = combined_score.loc[idx]

    results = []

    # 1. Buy & Hold SPY
    eq_bh = (1 + spy_ret).cumprod()
    results.append(strategy_metrics(eq_bh, "Buy & Hold SPY"))

    # 2. Macro Only (3/4 bullish -> SPY, else cash)
    macro_long = (macro_sig['macro_score'] >= 3).astype(int).shift(1).fillna(0)
    eq_macro = (1 + spy_ret * macro_long).cumprod()
    results.append(strategy_metrics(eq_macro, "Macro Only", macro_long.mean()))

    # 3. Commodity Only (breadth bullish -> SPY, else cash)
    comm_long = comm_sig['commodity_breadth'].shift(1).fillna(0)
    eq_comm = (1 + spy_ret * comm_long).cumprod()
    results.append(strategy_metrics(eq_comm, "Commodity Only", comm_long.mean()))

    # 4. Combined (macro + commodity score >= 4 of 7)
    combined_long = (combined_score >= 4).astype(int).shift(1).fillna(0)
    eq_combined = (1 + spy_ret * combined_long).cumprod()
    results.append(strategy_metrics(eq_combined, "Combined Macro+Commodity", combined_long.mean()))

    # 5. Full Rotation: SPY >= 5, GLD 3-4, TLT <= 2
    rotation_ret = pd.Series(0.0, index=idx)
    score_shifted = combined_score.shift(1).fillna(0)
    spy_mask = score_shifted >= 5
    gld_mask = (score_shifted >= 3) & (score_shifted < 5)
    tlt_mask = score_shifted <= 2
    rotation_ret[spy_mask] = spy_ret[spy_mask]
    rotation_ret[gld_mask] = gld_ret[gld_mask]
    rotation_ret[tlt_mask] = tlt_ret[tlt_mask]
    eq_rotation = (1 + rotation_ret).cumprod()
    tim = (spy_mask.sum() + gld_mask.sum() + tlt_mask.sum()) / len(idx)
    results.append(strategy_metrics(eq_rotation, "Full Rotation", tim))

    # Print results
    print("\n" + "=" * 80)
    print("COMMODITY-MACRO OVERLAY BACKTEST RESULTS")
    print(f"Period: {idx[0].date()} to {idx[-1].date()} ({results[0]['years']} years)")
    print("=" * 80)
    header = f"{'Strategy':<28} {'Return':>8} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'In Mkt':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<28} {r['total_return']:>7.1f}% {r['cagr']:>6.2f}% {r['sharpe']:>7.2f} {r['max_drawdown']:>7.1f}% {r['time_in_market']:>7.1f}%")
    print("=" * 80)

    # Signal analysis
    print("\nSIGNAL ANALYSIS:")
    print(f"  Macro score distribution: {macro_sig['macro_score'].value_counts().sort_index().to_dict()}")
    print(f"  Commodity breadth bullish: {comm_sig['commodity_breadth'].mean()*100:.1f}% of days")
    print(f"  Combined score mean: {combined_score.mean():.2f}")
    print(f"  Combined >= 4: {(combined_score >= 4).mean()*100:.1f}% of days")
    print(f"  Combined >= 5: {(combined_score >= 5).mean()*100:.1f}% of days")

    # Save
    output = {
        'run_date': datetime.now().isoformat(),
        'period': f"{idx[0].date()} to {idx[-1].date()}",
        'strategies': results,
        'signal_stats': {
            'macro_score_mean': round(macro_sig['macro_score'].mean(), 2),
            'commodity_breadth_pct': round(comm_sig['commodity_breadth'].mean() * 100, 1),
            'combined_score_mean': round(combined_score.mean(), 2),
        }
    }
    out_path = RESULTS_DIR / "commodity_overlay_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_backtest()
