#!/usr/bin/env python3
"""Test script for Maestro data source loaders."""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_fred():
    print("\n" + "="*60)
    print("FRED MACRO DATA LOADER")
    print("="*60)
    from datasource.fred_loader import MacroDataLoader

    loader = MacroDataLoader()

    # Yield curve spread
    spread = loader.get_series('T10Y2Y')
    print(f"\n10Y-2Y Spread: {len(spread)} obs, {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"  Latest: {spread.iloc[-1]:.2f}%")

    # Fed Funds
    ff = loader.get_series('FEDFUNDS')
    print(f"\nFed Funds: {len(ff)} obs, {ff.index[0].date()} to {ff.index[-1].date()}")
    print(f"  Latest: {ff.iloc[-1]:.2f}%")

    # CPI
    cpi = loader.get_series('CPIAUCSL')
    print(f"\nCPI: {len(cpi)} obs, {cpi.index[0].date()} to {cpi.index[-1].date()}")
    cpi_yoy = cpi.pct_change(12) * 100
    print(f"  Latest YoY: {cpi_yoy.iloc[-1]:.2f}%")

    # M2
    m2 = loader.get_series('M2SL')
    print(f"\nM2: {len(m2)} obs, {m2.index[0].date()} to {m2.index[-1].date()}")
    m2_yoy = m2.pct_change(12) * 100
    print(f"  Latest YoY: {m2_yoy.iloc[-1]:.2f}%")

    # Macro regime
    regime = loader.get_macro_regime()
    print(f"\nMacro Regime: {regime.shape[1]} indicators, {len(regime)} obs")
    print(f"  Latest values:")
    latest = regime.iloc[-1]
    for col in regime.columns:
        if not latest[col] != latest[col]:  # not NaN
            print(f"    {col}: {latest[col]:.2f}")

    return spread, cpi_yoy, m2_yoy


def test_fama_french():
    print("\n" + "="*60)
    print("FAMA-FRENCH FACTOR LOADER")
    print("="*60)
    from datasource.factor_loader import FactorDataLoader

    loader = FactorDataLoader()

    ff3 = loader.get_ff3()
    print(f"\nFF3 Factors: {ff3.shape[1]} cols, {len(ff3)} obs")
    print(f"  Range: {ff3.index[0].date()} to {ff3.index[-1].date()}")
    print(f"  Columns: {list(ff3.columns)}")
    print(f"\n  Summary stats (monthly returns):")
    print(ff3.describe().round(4).to_string())

    return ff3


def test_alphavantage():
    print("\n" + "="*60)
    print("ALPHA VANTAGE LOADER")
    print("="*60)
    from datasource.alphavantage_loader import AlphaVantageLoader

    loader = AlphaVantageLoader()
    if not loader.api_key:
        print("  ⚠ No API key set, skipping Alpha Vantage tests")
        return None

    print("  Fetching SPY daily...")
    spy = loader.get_daily('SPY', full=False)
    if not spy.empty:
        print(f"  SPY: {len(spy)} obs, {spy.index[0].date()} to {spy.index[-1].date()}")
        print(f"  Latest close: ${spy['close'].iloc[-1]:.2f}")
    else:
        print("  No data returned")
    return spy


def print_regime_snapshot(spread, cpi_yoy, m2_yoy):
    print("\n" + "="*60)
    print("MACRO REGIME SNAPSHOT")
    print("="*60)
    print(f"  Yield Curve (10Y-2Y): {spread.iloc[-1]:.2f}%  {'📈 Normal' if spread.iloc[-1] > 0 else '📉 INVERTED'}")
    print(f"  CPI YoY:              {cpi_yoy.iloc[-1]:.2f}%  {'🔥 Hot' if cpi_yoy.iloc[-1] > 3 else '❄️ Cool'}")
    print(f"  M2 YoY:               {m2_yoy.iloc[-1]:.2f}%  {'💧 Expanding' if m2_yoy.iloc[-1] > 0 else '🏜️ Contracting'}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        spread, cpi_yoy, m2_yoy = test_fred()
    except Exception as e:
        logger.error(f"FRED test failed: {e}")
        spread = cpi_yoy = m2_yoy = None

    try:
        ff3 = test_fama_french()
    except Exception as e:
        logger.error(f"Fama-French test failed: {e}")

    try:
        test_alphavantage()
    except Exception as e:
        logger.error(f"Alpha Vantage test failed: {e}")

    if spread is not None:
        print_regime_snapshot(spread, cpi_yoy, m2_yoy)

    print("\n✅ All tests complete.")
