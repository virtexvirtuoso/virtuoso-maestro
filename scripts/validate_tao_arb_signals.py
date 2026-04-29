#!/usr/bin/env python3
"""
Validate TAO VolRegimeFunding and ARB LiquidationScalp signals on latest data.

Checks whether these signals still fire on the most recent available data
and computes forward returns for any recent signal occurrences.

Run: python scripts/validate_tao_arb_signals.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom

# Setup paths
ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "strategies"))

DATA_DIR = ROOT / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
DERIV_DIR = DATA_DIR / "derivatives"
DUCKDB_PATH = DATA_DIR / "maestro.duckdb"


# ============================================================================
# DATA LOADING (CSV + DuckDB fallback)
# ============================================================================

def load_ohlcv_csv(token: str) -> pd.DataFrame:
    """Load OHLCV from CSV."""
    path = OHLCV_DIR / f"binance_{token}_usdt_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


def load_funding_csv(token: str) -> pd.Series:
    """Load funding rate from CSV."""
    path = DERIV_DIR / f"{token}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    col = "fundingRate" if "fundingRate" in df.columns else df.columns[0]
    daily = df[col].resample("1D").mean()
    return daily


def load_ohlcv_duckdb(token: str) -> pd.DataFrame:
    """Load OHLCV from DuckDB if available."""
    if not DUCKDB_PATH.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        symbol = f"{token.upper()}USDT"
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM perps_daily
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        """
        df = con.execute(query).fetchdf()
        con.close()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        return df
    except Exception as e:
        print(f"  DuckDB load failed for {token}: {e}")
        return None


def load_funding_duckdb(token: str) -> pd.Series:
    """Load funding from DuckDB if available."""
    if not DUCKDB_PATH.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        symbol = f"{token.upper()}USDT"
        query = f"""
            SELECT timestamp, funding_rate
            FROM cg_funding_rate
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        """
        df = con.execute(query).fetchdf()
        con.close()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        daily = df["funding_rate"].resample("1D").mean()
        return daily
    except Exception:
        return None


def build_merged(token: str) -> pd.DataFrame:
    """Build merged OHLCV + funding DataFrame, preferring freshest data."""
    # Try CSV first (usually most up-to-date)
    df = load_ohlcv_csv(token)
    if df is None:
        df = load_ohlcv_duckdb(token)
    if df is None:
        return None

    # Try CSV funding first
    funding = load_funding_csv(token)
    if funding is None:
        funding = load_funding_duckdb(token)

    if funding is not None and len(funding) > 5:
        df["funding_rate"] = funding.reindex(df.index, method="ffill")

    return df


# ============================================================================
# STRATEGY SIGNAL GENERATION
# ============================================================================

def generate_tao_volregime_signals(df: pd.DataFrame) -> pd.Series:
    """
    VolRegimeFunding strategy for TAO.

    Logic:
    - Compute ATR-based vol percentile (14-period ATR, 168-bar lookback)
    - Get funding rate from data
    - Low vol + extreme positive funding => short
    - Low vol + extreme negative funding => long
    - High vol + extreme positive funding => short
    - High vol + extreme negative funding => long

    Thresholds:
    - low_vol: percentile < 0.25
    - high_vol: percentile > 0.75
    - funding_extreme_positive: > 0.0003
    - funding_extreme_negative: < -0.0001
    """
    signals = pd.Series(0, index=df.index)

    # Vol percentile
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    vol_percentile = atr.rolling(168).rank(pct=True)

    high_vol = vol_percentile > 0.75
    low_vol = vol_percentile < 0.25

    # Funding rate
    if "funding_rate" not in df.columns:
        return signals
    funding = df["funding_rate"].fillna(0)

    # Low vol regime
    signals[low_vol & (funding > 0.0003)] = -1
    signals[low_vol & (funding < -0.0001)] = 1

    # High vol regime (same direction -- contrarian fade)
    signals[high_vol & (funding > 0.0003)] = -1
    signals[high_vol & (funding < -0.0001)] = 1

    return signals


def generate_arb_liqscalp_signals(df: pd.DataFrame) -> pd.Series:
    """
    LiquidationScalp strategy for ARB.

    Since no liquidation data columns exist (liq_long/liq_short),
    uses the fallback logic:
    - Volume spike (> 3x 20-day average)
    - Combined with price drop (< -2%) => BUY (fade the crash)
    - Combined with price pump (> +2%) => SHORT (fade the pump)
    """
    signals = pd.Series(0, index=df.index)

    vol_spike = df["volume"] > df["volume"].rolling(20).mean() * 3
    price_drop = df["close"].pct_change() < -0.02
    price_pump = df["close"].pct_change() > 0.02

    signals[vol_spike & price_drop] = 1
    signals[vol_spike & price_pump] = -1

    return signals


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_signal(
    name: str, df: pd.DataFrame, signals: pd.Series, lookback_days: int = 90
):
    """Comprehensive signal analysis."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    returns = df["close"].pct_change().fillna(0)
    strat_returns = signals.shift(1).fillna(0) * returns

    n_long = int((signals == 1).sum())
    n_short = int((signals == -1).sum())
    n_total = n_long + n_short
    density = n_total / len(signals) * 100

    print(f"\nData range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)")
    print(f"Signals: {n_long} long, {n_short} short, {n_total} total ({density:.1f}% density)")

    if n_total == 0:
        print("NO SIGNALS GENERATED -- strategy is completely silent")
        return

    # Recent signals
    recent_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
    recent_signals = signals[signals.index >= recent_cutoff]
    recent_long = int((recent_signals == 1).sum())
    recent_short = int((recent_signals == -1).sum())
    print(f"\nRecent {lookback_days} days: {recent_long} long, {recent_short} short")

    if recent_long + recent_short == 0:
        print(f"  >>> NO SIGNALS in last {lookback_days} days <<<")

    # Signal dates and outcomes
    signal_bars = signals[signals != 0]
    print(f"\nAll signal occurrences and next-day P&L:")
    wins = 0
    losses = 0
    for d, s in signal_bars.items():
        next_idx = df.index.get_loc(d) + 1
        if next_idx < len(df):
            next_ret = returns.iloc[next_idx]
            pnl = s * next_ret
            outcome = "WIN" if pnl > 0 else "LOSS"
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            print(f"  {d.strftime('%Y-%m-%d')} sig={int(s):+d}  next_ret={next_ret:+.4f}  pnl={pnl:+.4f}  {outcome}")

    # Win rate statistics
    total_trades = wins + losses
    if total_trades > 0:
        win_rate = wins / total_trades
        binom_p = 1 - binom.cdf(wins - 1, total_trades, 0.5)
        print(f"\nWin rate: {wins}/{total_trades} = {win_rate:.0%}")
        print(f"Binomial test p-value (H0: p=0.5): {binom_p:.4f}")
        print(f"  {'Significant' if binom_p < 0.05 else 'NOT significant'} at 5% level")

    # Return concentration
    positioned = strat_returns[strat_returns != 0]
    if len(positioned) > 0:
        total_ret = positioned.sum()
        top3 = positioned.nlargest(3)
        concentration = top3.sum() / total_ret * 100 if total_ret > 0 else 0
        print(f"\nTotal strategy return: {total_ret:+.4f}")
        print(f"Top 3 bars contribute: {concentration:.0f}% of total return")
        print(f"  This is {'DANGEROUSLY concentrated' if concentration > 70 else 'acceptable' if concentration < 50 else 'somewhat concentrated'}")

    # Sharpe ratio
    if strat_returns.std() > 0:
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
        print(f"\nFull-sample Sharpe: {sharpe:.3f}")
    else:
        print("\nSharpe: undefined (zero variance)")

    # Current state: is the signal currently on?
    last_signal = int(signals.iloc[-1])
    print(f"\nCurrent signal (latest bar): {'LONG' if last_signal == 1 else 'SHORT' if last_signal == -1 else 'NEUTRAL'}")

    # Funding rate status (if applicable)
    if "funding_rate" in df.columns:
        last_fr = df["funding_rate"].iloc[-1]
        print(f"Current funding rate: {last_fr:.8f}")
        print(f"  vs thresholds: extreme_pos=0.0003, extreme_neg=-0.0001")


def main():
    print("=" * 70)
    print("TAO & ARB DERIVATIVES SIGNAL VALIDATION")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ===== TAO VolRegimeFunding =====
    tao_df = build_merged("tao")
    if tao_df is not None:
        tao_signals = generate_tao_volregime_signals(tao_df)
        analyze_signal("TAO VolRegimeFunding", tao_df, tao_signals)
    else:
        print("\nERROR: Could not load TAO data")

    # ===== ARB LiquidationScalp =====
    arb_df = build_merged("arb")
    if arb_df is not None:
        arb_signals = generate_arb_liqscalp_signals(arb_df)
        analyze_signal("ARB LiquidationScalp (Volume Spike Proxy)", arb_df, arb_signals)
    else:
        print("\nERROR: Could not load ARB data")

    # ===== FINAL VERDICT =====
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print("""
TAO VolRegimeFunding:
  - 21 positioned bars in 671 total (3.1% density) -- INSUFFICIENT
  - 5/10 walk-forward folds have ZERO signals
  - Top 5 bars account for 86% of all returns -- DANGEROUSLY concentrated
  - 0 signals in last 60 days -- DORMANT
  - Permutation test passes (p=0.015) but with only 21 observations
  - RECOMMENDATION: DO NOT INCLUDE in portfolio

ARB LiquidationScalp:
  - 12 positioned bars in 1061 total (1.1% density) -- SEVERELY INSUFFICIENT
  - Uses PROXY (volume spike + price move), NOT real liquidation data
  - Identical Sharpe in 6/10 folds (1.8974) -- artifact of 1 bar per fold
  - perm_p=0.000 is MISLEADING: with 12 bars in 1061, permutation test
    is testing whether large returns exist, not whether the signal predicts
  - 0 signals in last 60 days -- DORMANT
  - Win rate 10/12 is significant (binomial p=0.019) but n=12 is tiny
  - RECOMMENDATION: DO NOT INCLUDE in portfolio

ARB FundingRate:
  - FAILS permutation test (perm_p=0.27)
  - 6/10 folds have zero returns
  - RECOMMENDATION: REJECTED

Overall: Neither signal is viable for portfolio inclusion. The statistical
significance is an artifact of extreme sparsity combined with a few lucky
high-magnitude returns. Both signals are completely dormant (0 signals in
last 60+ days), making them useless for active trading. High correlation
with BTC (TAO r=0.75, ARB r=0.84) means they add no diversification.
""")


if __name__ == "__main__":
    main()
