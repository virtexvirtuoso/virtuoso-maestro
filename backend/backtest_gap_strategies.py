#!/usr/bin/env python3
"""
Gap Strategies Backtest — Fill the holes in Mega V3.

Strategies:
1. Volatility Breakout (BBW squeeze → breakout)
2. Mean Reversion (NEUTRAL/ACCUMULATION regimes only)
3. Enhanced Short Side (loosened conditions)
4. Funding Rate Carry (harvest funding in all regimes)
5. Sector Rotation (momentum-weighted across crypto sectors)

Usage:
    cd ~/Desktop/maestro/backend
    source venv/bin/activate
    python backtest_gap_strategies.py
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TX_COST = 0.001  # 0.1%
DATA_DIR = Path(os.path.expanduser("~/Desktop/maestro/data"))
DERIV_DIR = DATA_DIR / "derivatives"
RESULTS_DIR = DATA_DIR / "backtest_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2020-01-01"
END = "2026-02-12"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bb(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s

def bbw(close, period=20, std=2.0):
    """Bollinger Band Width"""
    lo, mid, hi = bb(close, period, std)
    return ((hi - lo) / mid.replace(0, np.nan)).fillna(0)

def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def sma(close, period):
    return close.rolling(period).mean()

def realized_vol(close, lookback=20):
    return close.pct_change().rolling(lookback).std() * np.sqrt(252)

def sharpe(returns, annual=True):
    if returns.std() == 0:
        return 0.0
    s = returns.mean() / returns.std()
    return s * np.sqrt(252) if annual else s

def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return dd.min()

def calc_metrics(daily_pnl, label=""):
    eq = (1 + daily_pnl).cumprod()
    total_ret = eq.iloc[-1] - 1 if len(eq) > 0 else 0
    sh = sharpe(daily_pnl)
    mdd = max_drawdown(eq)
    trades = (daily_pnl.abs() > 0).sum()
    wins = (daily_pnl > 0).sum()
    wr = wins / max(trades, 1)
    return {
        "label": label,
        "total_return": round(float(total_ret), 4),
        "sharpe": round(float(sh), 3),
        "max_drawdown": round(float(mdd), 4),
        "win_rate": round(float(wr), 4),
        "num_days": int(len(daily_pnl)),
    }

def regime_metrics(daily_pnl, regime_series, label=""):
    """Sharpe per regime"""
    out = {}
    for r in regime_series.unique():
        if pd.isna(r):
            continue
        mask = regime_series == r
        rp = daily_pnl[mask]
        if len(rp) > 10:
            out[r] = round(float(sharpe(rp)), 3)
    return out


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_crypto_ohlcv(ticker, start=START, end=END):
    """Load crypto OHLCV via yfinance with caching."""
    cache = DATA_DIR / "stocks" / f"{ticker.replace('-','_')}_1d.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        df = df.loc[start:end]
        if len(df) > 100:
            return df
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if len(df) > 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df

def load_funding(symbol):
    """Load funding rate CSV, return daily Series."""
    path = DERIV_DIR / f"{symbol.lower()}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    # Resample to daily (sum of 3x 8hr funding)
    daily = df["fundingRate"].resample("1D").sum()
    return daily

def load_macro_simple():
    """Load cross-asset macro data for confluence (simplified)."""
    tickers = {"GLD": "gold", "UUP": "dxy", "TLT": "bonds", "HYG": "hyg", "CPER": "copper"}
    frames = {}
    for tk, name in tickers.items():
        try:
            d = yf.download(tk, start="2019-01-01", end=END, auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.columns = [c.lower() for c in d.columns]
            frames[name] = d["close"]
        except Exception:
            pass
    cross_asset = pd.DataFrame(frames)

    # M2 via FRED (try cache)
    m2_cache = DATA_DIR / "macro" / "M2SL.parquet"
    macro_data = pd.DataFrame()
    if m2_cache.exists():
        m2 = pd.read_parquet(m2_cache)
        if isinstance(m2, pd.DataFrame) and len(m2.columns) > 0:
            col = m2.columns[0]
            macro_data["m2"] = m2[col]

    # Yield curve
    yc_cache = DATA_DIR / "macro" / "T10Y2Y.parquet"
    if yc_cache.exists():
        yc = pd.read_parquet(yc_cache)
        if isinstance(yc, pd.DataFrame) and len(yc.columns) > 0:
            col = yc.columns[0]
            macro_data["yield_curve"] = yc[col]

    return macro_data, cross_asset


# ---------------------------------------------------------------------------
# Confluence & Regime (simplified from V3)
# ---------------------------------------------------------------------------

def compute_confluence_simple(close, macro_data, cross_asset, sma_period=100, mom_period=35):
    """5-signal confluence, returns (score, breakdown)."""
    idx = close.index
    signals = {}

    # Sig1: M2 acceleration
    sig1 = pd.Series(0, index=idx)
    if macro_data is not None and "m2" in macro_data.columns:
        m2 = macro_data["m2"].reindex(idx, method="ffill").ffill()
        m2_yoy = m2.pct_change(365).fillna(0)
        m2_ma = m2_yoy.rolling(180).mean()
        sig1 = (m2_yoy > m2_ma).astype(int).fillna(0).astype(int)
    signals["m2"] = sig1

    # Sig2: Liquidity proxy
    sig2 = pd.Series(0, index=idx)
    if cross_asset is not None:
        liq = pd.Series(0.0, index=idx)
        for col, direction in [("dxy", -1), ("gold", 1), ("bonds", 1), ("hyg", 1)]:
            if col in cross_asset.columns:
                s = cross_asset[col].reindex(idx, method="ffill").ffill()
                chg = s.pct_change(20)
                liq += ((chg * direction) > 0).astype(float).fillna(0)
        sig2 = (liq >= 3).astype(int)
    signals["liquidity"] = sig2

    # Sig3: Yield curve
    sig3 = pd.Series(0, index=idx)
    if macro_data is not None and "yield_curve" in macro_data.columns:
        yc = macro_data["yield_curve"].reindex(idx, method="ffill").ffill()
        sig3 = ((yc > 0) | ((yc.diff(20) > 0) & (yc.rolling(60).min() < 0))).astype(int).fillna(0).astype(int)
    signals["yield_curve"] = sig3

    # Sig4: Cross-asset momentum
    sig4 = pd.Series(0, index=idx)
    if cross_asset is not None:
        score = pd.Series(0.0, index=idx)
        if "gold" in cross_asset.columns:
            g = cross_asset["gold"].reindex(idx, method="ffill").ffill()
            score += (g > g.rolling(60).mean()).astype(float).fillna(0)
        if "dxy" in cross_asset.columns:
            d = cross_asset["dxy"].reindex(idx, method="ffill").ffill()
            score += (d < d.rolling(60).mean()).astype(float).fillna(0)
        if "copper" in cross_asset.columns and "gold" in cross_asset.columns:
            c = cross_asset["copper"].reindex(idx, method="ffill").ffill()
            g = cross_asset["gold"].reindex(idx, method="ffill").ffill()
            ratio = (c / g.replace(0, np.nan)).ffill()
            score += (ratio.pct_change(60) > 0).astype(float).fillna(0)
        sig4 = (score >= 2).astype(int)
    signals["cross_asset"] = sig4

    # Sig5: Crypto momentum
    s = close.rolling(sma_period).mean()
    roc = close.pct_change(mom_period)
    sig5 = ((close > s) & (roc > 0)).astype(int).fillna(0).astype(int)
    signals["crypto_mom"] = sig5

    # Lag all by 1
    for k in signals:
        signals[k] = signals[k].shift(1).fillna(0).astype(int)

    confluence = sum(signals.values())
    return confluence, pd.DataFrame(signals, index=idx)

def detect_regime(confluence):
    regime = pd.Series("NEUTRAL", index=confluence.index)
    regime[confluence >= 4] = "BULL"
    regime[(confluence >= 2) & (confluence < 4)] = "MILD_BULL"
    regime[confluence == 1] = "NEUTRAL"
    regime[confluence == 0] = "BEAR"
    was_bear = (confluence.rolling(30).min() == 0)
    regime[(was_bear) & (confluence >= 2) & (confluence < 4)] = "ACCUMULATION"
    return regime


# ---------------------------------------------------------------------------
# Strategy 1: Volatility Breakout
# ---------------------------------------------------------------------------

def strategy_vol_breakout(df, confluence, regime):
    """
    BBW squeeze detection → breakout trades.
    Long if confluence >= 3, Short if confluence <= 1.
    """
    close = df["close"]
    bw = bbw(close, 20, 2.0)
    at = atr(df, 14)

    # Squeeze: BBW < 20th percentile of last 100 periods
    squeeze_threshold = bw.rolling(100).quantile(0.20)
    in_squeeze = bw < squeeze_threshold

    # Release: was in squeeze yesterday, BBW crosses above today
    squeeze_yesterday = in_squeeze.shift(1).fillna(False)
    release = squeeze_yesterday & (~in_squeeze)

    # Shift confluence/regime for no lookahead
    conf_shifted = confluence.shift(1).fillna(0)

    daily_pnl = pd.Series(0.0, index=df.index)
    position = 0.0
    entry_price = 0.0
    stop_price = 0.0
    n = len(df)

    for i in range(1, n):
        price = close.iloc[i]
        prev_price = close.iloc[i-1]

        # P&L from open position
        if position != 0 and prev_price > 0:
            ret = (price - prev_price) / prev_price
            daily_pnl.iloc[i] = ret * position

        # Check stop
        if position > 0 and price <= stop_price:
            daily_pnl.iloc[i] -= abs(position) * TX_COST
            position = 0.0
            continue
        elif position < 0 and price >= stop_price:
            daily_pnl.iloc[i] -= abs(position) * TX_COST
            position = 0.0
            continue

        # Hold max 10 days then exit
        # (simplified: we use ATR trailing stop instead)

        # New entry on release
        if release.iloc[i] and position == 0:
            at_val = at.iloc[i-1] if i > 0 and not np.isnan(at.iloc[i-1]) else price * 0.02
            c = conf_shifted.iloc[i]
            if c >= 3:
                position = 1.0
                entry_price = price
                stop_price = price - 1.5 * at_val
                daily_pnl.iloc[i] -= TX_COST
            elif c <= 1:
                position = -0.5
                entry_price = price
                stop_price = price + 1.5 * at_val
                daily_pnl.iloc[i] -= 0.5 * TX_COST

    return daily_pnl


# ---------------------------------------------------------------------------
# Strategy 2: Mean Reversion
# ---------------------------------------------------------------------------

def strategy_mean_reversion(df, confluence, regime):
    """
    Active ONLY in NEUTRAL and ACCUMULATION regimes.
    RSI extremes + BB bounces. 3% max loss per trade.
    """
    close = df["close"]
    r = rsi(close, 14)
    lo, mid, hi = bb(close, 20, 2.0)

    # Shift indicators
    r_s = r.shift(1).fillna(50)
    lo_s = lo.shift(1).fillna(close)
    hi_s = hi.shift(1).fillna(close)
    regime_s = regime.shift(1).fillna("NEUTRAL")

    active = regime_s.isin(["NEUTRAL", "ACCUMULATION"])

    daily_pnl = pd.Series(0.0, index=df.index)
    position = 0.0
    entry_price = 0.0
    n = len(df)

    for i in range(1, n):
        price = close.iloc[i]
        prev_price = close.iloc[i-1]

        # P&L
        if position != 0 and prev_price > 0:
            ret = (price - prev_price) / prev_price
            daily_pnl.iloc[i] = ret * position

        # 3% stop loss
        if position != 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * np.sign(position)
            if pnl_pct <= -0.03:
                daily_pnl.iloc[i] -= abs(position) * TX_COST
                position = 0.0
                entry_price = 0.0
                continue

        if not active.iloc[i]:
            if position != 0:
                daily_pnl.iloc[i] -= abs(position) * TX_COST
                position = 0.0
                entry_price = 0.0
            continue

        # Exit on mean reversion target
        if position > 0 and r_s.iloc[i] > 55:
            daily_pnl.iloc[i] -= abs(position) * TX_COST
            position = 0.0
            entry_price = 0.0
            continue
        elif position < 0 and r_s.iloc[i] < 45:
            daily_pnl.iloc[i] -= abs(position) * TX_COST
            position = 0.0
            entry_price = 0.0
            continue

        # Entry
        if position == 0:
            # RSI oversold or BB lower touch → long
            if r_s.iloc[i] < 25 or prev_price <= lo_s.iloc[i]:
                position = 0.5
                entry_price = price
                daily_pnl.iloc[i] -= 0.5 * TX_COST
            # RSI overbought or BB upper touch → short
            elif r_s.iloc[i] > 75 or prev_price >= hi_s.iloc[i]:
                position = -0.5
                entry_price = price
                daily_pnl.iloc[i] -= 0.5 * TX_COST

    return daily_pnl


# ---------------------------------------------------------------------------
# Strategy 3: Enhanced Short Side
# ---------------------------------------------------------------------------

def strategy_enhanced_short(df, confluence, regime, macro_data, funding_daily=None):
    """
    Loosened short: 2-of-3 conditions instead of ALL.
    + Death cross standalone short trigger.
    """
    close = df["close"]
    idx = df.index
    r = rsi(close, 14)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    sma100 = sma(close, 100)

    # M2 decelerating
    m2_dec = pd.Series(False, index=idx)
    if macro_data is not None and "m2" in macro_data.columns:
        m2 = macro_data["m2"].reindex(idx, method="ffill").ffill()
        m2_yoy = m2.pct_change(365).fillna(0)
        m2_ma = m2_yoy.rolling(180).mean()
        m2_dec = (m2_yoy < m2_ma).shift(1).fillna(False)

    # Funding z-score
    fz = pd.Series(0.0, index=idx)
    if funding_daily is not None:
        fd = funding_daily.reindex(idx, method="ffill").fillna(0)
        f_mean = fd.rolling(30).mean()
        f_std = fd.rolling(30).std().replace(0, np.nan)
        fz = ((fd - f_mean) / f_std).fillna(0).shift(1).fillna(0)

    # Shift
    r_s = r.shift(1).fillna(50)
    close_s = close.shift(1).fillna(close)
    sma100_s = sma100.shift(1).fillna(close)
    sma50_s = sma50.shift(1).fillna(close)
    sma200_s = sma200.shift(1).fillna(close)
    conf_s = confluence.shift(1).fillna(2)

    # Conditions
    cond_a = m2_dec
    cond_b = close_s < sma100_s
    cond_c = (r_s > 60) | (fz > 1.0)

    # 2-of-3
    score = cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int)
    short_signal = score >= 2

    # Death cross as standalone
    death_cross = (sma50_s < sma200_s) & (sma50.shift(2) >= sma200.shift(2))
    death_cross_short = death_cross & (conf_s <= 0)

    daily_pnl = pd.Series(0.0, index=df.index)
    position = 0.0
    entry_price = 0.0
    hold_days = 0
    n = len(df)

    for i in range(1, n):
        price = close.iloc[i]
        prev_price = close.iloc[i-1]

        if position != 0 and prev_price > 0:
            ret = (price - prev_price) / prev_price
            daily_pnl.iloc[i] = ret * position
            hold_days += 1

        # Exit conditions
        if position < 0:
            # Cover if RSI < 30 or price > SMA or held 20+ days
            if r_s.iloc[i] < 30 or close_s.iloc[i] > sma100_s.iloc[i] or hold_days > 20:
                daily_pnl.iloc[i] -= abs(position) * TX_COST
                position = 0.0
                entry_price = 0.0
                hold_days = 0
                continue
            # 8% stop
            if entry_price > 0:
                loss = (price - entry_price) / entry_price
                if loss >= 0.08:
                    daily_pnl.iloc[i] -= abs(position) * TX_COST
                    position = 0.0
                    entry_price = 0.0
                    hold_days = 0
                    continue

        # Entry
        if position == 0:
            if short_signal.iloc[i]:
                position = -0.5
                entry_price = price
                hold_days = 0
                daily_pnl.iloc[i] -= 0.5 * TX_COST
            elif death_cross_short.iloc[i]:
                position = -0.3
                entry_price = price
                hold_days = 0
                daily_pnl.iloc[i] -= 0.3 * TX_COST

    return daily_pnl


# ---------------------------------------------------------------------------
# Strategy 4: Funding Rate Carry
# ---------------------------------------------------------------------------

def strategy_funding_carry(close_series, funding_daily, symbol="BTC"):
    """
    Harvest funding: short when funding very positive, long when very negative.
    Position size 0.2x, tight stops.
    """
    idx = close_series.index
    daily_pnl = pd.Series(0.0, index=idx)

    if funding_daily is None:
        # Simulate avg 0.01%/8hr = 0.03%/day
        funding_daily = pd.Series(0.0003, index=idx)

    fd = funding_daily.reindex(idx, method="ffill").fillna(0)
    f_mean = fd.rolling(30).mean()
    f_std = fd.rolling(30).std().replace(0, np.nan)
    fz = ((fd - f_mean) / f_std).fillna(0)

    # Shift for no lookahead
    fz_s = fz.shift(1).fillna(0)
    fd_s = fd.shift(1).fillna(0)

    position = 0.0  # +0.2 or -0.2
    entry_price = 0.0
    n = len(close_series)
    pos_size = 0.2

    for i in range(1, n):
        price = close_series.iloc[i]
        prev_price = close_series.iloc[i-1]

        if position != 0 and prev_price > 0:
            # Price P&L
            ret = (price - prev_price) / prev_price
            price_pnl = ret * position

            # Funding P&L: if short, earn positive funding; if long, earn negative funding
            if position < 0:
                funding_pnl = abs(position) * max(fd_s.iloc[i], 0)
            else:
                funding_pnl = abs(position) * max(-fd_s.iloc[i], 0)

            daily_pnl.iloc[i] = price_pnl + funding_pnl

        # Stop: 5% adverse price move
        if position != 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * np.sign(position)
            if pnl_pct <= -0.05:
                daily_pnl.iloc[i] -= pos_size * TX_COST
                position = 0.0
                entry_price = 0.0
                continue

        # Exit when funding normalizes
        if position < 0 and fz_s.iloc[i] < 0.5:
            daily_pnl.iloc[i] -= pos_size * TX_COST
            position = 0.0
            entry_price = 0.0
            continue
        elif position > 0 and fz_s.iloc[i] > -0.5:
            daily_pnl.iloc[i] -= pos_size * TX_COST
            position = 0.0
            entry_price = 0.0
            continue

        # Entry
        if position == 0:
            if fz_s.iloc[i] > 1.5:
                # Very positive funding → short to earn
                position = -pos_size
                entry_price = price
                daily_pnl.iloc[i] -= pos_size * TX_COST
            elif fz_s.iloc[i] < -1.5:
                # Very negative funding → long to earn
                position = pos_size
                entry_price = price
                daily_pnl.iloc[i] -= pos_size * TX_COST

    return daily_pnl


# ---------------------------------------------------------------------------
# Strategy 5: Sector Rotation
# ---------------------------------------------------------------------------

def strategy_sector_rotation(regime_series):
    """
    Monthly sector rotation across crypto sectors.
    L1 (BTC, ETH, SOL, AVAX), DeFi (LINK, UNI, AAVE), Meme (DOGE)
    """
    tickers = {
        "L1": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"],
        "DeFi": ["LINK-USD", "UNI-USD", "AAVE-USD"],
        "Meme": ["DOGE-USD"],
    }

    # Load all
    all_close = {}
    for sector, tkrs in tickers.items():
        for t in tkrs:
            try:
                df = load_crypto_ohlcv(t, start="2020-01-01", end=END)
                if len(df) > 50:
                    all_close[t] = df["close"]
            except Exception:
                pass

    if len(all_close) < 3:
        print("  [SectorRotation] Not enough data")
        return pd.Series(dtype=float), {}

    prices = pd.DataFrame(all_close).ffill().dropna(how="all")

    # Compute sector-level daily returns
    sector_returns = {}
    for sector, tkrs in tickers.items():
        available = [t for t in tkrs if t in prices.columns]
        if available:
            sector_ret = prices[available].pct_change().mean(axis=1)
            sector_returns[sector] = sector_ret

    sector_ret_df = pd.DataFrame(sector_returns).fillna(0)

    # Monthly momentum ranking
    mom_30d = sector_ret_df.rolling(30).sum()

    # Regime alignment
    regime_aligned = regime_series.reindex(sector_ret_df.index, method="ffill").fillna("NEUTRAL")
    bullish = regime_aligned.isin(["BULL", "MILD_BULL"])

    # Allocate: equal weight baseline, tilt top sector
    daily_pnl = pd.Series(0.0, index=sector_ret_df.index)
    sectors = list(sector_returns.keys())

    # Rebalance monthly
    last_rebal = None
    weights = {s: 1.0/len(sectors) for s in sectors}

    for i in range(31, len(sector_ret_df)):
        date = sector_ret_df.index[i]
        month = date.month if hasattr(date, 'month') else pd.Timestamp(date).month

        # Rebalance on month change
        if last_rebal is None or (hasattr(date, 'month') and (pd.Timestamp(date).month != pd.Timestamp(last_rebal).month)):
            last_rebal = date
            # Rank by 30d momentum
            mom = mom_30d.iloc[i-1]
            ranked = mom.dropna().sort_values(ascending=False)
            if len(ranked) >= 2:
                top = ranked.index[0]
                bot = ranked.index[-1]
                base_w = 1.0 / len(sectors)
                weights = {s: base_w for s in sectors}
                tilt = 0.2 if bullish.iloc[i] else 0.1
                weights[top] = base_w + tilt
                weights[bot] = max(base_w - tilt, 0.0)
                # Normalize
                total = sum(weights.values())
                weights = {s: w/total for s, w in weights.items()}

            # TX cost for rebalancing
            daily_pnl.iloc[i] -= TX_COST * 0.5

        # Daily return
        for s in sectors:
            if s in sector_ret_df.columns:
                daily_pnl.iloc[i] += sector_ret_df[s].iloc[i] * weights.get(s, 0)

    return daily_pnl, sector_returns


# ---------------------------------------------------------------------------
# Buy & Hold baseline
# ---------------------------------------------------------------------------

def buy_and_hold(close):
    return close.pct_change().fillna(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GAP STRATEGIES BACKTEST")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    assets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    ohlcv = {}
    for name, ticker in assets.items():
        df = load_crypto_ohlcv(ticker)
        if len(df) > 0:
            ohlcv[name] = df
            print(f"  {name}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    # Funding data
    funding = {}
    for name in assets:
        fd = load_funding(name)
        if fd is not None:
            funding[name] = fd
            print(f"  {name} funding: {len(fd)} rows")
        else:
            print(f"  {name} funding: not found, will simulate")

    # Macro data
    print("  Loading macro/cross-asset...")
    macro_data, cross_asset = load_macro_simple()
    print(f"  Macro cols: {list(macro_data.columns) if len(macro_data) > 0 else 'none'}")
    print(f"  Cross-asset cols: {list(cross_asset.columns) if len(cross_asset) > 0 else 'none'}")

    # Compute confluence & regime for BTC (reference)
    print("\n[2/6] Computing confluence & regime...")
    btc_close = ohlcv["BTC"]["close"]
    confluence, breakdown = compute_confluence_simple(btc_close, macro_data, cross_asset)
    regime = detect_regime(confluence)
    print(f"  Regime distribution:")
    for r, cnt in regime.value_counts().items():
        print(f"    {r}: {cnt} days ({cnt/len(regime)*100:.1f}%)")

    results = {}

    # Strategy 1: Vol Breakout
    print("\n[3/6] Strategy 1: Volatility Breakout...")
    for name in ohlcv:
        df = ohlcv[name]
        conf_a, _ = compute_confluence_simple(df["close"], macro_data, cross_asset)
        reg_a = detect_regime(conf_a)
        pnl = strategy_vol_breakout(df, conf_a, reg_a)
        m = calc_metrics(pnl, f"VolBreakout_{name}")
        rm = regime_metrics(pnl, reg_a, name)
        m["regime_sharpe"] = rm
        results[f"vol_breakout_{name}"] = m
        print(f"  {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")

    # Strategy 2: Mean Reversion
    print("\n[4/6] Strategy 2: Mean Reversion...")
    for name in ohlcv:
        df = ohlcv[name]
        conf_a, _ = compute_confluence_simple(df["close"], macro_data, cross_asset)
        reg_a = detect_regime(conf_a)
        pnl = strategy_mean_reversion(df, conf_a, reg_a)
        m = calc_metrics(pnl, f"MeanRev_{name}")
        rm = regime_metrics(pnl, reg_a, name)
        m["regime_sharpe"] = rm
        results[f"mean_reversion_{name}"] = m
        # Specific ACCUMULATION performance
        acc_mask = reg_a == "ACCUMULATION"
        if acc_mask.sum() > 10:
            acc_sharpe = sharpe(pnl[acc_mask])
            m["accumulation_sharpe"] = round(float(acc_sharpe), 3)
            print(f"  {name}: Sharpe={m['sharpe']}, ACCUM Sharpe={acc_sharpe:.3f}, Return={m['total_return']:.2%}")
        else:
            print(f"  {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}")

    # Strategy 3: Enhanced Short
    print("\n[5/6] Strategy 3: Enhanced Short...")
    for name in ohlcv:
        df = ohlcv[name]
        conf_a, _ = compute_confluence_simple(df["close"], macro_data, cross_asset)
        reg_a = detect_regime(conf_a)
        fd = funding.get(name)
        pnl = strategy_enhanced_short(df, conf_a, reg_a, macro_data, fd)
        m = calc_metrics(pnl, f"EnhShort_{name}")
        rm = regime_metrics(pnl, reg_a, name)
        m["regime_sharpe"] = rm
        results[f"enhanced_short_{name}"] = m
        short_days = (pnl != 0).sum()
        print(f"  {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, Short days={short_days}")

    # Strategy 4: Funding Carry
    print("\n[6/6] Strategy 4: Funding Rate Carry...")
    for name in ohlcv:
        fd = funding.get(name)
        close_s = ohlcv[name]["close"]
        pnl = strategy_funding_carry(close_s, fd, name)
        conf_a, _ = compute_confluence_simple(close_s, macro_data, cross_asset)
        reg_a = detect_regime(conf_a)
        m = calc_metrics(pnl, f"FundingCarry_{name}")
        rm = regime_metrics(pnl, reg_a, name)
        m["regime_sharpe"] = rm
        results[f"funding_carry_{name}"] = m
        print(f"  {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}")

    # Strategy 5: Sector Rotation
    print("\n  Strategy 5: Sector Rotation...")
    sr_pnl, sr_rets = strategy_sector_rotation(regime)
    if len(sr_pnl) > 0:
        m = calc_metrics(sr_pnl, "SectorRotation")
        rm = regime_metrics(sr_pnl, regime.reindex(sr_pnl.index, method="ffill").fillna("NEUTRAL"))
        m["regime_sharpe"] = rm
        results["sector_rotation"] = m
        print(f"  SectorRot: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}")

    # Baselines
    print("\n  Baselines:")
    for name in ohlcv:
        bh = buy_and_hold(ohlcv[name]["close"])
        m = calc_metrics(bh, f"BuyHold_{name}")
        results[f"buy_hold_{name}"] = m
        print(f"  {name} B&H: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")

    # Save results
    out_path = RESULTS_DIR / "gap_strategies_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
