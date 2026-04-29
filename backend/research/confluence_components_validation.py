"""
Systematic Validation of ALL Virtuoso Confluence Components
============================================================
Tests every indicator group's directional predictive power:

TOP-LEVEL (6 groups):
  1. Orderflow (0.30) — ALREADY PROVEN DEAD
  2. Orderbook (0.20) — depth, imbalance, OIR, liquidity, MPI, manipulation, absorption
  3. Volume (0.18) — ADL, CMF, OBV, relative volume, volume delta, VWAP
  4. Price Structure (0.15) — S/R, institutional zones, trend, volume profile, structure breaks, FVG
  5. Technical (0.10) — RSI, AO, Williams %R, ATR, CCI, ADX
  6. Sentiment (0.07) — funding rate, liquidations, L/S ratio, volatility

Since we don't have live orderbook/sentiment data historically, we test what we CAN:
- Technical indicators: fully testable from OHLCV
- Volume indicators: fully testable from OHLCV + orderflow data
- Price structure: partially testable (S/R, trend, volume profile)
- Sentiment: partially testable (funding from derivatives data)
- Orderbook: NOT testable (need L2 snapshots we don't have historically)

Methodology: Same as orderflow suite — 14-fold WF, non-overlapping, 10bps costs
Multi-TF (15m, 1h, 4h), Multi-horizon (5, 10, 30 bars)

Author: Maestro 🎼
Date: 2026-03-07
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")
import talib

from backend.config.data_paths import BARS_1M_V1

OF_DIR = BARS_1M_V1
OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
DERIV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/derivatives")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
COST_BPS = 10


def load_of_1h(asset):
    path = OF_DIR / f"{asset}_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trade_count": "sum", "dollar_volume": "sum",
           "buy_vol": "sum", "sell_vol": "sum", "delta": "sum"}
    for c in ["buy_dollar", "sell_dollar", "delta_dollar"]:
        if c in df.columns:
            agg[c] = "sum"
    df = df.resample("1h").agg(agg).dropna(subset=["open"])
    df["buy_pct"] = df["buy_vol"] / df["volume"].clip(lower=1e-10)
    return df


def load_ohlcv(asset_file, tf="1h"):
    """Load OHLCV, resample if needed."""
    path = OHLCV_DIR / asset_file
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in
                     pd.read_csv(path, nrows=1).columns else [0])
    df.columns = df.columns.str.lower()
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    return df


def load_derivatives(token):
    """Load derivatives data (funding, OI, LSR) from Coinalyze."""
    files = {
        "funding": DERIV_DIR / f"{token}_funding_rate.csv",
        "oi": DERIV_DIR / f"{token}_open_interest.csv",
        "lsr": DERIV_DIR / f"{token}_long_short_ratio.csv",
        "liquidations": DERIV_DIR / f"{token}_liquidations.csv",
    }
    data = {}
    for key, path in files.items():
        if path.exists():
            try:
                d = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in
                               pd.read_csv(path, nrows=1).columns else [0])
                d.columns = d.columns.str.lower()
                if "timestamp" in d.columns:
                    d = d.set_index("timestamp")
                elif "date" in d.columns:
                    d = d.set_index("date")
                data[key] = d
            except:
                pass
    return data


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════

def walk_forward(df, signal_col, hold_bars=30, n_folds=14):
    total = len(df)
    if total < 500:
        return {"error": "Too few bars"}
    
    date_range = df.index[-1] - df.index[0]
    months = max(date_range.days / 30.44, 1)
    bpm = int(total / months)
    min_train = bpm * 3
    test_size = bpm
    if test_size < 10:
        return {"error": "Test size too small"}
    
    available = total - min_train
    n_folds = min(n_folds, max(available // test_size, 0))
    if n_folds < 3:
        return {"error": "Insufficient data"}
    
    signals = df[signal_col].values
    closes = df["close"].values
    all_trades = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = min(test_start + test_size, total)
        if test_end > total:
            break
        
        i = test_start
        while i < test_end - hold_bars:
            sig = signals[i]
            if np.isnan(sig) or sig == 0:
                i += 1
                continue
            pos = np.sign(sig)
            raw = (closes[min(i + hold_bars, total - 1)] / closes[i] - 1) * pos
            net = raw - (COST_BPS / 10000)
            all_trades.append(net)
            i += hold_bars
    
    if len(all_trades) < 15:
        return {"error": "Too few trades", "n": len(all_trades)}
    
    net = np.array(all_trades)
    t_stat, p_val = sp_stats.ttest_1samp(net, 0)
    tpy = 8760 / hold_bars
    sharpe = (np.mean(net) / np.std(net)) * np.sqrt(tpy) if np.std(net) > 0 else 0
    
    bs = []
    for _ in range(500):
        s = np.random.choice(net, size=len(net), replace=True)
        bs.append((np.mean(s) / np.std(s)) * np.sqrt(tpy) if np.std(s) > 0 else 0)
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    
    pf = abs(np.sum(net[net > 0]) / np.sum(net[net <= 0])) if np.sum(net[net <= 0]) != 0 else float("inf")
    
    return {
        "n_trades": len(all_trades), "mean_pct": float(np.mean(net) * 100),
        "t_stat": float(t_stat), "p_value": float(p_val),
        "sharpe": float(sharpe), "sharpe_ci": [float(ci_lo), float(ci_hi)],
        "hit_rate": float(np.mean(net > 0)), "pf": float(pf),
        "total_ret_pct": float(np.sum(net) * 100),
    }


# ═══════════════════════════════════════════════════════════
# TECHNICAL INDICATOR SIGNALS
# ═══════════════════════════════════════════════════════════

def sig_rsi(df, period=14):
    """RSI mean-reversion: <30 = buy, >70 = sell."""
    rsi = talib.RSI(df["close"], timeperiod=period)
    signal = pd.Series(0.0, index=df.index)
    signal[rsi < 30] = 1.0
    signal[rsi > 70] = -1.0
    return signal.replace(0, np.nan).ffill().fillna(0)

def sig_rsi_momentum(df, period=14):
    """RSI momentum: >50 = buy, <50 = sell."""
    rsi = talib.RSI(df["close"], timeperiod=period)
    return ((rsi - 50) / 50).clip(-1, 1)

def sig_macd(df):
    """MACD crossover."""
    macd, signal, hist = talib.MACD(df["close"])
    return np.sign(hist).fillna(0)

def sig_macd_momentum(df):
    """MACD histogram magnitude as signal strength."""
    macd, signal, hist = talib.MACD(df["close"])
    mu = hist.rolling(50, min_periods=20).mean()
    sigma = hist.rolling(50, min_periods=20).std().clip(lower=1e-10)
    return ((hist - mu) / sigma).clip(-3, 3) / 3

def sig_ao(df):
    """Awesome Oscillator — 5-period vs 34-period midprice SMA."""
    mid = (df["high"] + df["low"]) / 2
    ao = mid.rolling(5).mean() - mid.rolling(34).mean()
    return np.sign(ao).fillna(0)

def sig_williams_r(df, period=14):
    """Williams %R: <-80 = buy, >-20 = sell (mean-reversion)."""
    wr = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=period)
    signal = pd.Series(0.0, index=df.index)
    signal[wr < -80] = 1.0
    signal[wr > -20] = -1.0
    return signal.replace(0, np.nan).ffill().fillna(0)

def sig_cci(df, period=20):
    """CCI: >100 = buy (momentum), <-100 = sell."""
    cci = talib.CCI(df["high"], df["low"], df["close"], timeperiod=period)
    return (cci / 200).clip(-1, 1)

def sig_adx_trend(df, period=14):
    """ADX trend strength × direction (using DI+ vs DI-)."""
    adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
    plus_di = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    minus_di = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    direction = np.sign(plus_di - minus_di)
    strength = (adx / 50).clip(0, 1)  # Normalize ADX to 0-1
    return (direction * strength).fillna(0)

def sig_bbands_reversion(df, period=20):
    """Bollinger Bands mean-reversion."""
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=period)
    pos_in_band = (df["close"] - lower) / (upper - lower).clip(lower=1e-10)
    # Below lower = buy, above upper = sell
    signal = -(pos_in_band - 0.5) * 2
    return signal.clip(-1, 1).fillna(0)

def sig_sma_cross(df, fast=20, slow=50):
    """SMA crossover — classic trend following."""
    sma_fast = df["close"].rolling(fast, min_periods=fast//2).mean()
    sma_slow = df["close"].rolling(slow, min_periods=slow//2).mean()
    return (sma_fast > sma_slow).astype(float) * 2 - 1

def sig_ema_cross(df, fast=12, slow=26):
    """EMA crossover."""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    return (ema_fast > ema_slow).astype(float) * 2 - 1

def sig_donchian_breakout(df, period=20):
    """Donchian channel breakout — price above highest high = buy."""
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    mid = (high_n + low_n) / 2
    return ((df["close"] > high_n.shift(1)).astype(float) - 
            (df["close"] < low_n.shift(1)).astype(float))

def sig_keltner_breakout(df, period=20, mult=2.0):
    """Keltner Channel breakout."""
    ema = df["close"].ewm(span=period).mean()
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return ((df["close"] > upper).astype(float) - (df["close"] < lower).astype(float))

def sig_stoch_rsi(df, period=14):
    """Stochastic RSI."""
    rsi = talib.RSI(df["close"], timeperiod=period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).clip(lower=1e-10)
    signal = pd.Series(0.0, index=df.index)
    signal[stoch_rsi < 0.2] = 1.0
    signal[stoch_rsi > 0.8] = -1.0
    return signal.replace(0, np.nan).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════
# VOLUME INDICATOR SIGNALS
# ═══════════════════════════════════════════════════════════

def sig_obv(df):
    """OBV trend — SMA20 > SMA50 on OBV."""
    obv = talib.OBV(df["close"], df["volume"])
    sma20 = obv.rolling(20).mean()
    sma50 = obv.rolling(50).mean()
    return (sma20 > sma50).astype(float) * 2 - 1

def sig_cmf(df, period=20):
    """Chaikin Money Flow."""
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).clip(lower=1e-10)
    mfv = mfm * df["volume"]
    cmf = mfv.rolling(period).sum() / df["volume"].rolling(period).sum().clip(lower=1e-10)
    return cmf.clip(-1, 1)

def sig_adl(df):
    """Accumulation/Distribution Line trend."""
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).clip(lower=1e-10)
    adl = (mfm * df["volume"]).cumsum()
    sma20 = adl.rolling(20).mean()
    sma50 = adl.rolling(50).mean()
    return (sma20 > sma50).astype(float) * 2 - 1

def sig_relative_volume(df, period=50):
    """Relative volume × price direction."""
    rvol = df["volume"] / df["volume"].rolling(period).mean().clip(lower=1e-10)
    direction = np.sign(df["close"].diff())
    # Only signal on high volume
    high_vol = (rvol > 1.5).astype(float)
    return (high_vol * direction).replace(0, np.nan).ffill().fillna(0)

def sig_vwap_position(df):
    """Price position relative to rolling VWAP."""
    cum_vol = df["volume"].cumsum()
    cum_vp = (df["close"] * df["volume"]).cumsum()
    vwap = cum_vp / cum_vol.clip(lower=1e-10)
    # Normalize distance from VWAP
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    dist = (df["close"] - vwap) / atr.clip(lower=1e-10)
    return dist.clip(-3, 3) / 3

def sig_volume_delta_trend(df):
    """Volume delta (buy-sell) trend — needs orderflow data."""
    if "delta" not in df.columns:
        return None
    cvd = df["delta"].cumsum()
    sma20 = cvd.rolling(20).mean()
    sma50 = cvd.rolling(50).mean()
    return (sma20 > sma50).astype(float) * 2 - 1


# ═══════════════════════════════════════════════════════════
# PRICE STRUCTURE SIGNALS
# ═══════════════════════════════════════════════════════════

def sig_trend_position(df):
    """Multi-MA trend position (like Virtuoso's trend_position)."""
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200, min_periods=100).mean()
    
    # Score: how many MAs is price above?
    above_20 = (df["close"] > sma20).astype(float)
    above_50 = (df["close"] > sma50).astype(float)
    above_200 = (df["close"] > sma200).astype(float)
    # Also MA alignment
    aligned_up = (sma20 > sma50).astype(float) * (sma50 > sma200).astype(float)
    aligned_down = (sma20 < sma50).astype(float) * (sma50 < sma200).astype(float)
    
    score = (above_20 + above_50 + above_200 + aligned_up - aligned_down) / 4
    return (score - 0.5) * 2  # Center at 0

def sig_structure_break(df, lookback=20):
    """Structure break detection — price breaking above/below recent range."""
    high_n = df["high"].rolling(lookback).max().shift(1)
    low_n = df["low"].rolling(lookback).min().shift(1)
    
    break_up = (df["close"] > high_n).astype(float)
    break_down = (df["close"] < low_n).astype(float)
    
    return (break_up - break_down).replace(0, np.nan).ffill().fillna(0)

def sig_atr_expansion(df, period=14):
    """ATR expansion = trending → follow direction."""
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    atr_sma = atr.rolling(50).mean()
    expanding = (atr > atr_sma * 1.2).astype(float)
    direction = np.sign(df["close"].diff(5))
    return (expanding * direction).replace(0, np.nan).ffill().fillna(0)

def sig_mean_reversion(df, period=20):
    """Price distance from SMA as mean-reversion signal."""
    sma = df["close"].rolling(period).mean()
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    z = (df["close"] - sma) / atr.clip(lower=1e-10)
    # Fade extremes
    signal = pd.Series(0.0, index=df.index)
    signal[z > 2] = -1.0
    signal[z < -2] = 1.0
    return signal.replace(0, np.nan).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════
# SENTIMENT / DERIVATIVES SIGNALS
# ═══════════════════════════════════════════════════════════

def sig_funding_contrarian(df):
    """High funding = short, low funding = long (contrarian)."""
    if "funding_rate" not in df.columns:
        return None
    fr = df["funding_rate"]
    mu = fr.rolling(50, min_periods=20).mean()
    sigma = fr.rolling(50, min_periods=20).std().clip(lower=1e-10)
    z = (fr - mu) / sigma
    return -z.clip(-3, 3) / 3  # Contrarian

def sig_oi_momentum(df):
    """OI increasing = momentum confirmation."""
    if "open_interest" not in df.columns:
        return None
    oi = df["open_interest"]
    oi_change = oi.pct_change(5)
    price_dir = np.sign(df["close"].diff(5))
    # OI up + price up = bullish momentum, OI up + price down = bearish momentum
    return (np.sign(oi_change) * price_dir).fillna(0)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # Check for TA-Lib
    try:
        import talib
    except ImportError:
        print("ERROR: TA-Lib not installed. Run: pip install TA-Lib")
        return
    
    # All signals organized by category
    SIGNALS = {
        # Technical (Virtuoso weight: 0.10)
        "tech_rsi_reversion": sig_rsi,
        "tech_rsi_momentum": sig_rsi_momentum,
        "tech_macd_cross": sig_macd,
        "tech_macd_momentum": sig_macd_momentum,
        "tech_ao": sig_ao,
        "tech_williams_r": sig_williams_r,
        "tech_cci": sig_cci,
        "tech_adx_trend": sig_adx_trend,
        "tech_bbands_revert": sig_bbands_reversion,
        "tech_sma_cross_20_50": sig_sma_cross,
        "tech_ema_cross_12_26": sig_ema_cross,
        "tech_donchian_break": sig_donchian_breakout,
        "tech_keltner_break": sig_keltner_breakout,
        "tech_stoch_rsi": sig_stoch_rsi,
        
        # Volume (Virtuoso weight: 0.18)
        "vol_obv_trend": sig_obv,
        "vol_cmf": sig_cmf,
        "vol_adl_trend": sig_adl,
        "vol_relative": sig_relative_volume,
        "vol_vwap_position": sig_vwap_position,
        
        # Price Structure (Virtuoso weight: 0.15)
        "ps_trend_position": sig_trend_position,
        "ps_structure_break": sig_structure_break,
        "ps_atr_expansion": sig_atr_expansion,
        "ps_mean_reversion": sig_mean_reversion,
    }
    
    # Find all assets with orderflow data (richest dataset)
    of_assets = sorted([f.stem.replace("_1m", "")
                        for f in OF_DIR.glob("*_1m.csv")
                        if "v2" not in f.stem])
    
    print("=" * 110)
    print("CONFLUENCE COMPONENT VALIDATION — WHAT PREDICTS PRICE DIRECTION?")
    print(f"Signals: {len(SIGNALS)} | Assets: {len(of_assets)}")
    print(f"Methodology: 14-fold WF, non-overlapping, {COST_BPS}bps, hold=5/10/30 bars @ 1h")
    print("=" * 110)
    
    all_rows = []
    
    for asset in of_assets:
        print(f"\n{'#' * 80}")
        print(f"# {asset.upper()}")
        print(f"{'#' * 80}")
        
        df = load_of_1h(asset)
        if df is None or len(df) < 500:
            print(f"  SKIP: insufficient data")
            continue
        
        print(f"  {len(df):,} bars | {df.index[0]} → {df.index[-1]}")
        
        for sig_name, sig_func in SIGNALS.items():
            try:
                result = sig_func(df)
                if result is None:
                    continue
                df[sig_name] = result
            except Exception as e:
                print(f"  {sig_name}: ERROR: {e}")
                continue
            
            for hold in [5, 10, 30]:
                wf = walk_forward(df, sig_name, hold_bars=hold)
                
                passes = not ("error" in wf) and wf.get("p_value", 1) < 0.05 and wf.get("sharpe", 0) > 0.5
                
                row = {"asset": asset, "signal": sig_name, "hold": hold,
                       "category": sig_name.split("_")[0]}
                if "error" not in wf:
                    row.update(wf)
                    row["verdict"] = "PASS" if passes else "FAIL"
                else:
                    row["verdict"] = "ERROR"
                
                all_rows.append(row)
                
                if passes:
                    print(f"  ✅ {sig_name} h={hold}: Sharpe={wf['sharpe']:+.2f} "
                          f"t={wf['t_stat']:.2f} p={wf['p_value']:.4f} n={wf['n_trades']}")
    
    # ═══════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════
    sdf = pd.DataFrame(all_rows)
    valid = sdf[sdf["verdict"].isin(["PASS", "FAIL"])]
    n_pass = (valid["verdict"] == "PASS").sum()
    n_total = len(valid)
    expected_fp = n_total * 0.05
    
    print(f"\n\n{'=' * 110}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 110}")
    print(f"Total tests: {n_total}")
    print(f"Passes: {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"Expected false positives at 5%: {expected_fp:.0f}")
    print(f"Pass rate vs random: {'ABOVE ⭐' if n_pass > expected_fp * 1.5 else 'AT/BELOW random'}")
    
    bonf = 0.05 / n_total
    bonf_pass = sum(1 for _, r in valid.iterrows() 
                    if r.get("p_value", 1) < bonf and r.get("sharpe", 0) > 0.5)
    print(f"Bonferroni corrected (p < {bonf:.6f}): {bonf_pass}")
    
    # By category
    print(f"\n--- BY CATEGORY ---")
    for cat in ["tech", "vol", "ps"]:
        cat_df = valid[valid["category"] == cat]
        cp = (cat_df["verdict"] == "PASS").sum()
        ct = len(cat_df)
        if ct > 0:
            cat_label = {"tech": "Technical (0.10)", "vol": "Volume (0.18)", "ps": "Price Structure (0.15)"}
            print(f"  {cat_label.get(cat, cat)}: {cp}/{ct} ({cp/ct*100:.1f}%)")
    
    # By signal (averaged across assets)
    print(f"\n--- SIGNAL RANKINGS (avg Sharpe across assets, h=30) ---")
    h30 = valid[valid["hold"] == 30]
    if len(h30) > 0:
        sig_avg = h30.groupby("signal").agg(
            avg_sharpe=("sharpe", "mean"),
            med_sharpe=("sharpe", "median"),
            pass_rate=("verdict", lambda x: (x == "PASS").mean()),
            n=("asset", "count"),
        ).sort_values("avg_sharpe", ascending=False)
        
        print(f"  {'Signal':<30s} {'Avg Sharpe':>10s} {'Med':>8s} {'Pass%':>7s}")
        print(f"  {'-'*60}")
        for sig, row in sig_avg.head(20).iterrows():
            marker = "⭐" if row["pass_rate"] > 0.15 else "🔸" if row["avg_sharpe"] > 0.3 else ""
            print(f"  {sig:<30s} {row['avg_sharpe']:>+10.2f} {row['med_sharpe']:>+8.2f} "
                  f"{row['pass_rate']:>6.0%} {marker}")
    
    # All passing tests
    if n_pass > 0:
        print(f"\n--- ALL PASSING TESTS ---")
        passes_df = valid[valid["verdict"] == "PASS"].sort_values("sharpe", ascending=False)
        print(f"  {'Asset':<12s} {'Signal':<30s} {'Hold':>5s} {'Sharpe':>8s} {'t':>6s} {'p':>8s} {'Hit%':>6s} {'PF':>6s}")
        for _, r in passes_df.iterrows():
            print(f"  {r['asset']:<12s} {r['signal']:<30s} {r['hold']:>5d} {r['sharpe']:>+8.2f} "
                  f"{r['t_stat']:>+6.2f} {r['p_value']:>8.4f} {r['hit_rate']:>5.0%} {r['pf']:>6.2f}")
    
    # Save
    out_json = RESULTS_DIR / "confluence_components_validation_20260307.json"
    sdf.to_json(out_json, orient="records", indent=2)
    
    out_csv = RESULTS_DIR / "confluence_components_summary_20260307.csv"
    sdf.to_csv(out_csv, index=False)
    
    out_md = RESULTS_DIR / "confluence_components_validation_20260307.md"
    with open(out_md, "w") as f:
        f.write("# Confluence Component Validation — What Predicts Direction?\n\n")
        f.write(f"**Date:** 2026-03-07\n")
        f.write(f"**Tests:** {n_total} | **Passes:** {n_pass} ({n_pass/n_total*100:.1f}%)\n")
        f.write(f"**Expected false positives:** {expected_fp:.0f} | **Bonferroni:** {bonf_pass}\n\n")
        
        if n_pass > 0:
            f.write("## Passing Tests\n\n")
            f.write("| Asset | Signal | Hold | Sharpe | t-stat | p-value | Hit% | PF |\n")
            f.write("|-------|--------|------|--------|--------|---------|------|----|\n")
            for _, r in passes_df.iterrows():
                f.write(f"| {r['asset']} | {r['signal']} | {r['hold']} | {r['sharpe']:+.2f} | "
                        f"{r['t_stat']:+.2f} | {r['p_value']:.4f} | {r['hit_rate']:.0%} | {r['pf']:.2f} |\n")
    
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
