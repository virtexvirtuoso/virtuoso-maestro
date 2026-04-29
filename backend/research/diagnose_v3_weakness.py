"""
Diagnose V3 weaknesses — understand why performance decayed 2023-2026
and what drives the low capital efficiency.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    compute_confluence, ASSET_CONFIGS, detect_regime
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
            if len(df) > 100:
                crypto_data[name] = df
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
    print("V3 WEAKNESS DIAGNOSIS")
    print("=" * 80)

    crypto_data, cross_asset_data, macro_data = load_data()

    # Compute confluence for BTC
    btc = crypto_data["BTC"]
    confluence, breakdown = compute_confluence(
        btc["close"], macro_data, cross_asset_data,
        sma_slow=100, momentum_period=35
    )
    regime = detect_regime(confluence)

    # 1. Confluence score distribution by year
    print("\n── CONFLUENCE SCORE DISTRIBUTION BY YEAR ──")
    df = pd.DataFrame({
        "confluence": confluence,
        "m2": breakdown["m2_accel"],
        "proxy": breakdown["liquidity_proxy"],
        "yc": breakdown["yield_curve"],
        "xam": breakdown["cross_asset_mom"],
        "crypto": breakdown["crypto_momentum"],
        "regime": regime,
    })
    df["year"] = df.index.year
    df["btc_ret"] = btc["close"].pct_change()

    print(f"\n{'Year':<6}", end="")
    for s in range(6):
        print(f"{'Score '+str(s):>9}", end="")
    print(f"{'Mean':>8} {'BTC%':>8}")
    print("-" * 70)

    for yr in sorted(df["year"].unique()):
        mask = df["year"] == yr
        yr_data = df.loc[mask, "confluence"]
        print(f"{yr:<6}", end="")
        for s in range(6):
            pct = (yr_data == s).mean() * 100
            print(f"{pct:>8.1f}%", end="")
        btc_yr = df.loc[mask, "btc_ret"].sum() * 100
        print(f"{yr_data.mean():>8.2f} {btc_yr:>8.1f}")

    # 2. Per-signal activation by year
    print("\n── PER-SIGNAL ACTIVATION RATE BY YEAR ──")
    signals = ["m2", "proxy", "yc", "xam", "crypto"]
    print(f"{'Year':<6}", end="")
    for s in signals:
        print(f"{s:>10}", end="")
    print()
    print("-" * 56)

    for yr in sorted(df["year"].unique()):
        mask = df["year"] == yr
        print(f"{yr:<6}", end="")
        for s in signals:
            pct = df.loc[mask, s].mean() * 100
            print(f"{pct:>9.1f}%", end="")
        print()

    # 3. BTC returns conditioned on confluence
    print("\n── BTC ANNUALIZED RETURN BY CONFLUENCE SCORE ──")
    for s in range(6):
        mask = df["confluence"] == s
        if mask.sum() < 10:
            continue
        ret = df.loc[mask, "btc_ret"]
        ann = ret.mean() * 252 * 100
        vol = ret.std() * np.sqrt(252) * 100
        sharpe = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        days = mask.sum()
        print(f"  Score {s}: {ann:>+8.1f}%/yr  Vol: {vol:>6.1f}%  Sharpe: {sharpe:>6.3f}  ({days} days, {days/len(df)*100:.1f}%)")

    # 4. What killed 2023-2024?
    print("\n── 2023-2024 DEEP DIVE ──")
    for yr in [2023, 2024, 2025]:
        mask = df["year"] == yr
        yr_df = df.loc[mask]
        print(f"\n  {yr}:")
        print(f"    Mean confluence: {yr_df['confluence'].mean():.2f}")
        print(f"    M2 active:      {yr_df['m2'].mean()*100:.1f}%")
        print(f"    Proxy active:   {yr_df['proxy'].mean()*100:.1f}%")
        print(f"    YC active:      {yr_df['yc'].mean()*100:.1f}%")
        print(f"    XAM active:     {yr_df['xam'].mean()*100:.1f}%")
        print(f"    Crypto active:  {yr_df['crypto'].mean()*100:.1f}%")
        print(f"    BTC return:     {yr_df['btc_ret'].sum()*100:>+.1f}%")
        print(f"    Days at score 0: {(yr_df['confluence']==0).sum()} ({(yr_df['confluence']==0).mean()*100:.1f}%)")
        print(f"    Days at score 3+: {(yr_df['confluence']>=3).sum()} ({(yr_df['confluence']>=3).mean()*100:.1f}%)")

    # 5. Missed moves analysis
    print("\n── MISSED MOVES (BTC up >5% in 20d while confluence < 2) ──")
    btc_20d = btc["close"].pct_change(20)
    missed = (btc_20d > 0.05) & (confluence < 2)
    missed_dates = btc["close"][missed].index
    for dt in missed_dates[-15:]:  # Last 15
        c = confluence.loc[dt]
        ret = btc_20d.loc[dt] * 100
        print(f"  {dt.date()}: BTC +{ret:.1f}% (20d), confluence={int(c)}")

    # 6. The M2 problem
    print("\n── M2 ACCELERATION STATUS ──")
    if "m2" in macro_data.columns:
        m2 = macro_data["m2"].reindex(btc.index, method="ffill").ffill()
        m2_yoy = m2.pct_change(365)
        m2_yoy_6m = m2_yoy.rolling(180).mean()
        m2_accel = m2_yoy > m2_yoy_6m

        print(f"  Current M2 YoY:     {m2_yoy.iloc[-1]*100:.2f}%")
        print(f"  Current 6mo avg:    {m2_yoy_6m.iloc[-1]*100:.2f}%")
        print(f"  M2 accelerating:    {'YES' if m2_accel.iloc[-1] else 'NO'}")
        print(f"  Last acceleration:  ", end="")
        # Find last True
        last_true = m2_accel[m2_accel].index[-1] if m2_accel.any() else "Never"
        print(f"{last_true}")
        print(f"  Days since accel:   {(btc.index[-1] - last_true).days if isinstance(last_true, pd.Timestamp) else 'N/A'}")

    # 7. What would help?
    print("\n── POTENTIAL IMPROVEMENTS ──")

    # Test: what if we use confluence >= 2 instead of leverage map?
    # i.e., go 1x long when score >= 2, flat otherwise
    simple_signal = (confluence >= 2).astype(int).shift(1).fillna(0)
    btc_ret = btc["close"].pct_change()
    simple_ret = simple_signal * btc_ret
    simple_eq = (1 + simple_ret.dropna()).cumprod()
    simple_sharpe = simple_ret.dropna().mean() / simple_ret.dropna().std() * np.sqrt(252) if simple_ret.dropna().std() > 0 else 0
    print(f"\n  Alt A: Simple threshold (confluence >= 2, 1x):")
    print(f"    Sharpe: {simple_sharpe:.3f}")
    print(f"    Total:  {(simple_eq.iloc[-1]-1)*100:.1f}%")
    print(f"    Exposure: {simple_signal.mean()*100:.1f}%")

    # Test: what if we loosen to >= 1?
    loose_signal = (confluence >= 1).astype(int).shift(1).fillna(0)
    loose_ret = loose_signal * btc_ret
    loose_eq = (1 + loose_ret.dropna()).cumprod()
    loose_sharpe = loose_ret.dropna().mean() / loose_ret.dropna().std() * np.sqrt(252) if loose_ret.dropna().std() > 0 else 0
    print(f"\n  Alt B: Looser threshold (confluence >= 1, 1x):")
    print(f"    Sharpe: {loose_sharpe:.3f}")
    print(f"    Total:  {(loose_eq.iloc[-1]-1)*100:.1f}%")
    print(f"    Exposure: {loose_signal.mean()*100:.1f}%")

    # Test: continuous sizing (score/5 as position size)
    cont_signal = (confluence / 5.0).shift(1).fillna(0)
    cont_ret = cont_signal * btc_ret
    cont_eq = (1 + cont_ret.dropna()).cumprod()
    cont_sharpe = cont_ret.dropna().mean() / cont_ret.dropna().std() * np.sqrt(252) if cont_ret.dropna().std() > 0 else 0
    print(f"\n  Alt C: Continuous sizing (score/5 = position):")
    print(f"    Sharpe: {cont_sharpe:.3f}")
    print(f"    Total:  {(cont_eq.iloc[-1]-1)*100:.1f}%")
    print(f"    Avg size: {cont_signal.mean()*100:.1f}%")

    # Test: what if we drop M2 requirement entirely?
    no_m2_conf = breakdown["liquidity_proxy"] + breakdown["yield_curve"] + breakdown["cross_asset_mom"] + breakdown["crypto_momentum"]
    no_m2_signal = (no_m2_conf >= 2).astype(int).shift(1).fillna(0)
    no_m2_ret = no_m2_signal * btc_ret
    no_m2_eq = (1 + no_m2_ret.dropna()).cumprod()
    no_m2_sharpe = no_m2_ret.dropna().mean() / no_m2_ret.dropna().std() * np.sqrt(252) if no_m2_ret.dropna().std() > 0 else 0
    print(f"\n  Alt D: Drop M2 signal (4-signal confluence >= 2):")
    print(f"    Sharpe: {no_m2_sharpe:.3f}")
    print(f"    Total:  {(no_m2_eq.iloc[-1]-1)*100:.1f}%")
    print(f"    Exposure: {no_m2_signal.mean()*100:.1f}%")

    # Test: what if M2 is a size BOOSTER not a gate?
    # Base: score of 4 non-M2 signals. M2 adds leverage.
    base_conf = breakdown["liquidity_proxy"] + breakdown["yield_curve"] + breakdown["cross_asset_mom"] + breakdown["crypto_momentum"]
    m2_boost = breakdown["m2_accel"]
    # Size: base_conf/4 * (1 + m2_boost * 0.5) → max 1.5x when M2 on
    boost_signal = (base_conf / 4.0 * (1 + m2_boost * 0.5)).shift(1).fillna(0)
    boost_signal = boost_signal.clip(0, 1.5)
    boost_ret = boost_signal * btc_ret
    boost_eq = (1 + boost_ret.dropna()).cumprod()
    boost_sharpe = boost_ret.dropna().mean() / boost_ret.dropna().std() * np.sqrt(252) if boost_ret.dropna().std() > 0 else 0
    print(f"\n  Alt E: M2 as booster not gate (4-signal base + M2 lever):")
    print(f"    Sharpe: {boost_sharpe:.3f}")
    print(f"    Total:  {(boost_eq.iloc[-1]-1)*100:.1f}%")
    print(f"    Avg size: {boost_signal.mean()*100:.1f}%")

    # Per-year comparison
    print(f"\n── PER-YEAR SHARPE COMPARISON ──")
    print(f"{'Year':<6} {'V3 Curr':>9} {'Alt A':>9} {'Alt B':>9} {'Alt C':>9} {'Alt D':>9} {'Alt E':>9} {'B&H':>9}")
    print("-" * 72)

    alts = {
        "V3 Curr": simple_signal * 0,  # placeholder
        "Alt A": simple_signal,
        "Alt B": loose_signal,
        "Alt C": cont_signal,
        "Alt D": no_m2_signal,
        "Alt E": boost_signal,
    }

    for yr in sorted(df["year"].unique()):
        mask = df["year"] == yr
        yr_btc = btc_ret[mask].dropna()
        if len(yr_btc) < 20:
            continue
        print(f"{yr:<6}", end="")
        for name, sig in alts.items():
            if name == "V3 Curr":
                # Use confluence-based adaptive sizing matching V3 logic
                lev_map = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
                v3_size = confluence.map(lev_map).shift(1).fillna(0)
                yr_ret = (v3_size * btc_ret)[mask].dropna()
            else:
                yr_ret = (sig * btc_ret)[mask].dropna()
            if len(yr_ret) < 20 or yr_ret.std() == 0:
                print(f"{'N/A':>9}", end="")
            else:
                s = yr_ret.mean() / yr_ret.std() * np.sqrt(252)
                print(f"{s:>9.3f}", end="")
        # B&H
        if len(yr_btc) > 20 and yr_btc.std() > 0:
            bh = yr_btc.mean() / yr_btc.std() * np.sqrt(252)
            print(f"{bh:>9.3f}")
        else:
            print(f"{'N/A':>9}")


if __name__ == "__main__":
    main()
