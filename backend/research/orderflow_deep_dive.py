"""
Deep dive into orderflow — challenging every assumption.

Assumptions to challenge:
1. "Aggregate delta has no predictive power" — maybe it's non-linear (extreme tails only)
2. "Bar-level is the right granularity" — maybe microstructure within bars matters
3. "Signals are independent of context" — maybe orderflow only matters at key levels
4. "1-bar forward is the right horizon" — maybe orderflow predicts 5-30 bars ahead
5. "Simple thresholds work" — maybe we need z-scores or percentile ranks
6. "All times are equal" — maybe session/time-of-day matters
7. "Commission kills everything" — maybe low-frequency signals survive
8. "CVD direction = signal" — maybe CVD ACCELERATION is the real signal
9. "Buy/sell split is informative" — maybe TRADE SIZE distribution matters more
10. "Aggregated metrics work" — maybe the SEQUENCE of buys/sells within a bar matters

Tests:
A. Extreme delta quintile analysis (non-linear effects)
B. Multi-horizon forward returns (1, 5, 10, 30 bars)
C. Conditional: orderflow at HSAKA structure points only
D. Time-of-day effects on delta predictiveness
E. CVD acceleration (2nd derivative) vs direction
F. Trade count anomalies as signals
G. Delta persistence / mean-reversion at different horizons
H. Information coefficient (IC) analysis
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import argrelextrema
from pathlib import Path

from backend.config.data_paths import BARS_1M_V1

OF_DIR = BARS_1M_V1
OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")


def load_of(asset="btcusdt", tf="1min"):
    df = pd.read_csv(OF_DIR / f"{asset}_1m.csv",
                     parse_dates=["timestamp"], index_col="timestamp")
    if tf != "1min":
        df = df.resample(tf).agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "buy_vol": "sum", "sell_vol": "sum",
            "delta": "sum", "trade_count": "sum",
            "dollar_volume": "sum",
        }).dropna(subset=["open"])
    df["return"] = df["close"].pct_change()
    df["buy_pct"] = df["buy_vol"] / df["volume"].clip(lower=1e-10)
    return df


def ic_analysis(signal, fwd_returns):
    """Information coefficient = rank correlation of signal with forward returns."""
    valid = ~np.isnan(signal) & ~np.isnan(fwd_returns) & np.isfinite(signal) & np.isfinite(fwd_returns)
    if valid.sum() < 100:
        return np.nan, np.nan
    corr, p = sp_stats.spearmanr(signal[valid], fwd_returns[valid])
    return corr, p


def main():
    print("=" * 80)
    print("ORDERFLOW DEEP DIVE — CHALLENGING ASSUMPTIONS")
    print("=" * 80)
    
    for asset in ["btcusdt", "ethusdt"]:
        print(f"\n{'#'*80}")
        print(f"# {asset.upper()}")
        print(f"{'#'*80}")
        
        # Load at multiple TFs
        df1m = load_of(asset, "1min")
        df5m = load_of(asset, "5min")
        df15m = load_of(asset, "15min")
        df1h = load_of(asset, "1h")
        df4h = load_of(asset, "4h")
        
        for tf_name, df in [("5m", df5m), ("15m", df15m), ("1h", df1h), ("4h", df4h)]:
            print(f"\n{'='*70}")
            print(f"=== {asset.upper()} {tf_name} ({len(df):,} bars) ===")
            
            # ════════════════════════════════════════
            # TEST A: Non-linear effects (extreme delta quintiles)
            # ════════════════════════════════════════
            print(f"\n  A. EXTREME DELTA → FORWARD RETURNS (non-linear)")
            for horizon in [1, 5, 10, 30]:
                fwd = df["close"].pct_change(horizon).shift(-horizon).values
                try:
                    deciles = pd.qcut(df["delta"], 10, labels=False, duplicates="drop")
                except:
                    continue
                print(f"     Horizon={horizon} bars:")
                for d in [0, 1, 8, 9]:
                    mask = deciles == d
                    if mask.sum() < 20:
                        continue
                    mr = np.nanmean(fwd[mask]) * 100
                    se = np.nanstd(fwd[mask]) / np.sqrt(mask.sum()) * 100
                    t = mr / se if se > 0 else 0
                    sig = "**" if abs(t) > 2 else "*" if abs(t) > 1.65 else ""
                    print(f"       D{d} (n={mask.sum()}): {mr:+.4f}% ± {se:.4f}% (t={t:.2f}) {sig}")
            
            # ════════════════════════════════════════
            # TEST B: Information Coefficient at multiple horizons
            # ════════════════════════════════════════
            print(f"\n  B. INFORMATION COEFFICIENT (rank corr delta→fwd_ret)")
            delta_vals = df["delta"].values
            buy_pct_vals = df["buy_pct"].values
            trade_count_vals = df["trade_count"].values
            
            # Delta z-score
            delta_z = (delta_vals - pd.Series(delta_vals).rolling(50).mean().values) / \
                      pd.Series(delta_vals).rolling(50).std().values.clip(min=1e-10)
            
            # CVD acceleration
            cvd = np.cumsum(delta_vals)
            cvd_vel = np.diff(cvd, prepend=0)  # = delta
            cvd_acc = np.diff(cvd_vel, prepend=0)  # 2nd derivative
            
            # Trade intensity (count z-score)
            tc_z = (trade_count_vals - pd.Series(trade_count_vals).rolling(50).mean().values) / \
                   pd.Series(trade_count_vals).rolling(50).std().values.clip(min=1e-10)
            
            signals_to_test = {
                "delta": delta_vals,
                "delta_z": delta_z,
                "buy_pct": buy_pct_vals,
                "cvd_accel": cvd_acc,
                "trade_count_z": tc_z,
                "delta*vol": delta_vals * df["volume"].values,
            }
            
            print(f"     {'Signal':<18s}", end="")
            for h in [1, 3, 5, 10, 30, 60]:
                print(f"  h={h:>3d}", end="")
            print()
            
            for name, sig_vals in signals_to_test.items():
                print(f"     {name:<18s}", end="")
                for h in [1, 3, 5, 10, 30, 60]:
                    fwd = df["close"].pct_change(h).shift(-h).values
                    ic, p = ic_analysis(sig_vals, fwd)
                    star = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"  {ic:+.4f}{star:<2s}", end="")
                print()
            
            # ════════════════════════════════════════
            # TEST C: Conditional — orderflow at structure points
            # ════════════════════════════════════════
            print(f"\n  C. ORDERFLOW AT STRUCTURE POINTS")
            high_vals = df["high"].values
            low_vals = df["low"].values
            close_vals = df["close"].values
            
            # Detect swing points
            order = max(3, min(14, len(df) // 500))
            sh = argrelextrema(high_vals, np.greater_equal, order=order)[0]
            sl = argrelextrema(low_vals, np.less_equal, order=order)[0]
            
            # At swing highs: does delta predict reversal?
            if len(sh) > 20:
                delta_at_highs = delta_vals[sh]
                fwd_at_highs = df["close"].pct_change(5).shift(-5).values[sh]
                valid = ~np.isnan(fwd_at_highs)
                if valid.sum() > 10:
                    ic, p = sp_stats.spearmanr(delta_at_highs[valid], fwd_at_highs[valid])
                    # Negative IC at highs = high delta (buying) → price drops = trapped buyers
                    print(f"     At swing highs (n={valid.sum()}): IC={ic:+.4f} p={p:.4f}")
                    # Split by delta sign
                    buy_dom = delta_at_highs > 0
                    sell_dom = delta_at_highs < 0
                    if buy_dom[valid].sum() > 5:
                        mr = np.nanmean(fwd_at_highs[valid & buy_dom]) * 100
                        print(f"       Buy-dominant at high → 5-bar fwd: {mr:+.4f}%")
                    if sell_dom[valid].sum() > 5:
                        mr = np.nanmean(fwd_at_highs[valid & sell_dom]) * 100
                        print(f"       Sell-dominant at high → 5-bar fwd: {mr:+.4f}%")
            
            if len(sl) > 20:
                delta_at_lows = delta_vals[sl]
                fwd_at_lows = df["close"].pct_change(5).shift(-5).values[sl]
                valid = ~np.isnan(fwd_at_lows)
                if valid.sum() > 10:
                    ic, p = sp_stats.spearmanr(delta_at_lows[valid], fwd_at_lows[valid])
                    print(f"     At swing lows (n={valid.sum()}): IC={ic:+.4f} p={p:.4f}")
                    buy_dom = delta_at_lows > 0
                    sell_dom = delta_at_lows < 0
                    if buy_dom[valid].sum() > 5:
                        mr = np.nanmean(fwd_at_lows[valid & buy_dom]) * 100
                        print(f"       Buy-dominant at low → 5-bar fwd: {mr:+.4f}%")
                    if sell_dom[valid].sum() > 5:
                        mr = np.nanmean(fwd_at_lows[valid & sell_dom]) * 100
                        print(f"       Sell-dominant at low → 5-bar fwd: {mr:+.4f}%")
            
            # ════════════════════════════════════════
            # TEST D: Time-of-day effects (UTC hours)
            # ════════════════════════════════════════
            if tf_name in ["5m", "15m", "1h"]:
                print(f"\n  D. TIME-OF-DAY DELTA PREDICTIVENESS")
                df["hour"] = df.index.hour
                fwd5 = df["close"].pct_change(5).shift(-5).values
                
                # Group hours into sessions
                sessions = {
                    "Asia (0-8 UTC)": range(0, 8),
                    "Europe (8-14 UTC)": range(8, 14),
                    "US (14-21 UTC)": range(14, 21),
                    "Off-hours (21-0 UTC)": range(21, 24),
                }
                for session, hours in sessions.items():
                    mask = df["hour"].isin(hours).values
                    if mask.sum() < 100:
                        continue
                    ic, p = ic_analysis(delta_vals[mask], fwd5[mask])
                    star = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"     {session:<25s}: IC={ic:+.4f} (p={p:.4f}) {star}")
            
            # ════════════════════════════════════════
            # TEST E: Delta persistence / mean-reversion
            # ════════════════════════════════════════
            print(f"\n  E. DELTA AUTOCORRELATION (persistence vs mean-reversion)")
            for lag in [1, 5, 10, 20, 50]:
                ac = pd.Series(delta_vals).autocorr(lag)
                print(f"     lag={lag:>3d}: autocorr={ac:+.4f}")
            
            # ════════════════════════════════════════
            # TEST F: Extreme events — large delta spikes
            # ════════════════════════════════════════
            print(f"\n  F. EXTREME DELTA SPIKES (|z| > 3) → forward returns")
            delta_z_clean = delta_z[~np.isnan(delta_z)]
            n_extreme = np.sum(np.abs(delta_z) > 3)
            print(f"     Extreme events: {n_extreme} ({n_extreme/len(delta_z)*100:.1f}%)")
            
            for horizon in [1, 5, 10, 30]:
                fwd = df["close"].pct_change(horizon).shift(-horizon).values
                
                # Extreme sell (z < -3) → fade = expect bounce?
                sell_spike = delta_z < -3
                if np.sum(sell_spike) > 10:
                    mr = np.nanmean(fwd[sell_spike]) * 100
                    se = np.nanstd(fwd[sell_spike]) / np.sqrt(np.sum(sell_spike)) * 100
                    t = mr / se if se > 0 else 0
                    sig = "**" if abs(t) > 2 else "*" if abs(t) > 1.65 else ""
                    print(f"     Sell spike (z<-3, n={np.sum(sell_spike)}) h={horizon}: {mr:+.4f}% t={t:.2f} {sig}")
                
                # Extreme buy (z > 3) → fade = expect pullback?
                buy_spike = delta_z > 3
                if np.sum(buy_spike) > 10:
                    mr = np.nanmean(fwd[buy_spike]) * 100
                    se = np.nanstd(fwd[buy_spike]) / np.sqrt(np.sum(buy_spike)) * 100
                    t = mr / se if se > 0 else 0
                    sig = "**" if abs(t) > 2 else "*" if abs(t) > 1.65 else ""
                    print(f"     Buy spike  (z>+3, n={np.sum(buy_spike)}) h={horizon}: {mr:+.4f}% t={t:.2f} {sig}")
            
            # ════════════════════════════════════════
            # TEST G: Trade count as information signal
            # ════════════════════════════════════════
            print(f"\n  G. TRADE COUNT EXTREMES → forward returns")
            for horizon in [1, 5, 10]:
                fwd = df["close"].pct_change(horizon).shift(-horizon).values
                try:
                    tc_deciles = pd.qcut(trade_count_vals, 10, labels=False, duplicates="drop")
                except:
                    continue
                # Extremely high trade count
                high_tc = tc_deciles >= 9
                low_tc = tc_deciles <= 0
                if np.sum(high_tc) > 10:
                    mr_h = np.nanmean(fwd[high_tc]) * 100
                    mr_l = np.nanmean(fwd[low_tc]) * 100
                    print(f"     h={horizon}: High TC → {mr_h:+.4f}%, Low TC → {mr_l:+.4f}%, spread={mr_h-mr_l:+.4f}%")
            
            # ════════════════════════════════════════
            # TEST H: Cumulative delta regime (trending CVD)
            # ════════════════════════════════════════
            print(f"\n  H. CVD REGIME → forward returns")
            cvd = np.cumsum(delta_vals)
            cvd_sma20 = pd.Series(cvd).rolling(20).mean().values
            cvd_sma50 = pd.Series(cvd).rolling(50).mean().values
            
            # CVD trending up (sma20 > sma50) vs down
            cvd_bull = cvd_sma20 > cvd_sma50
            cvd_bear = cvd_sma20 < cvd_sma50
            
            for horizon in [1, 5, 10, 30]:
                fwd = df["close"].pct_change(horizon).shift(-horizon).values
                if np.sum(cvd_bull) > 50 and np.sum(cvd_bear) > 50:
                    mr_bull = np.nanmean(fwd[cvd_bull]) * 100
                    mr_bear = np.nanmean(fwd[cvd_bear]) * 100
                    se_bull = np.nanstd(fwd[cvd_bull]) / np.sqrt(np.sum(cvd_bull)) * 100
                    se_bear = np.nanstd(fwd[cvd_bear]) / np.sqrt(np.sum(cvd_bear)) * 100
                    diff = mr_bull - mr_bear
                    # Welch t-test for difference
                    t_diff = diff / np.sqrt(se_bull**2 + se_bear**2) if (se_bull**2 + se_bear**2) > 0 else 0
                    sig = "**" if abs(t_diff) > 2 else "*" if abs(t_diff) > 1.65 else ""
                    print(f"     h={horizon:>3d}: CVD↑={mr_bull:+.4f}% CVD↓={mr_bear:+.4f}% diff={diff:+.4f}% (t={t_diff:.2f}) {sig}")
    
    print(f"\n{'='*80}")
    print("DEEP DIVE COMPLETE")
    print("If ANY test shows t>2 consistently across assets, we have a lead.")


if __name__ == "__main__":
    main()
