"""
Hybrid Strategies Backtest — 5 sector rotation & cross-asset strategies.
Tests BTC Dominance Rotation, TradFi Lead Signals, Fama-French Bridge,
Liquidation-Driven Entries, and Multi-Timeframe Confluence.
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
from datasource.factor_loader import FactorDataLoader

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DERIV_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/derivatives"))

TX_COST = 0.001  # 0.1% per trade

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
TRADFI_TICKERS = {"GLD": "GLD", "UUP": "UUP", "TLT": "TLT", "VIX": "^VIX"}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    stock = StockDataLoader()
    fred = MacroDataLoader()
    factor = FactorDataLoader()

    # Crypto OHLCV
    crypto = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2019-01-01")
            if len(df) > 100:
                crypto[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # TradFi
    tradfi = {}
    for name, ticker in TRADFI_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2019-01-01")
            tradfi[name] = df["close"]
            print(f"  {name}: {len(df)} days")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Macro (M2 for liquidation overlay)
    try:
        macro = fred.get_multiple({"m2": "M2SL"}, start_date="2015-01-01")
        macro = macro.ffill()
        print(f"  Macro M2: {len(macro)} days")
    except:
        macro = pd.DataFrame()

    # Fama-French
    try:
        ff5 = factor.get_ff5()
        mom = factor.get_momentum()
        print(f"  FF5: {len(ff5)} months, Momentum: {len(mom)} months")
    except Exception as e:
        ff5, mom = pd.DataFrame(), pd.DataFrame()
        print(f"  FF factors: FAILED - {e}")

    # Derivatives (taker data as liquidation proxy)
    deriv = {}
    for asset in ["btc", "eth", "sol"]:
        path = DERIV_DIR / f"{asset}_taker.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)
            # Resample to daily
            daily = df.resample("1D").agg({"buyVol": "sum", "sellVol": "sum", "buySellRatio": "mean"}).dropna()
            deriv[asset.upper()] = daily
            print(f"  Derivatives {asset.upper()}: {len(daily)} days")

    return crypto, tradfi, macro, ff5, mom, deriv


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(returns: pd.Series) -> dict:
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["total_return", "cagr", "sharpe", "sortino", "max_dd", "calmar", "win_rate", "n_trades"]}
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
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0
    # Count trades (signal changes)
    positions = (returns != 0).astype(int)
    n_trades = int(positions.diff().abs().sum() / 2)
    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": n_trades,
    }


def apply_signals_to_returns(signals: pd.Series, asset_returns: pd.Series, tx_cost=TX_COST) -> pd.Series:
    """Apply signal series to asset returns with transaction costs. Signal at t → return at t+1."""
    sig = signals.reindex(asset_returns.index).fillna(0)
    # Shift signal by 1 to avoid lookahead
    pos = sig.shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    ret = pos * asset_returns - trades * tx_cost
    return ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: BTC DOMINANCE ROTATION
# ═══════════════════════════════════════════════════════════════════════

def strategy_btc_dominance(crypto):
    print("\n" + "=" * 70)
    print("STRATEGY 1: BTC DOMINANCE ROTATION")
    print("=" * 70)

    btc_close = crypto["BTC"]["close"]
    # Build alt basket (equal weight ETH + SOL where available)
    alts = []
    for a in ["ETH", "SOL"]:
        if a in crypto:
            alts.append(crypto[a]["close"].rename(a))

    if not alts:
        print("  No alt data!")
        return pd.Series(dtype=float), {}

    alt_basket = pd.concat(alts, axis=1)
    # Normalize each to 1 at first common date
    common_start = alt_basket.dropna().index[0]
    alt_basket = alt_basket.loc[common_start:]
    btc_close = btc_close.loc[common_start:]

    alt_norm = alt_basket.div(alt_basket.iloc[0])
    alt_eq = alt_norm.mean(axis=1)  # equal-weight alt index
    btc_norm = btc_close / btc_close.iloc[0]

    # BTC dominance proxy: BTC relative strength vs alts
    btc_vs_alt = btc_norm / alt_eq
    btc_vs_alt_ma30 = btc_vs_alt.rolling(30).mean()

    # Signal: BTC.D rising → long BTC, BTC.D falling → long alts
    # We create portfolio returns
    btc_ret = btc_close.pct_change()
    alt_returns = {}
    for a in ["ETH", "SOL"]:
        if a in crypto:
            alt_returns[a] = crypto[a]["close"].pct_change()

    alt_ret = pd.DataFrame(alt_returns).mean(axis=1)

    # Align
    idx = btc_ret.dropna().index.intersection(alt_ret.dropna().index).intersection(btc_vs_alt_ma30.dropna().index)
    btc_ret = btc_ret.reindex(idx)
    alt_ret = alt_ret.reindex(idx)

    # BTC dominance rising = BTC outperforming = btc_vs_alt > MA30
    btc_dom_rising = (btc_vs_alt > btc_vs_alt_ma30).reindex(idx).astype(int)

    # Also use ETH/BTC < 50-day MA signal
    if "ETH" in crypto:
        eth_btc = crypto["ETH"]["close"] / btc_close
        eth_btc_ma50 = eth_btc.rolling(50).mean()
        btc_preferred = (eth_btc < eth_btc_ma50).reindex(idx).astype(int)
        # Combine: BTC dom rising OR ETH/BTC weak → overweight BTC
        btc_signal = ((btc_dom_rising + btc_preferred) >= 1).astype(int)
    else:
        btc_signal = btc_dom_rising

    # Portfolio: when BTC signal → 100% BTC, else 100% alt basket
    # Shift to avoid lookahead
    pos = btc_signal.shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    strategy_ret = pos * btc_ret + (1 - pos) * alt_ret - trades * TX_COST

    strategy_ret = strategy_ret.dropna()
    strategy_ret.name = "BTC_Dom_Rotation"
    metrics = compute_metrics(strategy_ret)

    # Also compute per-asset
    for label, ret in [("BTC_only", btc_ret), ("Alt_only", alt_ret)]:
        m = compute_metrics(ret.loc[strategy_ret.index])
        print(f"  {label}: Sharpe={m['sharpe']}, Return={m['total_return']}%")

    print(f"  Strategy: Sharpe={metrics['sharpe']}, Return={metrics['total_return']}%, MaxDD={metrics['max_dd']}%")
    print(f"  BTC allocation: {pos.mean()*100:.1f}% of time")

    return strategy_ret, metrics


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: TRADFI-TO-CRYPTO LEAD SIGNALS
# ═══════════════════════════════════════════════════════════════════════

def strategy_tradfi_lead(crypto, tradfi):
    print("\n" + "=" * 70)
    print("STRATEGY 2: TRADFI-TO-CRYPTO LEAD SIGNALS")
    print("=" * 70)

    # Equal-weight crypto returns
    crypto_rets = {}
    for name in crypto:
        crypto_rets[name] = crypto[name]["close"].pct_change()
    crypto_ret = pd.DataFrame(crypto_rets).mean(axis=1).dropna()

    signals = {}

    # Signal A: GLD breakout above 20-day high → long crypto 5 days later
    if "GLD" in tradfi:
        gld = tradfi["GLD"]
        gld_20h = gld.rolling(20).max().shift(1)  # shift to avoid lookahead on the high itself
        gld_breakout = (gld > gld_20h).astype(int)
        # Delay by 5 days
        gld_signal = gld_breakout.shift(5).reindex(crypto_ret.index).fillna(0)
        signals["GLD_breakout"] = gld_signal

    # Signal B: UUP breakdown below 20-day low → long crypto
    if "UUP" in tradfi:
        uup = tradfi["UUP"]
        uup_20l = uup.rolling(20).min().shift(1)
        uup_breakdown = (uup < uup_20l).astype(int)
        uup_signal = uup_breakdown.shift(5).reindex(crypto_ret.index).fillna(0)
        signals["UUP_breakdown"] = uup_signal

    # Signal C: TLT crash (bonds selling off) → risk-on → long crypto
    if "TLT" in tradfi:
        tlt = tradfi["TLT"]
        tlt_ret = tlt.pct_change()
        tlt_20d_ret = tlt_ret.rolling(20).sum()
        # TLT crashing = 20d return < -5%
        tlt_crash = (tlt_20d_ret < -0.05).astype(int)
        tlt_signal = tlt_crash.shift(3).reindex(crypto_ret.index).fillna(0)
        signals["TLT_crash"] = tlt_signal

    # Signal D: VIX spike > 30 then reversion (VIX drops below 25 after being > 30)
    if "VIX" in tradfi:
        vix = tradfi["VIX"]
        was_high = (vix.rolling(10).max() > 30).astype(int)
        now_low = (vix < 25).astype(int)
        vix_reversion = (was_high & now_low).astype(int)
        vix_signal = vix_reversion.shift(1).reindex(crypto_ret.index).fillna(0)
        signals["VIX_reversion"] = vix_signal

    # Test each individually
    individual_results = {}
    for name, sig in signals.items():
        ret = apply_signals_to_returns(sig, crypto_ret)
        m = compute_metrics(ret)
        individual_results[name] = m
        print(f"  {name}: Sharpe={m['sharpe']}, Return={m['total_return']}%, Trades={m['n_trades']}")

    # Combined: go long when ANY signal fires
    if signals:
        combined = pd.DataFrame(signals)
        combined_signal = (combined.sum(axis=1) > 0).astype(int)
        strategy_ret = apply_signals_to_returns(combined_signal, crypto_ret)
        strategy_ret.name = "TradFi_Lead"
        metrics = compute_metrics(strategy_ret)
        print(f"\n  COMBINED: Sharpe={metrics['sharpe']}, Return={metrics['total_return']}%, MaxDD={metrics['max_dd']}%")
        print(f"  In-market: {combined_signal.mean()*100:.1f}% of time")
        metrics["individual_signals"] = individual_results
    else:
        strategy_ret = pd.Series(dtype=float)
        metrics = {}

    return strategy_ret, metrics


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 3: FAMA-FRENCH FACTOR BRIDGE
# ═══════════════════════════════════════════════════════════════════════

def strategy_ff_bridge(crypto, ff5, mom):
    print("\n" + "=" * 70)
    print("STRATEGY 3: FAMA-FRENCH FACTOR BRIDGE")
    print("=" * 70)

    if ff5.empty:
        print("  No FF5 data!")
        return pd.Series(dtype=float), {}

    # Monthly crypto returns
    btc_monthly = crypto["BTC"]["close"].resample("ME").last().pct_change().dropna()

    # Merge FF factors with crypto
    # FF data already in decimal from loader; align month-start to month-end
    ff = ff5.copy()
    ff.index = ff.index.to_period('M').to_timestamp('M')  # align to month-end
    if not mom.empty:
        mom_col = mom.columns[0] if len(mom.columns) > 0 else None
        if mom_col:
            ff["MOM"] = mom[mom_col] / 100.0

    # Align on monthly
    common = ff.index.intersection(btc_monthly.index)
    if len(common) < 12:
        print(f"  Only {len(common)} overlapping months, need 12+")
        return pd.Series(dtype=float), {}

    ff_aligned = ff.reindex(common)
    btc_m = btc_monthly.reindex(common)

    # Correlate each factor with NEXT month crypto return
    print("\n  Factor → Next-Month BTC Correlation:")
    correlations = {}
    for col in ff_aligned.columns:
        if col == "RF":
            continue
        # Factor at month t vs BTC return at month t+1
        corr = ff_aligned[col].iloc[:-1].reset_index(drop=True).corr(btc_m.iloc[1:].reset_index(drop=True))
        correlations[col] = round(corr, 4)
        star = " ★" if abs(corr) > 0.15 else ""
        print(f"    {col:>10}: {corr:+.4f}{star}")

    # Build factor-timed strategy using factors with |corr| > 0.1
    useful_factors = {k: v for k, v in correlations.items() if abs(v) > 0.1}
    print(f"\n  Useful factors (|corr|>0.1): {list(useful_factors.keys())}")

    if not useful_factors:
        print("  No predictive factors found. Using momentum as default.")
        if "MOM" in ff_aligned.columns:
            useful_factors = {"MOM": correlations.get("MOM", 0.1)}
        else:
            useful_factors = {"Mkt-RF": correlations.get("Mkt-RF", 0.1)}

    # Composite signal: z-score of each factor * sign(correlation)
    factor_signals = pd.DataFrame(index=common)
    for fac, corr_val in useful_factors.items():
        if fac in ff_aligned.columns:
            z = (ff_aligned[fac] - ff_aligned[fac].rolling(12).mean()) / ff_aligned[fac].rolling(12).std()
            factor_signals[fac] = z * np.sign(corr_val)

    composite = factor_signals.mean(axis=1)
    # Long when composite > 0
    monthly_signal = (composite > 0).astype(int)

    # Expand to daily for BTC
    btc_daily_ret = crypto["BTC"]["close"].pct_change()
    daily_signal = monthly_signal.reindex(btc_daily_ret.index, method="ffill").fillna(0)

    strategy_ret = apply_signals_to_returns(daily_signal, btc_daily_ret)
    strategy_ret = strategy_ret.dropna()
    strategy_ret.name = "FF_Bridge"
    metrics = compute_metrics(strategy_ret)
    metrics["factor_correlations"] = correlations
    metrics["useful_factors"] = list(useful_factors.keys())

    print(f"\n  Strategy: Sharpe={metrics['sharpe']}, Return={metrics['total_return']}%, MaxDD={metrics['max_dd']}%")
    print(f"  In-market: {daily_signal.mean()*100:.1f}% of time")

    return strategy_ret, metrics


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 4: LIQUIDATION-DRIVEN ENTRIES (using taker data as proxy)
# ═══════════════════════════════════════════════════════════════════════

def strategy_liquidation(crypto, deriv, macro):
    print("\n" + "=" * 70)
    print("STRATEGY 4: LIQUIDATION-DRIVEN ENTRIES (taker proxy)")
    print("=" * 70)

    all_rets = []
    for asset in ["BTC", "ETH"]:
        if asset not in crypto or asset not in deriv:
            print(f"  {asset}: missing data, skipping")
            continue

        close = crypto[asset]["close"]
        ret = close.pct_change()
        taker = deriv[asset]

        # Align
        idx = ret.dropna().index.intersection(taker.dropna(how="all").index)
        if len(idx) < 30:
            print(f"  {asset}: only {len(idx)} overlapping days, skipping")
            continue

        ret = ret.reindex(idx)
        sell_vol = taker["sellVol"].reindex(idx)
        buy_vol = taker["buyVol"].reindex(idx)

        # Extreme sell volume = long liquidation cascade (longs got wiped)
        sell_z = (sell_vol - sell_vol.rolling(60).mean()) / sell_vol.rolling(60).std()
        # Extreme buy volume = short squeeze
        buy_z = (buy_vol - buy_vol.rolling(60).mean()) / buy_vol.rolling(60).std()

        # Buy after extreme selling (> 2 std), short after extreme buying (> 2 std)
        long_signal = (sell_z > 2).astype(int)
        short_signal = (buy_z > 2).astype(int) * -1

        signal = long_signal + short_signal
        # Clamp
        signal = signal.clip(-1, 1)

        # M2 overlay: only go long when M2 is accelerating
        if not macro.empty and "m2" in macro.columns:
            m2 = macro["m2"].reindex(idx, method="ffill")
            m2_accel = (m2.pct_change(90) > 0).astype(int)  # 3-month M2 growth positive
            # Only allow longs when M2 accelerating
            signal = signal.where(~((signal > 0) & (m2_accel == 0)), 0)

        # Hold position for 5 days after signal
        hold_signal = signal.copy()
        for i in range(1, 6):
            hold_signal = hold_signal.where(hold_signal != 0, signal.shift(i))
        hold_signal = hold_signal.fillna(0).clip(-1, 1)

        strat_ret = apply_signals_to_returns(hold_signal, ret)
        all_rets.append(strat_ret)
        m = compute_metrics(strat_ret)
        long_trades = int((long_signal > 0).sum())
        short_trades = int((short_signal < 0).sum())
        print(f"  {asset}: Sharpe={m['sharpe']}, Return={m['total_return']}%, Long signals={long_trades}, Short signals={short_trades}")

    if all_rets:
        # Equal weight BTC + ETH
        combined = pd.concat(all_rets, axis=1).mean(axis=1).dropna()
        combined.name = "Liquidation_Entries"
        metrics = compute_metrics(combined)
        print(f"\n  Combined: Sharpe={metrics['sharpe']}, Return={metrics['total_return']}%, MaxDD={metrics['max_dd']}%")
    else:
        combined = pd.Series(dtype=float)
        metrics = {}

    return combined, metrics


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 5: MULTI-TIMEFRAME CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════

def strategy_multi_timeframe(crypto, tradfi, macro):
    print("\n" + "=" * 70)
    print("STRATEGY 5: MULTI-TIMEFRAME CONFLUENCE")
    print("=" * 70)

    all_rets = []
    for asset in ["BTC", "ETH", "SOL"]:
        if asset not in crypto:
            continue

        close = crypto[asset]["close"]
        ret = close.pct_change()

        # Daily trend: 20-day EMA slope (proxy for V3 daily signal)
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        daily_trend = (ema20 > ema50).astype(int) * 2 - 1  # +1 or -1

        # Weekly momentum: 5-day ROC > 0
        roc5 = close.pct_change(5)
        weekly_mom = (roc5 > 0).astype(int) * 2 - 1

        # Monthly regime: 63-day (3-month) trend + macro overlay
        roc63 = close.pct_change(63)
        monthly_trend = (roc63 > 0).astype(int) * 2 - 1

        # Optional macro: if VIX available, risk-off when VIX > 30
        if "VIX" in tradfi:
            vix = tradfi["VIX"].reindex(close.index, method="ffill")
            risk_off = (vix > 30)
            monthly_trend = monthly_trend.where(~risk_off, -1)

        # Confluence: all 3 must agree
        confluence = daily_trend + weekly_mom + monthly_trend
        # +3 = all bullish → long, -3 = all bearish → short, else flat
        signal = pd.Series(0, index=close.index)
        signal[confluence == 3] = 1
        signal[confluence == -3] = -1

        strat_ret = apply_signals_to_returns(signal, ret)
        all_rets.append(strat_ret)
        m = compute_metrics(strat_ret)
        long_pct = (signal == 1).mean() * 100
        short_pct = (signal == -1).mean() * 100
        print(f"  {asset}: Sharpe={m['sharpe']}, Return={m['total_return']}%, Long={long_pct:.1f}%, Short={short_pct:.1f}%")

    if all_rets:
        combined = pd.concat(all_rets, axis=1).mean(axis=1).dropna()
        combined.name = "MultiTF_Confluence"
        metrics = compute_metrics(combined)
        print(f"\n  Combined: Sharpe={metrics['sharpe']}, Return={metrics['total_return']}%, MaxDD={metrics['max_dd']}%")
    else:
        combined = pd.Series(dtype=float)
        metrics = {}

    return combined, metrics


# ═══════════════════════════════════════════════════════════════════════
# V3 PROXY (simple trend-following baseline)
# ═══════════════════════════════════════════════════════════════════════

def v3_proxy(crypto):
    """Simple trend-following as V3 proxy for correlation comparison."""
    all_rets = []
    for asset in crypto:
        close = crypto[asset]["close"]
        ret = close.pct_change()
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        signal = (ema20 > ema50).astype(int)
        strat_ret = apply_signals_to_returns(signal, ret)
        all_rets.append(strat_ret)
    if all_rets:
        return pd.concat(all_rets, axis=1).mean(axis=1).dropna()
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    crypto, tradfi, macro, ff5, mom, deriv = load_all_data()

    # V3 proxy
    v3_ret = v3_proxy(crypto)
    v3_metrics = compute_metrics(v3_ret)
    print(f"\n  V3 Proxy: Sharpe={v3_metrics['sharpe']}, Return={v3_metrics['total_return']}%")

    # Run all strategies
    results = {}
    returns_dict = {"V3_Proxy": v3_ret}

    # Strategy 1
    ret1, m1 = strategy_btc_dominance(crypto)
    if len(ret1) > 0:
        results["S1_BTC_Dom_Rotation"] = m1
        returns_dict["S1_BTC_Dom_Rotation"] = ret1

    # Strategy 2
    ret2, m2 = strategy_tradfi_lead(crypto, tradfi)
    if len(ret2) > 0:
        results["S2_TradFi_Lead"] = m2
        returns_dict["S2_TradFi_Lead"] = ret2

    # Strategy 3
    ret3, m3 = strategy_ff_bridge(crypto, ff5, mom)
    if len(ret3) > 0:
        results["S3_FF_Bridge"] = m3
        returns_dict["S3_FF_Bridge"] = ret3

    # Strategy 4
    ret4, m4 = strategy_liquidation(crypto, deriv, macro)
    if len(ret4) > 0:
        results["S4_Liquidation"] = m4
        returns_dict["S4_Liquidation"] = ret4

    # Strategy 5
    ret5, m5 = strategy_multi_timeframe(crypto, tradfi, macro)
    if len(ret5) > 0:
        results["S5_MultiTF"] = m5
        returns_dict["S5_MultiTF"] = ret5

    # ─── COMPARISON TABLE ───
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Sharpe':>8} {'Return%':>10} {'CAGR%':>8} {'MaxDD%':>8} {'Sortino':>8} {'WinRate':>8}")
    print("-" * 80)

    # V3 first
    v = v3_metrics
    print(f"{'V3_Proxy':<25} {v['sharpe']:>8.3f} {v['total_return']:>10.1f} {v['cagr']:>8.1f} {v['max_dd']:>8.1f} {v['sortino']:>8.3f} {v['win_rate']:>7.1f}%")

    for name, m in results.items():
        # Skip nested dicts for printing
        sharpe = m.get("sharpe", 0)
        print(f"{name:<25} {sharpe:>8.3f} {m.get('total_return',0):>10.1f} {m.get('cagr',0):>8.1f} {m.get('max_dd',0):>8.1f} {m.get('sortino',0):>8.3f} {m.get('win_rate',0):>7.1f}%")

    # ─── CORRELATION MATRIX ───
    print("\n" + "=" * 70)
    print("RETURN CORRELATION MATRIX")
    print("=" * 70)

    # Align all return series
    ret_df = pd.DataFrame()
    for name, ret in returns_dict.items():
        if len(ret) > 0:
            ret_df[name] = ret

    if len(ret_df.columns) > 1:
        corr = ret_df.corr()
        print(corr.round(3).to_string())

        # Diversification analysis
        print("\n  Correlation with V3:")
        for col in corr.columns:
            if col != "V3_Proxy" and "V3_Proxy" in corr.index:
                c = corr.loc["V3_Proxy", col]
                div = "HIGH DIV ✓" if abs(c) < 0.3 else ("MODERATE" if abs(c) < 0.5 else "REDUNDANT ✗")
                print(f"    {col}: {c:+.3f} → {div}")

    # ─── RECOMMENDATION ───
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Rank by Sharpe, prefer low V3 correlation
    scored = []
    for name, m in results.items():
        sharpe = m.get("sharpe", 0)
        v3_corr = corr.loc["V3_Proxy", name] if "V3_Proxy" in corr.index and name in corr.columns else 1.0
        # Score = Sharpe * (1 - |v3_corr|) — rewards high Sharpe + low correlation
        score = sharpe * (1 - abs(v3_corr) * 0.5)
        scored.append((name, sharpe, v3_corr, score))
        print(f"  {name}: Sharpe={sharpe:.3f}, V3_corr={v3_corr:+.3f}, Score={score:.3f}")

    scored.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  TOP PICKS (by diversification-adjusted score):")
    for i, (name, sharpe, v3c, score) in enumerate(scored[:3]):
        print(f"    {i+1}. {name} (score={score:.3f})")

    # ─── SAVE RESULTS ───
    output = {
        "timestamp": datetime.now().isoformat(),
        "v3_proxy": v3_metrics,
        "strategies": {},
        "correlation_matrix": corr.to_dict() if len(ret_df.columns) > 1 else {},
        "recommendations": [s[0] for s in scored[:3]],
    }
    for name, m in results.items():
        # Clean non-serializable items
        clean = {}
        for k, v in m.items():
            if isinstance(v, (dict, list)):
                clean[k] = v
            elif isinstance(v, (int, float)):
                clean[k] = v
            else:
                clean[k] = str(v)
        output["strategies"][name] = clean

    out_path = RESULTS_DIR / "hybrid_strategies_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
