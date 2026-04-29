"""
V3.2 Enhancement Research — Built on V3.1-H2 Base

Tests 17 variants across 5 groups:
A: Bear Capitalization (shorting)
B: Dip-Buying (position addition)
C: Pyramiding
D: Tighter Drawdown Protection
E: Best Combinations

Each variant post-processes V3.1-H2's per-asset leverage + protection layers.
Full walk-forward validation on all variants.
"""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from copy import deepcopy

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v31 import (
    LEVERAGE_MAP, WEIGHTS, VOL_CEILING, VOL_LOOKBACK, BEAR_FILTER_DAYS,
    PORTFOLIO_TRAIL_STOP, TRAIL_REDUCE_FACTOR, TRAIL_RECOVERY_DAYS,
    TRAIL_RECOVERY_THRESHOLD, SCORE4_CRYPTO_MOM_OVERRIDE,
)
from strategies.composite.mega_strategy_v3 import (
    compute_confluence, detect_regime, ASSET_CONFIGS,
    _rsi, _bb, _atr, _realized_vol, TX_COST,
)

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}


# ── Data Loading (same as backtest_mega_v31.py) ──────────────────

def load_data():
    print("Loading data...")
    stock = StockDataLoader()
    fred = MacroDataLoader()
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col] = df["close"]
        except Exception as e:
            print(f"  {col}: FAILED - {e}")
    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    print(f"  Macro: {len(macro_data)} days")
    return crypto_data, cross_asset_data, macro_data


# ── Metrics ──────────────────────────────────────────────────────

def compute_metrics(returns, leverage_series=None):
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["sharpe","cagr","max_dd","sortino","calmar","win_rate",
                                   "mean_leverage","flat_pct","ret_2022","ret_2024"]}
    eq = (1 + returns).cumprod()
    n_yr = len(returns) / 252
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0

    mean_lev = 0.0
    flat_pct = 0.0
    if leverage_series is not None and len(leverage_series) > 0:
        mean_lev = float(leverage_series.abs().mean())
        flat_pct = float((leverage_series.abs() < 0.01).mean() * 100)

    # Period returns
    ret_2022 = 0.0
    ret_2024 = 0.0
    for yr, attr in [(2022, "ret_2022"), (2024, "ret_2024")]:
        mask = returns.index.year == yr
        if mask.sum() > 10:
            yr_eq = (1 + returns[mask]).cumprod()
            val = float(yr_eq.iloc[-1] - 1) * 100
            if attr == "ret_2022": ret_2022 = val
            else: ret_2024 = val

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "mean_leverage": round(mean_lev, 3),
        "flat_pct": round(flat_pct, 1),
        "ret_2022": round(ret_2022, 1),
        "ret_2024": round(ret_2024, 1),
    }


# ── Walk-Forward ─────────────────────────────────────────────────

def walk_forward(returns, leverage_series=None):
    """14-fold walk-forward: 1yr min train, 6mo test, rolling ~3mo."""
    ret = returns.dropna()
    n = len(ret)
    min_train = 365
    test_size = 126
    step = 63  # ~3 months

    fold_sharpes = []
    folds = []

    for fold_idx in range(14):
        train_end = min_train + fold_idx * step
        test_end = train_end + test_size
        if test_end > n:
            break
        test_ret = ret.iloc[train_end:test_end]
        if len(test_ret) < 50 or test_ret.std() == 0:
            continue
        fs = float(test_ret.mean() / test_ret.std() * np.sqrt(252))
        fold_sharpes.append(fs)
        folds.append({
            "fold": fold_idx + 1,
            "test_start": str(ret.index[train_end].date()),
            "test_end": str(ret.index[min(test_end-1, n-1)].date()),
            "sharpe": round(fs, 3),
        })

    if len(fold_sharpes) < 3:
        return {"mean_oos_sharpe": 0, "median_oos_sharpe": 0, "n_folds": 0,
                "positive_folds": 0, "folds": folds}

    mean_s = np.mean(fold_sharpes)
    std_s = np.std(fold_sharpes, ddof=1)
    t_stat = mean_s / (std_s / np.sqrt(len(fold_sharpes))) if std_s > 0 else 0
    p_val = 1 - stats.t.cdf(t_stat, df=len(fold_sharpes) - 1)

    return {
        "mean_oos_sharpe": round(mean_s, 3),
        "median_oos_sharpe": round(float(np.median(fold_sharpes)), 3),
        "std_oos_sharpe": round(std_s, 3),
        "n_folds": len(fold_sharpes),
        "positive_folds": sum(1 for s in fold_sharpes if s > 0),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "folds": folds,
    }


# ── Core Simulation Engine ───────────────────────────────────────
# This replicates V3.1's run_full_strategy but allows post-processing
# modifications to the leverage series before computing P&L.

def precompute_asset_data(crypto_data, macro_data, cross_asset_data):
    """Pre-compute confluence, leverage, returns, and indicators for all assets."""
    assets = list(crypto_data.keys())
    common_idx = crypto_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(crypto_data[a].index)

    result = {}
    for asset in assets:
        df = crypto_data[asset].reindex(common_idx)
        close = df["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])

        conf, bd = compute_confluence(
            close, macro_data, cross_asset_data,
            sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"]
        )
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)

        # Base leverage from H2 config
        base_lev = conf.map(lambda c: LEVERAGE_MAP.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), SCORE4_CRYPTO_MOM_OVERRIDE)
        base_lev = base_lev.shift(1).fillna(0)

        # Technical indicators for enhancements
        rsi = _rsi(close, 14).reindex(common_idx).fillna(50)
        bb_lower, bb_mid, bb_upper = _bb(close, 20, 2.0)
        bb_lower = bb_lower.reindex(common_idx).fillna(close)
        atr = _atr(df, 20).reindex(common_idx).fillna(0)
        rolling_high_20 = close.rolling(20).max().reindex(common_idx)
        ret = close.pct_change().reindex(common_idx).fillna(0)

        result[asset] = {
            "close": close, "ret": ret, "conf": conf, "bd": bd,
            "base_lev": base_lev, "rsi": rsi, "bb_lower": bb_lower,
            "atr": atr, "rolling_high_20": rolling_high_20, "df": df,
        }

    result["_common_idx"] = common_idx
    result["_assets"] = assets

    # BTC regime
    btc_conf = result.get("BTC", result[assets[0]])["conf"]
    result["_regime"] = detect_regime(btc_conf)
    result["_btc_conf"] = btc_conf

    return result


def simulate_variant(precomp, variant_fn, variant_name="",
                     vol_ceiling=VOL_CEILING, vol_lookback=VOL_LOOKBACK,
                     bear_filter_days=BEAR_FILTER_DAYS,
                     portfolio_trail_stop=PORTFOLIO_TRAIL_STOP,
                     trail_reduce_factor=TRAIL_REDUCE_FACTOR,
                     trail_recovery_days=TRAIL_RECOVERY_DAYS,
                     trail_recovery_threshold=TRAIL_RECOVERY_THRESHOLD,
                     per_asset_trail_stops=None,
                     dynamic_atr_trail=False,
                     atr_trail_mult=1.5):
    """
    Simulate a variant by running V3.1-H2 base with optional modifications.

    variant_fn: function(precomp, i, per_asset_exposures, equity, peak_eq) -> modified per_asset_exposures
        Takes current state, returns modified {asset: exposure} dict.
        If None, runs base H2.
    """
    common_idx = precomp["_common_idx"]
    assets = precomp["_assets"]
    btc_conf = precomp["_btc_conf"]
    w = WEIGHTS
    n = len(common_idx)

    daily_pnl = np.zeros(n)
    total_leverage = np.zeros(n)

    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0
    prev_total_exposure = 0.0

    # Per-asset tracking for pyramiding/dip-buying
    entry_prices = {a: 0.0 for a in assets}
    position_pnl_pct = {a: 0.0 for a in assets}
    last_pyramid_prices = {a: 0.0 for a in assets}

    for i in range(1, n):
        # ── Base per-asset exposure ──
        per_asset_exp = {}
        for asset in assets:
            asset_lev = float(precomp[asset]["base_lev"].iloc[i])
            per_asset_exp[asset] = w.get(asset, 0.25) * asset_lev

        # ── Layer 2: Vol ceiling ──
        if vol_ceiling > 0:
            btc_close = precomp.get("BTC", precomp[assets[0]])["close"]
            if i >= vol_lookback:
                window = btc_close.iloc[max(0, i - vol_lookback):i]
                rvol = window.pct_change().std() * np.sqrt(252)
                if rvol > vol_ceiling:
                    for a in assets:
                        per_asset_exp[a] *= 0.5

        # ── Layer 3: Bear filter ──
        btc_score = int(btc_conf.iloc[i - 1])
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0
        if consecutive_low >= bear_filter_days:
            for a in assets:
                per_asset_exp[a] = 0.0

        # ── Layer 4: Portfolio trailing stop ──
        if portfolio_trail_stop > 0:
            dd = 1 - equity / peak_eq
            if dd > portfolio_trail_stop:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start_idx = i
                for a in assets:
                    per_asset_exp[a] *= trail_reduce_factor
            elif in_drawdown:
                days_since = i - dd_start_idx
                if days_since > trail_recovery_days and equity > peak_eq * trail_recovery_threshold:
                    in_drawdown = False

        # ── Per-asset trailing stops (D3) ──
        if per_asset_trail_stops:
            for asset in assets:
                stop_pct = per_asset_trail_stops.get(asset, 0.15)
                close_val = float(precomp[asset]["close"].iloc[i])
                if entry_prices[asset] > 0 and per_asset_exp[asset] > 0:
                    peak_price = max(entry_prices[asset], close_val)
                    dd_from_peak = 1 - close_val / peak_price
                    if dd_from_peak > stop_pct:
                        per_asset_exp[asset] = 0.0

        # ── Dynamic ATR trail (D4) ──
        if dynamic_atr_trail:
            for asset in assets:
                if per_asset_exp[asset] > 0:
                    atr_val = float(precomp[asset]["atr"].iloc[i])
                    close_val = float(precomp[asset]["close"].iloc[i])
                    if atr_val > 0 and entry_prices[asset] > 0:
                        stop_dist = atr_trail_mult * atr_val
                        if entry_prices[asset] - close_val > stop_dist:
                            per_asset_exp[asset] = 0.0

        # ── Apply variant modifications ──
        if variant_fn is not None:
            per_asset_exp = variant_fn(
                precomp, i, per_asset_exp, equity, peak_eq,
                entry_prices, position_pnl_pct, last_pyramid_prices,
                consecutive_low,
            )

        # ── Compute total exposure and P&L ──
        target_exposure = sum(per_asset_exp.values())
        port_ret = 0.0
        for asset in assets:
            exp = per_asset_exp[asset]
            ret = float(precomp[asset]["ret"].iloc[i])
            port_ret += exp * ret

            # Track entry prices
            close_val = float(precomp[asset]["close"].iloc[i])
            if abs(exp) > 0.01:
                if entry_prices[asset] == 0:
                    entry_prices[asset] = close_val
                    last_pyramid_prices[asset] = close_val
                position_pnl_pct[asset] = (close_val / entry_prices[asset] - 1) if entry_prices[asset] > 0 else 0
            else:
                entry_prices[asset] = 0.0
                position_pnl_pct[asset] = 0.0
                last_pyramid_prices[asset] = 0.0

        # Transaction costs
        lev_change = abs(target_exposure - prev_total_exposure)
        port_ret -= lev_change * TX_COST
        prev_total_exposure = target_exposure

        daily_pnl[i] = port_ret
        total_leverage[i] = target_exposure
        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)

    pnl_series = pd.Series(daily_pnl, index=common_idx)
    lev_series = pd.Series(total_leverage, index=common_idx)
    return pnl_series, lev_series


# ══════════════════════════════════════════════════════════════════
# VARIANT FUNCTIONS
# Each takes (precomp, i, per_asset_exp, equity, peak_eq,
#             entry_prices, position_pnl_pct, last_pyramid_prices,
#             consecutive_low) -> modified per_asset_exp
# ══════════════════════════════════════════════════════════════════

# ── Group A: Bear Capitalization ──

def variant_A1(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """Score 0 → 0.3x SHORT (score 1 stays flat)"""
    assets = precomp["_assets"]
    btc_conf = precomp["_btc_conf"]
    score = int(btc_conf.iloc[i-1]) if i > 0 else 0
    if score == 0:
        for a in assets:
            exp[a] = -WEIGHTS.get(a, 0.25) * 0.3
    return exp

def variant_A2(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """Score 0 → 0.5x SHORT, Score 1 → 0.2x SHORT"""
    btc_conf = precomp["_btc_conf"]
    score = int(btc_conf.iloc[i-1]) if i > 0 else 0
    if score == 0:
        for a in precomp["_assets"]:
            exp[a] = -WEIGHTS.get(a, 0.25) * 0.5
    elif score == 1:
        for a in precomp["_assets"]:
            exp[a] = -WEIGHTS.get(a, 0.25) * 0.2
    return exp

def variant_A3(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """Score 0 → 0.3x SHORT + fast bear detection (15d instead of 30d)"""
    btc_conf = precomp["_btc_conf"]
    score = int(btc_conf.iloc[i-1]) if i > 0 else 0
    if score == 0:
        for a in precomp["_assets"]:
            exp[a] = -WEIGHTS.get(a, 0.25) * 0.3
    # Fast bear: go flat after 15 consecutive days at score <= 1 (already handled by bear_filter,
    # but we override the base exposure which was set to 0 by 30d filter)
    # The 15d filter effect is handled by passing bear_filter_days=15 to simulate_variant
    return exp


# ── Group B: Dip-Buying (position addition) ──

def variant_B1(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """When already long AND RSI(14) < 30 → add 0.3x (capped at 2.5x total)"""
    total = sum(exp.values())
    if total > 0.01:  # already long
        for a in precomp["_assets"]:
            if exp[a] > 0.01:
                rsi_val = float(precomp[a]["rsi"].iloc[i-1]) if i > 0 else 50
                if rsi_val < 30:
                    addition = WEIGHTS.get(a, 0.25) * 0.3
                    exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
    return exp

def variant_B2(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """When already long AND price touches lower BB(20,2) → add 0.2x (capped at 2.5x)"""
    total = sum(exp.values())
    if total > 0.01:
        for a in precomp["_assets"]:
            if exp[a] > 0.01:
                close_val = float(precomp[a]["close"].iloc[i-1]) if i > 0 else 0
                bb_val = float(precomp[a]["bb_lower"].iloc[i-1]) if i > 0 else 0
                if close_val > 0 and close_val <= bb_val:
                    addition = WEIGHTS.get(a, 0.25) * 0.2
                    exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
    return exp

def variant_B3(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """When already long AND drawdown from recent high > 10% → add 0.3x (capped at 2.5x)"""
    total = sum(exp.values())
    if total > 0.01:
        for a in precomp["_assets"]:
            if exp[a] > 0.01:
                close_val = float(precomp[a]["close"].iloc[i])
                rh = float(precomp[a]["rolling_high_20"].iloc[i-1]) if i > 0 else close_val
                if rh > 0:
                    dd = 1 - close_val / rh
                    if dd > 0.10:
                        addition = WEIGHTS.get(a, 0.25) * 0.3
                        exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
    return exp


# ── Group C: Pyramiding ──

def variant_C1(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """When position profitable > 5% AND RSI > 60 → add 0.2x (capped at 2.5x)"""
    for a in precomp["_assets"]:
        if exp[a] > 0.01 and pnl_pct.get(a, 0) > 0.05:
            rsi_val = float(precomp[a]["rsi"].iloc[i-1]) if i > 0 else 50
            if rsi_val > 60:
                addition = WEIGHTS.get(a, 0.25) * 0.2
                exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
    return exp

def variant_C2(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """When price breaks above 20d high AND already long → add 0.2x (capped at 2.5x)"""
    for a in precomp["_assets"]:
        if exp[a] > 0.01:
            close_val = float(precomp[a]["close"].iloc[i])
            rh = float(precomp[a]["rolling_high_20"].iloc[i-1]) if i > 0 else 0
            if rh > 0 and close_val > rh:
                addition = WEIGHTS.get(a, 0.25) * 0.2
                exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
    return exp

def variant_C3(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
    """Trailing pyramid: every +10% from last pyramid price, add 0.15x (capped at 2.5x)"""
    for a in precomp["_assets"]:
        if exp[a] > 0.01 and lpp.get(a, 0) > 0:
            close_val = float(precomp[a]["close"].iloc[i])
            gain_from_last = (close_val / lpp[a] - 1) if lpp[a] > 0 else 0
            if gain_from_last > 0.10:
                addition = WEIGHTS.get(a, 0.25) * 0.15
                exp[a] = min(exp[a] + addition, WEIGHTS.get(a, 0.25) * 2.5)
                lpp[a] = close_val  # reset pyramid price
    return exp


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("#" * 70)
    print("#  V3.2 Enhancement Research")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    crypto_data, cross_asset_data, macro_data = load_data()
    if len(crypto_data) < 2:
        print("ERROR: insufficient data"); return

    print("\nPre-computing asset data...")
    precomp = precompute_asset_data(crypto_data, macro_data, cross_asset_data)

    # ── Define all variants ──
    variants = {}

    # Baseline
    variants["H2-Baseline"] = {"fn": None}

    # Group A: Bear Capitalization
    variants["A1: Short-0.3x@score0"] = {"fn": variant_A1}
    variants["A2: Short-0.5x@0+0.2x@1"] = {"fn": variant_A2}
    variants["A3: Short-0.3x+FastBear15d"] = {"fn": variant_A3, "bear_filter_days": 15}

    # Group B: Dip-Buying
    variants["B1: DipBuy-RSI30+0.3x"] = {"fn": variant_B1}
    variants["B2: DipBuy-BB+0.2x"] = {"fn": variant_B2}
    variants["B3: DipBuy-DD10%+0.3x"] = {"fn": variant_B3}

    # Group C: Pyramiding
    variants["C1: Pyramid-Profit5%+RSI60"] = {"fn": variant_C1}
    variants["C2: Pyramid-20dBreakout"] = {"fn": variant_C2}
    variants["C3: Pyramid-Trail10%"] = {"fn": variant_C3}

    # Group D: Tighter Drawdown Protection
    variants["D1: Trail8%+VolCeil60%"] = {"fn": None, "portfolio_trail_stop": 0.08, "vol_ceiling": 0.60}
    variants["D2: Trail6%+VolCeil70%+Bear20d"] = {"fn": None, "portfolio_trail_stop": 0.06, "vol_ceiling": 0.70, "bear_filter_days": 20}
    variants["D3: PerAssetTrails"] = {"fn": None, "per_asset_trail_stops": {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08, "LINK": 0.08}}
    variants["D4: DynamicATRTrail"] = {"fn": None, "dynamic_atr_trail": True, "atr_trail_mult": 1.5}

    # Run all individual variants first
    results = {}
    print(f"\nRunning {len(variants)} variants...\n")

    for name, cfg in variants.items():
        t1 = time.time()
        fn = cfg.pop("fn", None)
        # Separate sim kwargs
        sim_kwargs = {}
        for k in ["vol_ceiling","vol_lookback","bear_filter_days","portfolio_trail_stop",
                   "trail_reduce_factor","trail_recovery_days","trail_recovery_threshold",
                   "per_asset_trail_stops","dynamic_atr_trail","atr_trail_mult"]:
            if k in cfg:
                sim_kwargs[k] = cfg[k]

        pnl, lev = simulate_variant(precomp, fn, name, **sim_kwargs)
        is_metrics = compute_metrics(pnl, lev)
        wf = walk_forward(pnl, lev)

        results[name] = {"is": is_metrics, "oos": wf}
        cfg["fn"] = fn  # restore

        elapsed = time.time() - t1
        print(f"  {name:<35} IS Sharpe={is_metrics['sharpe']:>6.3f}  "
              f"CAGR={is_metrics['cagr']:>6.1f}%  MaxDD={is_metrics['max_dd']:>7.1f}%  "
              f"OOS Sharpe={wf['mean_oos_sharpe']:>6.3f}  ({elapsed:.1f}s)")

    # ── Group E: Combinations ──
    # Determine best from each group based on OOS Sharpe
    def best_of_group(prefix):
        group = {k: v for k, v in results.items() if k.startswith(prefix)}
        if not group:
            return None
        return max(group.keys(), key=lambda k: results[k]["oos"]["mean_oos_sharpe"])

    best_a = best_of_group("A")
    best_b = best_of_group("B")
    best_c = best_of_group("C")
    best_d = best_of_group("D")

    print(f"\n  Best A: {best_a}")
    print(f"  Best B: {best_b}")
    print(f"  Best C: {best_c}")
    print(f"  Best D: {best_d}")

    # Get variant functions by name
    variant_map = {
        "A1: Short-0.3x@score0": variant_A1,
        "A2: Short-0.5x@0+0.2x@1": variant_A2,
        "A3: Short-0.3x+FastBear15d": variant_A3,
        "B1: DipBuy-RSI30+0.3x": variant_B1,
        "B2: DipBuy-BB+0.2x": variant_B2,
        "B3: DipBuy-DD10%+0.3x": variant_B3,
        "C1: Pyramid-Profit5%+RSI60": variant_C1,
        "C2: Pyramid-20dBreakout": variant_C2,
        "C3: Pyramid-Trail10%": variant_C3,
    }

    # Get D config
    d_configs = {
        "D1: Trail8%+VolCeil60%": {"portfolio_trail_stop": 0.08, "vol_ceiling": 0.60},
        "D2: Trail6%+VolCeil70%+Bear20d": {"portfolio_trail_stop": 0.06, "vol_ceiling": 0.70, "bear_filter_days": 20},
        "D3: PerAssetTrails": {"per_asset_trail_stops": {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08, "LINK": 0.08}},
        "D4: DynamicATRTrail": {"dynamic_atr_trail": True, "atr_trail_mult": 1.5},
    }

    def make_combo_fn(fn_list):
        """Chain multiple variant functions."""
        def combo(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low):
            for fn in fn_list:
                if fn is not None:
                    exp = fn(precomp, i, exp, equity, peak_eq, ep, pnl_pct, lpp, cons_low)
            return exp
        return combo

    # E1: Best bear short + best dip-buy + H2 protections
    e1_fns = [variant_map.get(best_a), variant_map.get(best_b)]
    e1_fns = [f for f in e1_fns if f is not None]

    # E2: Best bear short + best pyramid + tighter drawdown
    e2_fns = [variant_map.get(best_a), variant_map.get(best_c)]
    e2_fns = [f for f in e2_fns if f is not None]
    e2_kwargs = d_configs.get(best_d, {}) if best_d else {}

    # E3: Best dip-buy + best pyramid + tighter drawdown
    e3_fns = [variant_map.get(best_b), variant_map.get(best_c)]
    e3_fns = [f for f in e3_fns if f is not None]
    e3_kwargs = d_configs.get(best_d, {}) if best_d else {}

    # E4: Kitchen sink
    e4_fns = [variant_map.get(best_a), variant_map.get(best_b), variant_map.get(best_c)]
    e4_fns = [f for f in e4_fns if f is not None]
    e4_kwargs = d_configs.get(best_d, {}) if best_d else {}

    combo_variants = {
        "E1: BestShort+BestDip": (make_combo_fn(e1_fns), {}),
        "E2: BestShort+BestPyramid+TightDD": (make_combo_fn(e2_fns), e2_kwargs),
        "E3: BestDip+BestPyramid+TightDD": (make_combo_fn(e3_fns), e3_kwargs),
        "E4: KitchenSink": (make_combo_fn(e4_fns), e4_kwargs),
    }

    print(f"\nRunning {len(combo_variants)} combination variants...\n")
    for name, (fn, kwargs) in combo_variants.items():
        t1 = time.time()
        pnl, lev = simulate_variant(precomp, fn, name, **kwargs)
        is_metrics = compute_metrics(pnl, lev)
        wf = walk_forward(pnl, lev)
        results[name] = {"is": is_metrics, "oos": wf}
        elapsed = time.time() - t1
        print(f"  {name:<35} IS Sharpe={is_metrics['sharpe']:>6.3f}  "
              f"CAGR={is_metrics['cagr']:>6.1f}%  MaxDD={is_metrics['max_dd']:>7.1f}%  "
              f"OOS Sharpe={wf['mean_oos_sharpe']:>6.3f}  ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 160)
    print("RANKED SUMMARY — ALL V3.2 VARIANTS (sorted by OOS Sharpe)")
    print("=" * 160)
    header = (f"{'Rank':<5} {'Variant':<36} {'IS Shrp':>8} {'OOS Shrp':>9} {'CAGR%':>7} {'MaxDD%':>8} "
              f"{'Sortino':>8} {'Calmar':>7} {'WinR%':>6} {'MnLev':>6} {'Flat%':>6} "
              f"{'2022':>7} {'2024':>7} {'Target':>7}")
    print(header)
    print("-" * 160)

    sorted_variants = sorted(results.keys(),
                            key=lambda k: results[k]["oos"]["mean_oos_sharpe"],
                            reverse=True)

    for rank, name in enumerate(sorted_variants, 1):
        r = results[name]
        m = r["is"]
        oos = r["oos"]["mean_oos_sharpe"]
        # Check if meets target: MaxDD > -25% AND Sharpe > 1.0
        meets = "✓" if m["max_dd"] > -25 and m["sharpe"] > 1.0 else ""
        if m["max_dd"] > -25 and oos > 1.0:
            meets = "★"

        print(f"{rank:<5} {name:<36} {m['sharpe']:>8.3f} {oos:>9.3f} {m['cagr']:>7.1f} {m['max_dd']:>8.1f} "
              f"{m['sortino']:>8.3f} {m['calmar']:>7.3f} {m['win_rate']:>6.1f} {m['mean_leverage']:>6.3f} "
              f"{m['flat_pct']:>6.1f} {m['ret_2022']:>7.1f} {m['ret_2024']:>7.1f} {meets:>7}")

    # Highlight meeting targets
    print("\n" + "=" * 80)
    print("VARIANTS MEETING TARGETS (MaxDD > -25% AND IS Sharpe > 1.0)")
    print("=" * 80)
    target_met = [k for k in sorted_variants
                  if results[k]["is"]["max_dd"] > -25 and results[k]["is"]["sharpe"] > 1.0]
    if target_met:
        for name in target_met:
            m = results[name]["is"]
            oos = results[name]["oos"]["mean_oos_sharpe"]
            print(f"  {name:<36} IS={m['sharpe']:.3f} OOS={oos:.3f} CAGR={m['cagr']:.1f}% MaxDD={m['max_dd']:.1f}%")
    else:
        print("  None meet both targets. Closest:")
        # Show top 3 by combined score
        def score(k):
            m = results[k]["is"]
            oos = results[k]["oos"]["mean_oos_sharpe"]
            dd_score = max(0, 1 - abs(m["max_dd"] + 25) / 25)  # closer to -25% = higher
            return oos + dd_score
        top3 = sorted(results.keys(), key=score, reverse=True)[:3]
        for name in top3:
            m = results[name]["is"]
            oos = results[name]["oos"]["mean_oos_sharpe"]
            print(f"  {name:<36} IS={m['sharpe']:.3f} OOS={oos:.3f} CAGR={m['cagr']:.1f}% MaxDD={m['max_dd']:.1f}%")

    # ── Walk-forward detail for top 3 ──
    print("\n" + "=" * 80)
    print("WALK-FORWARD DETAIL — TOP 3 BY OOS SHARPE")
    print("=" * 80)
    for name in sorted_variants[:3]:
        wf = results[name]["oos"]
        print(f"\n  {name}: OOS Sharpe={wf['mean_oos_sharpe']:.3f} (median={wf['median_oos_sharpe']:.3f})")
        print(f"    Folds: {wf['n_folds']}, Positive: {wf['positive_folds']}, p-value: {wf['p_value']:.4f}")
        for f in wf["folds"]:
            marker = " ★" if f["sharpe"] > 0 else ""
            print(f"    F{f['fold']:>2}: {f['test_start']} → {f['test_end']}  Sharpe={f['sharpe']:>7.3f}{marker}")

    # ── Save ──
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "V3.1-H2",
        "best_per_group": {"A": best_a, "B": best_b, "C": best_c, "D": best_d},
        "variants": {},
    }
    for name in sorted_variants:
        save_data["variants"][name] = results[name]

    out_path = RESULTS_DIR / "v32_enhancement_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("#" * 70)
    print("#  V3.2 RESEARCH COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
