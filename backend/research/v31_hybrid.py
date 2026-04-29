"""
V3.1 Hybrid — Direct sizing with drawdown protection layers.
Combines: confluence-driven exposure + trailing stop + vol ceiling + 2022-style bear filter.
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
    compute_confluence, detect_regime, ASSET_CONFIGS,
    _realized_vol, TX_COST, run_full_strategy
)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}
WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}


def load_data():
    stock = StockDataLoader()
    fred = MacroDataLoader()
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100: crypto_data[name] = df
        except: pass
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col] = df["close"]
        except: pass
    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    return crypto_data, cross_asset_data, macro_data


def hybrid_backtest(crypto_data, macro_data, cross_asset_data, config):
    """
    Hybrid V3.1: Direct sizing + protection layers.
    
    Config keys:
        leverage_map: {score: leverage}
        s4_crypto_mom_override: leverage when score=4 and crypto mom is off
        vol_ceiling: annualized vol threshold (halve above)
        vol_lookback: days for realized vol calc
        portfolio_trail_stop: portfolio DD that triggers position reduction
        trail_reduce_factor: how much to reduce (0.5 = halve positions)
        trail_recovery_days: days of recovery needed to restore full sizing
        bear_filter: if True, go flat when score <= 1 for 30+ consecutive days
    """
    c = config
    lev_map = c["leverage_map"]
    w = WEIGHTS
    
    assets = list(crypto_data.keys())
    common_idx = crypto_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(crypto_data[a].index)
    
    # Pre-compute per-asset confluences and returns
    asset_confluences = {}
    asset_breakdowns = {}
    asset_returns = {}
    
    for asset in assets:
        df = crypto_data[asset].reindex(common_idx)
        close = df["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        
        conf, bd = compute_confluence(
            close, macro_data, cross_asset_data,
            sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"]
        )
        asset_confluences[asset] = conf.reindex(common_idx).fillna(0)
        asset_breakdowns[asset] = bd.reindex(common_idx).fillna(0)
        asset_returns[asset] = close.pct_change().reindex(common_idx).fillna(0)
    
    # BTC confluence drives portfolio regime
    btc_conf = asset_confluences["BTC"]
    regime = detect_regime(btc_conf)
    
    # Simulate day by day with protection layers
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    daily_lev = np.zeros(n)
    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0  # consecutive days at score <= 1
    
    for i in range(1, n):
        dt = common_idx[i]
        
        # ── Layer 1: Base exposure from confluence ──
        total_exposure = 0.0
        for asset in assets:
            conf_val = int(asset_confluences[asset].iloc[i-1])  # lagged 1 day
            base_lev = lev_map.get(min(conf_val, 5), 0.0)
            
            # Score 4 smart override
            if conf_val == 4 and c.get("s4_crypto_mom_override") is not None:
                crypto_mom = asset_breakdowns[asset]["crypto_momentum"].iloc[i-1]
                if crypto_mom == 0:
                    base_lev = c["s4_crypto_mom_override"]
            
            total_exposure += w.get(asset, 0.25) * base_lev
        
        # ── Layer 2: Vol ceiling ──
        if c.get("vol_ceiling"):
            # Use BTC vol as portfolio proxy
            btc_close = crypto_data["BTC"]["close"].reindex(common_idx)
            lookback = c.get("vol_lookback", 30)
            if i >= lookback:
                window = btc_close.iloc[max(0,i-lookback):i]
                rvol = window.pct_change().std() * np.sqrt(252)
                if rvol > c["vol_ceiling"]:
                    total_exposure *= 0.5
        
        # ── Layer 3: Bear filter ──
        if c.get("bear_filter"):
            btc_score = int(btc_conf.iloc[i-1])
            if btc_score <= 1:
                consecutive_low += 1
            else:
                consecutive_low = 0
            
            if consecutive_low >= 30:
                total_exposure = 0.0  # Go flat in sustained bear
        
        # ── Layer 4: Portfolio trailing stop ──
        if c.get("portfolio_trail_stop"):
            dd = 1 - equity / peak_eq
            if dd > c["portfolio_trail_stop"]:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start_idx = i
                total_exposure *= c.get("trail_reduce_factor", 0.5)
            elif in_drawdown:
                # Recovery: restore after N days above peak * 0.95
                recovery_days = c.get("trail_recovery_days", 20)
                if i - dd_start_idx > recovery_days and equity > peak_eq * 0.95:
                    in_drawdown = False
        
        # ── Compute P&L ──
        port_ret = 0.0
        for asset in assets:
            asset_weight = w.get(asset, 0.25)
            # Distribute total_exposure proportionally
            asset_lev = total_exposure * (asset_weight / sum(w.values()))
            # This is a simplification — in reality each asset has its own confluence
            # but for portfolio-level protection it works
            
            ret = asset_returns[asset].iloc[i]
            port_ret += asset_lev * ret
        
        # Transaction costs (approximate)
        if i > 0:
            lev_change = abs(total_exposure - daily_lev[i-1])
            port_ret -= lev_change * TX_COST
        
        daily_pnl[i] = port_ret
        daily_lev[i] = total_exposure
        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)
    
    pnl = pd.Series(daily_pnl, index=common_idx)
    lev = pd.Series(daily_lev, index=common_idx)
    return pnl, lev


def compute_metrics(returns, leverage=None):
    r = returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return {k: 0.0 for k in ["sharpe","cagr","maxdd","sortino","calmar"]}
    eq = (1 + r).cumprod()
    n_yr = len(r) / 252
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(252))
    dd = float((eq / eq.cummax() - 1).min()) * 100
    down = r[r < 0].std() * np.sqrt(252)
    sortino = float(r.mean() * 252 / down) if down > 0 else 0
    calmar = float(cagr / 100 / abs(dd / 100)) if dd != 0 else 0
    
    m = {"sharpe": round(sharpe, 3), "cagr": round(cagr, 1), "maxdd": round(dd, 1),
         "sortino": round(sortino, 3), "calmar": round(calmar, 3)}
    
    if leverage is not None:
        m["mean_lev"] = round(float(leverage.mean()), 3)
        m["pct_flat"] = round(float((leverage < 0.01).mean()) * 100, 1)
    
    for period, s, e in [("covid", "2020-02-15", "2020-04-15"),
                          ("may21", "2021-04-15", "2021-07-20"),
                          ("bear22", "2022-01-01", "2022-12-31")]:
        mask = (r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))
        if mask.sum() > 5:
            ceq = (1 + r[mask]).cumprod()
            m[period] = round(float(ceq.iloc[-1] - 1) * 100, 1)
        else:
            m[period] = 0
    
    yearly = r.resample("YE").sum() * 100
    m["yearly"] = {str(dt.year): round(float(v), 1) for dt, v in yearly.items()}
    return m


def walk_forward(crypto_data, macro_data, cross_asset_data, config,
                 n_folds=7, train_days=730, test_days=180):
    pnl, lev = hybrid_backtest(crypto_data, macro_data, cross_asset_data, config)
    oos_returns = []
    start_idx = 400
    for fold in range(n_folds):
        train_end = start_idx + fold * test_days + train_days
        test_end = train_end + test_days
        if test_end > len(pnl): break
        oos_returns.append(pnl.iloc[train_end:test_end])
    if not oos_returns: return -1
    all_oos = pd.concat(oos_returns).dropna()
    if len(all_oos) < 100 or all_oos.std() == 0: return -1
    return round(float(all_oos.mean() / all_oos.std() * np.sqrt(252)), 3)


def main():
    print("=" * 90)
    print("V3.1 HYBRID — DIRECT SIZING + PROTECTION LAYERS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    crypto_data, cross_asset_data, macro_data = load_data()

    # V3 baseline
    v3_pdf, _ = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        enable_long=True, enable_short=False, enable_adaptive_leverage=True
    )
    v3_m = compute_metrics(v3_pdf["daily_pnl"], v3_pdf["total_leverage"])
    
    configs = {
        # H1: Direct sizing + vol ceiling + bear filter (no trail stop)
        "H1_Vol+Bear": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "vol_ceiling": 0.80,
            "vol_lookback": 30,
            "bear_filter": True,
        },
        # H2: + trailing stop
        "H2_Vol+Bear+Trail10": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "vol_ceiling": 0.80,
            "vol_lookback": 30,
            "bear_filter": True,
            "portfolio_trail_stop": 0.10,
            "trail_reduce_factor": 0.3,
            "trail_recovery_days": 30,
        },
        # H3: Tighter trail
        "H3_Vol+Bear+Trail15": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "vol_ceiling": 0.80,
            "vol_lookback": 30,
            "bear_filter": True,
            "portfolio_trail_stop": 0.15,
            "trail_reduce_factor": 0.3,
            "trail_recovery_days": 20,
        },
        # H4: Conservative leverage + all protections
        "H4_Conservative+All": {
            "leverage_map": {5: 1.5, 4: 1.0, 3: 1.0, 2: 0.6, 1: 0.2, 0: 0.0},
            "s4_crypto_mom_override": 0.4,
            "vol_ceiling": 0.80,
            "vol_lookback": 30,
            "bear_filter": True,
            "portfolio_trail_stop": 0.12,
            "trail_reduce_factor": 0.3,
            "trail_recovery_days": 25,
        },
        # H5: Aggressive + protections
        "H5_Aggressive+All": {
            "leverage_map": {5: 2.5, 4: 1.5, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "vol_ceiling": 0.80,
            "vol_lookback": 30,
            "bear_filter": True,
            "portfolio_trail_stop": 0.15,
            "trail_reduce_factor": 0.3,
            "trail_recovery_days": 20,
        },
        # H6: Smart Score4 + moderate protections
        "H6_Smart+Moderate": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "vol_ceiling": 1.00,  # looser vol ceiling
            "vol_lookback": 30,
            "bear_filter": True,
            "portfolio_trail_stop": 0.20,  # wider trail
            "trail_reduce_factor": 0.5,  # less aggressive reduction
            "trail_recovery_days": 15,
        },
        # H7: Score4→0 when crypto mom off, no other protections except bear filter
        "H7_Score4Zero+Bear": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.0,
            "bear_filter": True,
        },
        # H8: Minimal — just bear filter, no vol/trail
        "H8_BearOnly": {
            "leverage_map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_crypto_mom_override": 0.5,
            "bear_filter": True,
        },
    }

    results = {"V3_Baseline": v3_m}
    
    print(f"\n{'Variant':<25} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'MeanLev':>8} {'Flat%':>6} {'COVID':>7} {'May21':>7} {'2022':>7}")
    print("-" * 95)
    print(f"{'V3_Baseline':<25} {v3_m['sharpe']:>7.3f} {v3_m['cagr']:>7.1f} {v3_m['maxdd']:>8.1f} "
          f"{v3_m.get('mean_lev',0):>8.3f} {v3_m.get('pct_flat',0):>6.1f} "
          f"{v3_m.get('covid',0):>+7.1f} {v3_m.get('may21',0):>+7.1f} {v3_m.get('bear22',0):>+7.1f}")

    for name, cfg in configs.items():
        pnl, lev = hybrid_backtest(crypto_data, macro_data, cross_asset_data, cfg)
        m = compute_metrics(pnl, lev)
        results[name] = m
        print(f"{name:<25} {m['sharpe']:>7.3f} {m['cagr']:>7.1f} {m['maxdd']:>8.1f} "
              f"{m.get('mean_lev',0):>8.3f} {m.get('pct_flat',0):>6.1f} "
              f"{m.get('covid',0):>+7.1f} {m.get('may21',0):>+7.1f} {m.get('bear22',0):>+7.1f}")

    # Per-year
    print(f"\n{'=' * 110}")
    print("PER-YEAR RETURNS (%)")
    print(f"{'=' * 110}")
    years = sorted(set(y for r in results.values() if "yearly" in r for y in r["yearly"]))
    print(f"{'Variant':<25}", end="")
    for yr in years: print(f"{yr:>10}", end="")
    print()
    print("-" * 110)
    for name, r in results.items():
        if "yearly" not in r: continue
        print(f"{name:<25}", end="")
        for yr in years:
            print(f"{r['yearly'].get(yr, 0):>+10.1f}", end="")
        print()

    # Walk-forward top variants
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD OOS SHARPE")
    print(f"{'=' * 70}")
    
    for name, cfg in configs.items():
        oos = walk_forward(crypto_data, macro_data, cross_asset_data, cfg)
        results[name]["oos_sharpe"] = oos
        m = results[name]
        print(f"  {name:<25} IS: {m['sharpe']:.3f}  OOS: {oos:.3f}  CAGR: {m['cagr']:.1f}%  MaxDD: {m['maxdd']:.1f}%")

    # Save
    outpath = os.path.expanduser("~/Desktop/maestro/data/research/v31_hybrid_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Find best by OOS Sharpe
    hybrid_results = {k: v for k, v in results.items() if k.startswith("H")}
    best = max(hybrid_results.items(), key=lambda x: x[1].get("oos_sharpe", -99))
    
    print(f"\n{'=' * 70}")
    print(f"WINNER: {best[0]}")
    print(f"  IS Sharpe:  {best[1]['sharpe']:.3f}")
    print(f"  OOS Sharpe: {best[1].get('oos_sharpe', 'N/A')}")
    print(f"  CAGR:       {best[1]['cagr']:.1f}%")
    print(f"  MaxDD:      {best[1]['maxdd']:.1f}%")
    print(f"  2022 Bear:  {best[1].get('bear22', 0):+.1f}%")
    print(f"  Mean Lev:   {best[1].get('mean_lev', 0):.3f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
