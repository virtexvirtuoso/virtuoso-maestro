"""
V3.1 Direct Sizing — bypass the dip-buying entry logic entirely.
Confluence score directly sets portfolio exposure.
Compare against V3 baseline and various configurations.
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
    _realized_vol, TX_COST, DEFAULT_SAFETY_PARAMS,
    run_full_strategy
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


def direct_sizing_backtest(crypto_data, macro_data, cross_asset_data,
                           leverage_map, label, weights=None,
                           score4_crypto_mom_override=None,
                           vol_ceiling=0.80, dd_limit=0.15,
                           trail_stop=None):
    """
    Direct sizing: confluence score → portfolio exposure.
    No dip-buying logic. No RSI filters. No pyramiding.
    Just: what does the confluence say → that's your position.
    
    Optional: trail_stop applies a portfolio-level trailing stop.
    Optional: score4_crypto_mom_override reduces leverage when score=4 
              and crypto momentum is the dissenting signal.
    """
    w = weights or WEIGHTS
    
    # Get common index
    assets = list(crypto_data.keys())
    common_idx = crypto_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(crypto_data[a].index)
    
    portfolio_pnl = pd.Series(0.0, index=common_idx)
    portfolio_lev = pd.Series(0.0, index=common_idx)
    portfolio_regime = None
    portfolio_confluence = None
    
    for asset in assets:
        df = crypto_data[asset].reindex(common_idx)
        close = df["close"]
        ret = close.pct_change().fillna(0)
        
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        
        # Compute confluence
        confluence, breakdown = compute_confluence(
            close, macro_data, cross_asset_data,
            sma_slow=cfg["sma_slow"],
            momentum_period=cfg["momentum_period"],
        )
        confluence = confluence.reindex(common_idx).fillna(0)
        breakdown = breakdown.reindex(common_idx).fillna(0)
        
        if asset == "BTC":
            portfolio_regime = detect_regime(confluence)
            portfolio_confluence = confluence
        
        # Base leverage from map
        lev = confluence.map(lambda c: leverage_map.get(min(int(c), 5), 0.0))
        
        # Score 4 override: if crypto momentum is the dissenter, reduce
        if score4_crypto_mom_override is not None:
            is_score4 = confluence == 4
            crypto_mom_off = breakdown["crypto_momentum"] == 0
            override_mask = is_score4 & crypto_mom_off
            lev = lev.where(~override_mask, score4_crypto_mom_override)
        
        # Vol ceiling
        rvol = _realized_vol(close, 30)
        high_vol = rvol > vol_ceiling
        lev = lev.where(~high_vol, lev * 0.5)
        
        # Shift leverage by 1 (no lookahead on confluence)
        lev = lev.shift(1).fillna(0)
        
        # Position = weight * leverage
        position = w.get(asset, 0.25) * lev
        
        # Transaction costs
        pos_change = position.diff().abs().fillna(0)
        costs = pos_change * TX_COST
        
        # P&L
        asset_pnl = position * ret - costs
        portfolio_pnl += asset_pnl
        portfolio_lev += position.abs()
    
    # Portfolio-level trailing stop
    if trail_stop is not None:
        eq = (1 + portfolio_pnl).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1
        
        # When DD exceeds limit, reduce to 50% for rest of that drawdown
        in_deep_dd = dd < -trail_stop
        # Simple: zero out positions during deep DD
        portfolio_pnl_adj = portfolio_pnl.copy()
        portfolio_pnl_adj[in_deep_dd] *= 0.5
        portfolio_pnl = portfolio_pnl_adj
    
    # DD limit: hard stop
    eq = (1 + portfolio_pnl).cumprod()
    
    return portfolio_pnl, portfolio_lev, portfolio_regime, portfolio_confluence


def compute_metrics(returns, leverage=None):
    r = returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return {k: 0.0 for k in ["sharpe","cagr","maxdd","sortino","calmar","mean_lev","pct_flat"]}
    eq = (1 + r).cumprod()
    n_yr = len(r) / 252
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(252))
    dd = float((eq / eq.cummax() - 1).min()) * 100
    down = r[r < 0].std() * np.sqrt(252)
    sortino = float(r.mean() * 252 / down) if down > 0 else 0
    calmar = float(cagr / 100 / abs(dd / 100)) if dd != 0 else 0
    
    m = {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 1),
        "maxdd": round(dd, 1),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
    }
    
    if leverage is not None:
        lev = leverage.reindex(r.index).fillna(0)
        m["mean_lev"] = round(float(lev.mean()), 3)
        m["pct_flat"] = round(float((lev < 0.01).mean()) * 100, 1)
    
    # Crash periods
    for period, s, e in [("covid", "2020-02-15", "2020-04-15"),
                          ("may21", "2021-04-15", "2021-07-20"),
                          ("bear22", "2022-01-01", "2022-12-31")]:
        mask = (r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))
        if mask.sum() > 5:
            ceq = (1 + r[mask]).cumprod()
            m[period] = round(float(ceq.iloc[-1] - 1) * 100, 1)
        else:
            m[period] = 0
    
    # Per-year
    yearly = r.resample("YE").sum() * 100
    m["yearly"] = {str(dt.year): round(float(v), 1) for dt, v in yearly.items()}
    
    return m


def walk_forward(crypto_data, macro_data, cross_asset_data,
                 leverage_map, n_folds=7, train_days=730, test_days=180,
                 score4_crypto_mom_override=None):
    """Walk-forward OOS Sharpe for direct sizing."""
    pnl, lev, _, _ = direct_sizing_backtest(
        crypto_data, macro_data, cross_asset_data,
        leverage_map=leverage_map,
        label="wf",
        score4_crypto_mom_override=score4_crypto_mom_override,
    )
    
    oos_returns = []
    start_idx = 400
    for fold in range(n_folds):
        train_end = start_idx + fold * test_days + train_days
        test_end = train_end + test_days
        if test_end > len(pnl):
            break
        oos = pnl.iloc[train_end:test_end]
        oos_returns.append(oos)
    
    if not oos_returns:
        return -1
    all_oos = pd.concat(oos_returns).dropna()
    if len(all_oos) < 100 or all_oos.std() == 0:
        return -1
    return round(float(all_oos.mean() / all_oos.std() * np.sqrt(252)), 3)


def main():
    print("=" * 90)
    print("V3.1 DIRECT SIZING — BYPASS DIP-BUYING LOGIC")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    crypto_data, cross_asset_data, macro_data = load_data()
    print(f"Assets: {list(crypto_data.keys())}")

    # ── V3 Baseline for comparison ──
    print("\n── V3 BASELINE (original dip-buying logic) ──")
    v3_pdf, _ = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        enable_long=True, enable_short=False, enable_adaptive_leverage=True
    )
    v3_m = compute_metrics(v3_pdf["daily_pnl"], v3_pdf["total_leverage"])
    print(f"  V3 S4: Sharpe {v3_m['sharpe']:.3f}, CAGR {v3_m['cagr']:.1f}%, MaxDD {v3_m['maxdd']:.1f}%, "
          f"MeanLev {v3_m['mean_lev']:.3f}, Flat {v3_m['pct_flat']:.0f}%")

    # ── Direct Sizing Variants ──
    variants = {
        # Original leverage map, direct sizing
        "DS_Original_Map": {
            "map": {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0},
        },
        # Score 4 demoted
        "DS_Score4_Demoted": {
            "map": {5: 2.0, 4: 0.6, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0},
        },
        # Score 4 demoted ONLY when crypto mom is off
        "DS_Score4_CryptoMom": {
            "map": {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0},
            "s4_override": 0.6,
        },
        # Conservative
        "DS_Conservative": {
            "map": {5: 1.5, 4: 1.0, 3: 0.8, 2: 0.5, 1: 0.2, 0: 0.0},
        },
        # Aggressive
        "DS_Aggressive": {
            "map": {5: 2.5, 4: 1.5, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.0},
        },
        # Score4 smart: demote crypto-mom-off, keep others
        "DS_Smart_Score4": {
            "map": {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
            "s4_override": 0.5,
        },
        # Bump low scores, demote score 4
        "DS_Balanced": {
            "map": {5: 2.0, 4: 0.8, 3: 1.2, 2: 0.8, 1: 0.5, 0: 0.0},
        },
        # Maximum capital efficiency
        "DS_Max_Efficiency": {
            "map": {5: 2.0, 4: 1.0, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.3},
        },
    }

    results = {"V3_S4_baseline": v3_m}

    print(f"\n── DIRECT SIZING VARIANTS ──")
    for name, cfg in variants.items():
        pnl, lev, regime, conf = direct_sizing_backtest(
            crypto_data, macro_data, cross_asset_data,
            leverage_map=cfg["map"],
            label=name,
            score4_crypto_mom_override=cfg.get("s4_override"),
        )
        m = compute_metrics(pnl, lev)
        results[name] = m
        print(f"  {name}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%, MaxDD {m['maxdd']:.1f}%, "
              f"MeanLev {m.get('mean_lev',0):.3f}, Flat {m.get('pct_flat',0):.0f}%")

    # ── Comparison Table ──
    print(f"\n{'=' * 130}")
    print("FULL COMPARISON")
    print(f"{'=' * 130}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'Sortino':>8} {'Calmar':>7} "
          f"{'MeanLev':>8} {'Flat%':>6} {'COVID':>7} {'May21':>7} {'2022':>7}")
    print("-" * 130)
    
    for name, r in results.items():
        print(f"{name:<25} {r['sharpe']:>7.3f} {r['cagr']:>7.1f} {r['maxdd']:>8.1f} "
              f"{r.get('sortino',0):>8.3f} {r.get('calmar',0):>7.3f} "
              f"{r.get('mean_lev',0):>8.3f} {r.get('pct_flat',0):>6.1f} "
              f"{r.get('covid',0):>+7.1f} {r.get('may21',0):>+7.1f} {r.get('bear22',0):>+7.1f}")

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
            v = r["yearly"].get(yr, 0)
            print(f"{v:>+10.1f}", end="")
        print()

    # ── Walk-Forward for top 3 ──
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD OOS SHARPE (Top variants)")
    print(f"{'=' * 70}")
    
    # Pick top 3 by IS Sharpe (excluding baseline)
    ds_results = {k: v for k, v in results.items() if k.startswith("DS_")}
    top3 = sorted(ds_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)[:4]
    
    for name, r in top3:
        cfg = variants[name]
        oos = walk_forward(crypto_data, macro_data, cross_asset_data,
                          leverage_map=cfg["map"],
                          score4_crypto_mom_override=cfg.get("s4_override"))
        r["oos_sharpe"] = oos
        print(f"  {name}: IS Sharpe {r['sharpe']:.3f}, OOS Sharpe {oos:.3f}, CAGR {r['cagr']:.1f}%")

    # Save
    outpath = os.path.expanduser("~/Desktop/maestro/data/research/v31_direct_sizing_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Winner
    best_ds = max([(n, r) for n, r in ds_results.items()],
                  key=lambda x: x[1].get("oos_sharpe", x[1]["sharpe"]))
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED: {best_ds[0]}")
    print(f"  Sharpe: {best_ds[1]['sharpe']:.3f}, CAGR: {best_ds[1]['cagr']:.1f}%, MaxDD: {best_ds[1]['maxdd']:.1f}%")
    if "oos_sharpe" in best_ds[1]:
        print(f"  OOS Sharpe: {best_ds[1]['oos_sharpe']:.3f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
