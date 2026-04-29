"""
Portfolio Expansion V2 — Test adding RENDER, BNB, DOGE, XRP, ADA, ZEC to V3.1-H2 core 4.
7 configurations, 13-fold walk-forward validation.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    compute_confluence, detect_regime, ASSET_CONFIGS, TX_COST,
)
from strategies.composite.mega_strategy_v31 import (
    LEVERAGE_MAP, SCORE4_CRYPTO_MOM_OVERRIDE,
    VOL_CEILING, VOL_LOOKBACK, BEAR_FILTER_DAYS,
    PORTFOLIO_TRAIL_STOP, TRAIL_REDUCE_FACTOR,
    TRAIL_RECOVERY_DAYS, TRAIL_RECOVERY_THRESHOLD,
)

# ── Data mappings (same as backtest_mega_v31.py) ──
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}

ALL_CRYPTO = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD",
    "RENDER": "RNDR-USD", "BNB": "BNB-USD", "DOGE": "DOGE-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "ZEC": "ZEC-USD",
}

FOLDS = [
    ("2021-04-10","2021-08-13"), ("2021-08-14","2021-12-17"), ("2021-12-18","2022-04-22"),
    ("2022-04-23","2022-08-26"), ("2022-08-27","2022-12-30"), ("2022-12-31","2023-05-05"),
    ("2023-05-06","2023-09-08"), ("2023-09-09","2024-01-12"), ("2024-01-13","2024-05-17"),
    ("2024-05-18","2024-09-20"), ("2024-09-21","2025-01-24"), ("2025-01-25","2025-05-30"),
    ("2025-05-31","2025-10-03"),
]

CONFIGS = {
    "C1: Original 4": {
        "assets": ["BTC","ETH","SOL","LINK"],
        "weights": {"BTC":0.40,"ETH":0.25,"SOL":0.20,"LINK":0.15},
        "momentum_tilt": False,
    },
    "C2: Core4+RENDER+BNB": {
        "assets": ["BTC","ETH","SOL","LINK","RENDER","BNB"],
        "weights": {"BTC":0.30,"ETH":0.18,"SOL":0.14,"LINK":0.10,"RENDER":0.14,"BNB":0.14},
        "momentum_tilt": False,
    },
    "C3: Core4+DOGE+XRP+ADA": {
        "assets": ["BTC","ETH","SOL","LINK","DOGE","XRP","ADA"],
        "weights": {"BTC":0.30,"ETH":0.15,"SOL":0.12,"LINK":0.08,"DOGE":0.12,"XRP":0.12,"ADA":0.11},
        "momentum_tilt": False,
    },
    "C4: Full 10-asset": {
        "assets": ["BTC","ETH","SOL","LINK","RENDER","BNB","DOGE","XRP","ADA","ZEC"],
        "weights": {"BTC":0.25,"ETH":0.12,"SOL":0.10,"LINK":0.06,"RENDER":0.10,"BNB":0.10,"DOGE":0.08,"XRP":0.08,"ADA":0.06,"ZEC":0.05},
        "momentum_tilt": False,
    },
    "C5: Equal weight 10": {
        "assets": ["BTC","ETH","SOL","LINK","RENDER","BNB","DOGE","XRP","ADA","ZEC"],
        "weights": {a: 0.10 for a in ["BTC","ETH","SOL","LINK","RENDER","BNB","DOGE","XRP","ADA","ZEC"]},
        "momentum_tilt": False,
    },
    "C6: Momentum-tilted 10": {
        "assets": ["BTC","ETH","SOL","LINK","RENDER","BNB","DOGE","XRP","ADA","ZEC"],
        "weights": {"BTC":0.25,"ETH":0.12,"SOL":0.10,"LINK":0.06,"RENDER":0.10,"BNB":0.10,"DOGE":0.08,"XRP":0.08,"ADA":0.06,"ZEC":0.05},
        "momentum_tilt": True,
    },
    "C7: Top6 by Sharpe": {
        "assets": ["BTC","ETH","RENDER","BNB","DOGE","ZEC"],
        "weights": {"BTC":0.30,"ETH":0.20,"RENDER":0.15,"BNB":0.15,"DOGE":0.10,"ZEC":0.10},
        "momentum_tilt": False,
    },
}


def load_data():
    print("Loading data...")
    stock = StockDataLoader()
    fred = MacroDataLoader()

    crypto_data = {}
    for name, ticker in ALL_CRYPTO.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
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
    print(f"  Macro: {len(macro_data)} rows, cols: {list(macro_data.columns)}")
    return crypto_data, cross_asset_data, macro_data


def run_portfolio_v31(
    asset_data, macro_data, cross_asset_data, assets, weights,
    momentum_tilt=False, start_date=None, end_date=None,
):
    """Run V3.1-H2 strategy for a portfolio config."""
    # Common index
    common_idx = asset_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(asset_data[a].index)

    if start_date:
        common_idx = common_idx[common_idx >= pd.Timestamp(start_date)]
    if end_date:
        common_idx = common_idx[common_idx <= pd.Timestamp(end_date)]

    if len(common_idx) < 30:
        return None, None

    # Pre-compute per-asset confluence + leverage
    per_asset_lev = {}
    per_asset_ret = {}
    per_asset_close = {}
    per_asset_conf = {}

    for asset in assets:
        df = asset_data[asset].reindex(common_idx)
        close = df["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])

        conf, bd = compute_confluence(
            close, macro_data, cross_asset_data,
            sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"]
        )
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)

        base_lev = conf.map(lambda c: LEVERAGE_MAP.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), SCORE4_CRYPTO_MOM_OVERRIDE)
        base_lev = base_lev.shift(1).fillna(0)

        per_asset_lev[asset] = base_lev
        per_asset_ret[asset] = close.pct_change().reindex(common_idx).fillna(0)
        per_asset_close[asset] = close
        per_asset_conf[asset] = conf

    btc_conf = per_asset_conf.get("BTC", per_asset_conf[assets[0]])

    # Momentum tilt: rebalance weights monthly by 60d Sharpe
    if momentum_tilt:
        active_weights = _compute_momentum_weights(per_asset_ret, weights, common_idx)
    else:
        active_weights = None

    # Day-by-day simulation
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    total_leverage = np.zeros(n)
    per_asset_contrib = {a: np.zeros(n) for a in assets}

    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0
    prev_exposure = 0.0

    for i in range(1, n):
        # Get weights for this day
        if active_weights is not None:
            w = active_weights.iloc[i]
        else:
            w = weights

        # Base exposure
        target_exposure = 0.0
        for asset in assets:
            target_exposure += w[asset] * per_asset_lev[asset].iloc[i]

        # Vol ceiling
        btc_close = per_asset_close.get("BTC", per_asset_close[assets[0]])
        if i >= VOL_LOOKBACK:
            window = btc_close.iloc[max(0, i - VOL_LOOKBACK):i]
            rvol = window.pct_change().std() * np.sqrt(252)
            if rvol > VOL_CEILING:
                target_exposure *= 0.5

        # Bear filter
        btc_score = int(btc_conf.iloc[i - 1])
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0
        if consecutive_low >= BEAR_FILTER_DAYS:
            target_exposure = 0.0

        # Portfolio trailing stop
        dd = 1 - equity / peak_eq
        if dd > PORTFOLIO_TRAIL_STOP:
            if not in_drawdown:
                in_drawdown = True
                dd_start_idx = i
            target_exposure *= TRAIL_REDUCE_FACTOR
        elif in_drawdown:
            days_since = i - dd_start_idx
            if days_since > TRAIL_RECOVERY_DAYS and equity > peak_eq * TRAIL_RECOVERY_THRESHOLD:
                in_drawdown = False

        # PnL
        port_ret = 0.0
        weight_sum = sum(w[a] for a in assets)
        for asset in assets:
            asset_exposure = target_exposure * (w[asset] / weight_sum) if weight_sum > 0 else 0
            ret = per_asset_ret[asset].iloc[i]
            contrib = asset_exposure * ret
            port_ret += contrib
            per_asset_contrib[asset][i] = contrib

        # Transaction costs
        lev_change = abs(target_exposure - prev_exposure)
        port_ret -= lev_change * TX_COST
        prev_exposure = target_exposure

        daily_pnl[i] = port_ret
        total_leverage[i] = target_exposure
        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)

    portfolio_df = pd.DataFrame({
        "daily_pnl": daily_pnl,
        "equity": np.cumprod(1 + daily_pnl),
        "total_leverage": total_leverage,
    }, index=common_idx)

    contrib_df = pd.DataFrame(per_asset_contrib, index=common_idx)

    return portfolio_df, contrib_df


def _compute_momentum_weights(per_asset_ret, base_weights, common_idx):
    """Monthly rebalance by 60d Sharpe, 0-2x scaling of base weights."""
    assets = list(base_weights.keys())
    result = pd.DataFrame(index=common_idx, columns=assets, dtype=float)

    # Fill with base weights initially
    for a in assets:
        result[a] = base_weights[a]

    # Rebalance monthly
    months = pd.Series(common_idx).dt.to_period("M").unique()
    for m in months:
        mask = pd.Series(common_idx).dt.to_period("M") == m
        month_idx = common_idx[mask.values]
        if len(month_idx) == 0:
            continue

        first_day = month_idx[0]
        loc = common_idx.get_loc(first_day)
        if loc < 60:
            continue

        # Compute 60d Sharpe for each asset
        sharpes = {}
        for a in assets:
            window = per_asset_ret[a].iloc[loc-60:loc]
            if window.std() > 0:
                sharpes[a] = float(window.mean() / window.std() * np.sqrt(252))
            else:
                sharpes[a] = 0.0

        # Scale: map Sharpe to 0-2x multiplier (Sharpe 0 -> 0.5x, Sharpe 2 -> 2x, linear clamp)
        scaled = {}
        for a in assets:
            mult = np.clip(0.5 + sharpes[a] * 0.75, 0.0, 2.0)
            scaled[a] = base_weights[a] * mult

        # Normalize to sum to 1
        total = sum(scaled.values())
        if total > 0:
            for a in assets:
                result.loc[month_idx, a] = scaled[a] / total
        else:
            for a in assets:
                result.loc[month_idx, a] = 1.0 / len(assets)

    return result.astype(float)


def compute_metrics(returns):
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["cagr","sharpe","sortino","max_dd","total_return"]}
    eq = (1 + returns).cumprod()
    n_yr = len(returns) / 252
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
    }


def yearly_returns(returns):
    yearly = {}
    for yr, grp in returns.groupby(returns.index.year):
        eq = (1 + grp).cumprod()
        yearly[str(yr)] = round(float(eq.iloc[-1] - 1) * 100, 2)
    return yearly


def walk_forward(portfolio_fn, asset_data, macro_data, cross_asset_data, config):
    """13-fold non-overlapping walk-forward."""
    assets = config["assets"]
    weights = config["weights"]
    momentum_tilt = config["momentum_tilt"]

    # Find earliest date when all assets have data
    earliest = max(asset_data[a].index[0] for a in assets)

    fold_results = []
    fold_sharpes = []

    for fold_idx, (start, end) in enumerate(FOLDS):
        if pd.Timestamp(start) < earliest:
            continue

        # IS: all data before fold start
        is_end = pd.Timestamp(start) - pd.Timedelta(days=1)
        is_port, _ = run_portfolio_v31(
            asset_data, macro_data, cross_asset_data, assets, weights,
            momentum_tilt=momentum_tilt, end_date=is_end,
        )

        # OOS: fold period
        oos_port, _ = run_portfolio_v31(
            asset_data, macro_data, cross_asset_data, assets, weights,
            momentum_tilt=momentum_tilt, start_date=start, end_date=end,
        )

        if oos_port is None or len(oos_port) < 20:
            continue

        oos_ret = oos_port["daily_pnl"]
        oos_metrics = compute_metrics(oos_ret)

        fold_results.append({
            "fold": fold_idx + 1,
            "period": f"{start} to {end}",
            "days": len(oos_ret),
            "sharpe": oos_metrics["sharpe"],
            "return_pct": oos_metrics["total_return"],
            "max_dd": oos_metrics["max_dd"],
        })
        fold_sharpes.append(oos_metrics["sharpe"])

    if len(fold_sharpes) < 3:
        return {"n_folds": len(fold_sharpes), "mean_oos_sharpe": 0, "p_value": 1, "folds": fold_results}

    mean_s = np.mean(fold_sharpes)
    std_s = np.std(fold_sharpes, ddof=1)
    t_stat = mean_s / (std_s / np.sqrt(len(fold_sharpes))) if std_s > 0 else 0
    p_value = float(1 - stats.t.cdf(t_stat, df=len(fold_sharpes) - 1))
    positive = sum(1 for s in fold_sharpes if s > 0)

    return {
        "n_folds": len(fold_sharpes),
        "positive_folds": f"{positive}/{len(fold_sharpes)}",
        "mean_oos_sharpe": round(mean_s, 3),
        "median_oos_sharpe": round(float(np.median(fold_sharpes)), 3),
        "std_oos_sharpe": round(std_s, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
        "folds": fold_results,
    }


def main():
    print("\n" + "#" * 70)
    print("#  Portfolio Expansion V2 — V3.1-H2 with additional assets")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    crypto_data, cross_asset_data, macro_data = load_data()

    results = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  {cfg_name}")
        print(f"  Assets: {cfg['assets']}")
        print(f"  Weights: {cfg['weights']}")
        print(f"{'='*70}")

        # Check all assets available
        missing = [a for a in cfg["assets"] if a not in crypto_data]
        if missing:
            print(f"  SKIPPED — missing: {missing}")
            continue

        # Full period run
        port_df, contrib_df = run_portfolio_v31(
            crypto_data, macro_data, cross_asset_data,
            cfg["assets"], cfg["weights"],
            momentum_tilt=cfg["momentum_tilt"],
            end_date="2026-02-01",
        )

        if port_df is None:
            print("  SKIPPED — insufficient data")
            continue

        is_metrics = compute_metrics(port_df["daily_pnl"])
        yr_ret = yearly_returns(port_df["daily_pnl"])

        print(f"  IS: Sharpe={is_metrics['sharpe']:.3f}  CAGR={is_metrics['cagr']:.1f}%  "
              f"MaxDD={is_metrics['max_dd']:.1f}%  Sortino={is_metrics['sortino']:.3f}")
        print(f"  Yearly: {yr_ret}")

        # Asset contribution correlation
        if contrib_df is not None and len(contrib_df) > 100:
            corr = contrib_df.corr()
            corr_dict = {f"{a1}-{a2}": round(float(corr.loc[a1,a2]), 3)
                         for i, a1 in enumerate(cfg["assets"])
                         for a2 in cfg["assets"][i+1:]}
        else:
            corr_dict = {}

        # Walk-forward
        print("  Running walk-forward...")
        wf = walk_forward(run_portfolio_v31, crypto_data, macro_data, cross_asset_data, cfg)

        print(f"  OOS: mean Sharpe={wf['mean_oos_sharpe']:.3f}  "
              f"median={wf.get('median_oos_sharpe',0):.3f}  "
              f"p={wf['p_value']:.4f}  positive={wf.get('positive_folds','?')}")

        results[cfg_name] = {
            "is_metrics": is_metrics,
            "yearly_returns": yr_ret,
            "walk_forward": wf,
            "asset_contrib_corr": corr_dict,
            "assets": cfg["assets"],
            "weights": cfg["weights"],
        }

    # Save
    out_dir = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "portfolio_expansion_v2_results.json"

    save_data = {"timestamp": datetime.now().isoformat(), "configs": results}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Ranked summary
    print("\n" + "=" * 100)
    print("RANKED SUMMARY BY OOS SHARPE")
    print("=" * 100)
    ranked = sorted(results.items(), key=lambda x: x[1]["walk_forward"]["mean_oos_sharpe"], reverse=True)
    print(f"{'Rank':<5} {'Config':<30} {'OOS Sharpe':>10} {'OOS Med':>8} {'p-val':>7} "
          f"{'IS Sharpe':>10} {'IS CAGR%':>9} {'IS MaxDD%':>10} {'Pos Folds':>10}")
    print("-" * 100)
    for rank, (name, r) in enumerate(ranked, 1):
        wf = r["walk_forward"]
        m = r["is_metrics"]
        print(f"{rank:<5} {name:<30} {wf['mean_oos_sharpe']:>10.3f} {wf.get('median_oos_sharpe',0):>8.3f} "
              f"{wf['p_value']:>7.4f} {m['sharpe']:>10.3f} {m['cagr']:>9.1f} {m['max_dd']:>10.1f} "
              f"{wf.get('positive_folds','?'):>10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
