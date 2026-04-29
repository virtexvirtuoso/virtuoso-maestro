"""
Portfolio Expansion Backtest: Add UNI, DOGE, ADA to V3.1-H2
Tests 7 portfolio configurations with 13-fold walk-forward validation.
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
from strategies.composite.mega_strategy_v31 import run_full_strategy, LEVERAGE_MAP
from strategies.composite.mega_strategy_v3 import ASSET_CONFIGS

# ── Configuration ─────────────────────────────────────────────────

CRYPTO_TICKERS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD",
    "UNI": "UNI1-USD", "DOGE": "DOGE-USD", "ADA": "ADA-USD",
}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}

DATE_START = "2020-09-01"
DATE_END = "2026-02-01"
TX_COST = 0.001

# Asset configs for new assets (use BTC defaults with adjusted params)
for asset in ["UNI", "DOGE", "ADA"]:
    if asset not in ASSET_CONFIGS:
        ASSET_CONFIGS[asset] = dict(
            sma_slow=100, momentum_period=35, rsi_entry=52,
            trail_stop_pct=0.12, max_position=1.50, initial_size=0.60,
            rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
            roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
            atr_exit_mult=2.0, macro_weight=1.0,
        )

CONFIGS = {
    "Config1_Original4": {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15},
    "Config2_5Asset_UNI": {"BTC": 0.35, "ETH": 0.20, "SOL": 0.15, "LINK": 0.10, "UNI": 0.20},
    "Config3_6Asset_UNI_DOGE": {"BTC": 0.30, "ETH": 0.18, "SOL": 0.12, "LINK": 0.10, "UNI": 0.15, "DOGE": 0.15},
    "Config4_7Asset": {"BTC": 0.28, "ETH": 0.15, "SOL": 0.12, "LINK": 0.08, "UNI": 0.15, "DOGE": 0.12, "ADA": 0.10},
    "Config5_EqualWeight7": {a: 1/7 for a in ["BTC", "ETH", "SOL", "LINK", "UNI", "DOGE", "ADA"]},
    "Config6_Momentum": {"BTC": 0.28, "ETH": 0.15, "SOL": 0.12, "LINK": 0.08, "UNI": 0.15, "DOGE": 0.12, "ADA": 0.10},  # base, adjusted dynamically
    "Config7_BTCHeavy7": {"BTC": 0.40, "ETH": 0.15, "SOL": 0.10, "LINK": 0.05, "UNI": 0.12, "DOGE": 0.10, "ADA": 0.08},
}


def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    stock = StockDataLoader()
    fred = MacroDataLoader()

    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2019-01-01")
            # Filter to date range
            df = df[(df.index >= DATE_START) & (df.index <= DATE_END)]
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"  {name}: only {len(df)} days, skipping")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2019-01-01")
            cross_asset_data[col] = df["close"]
            print(f"  {col} ({ticker}): {len(df)} days")
        except Exception as e:
            print(f"  {col}: FAILED - {e}")

    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    print(f"  Macro: {len(macro_data)} days, columns: {list(macro_data.columns)}")
    return crypto_data, cross_asset_data, macro_data


def compute_metrics(returns):
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate","mean_leverage"]}
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
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0
    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
    }


def run_momentum_weighted(crypto_data, macro_data, cross_asset_data, base_weights):
    """Config 6: Momentum-weighted with 60d Sharpe scaling, rebalanced monthly."""
    assets = list(base_weights.keys())
    # Filter to available assets
    assets = [a for a in assets if a in crypto_data]
    
    # Get common index
    common_idx = crypto_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(crypto_data[a].index)
    
    # Pre-compute per-asset returns
    asset_returns = {}
    for a in assets:
        asset_returns[a] = crypto_data[a]["close"].reindex(common_idx).pct_change().fillna(0)
    
    # Run strategy with dynamic weights - rebalance monthly
    # First run with base weights to get confluence/leverage
    from strategies.composite.mega_strategy_v31 import run_full_strategy as _run
    from strategies.composite.mega_strategy_v3 import compute_confluence
    
    # Compute per-asset confluence and leverage
    per_asset_lev = {}
    per_asset_ret = {}
    for asset in assets:
        df = crypto_data[asset].reindex(common_idx)
        close = df["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        conf, bd = compute_confluence(close, macro_data, cross_asset_data,
                                       sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"])
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)
        base_lev = conf.map(lambda c: LEVERAGE_MAP.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), 0.5)
        base_lev = base_lev.shift(1).fillna(0)
        per_asset_lev[asset] = base_lev
        per_asset_ret[asset] = asset_returns[asset]
    
    # BTC confluence for regime/protection layers
    btc_close = crypto_data["BTC"]["close"].reindex(common_idx)
    btc_cfg = ASSET_CONFIGS["BTC"]
    btc_conf, _ = compute_confluence(btc_close, macro_data, cross_asset_data,
                                      sma_slow=btc_cfg["sma_slow"], momentum_period=btc_cfg["momentum_period"])
    btc_conf = btc_conf.reindex(common_idx).fillna(0)
    
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    total_leverage = np.zeros(n)
    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0
    prev_exposure = 0.0
    
    # Compute dynamic weights monthly
    current_weights = {a: base_weights.get(a, 1/len(assets)) for a in assets}
    last_rebalance = 0
    
    for i in range(1, n):
        # Monthly rebalance of weights based on 60d Sharpe
        if i - last_rebalance >= 21 and i >= 60:
            sharpe_scores = {}
            for a in assets:
                ret_window = per_asset_ret[a].iloc[max(0, i-60):i]
                if ret_window.std() > 0:
                    sharpe_scores[a] = float(ret_window.mean() / ret_window.std() * np.sqrt(252))
                else:
                    sharpe_scores[a] = 0.0
            
            # Scale base weights by momentum: 0-2x based on Sharpe rank
            new_weights = {}
            for a in assets:
                base_w = base_weights.get(a, 1/len(assets))
                # Normalize Sharpe to 0-2 scale
                s = sharpe_scores[a]
                scale = np.clip(0.5 + s / 2.0, 0.0, 2.0)  # Center at 1.0, range 0-2
                new_weights[a] = base_w * scale
            
            # Normalize to sum to 1
            total_w = sum(new_weights.values())
            if total_w > 0:
                current_weights = {a: w/total_w for a, w in new_weights.items()}
            last_rebalance = i
        
        # Target exposure from per-asset confluence
        target_exposure = 0.0
        for asset in assets:
            asset_lev = per_asset_lev[asset].iloc[i]
            target_exposure += current_weights.get(asset, 0) * asset_lev
        
        # Vol ceiling
        if i >= 30:
            window = btc_close.iloc[max(0, i-30):i]
            rvol = window.pct_change().std() * np.sqrt(252)
            if rvol > 0.80:
                target_exposure *= 0.5
        
        # Bear filter
        btc_score = int(btc_conf.iloc[i-1])
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0
        if consecutive_low >= 30:
            target_exposure = 0.0
        
        # Portfolio trailing stop
        dd = 1 - equity / peak_eq
        if dd > 0.10:
            if not in_drawdown:
                in_drawdown = True
                dd_start_idx = i
            target_exposure *= 0.3
        elif in_drawdown:
            days_since = i - dd_start_idx
            if days_since > 30 and equity > peak_eq * 0.95:
                in_drawdown = False
        
        # Compute P&L
        port_ret = 0.0
        for asset in assets:
            asset_exposure = target_exposure * current_weights.get(asset, 0)
            ret = per_asset_ret[asset].iloc[i]
            port_ret += asset_exposure * ret
        
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
    
    return portfolio_df


def walk_forward_13fold(returns, label=""):
    """13-fold non-overlapping 6-month walk-forward."""
    ret = returns.dropna()
    n = len(ret)
    
    test_size = 126  # ~6 months
    min_train = 365  # 1 year minimum
    
    folds = []
    fold_sharpes = []
    fold_returns = []
    
    for fold in range(13):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size
        if test_end > n:
            break
        
        test_ret = ret.iloc[train_end:test_end]
        if len(test_ret) < 50 or test_ret.std() == 0:
            continue
        
        fold_sharpe = float(test_ret.mean() / test_ret.std() * np.sqrt(252))
        fold_total = float((1 + test_ret).cumprod().iloc[-1] - 1) * 100
        
        start_date = ret.index[train_end].strftime("%Y-%m-%d")
        end_date = ret.index[min(test_end - 1, n - 1)].strftime("%Y-%m-%d")
        
        folds.append({
            "fold": fold + 1,
            "test_start": start_date,
            "test_end": end_date,
            "sharpe": round(fold_sharpe, 3),
            "return_pct": round(fold_total, 1),
            "days": len(test_ret),
        })
        fold_sharpes.append(fold_sharpe)
        fold_returns.append(fold_total)
    
    if len(fold_sharpes) < 3:
        return {"folds": folds, "n_folds": len(folds), "mean_oos_sharpe": 0, "p_value": 1}
    
    mean_sharpe = np.mean(fold_sharpes)
    median_sharpe = np.median(fold_sharpes)
    std_sharpe = np.std(fold_sharpes, ddof=1)
    t_stat = mean_sharpe / (std_sharpe / np.sqrt(len(fold_sharpes))) if std_sharpe > 0 else 0
    p_value = 1 - stats.t.cdf(t_stat, df=len(fold_sharpes) - 1)
    positive_folds = sum(1 for s in fold_sharpes if s > 0)
    
    # Bootstrap CI
    boot_means = [np.mean(np.random.choice(fold_sharpes, size=len(fold_sharpes), replace=True)) for _ in range(5000)]
    
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_oos_sharpe": round(mean_sharpe, 3),
        "median_oos_sharpe": round(median_sharpe, 3),
        "std_oos_sharpe": round(std_sharpe, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
        "ci_95_lower": round(np.percentile(boot_means, 2.5), 3),
        "ci_95_upper": round(np.percentile(boot_means, 97.5), 3),
        "positive_folds": f"{positive_folds}/{len(fold_sharpes)}",
        "mean_fold_return": round(np.mean(fold_returns), 1),
    }


def yearly_returns(returns):
    """Compute yearly returns."""
    yearly = {}
    for yr in range(2021, 2026):
        mask = returns.index.year == yr
        if mask.sum() < 10:
            continue
        yr_ret = returns[mask]
        eq = (1 + yr_ret).cumprod()
        yearly[str(yr)] = round(float(eq.iloc[-1] - 1) * 100, 1)
    return yearly


def correlation_matrix(crypto_data, weights, common_idx):
    """Correlation between asset daily returns."""
    rets = pd.DataFrame()
    for a in weights:
        if a in crypto_data:
            rets[a] = crypto_data[a]["close"].reindex(common_idx).pct_change().fillna(0)
    corr = rets.corr()
    return {a: {b: round(float(corr.loc[a, b]), 3) for b in corr.columns} for a in corr.index}


def diversification_benefit(crypto_data, macro_data, cross_asset_data, weights, portfolio_sharpe):
    """Compare portfolio Sharpe to weighted average of individual Sharpes."""
    individual_sharpes = {}
    for asset in weights:
        if asset not in crypto_data:
            continue
        # Run single-asset strategy
        single_weights = {asset: 1.0}
        single_data = {asset: crypto_data[asset]}
        try:
            pf, _ = run_full_strategy(single_data, macro_data, cross_asset_data, weights=single_weights)
            ret = pf["daily_pnl"]
            if ret.std() > 0:
                individual_sharpes[asset] = float(ret.mean() / ret.std() * np.sqrt(252))
            else:
                individual_sharpes[asset] = 0.0
        except:
            individual_sharpes[asset] = 0.0
    
    weighted_avg = sum(weights.get(a, 0) * individual_sharpes.get(a, 0) for a in weights)
    benefit = portfolio_sharpe - weighted_avg if weighted_avg != 0 else 0
    
    return {
        "individual_sharpes": {k: round(v, 3) for k, v in individual_sharpes.items()},
        "weighted_avg_sharpe": round(weighted_avg, 3),
        "portfolio_sharpe": round(portfolio_sharpe, 3),
        "diversification_benefit": round(benefit, 3),
        "benefit_ratio": round(portfolio_sharpe / weighted_avg, 3) if weighted_avg > 0 else 0,
    }


def main():
    print("\n" + "#" * 70)
    print("#  Portfolio Expansion Backtest — V3.1-H2 + UNI/DOGE/ADA")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    crypto_data, cross_asset_data, macro_data = load_data()
    
    # Verify we have all 7 assets
    required = ["BTC", "ETH", "SOL", "LINK", "UNI", "DOGE", "ADA"]
    missing = [a for a in required if a not in crypto_data]
    if missing:
        print(f"ERROR: Missing assets: {missing}")
        # Try alternate UNI ticker
        if "UNI" in missing:
            print("  Trying UNI-USD instead of UNI1-USD...")
            stock = StockDataLoader()
            try:
                df = stock.get_ohlcv("UNI-USD", "1d", start_date="2019-01-01")
                df = df[(df.index >= DATE_START) & (df.index <= DATE_END)]
                if len(df) > 100:
                    crypto_data["UNI"] = df
                    print(f"  UNI (UNI-USD): {len(df)} days")
                    missing.remove("UNI")
            except Exception as e:
                print(f"  UNI-USD also failed: {e}")
        if missing:
            print(f"FATAL: Still missing: {missing}")
            return
    
    # Common index across ALL 7 assets
    common_idx = crypto_data[required[0]].index
    for a in required[1:]:
        common_idx = common_idx.intersection(crypto_data[a].index)
    print(f"\nCommon index: {len(common_idx)} days ({common_idx[0].date()} to {common_idx[-1].date()})")
    
    results = {}
    
    for config_name, weights in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  Running {config_name}")
        print(f"  Assets: {list(weights.keys())}")
        print(f"  Weights: {weights}")
        print(f"{'='*70}")
        
        # Filter crypto_data to only needed assets
        config_assets = {a: crypto_data[a] for a in weights if a in crypto_data}
        
        if config_name == "Config6_Momentum":
            # Special handling for momentum-weighted
            portfolio_df = run_momentum_weighted(crypto_data, macro_data, cross_asset_data, weights)
        else:
            portfolio_df, per_asset = run_full_strategy(
                config_assets, macro_data, cross_asset_data, weights=weights
            )
        
        ret = portfolio_df["daily_pnl"]
        
        # IS metrics (full period)
        is_metrics = compute_metrics(ret)
        is_metrics["mean_leverage"] = round(float(portfolio_df["total_leverage"].mean()), 3)
        
        # Walk-forward
        wf = walk_forward_13fold(ret, config_name)
        
        # Yearly returns
        yr = yearly_returns(ret)
        
        # Correlation matrix
        config_common = portfolio_df.index
        corr = correlation_matrix(crypto_data, weights, config_common)
        
        # Diversification benefit
        div_benefit = diversification_benefit(crypto_data, macro_data, cross_asset_data, weights, is_metrics["sharpe"])
        
        results[config_name] = {
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "n_assets": len(weights),
            "is_metrics": is_metrics,
            "oos_walk_forward": wf,
            "yearly_returns": yr,
            "correlation_matrix": corr,
            "diversification_benefit": div_benefit,
        }
        
        # Print summary
        print(f"  IS:  Sharpe={is_metrics['sharpe']:.3f}  CAGR={is_metrics['cagr']:.1f}%  MaxDD={is_metrics['max_dd']:.1f}%  Sortino={is_metrics['sortino']:.3f}  Calmar={is_metrics['calmar']:.3f}  WR={is_metrics['win_rate']:.1f}%  Lev={is_metrics['mean_leverage']:.3f}")
        print(f"  OOS: Mean Sharpe={wf['mean_oos_sharpe']:.3f}  Median={wf.get('median_oos_sharpe',0):.3f}  p={wf['p_value']:.4f}  Positive={wf.get('positive_folds','N/A')}")
        print(f"  Yearly: {yr}")
        print(f"  Div Benefit: portfolio={div_benefit['portfolio_sharpe']:.3f} vs wt_avg={div_benefit['weighted_avg_sharpe']:.3f} → benefit={div_benefit['diversification_benefit']:.3f}")
    
    # ── Ranked Summary ──
    print("\n" + "#" * 90)
    print("#  RANKED SUMMARY BY OOS SHARPE")
    print("#" * 90)
    
    ranked = sorted(results.items(), key=lambda x: x[1]["oos_walk_forward"]["mean_oos_sharpe"], reverse=True)
    
    print(f"\n{'Rank':<5} {'Config':<30} {'#Assets':>7} {'IS_Sharpe':>10} {'OOS_Sharpe':>11} {'OOS_Med':>8} {'p-val':>7} {'CAGR%':>7} {'MaxDD%':>7} {'DivBen':>7}")
    print("-" * 110)
    
    for rank, (name, data) in enumerate(ranked, 1):
        is_m = data["is_metrics"]
        oos = data["oos_walk_forward"]
        div = data["diversification_benefit"]
        marker = " ⭐ BEST" if rank == 1 else ""
        print(f"{rank:<5} {name:<30} {data['n_assets']:>7} {is_m['sharpe']:>10.3f} {oos['mean_oos_sharpe']:>11.3f} {oos.get('median_oos_sharpe',0):>8.3f} {oos['p_value']:>7.4f} {is_m['cagr']:>7.1f} {is_m['max_dd']:>7.1f} {div['diversification_benefit']:>7.3f}{marker}")
    
    best_name = ranked[0][0]
    best_data = ranked[0][1]
    print(f"\n🏆 BEST CONFIG: {best_name}")
    print(f"   OOS Sharpe: {best_data['oos_walk_forward']['mean_oos_sharpe']:.3f}")
    print(f"   IS Sharpe: {best_data['is_metrics']['sharpe']:.3f}")
    print(f"   CAGR: {best_data['is_metrics']['cagr']:.1f}%")
    print(f"   MaxDD: {best_data['is_metrics']['max_dd']:.1f}%")
    print(f"   Positive folds: {best_data['oos_walk_forward'].get('positive_folds', 'N/A')}")
    print(f"   p-value: {best_data['oos_walk_forward']['p_value']:.4f}")
    
    # KEY QUESTION
    print(f"\n{'='*70}")
    print("KEY QUESTION: Does adding more assets improve risk-adjusted returns?")
    print(f"{'='*70}")
    
    for name, data in ranked:
        n = data["n_assets"]
        oos = data["oos_walk_forward"]["mean_oos_sharpe"]
        div = data["diversification_benefit"]["diversification_benefit"]
        print(f"  {n}-asset {name}: OOS Sharpe={oos:.3f}, Div Benefit={div:+.3f}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "date_range": f"{DATE_START} to {DATE_END}",
        "methodology": "13-fold non-overlapping 6-month walk-forward, 0.1% tx costs",
        "results": results,
        "ranking": [{"rank": i+1, "config": name, "oos_sharpe": data["oos_walk_forward"]["mean_oos_sharpe"]} 
                     for i, (name, data) in enumerate(ranked)],
        "best_config": best_name,
        "conclusion": f"Best config is {best_name} with OOS Sharpe {best_data['oos_walk_forward']['mean_oos_sharpe']:.3f}",
    }
    
    out_dir = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "portfolio_expansion_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
