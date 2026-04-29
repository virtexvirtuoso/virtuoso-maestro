#!/usr/bin/env python3
"""
Scalping Strategy Full Exploration: Baseline + Macro-Filtered Hybrids + Optuna Optimization
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# --- Strategy imports ---
from strategies.scalping.ema_ribbon import generate_signals as ema_ribbon_signals
from strategies.scalping.grid_trading import generate_signals as grid_signals
from strategies.scalping.momentum_breakout import generate_signals as momentum_breakout_signals
from strategies.scalping.quickie import generate_signals as quickie_signals
from strategies.scalping.scalp_rsi import generate_signals as scalp_rsi_signals
from strategies.scalping.smooth_scalp import generate_signals as smooth_scalp_signals
from strategies.scalping.stoch_rsi import generate_signals as stoch_rsi_signals
from strategies.scalping.vwap import generate_signals as vwap_signals

# Macro framework
from strategies.composite.macro_score_builder import compute_macro_score_from_df
from strategies.composite.mega_strategy_v3 import compute_confluence, detect_regime, DEFAULT_LEVERAGE_MAP

STRATEGIES = {
    "EMARibbon": ema_ribbon_signals,
    "GridTrading": grid_signals,
    "MomentumBreakout": momentum_breakout_signals,
    "Quickie": quickie_signals,
    "ScalpRSI": scalp_rsi_signals,
    "SmoothScalp": smooth_scalp_signals,
    "StochRSI": stoch_rsi_signals,
    "VWAP": vwap_signals,
}

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]
TX_COST = 0.001  # 0.1% per trade

# ─── Data Loading ───────────────────────────────────────────────────────
def load_crypto_data(ticker: str, start="2020-01-01", end="2026-02-12") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            # Try to find it
            for c in df.columns:
                if col in str(c).lower():
                    df[col] = df[c]
                    break
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    return df

def load_fred_data(start="2019-01-01", end="2026-02-12") -> pd.DataFrame:
    """Load FRED macro data for macro score computation."""
    try:
        import fredapi
        FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
        if FRED_API_KEY:
            fred = fredapi.Fred(api_key=FRED_API_KEY)
            series_map = {
                "yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS",
                "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2",
            }
            frames = {}
            for name, sid in series_map.items():
                try:
                    s = fred.get_series(sid, observation_start=start, observation_end=end)
                    frames[name] = s
                except:
                    pass
            if frames:
                return pd.DataFrame(frames).ffill()
    except ImportError:
        pass
    
    # Fallback: synthetic macro data based on known regimes
    print("  [FRED unavailable — using synthetic macro score]")
    idx = pd.date_range(start, end, freq="D")
    np.random.seed(42)
    # Simulate macro score directly (skip raw FRED)
    return None

def load_cross_asset_data(start="2019-01-01", end="2026-02-12") -> pd.DataFrame:
    """Load cross-asset data for confluence scoring."""
    import yfinance as yf
    tickers = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
    frames = {}
    for name, tk in tickers.items():
        try:
            d = yf.download(tk, start=start, end=end, progress=False)
            col = [c for c in d.columns if 'close' in str(c).lower() or 'Close' in str(c)]
            if len(col) > 0:
                frames[name] = d[col[0]]
            else:
                frames[name] = d.iloc[:, 3]  # 4th col usually close
        except:
            pass
    if frames:
        return pd.DataFrame(frames).ffill()
    return None


# ─── Backtest Engine ────────────────────────────────────────────────────
def backtest_signals(df: pd.DataFrame, raw_signals: pd.Series, 
                     position_sizing: Optional[pd.Series] = None) -> Dict:
    """
    Simple long/short backtest with transaction costs.
    raw_signals: {1, 0, -1} — shifted by 1 to avoid lookahead.
    position_sizing: optional multiplier (0-2) for adaptive sizing.
    """
    signals = raw_signals.shift(1).fillna(0).astype(int)  # NO LOOKAHEAD
    close = df['close'].copy()
    
    if position_sizing is not None:
        sizing = position_sizing.shift(1).fillna(1.0).reindex(signals.index, method='ffill').fillna(1.0)
    else:
        sizing = pd.Series(1.0, index=signals.index)
    
    # Compute returns
    daily_ret = close.pct_change().fillna(0)
    
    # Position changes for tx costs
    effective_pos = signals * sizing
    pos_changes = effective_pos.diff().abs().fillna(0)
    tx_costs = pos_changes * TX_COST
    
    # Strategy returns
    strat_ret = effective_pos * daily_ret - tx_costs
    equity = (1 + strat_ret).cumprod()
    
    # Metrics
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
    
    # Sharpe
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    
    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    
    # Win rate & trades
    trade_starts = signals.diff().fillna(0) != 0
    n_trades = int(trade_starts.sum())
    
    # Per-trade P&L for win rate
    trade_id = trade_starts.cumsum()
    if n_trades > 0:
        trade_pnl = strat_ret.groupby(trade_id).sum()
        win_rate = float((trade_pnl > 0).mean())
        avg_duration = float(len(signals) / n_trades)
    else:
        win_rate = 0.0
        avg_duration = 0.0
    
    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "n_trades": n_trades,
        "avg_trade_days": round(avg_duration, 1),
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
    }


# ─── Phase 1: Baseline ─────────────────────────────────────────────────
def phase1_baseline(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PHASE 1: BASELINE TEST — ALL 8 SCALPING STRATEGIES")
    print("="*70)
    
    results = []
    for strat_name, strat_fn in STRATEGIES.items():
        for ticker in TICKERS:
            df = data[ticker]
            try:
                signals = strat_fn(df)
                metrics = backtest_signals(df, signals)
                metrics["strategy"] = strat_name
                metrics["ticker"] = ticker
                metrics["variant"] = "baseline"
                results.append(metrics)
                print(f"  {strat_name:20s} | {ticker:8s} | Sharpe={metrics['sharpe']:+.3f} | Ret={metrics['total_return']:+.1f}% | DD={metrics['max_dd']:.1f}% | WR={metrics['win_rate']:.1f}% | Trades={metrics['n_trades']}")
            except Exception as e:
                print(f"  {strat_name:20s} | {ticker:8s} | ERROR: {e}")
                results.append({"strategy": strat_name, "ticker": ticker, "variant": "baseline",
                                "sharpe": 0, "total_return": 0, "max_dd": 0, "win_rate": 0, "n_trades": 0,
                                "avg_trade_days": 0, "ann_return": 0, "ann_vol": 0})
    
    results_df = pd.DataFrame(results)
    
    # Rank by average Sharpe across tickers
    avg_sharpe = results_df.groupby("strategy")["sharpe"].mean().sort_values(ascending=False)
    print(f"\n{'='*50}")
    print("RANKING BY AVERAGE SHARPE:")
    for i, (strat, sharpe) in enumerate(avg_sharpe.items()):
        print(f"  #{i+1}: {strat:20s} → Avg Sharpe = {sharpe:+.3f}")
    
    return results_df


# ─── Phase 2: Macro Hybrids ────────────────────────────────────────────
def phase2_hybrids(data: Dict[str, pd.DataFrame], baseline_df: pd.DataFrame,
                   macro_scores: Dict[str, pd.Series], 
                   confluences: Dict[str, pd.Series],
                   regimes: Dict[str, pd.Series]) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PHASE 2: MACRO-FILTERED SCALPING HYBRIDS")
    print("="*70)
    
    # Top 4 by average Sharpe
    avg_sharpe = baseline_df.groupby("strategy")["sharpe"].mean().sort_values(ascending=False)
    top4 = list(avg_sharpe.index[:4])
    print(f"  Top 4 strategies: {top4}")
    
    results = []
    for strat_name in top4:
        strat_fn = STRATEGIES[strat_name]
        for ticker in TICKERS:
            df = data[ticker]
            signals = strat_fn(df)
            
            macro_s = macro_scores.get(ticker)
            confl = confluences.get(ticker)
            regime = regimes.get(ticker)
            
            # Hybrid 1: Macro Gate (macro score >= 3)
            if macro_s is not None:
                macro_aligned = macro_s.reindex(df.index, method='ffill').fillna(3)
                gated = signals.copy()
                gated[macro_aligned < 3] = 0
                m = backtest_signals(df, gated)
                m.update({"strategy": strat_name, "ticker": ticker, "variant": "macro_gate"})
                results.append(m)
                print(f"  {strat_name:20s} | {ticker:8s} | macro_gate    | Sharpe={m['sharpe']:+.3f} | Ret={m['total_return']:+.1f}%")
            
            # Hybrid 2: Regime Filter (BULL/MILD_BULL only)
            if regime is not None:
                regime_aligned = regime.reindex(df.index, method='ffill').fillna("NEUTRAL")
                filtered = signals.copy()
                filtered[~regime_aligned.isin(["BULL", "MILD_BULL", "ACCUMULATION"])] = 0
                m = backtest_signals(df, filtered)
                m.update({"strategy": strat_name, "ticker": ticker, "variant": "regime_filter"})
                results.append(m)
                print(f"  {strat_name:20s} | {ticker:8s} | regime_filter | Sharpe={m['sharpe']:+.3f} | Ret={m['total_return']:+.1f}%")
            
            # Hybrid 3: Adaptive Sizing (scale by confluence)
            if confl is not None:
                confl_aligned = confl.reindex(df.index, method='ffill').fillna(0)
                sizing = confl_aligned.map(DEFAULT_LEVERAGE_MAP).fillna(0.3)
                m = backtest_signals(df, signals, position_sizing=sizing)
                m.update({"strategy": strat_name, "ticker": ticker, "variant": "adaptive_sizing"})
                results.append(m)
                print(f"  {strat_name:20s} | {ticker:8s} | adaptive_size | Sharpe={m['sharpe']:+.3f} | Ret={m['total_return']:+.1f}%")
    
    return pd.DataFrame(results)


# ─── Phase 3: Comparison ───────────────────────────────────────────────
def phase3_compare(baseline_df: pd.DataFrame, hybrid_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PHASE 3: FULL COMPARISON TABLE")
    print("="*70)
    
    all_df = pd.concat([baseline_df, hybrid_df], ignore_index=True)
    
    # Summary table: avg across tickers
    summary = all_df.groupby(["strategy", "variant"]).agg({
        "sharpe": "mean", "total_return": "mean", "max_dd": "mean",
        "win_rate": "mean", "n_trades": "mean"
    }).round(3).sort_values("sharpe", ascending=False)
    
    print("\n" + summary.to_string())
    
    # Analysis
    print("\n\n--- ANALYSIS ---")
    
    # Which strategies benefit most from macro filtering?
    top4_strats = baseline_df.groupby("strategy")["sharpe"].mean().sort_values(ascending=False).index[:4]
    print("\nMacro filtering benefit (Sharpe improvement over baseline):")
    for strat in top4_strats:
        base_sharpe = all_df[(all_df.strategy == strat) & (all_df.variant == "baseline")]["sharpe"].mean()
        for variant in ["macro_gate", "regime_filter", "adaptive_sizing"]:
            v_data = all_df[(all_df.strategy == strat) & (all_df.variant == variant)]
            if len(v_data) > 0:
                v_sharpe = v_data["sharpe"].mean()
                delta = v_sharpe - base_sharpe
                print(f"  {strat:20s} | {variant:15s} | Δ Sharpe = {delta:+.3f}")
    
    # Best hybrid mode overall
    print("\nBest hybrid mode (avg Sharpe across all strategies):")
    for variant in ["baseline", "macro_gate", "regime_filter", "adaptive_sizing"]:
        v_data = all_df[all_df.variant == variant]
        if len(v_data) > 0:
            print(f"  {variant:20s} → Avg Sharpe = {v_data['sharpe'].mean():+.3f}")
    
    # Win rate improvement
    print("\nWin rate improvement from macro filtering:")
    for variant in ["macro_gate", "regime_filter", "adaptive_sizing"]:
        base_wr = all_df[all_df.variant == "baseline"]["win_rate"].mean()
        v_wr = all_df[all_df.variant == variant]["win_rate"].mean()
        if not np.isnan(v_wr):
            print(f"  {variant:20s} → Δ WR = {v_wr - base_wr:+.1f}%")
    
    return all_df


# ─── Phase 4: Optuna Optimization ──────────────────────────────────────
def phase4_optimize(data: Dict[str, pd.DataFrame], all_df: pd.DataFrame,
                    macro_scores: Dict, confluences: Dict, regimes: Dict) -> Dict:
    print("\n" + "="*70)
    print("PHASE 4: OPTUNA OPTIMIZATION OF BEST HYBRID")
    print("="*70)
    
    # Find best combo
    best_row = all_df.groupby(["strategy", "variant"])["sharpe"].mean().sort_values(ascending=False).reset_index().iloc[0]
    best_strat = best_row["strategy"]
    best_variant = best_row["variant"]
    print(f"  Best combo: {best_strat} + {best_variant} (Avg Sharpe = {best_row['sharpe']:.3f})")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not available, skipping optimization")
        return {"best_strategy": best_strat, "best_variant": best_variant, "optimized": False}
    
    # Strategy-specific param spaces
    PARAM_SPACES = {
        "EMARibbon": {},  # No tunable params
        "GridTrading": lambda t: {"lookback": t.suggest_int("lookback", 20, 100), "num_grids": t.suggest_int("num_grids", 3, 10)},
        "MomentumBreakout": lambda t: {"lookback": t.suggest_int("lookback", 5, 30), "vol_mult": t.suggest_float("vol_mult", 1.0, 3.0)},
        "Quickie": lambda t: {"period": t.suggest_int("period", 5, 30), "threshold": t.suggest_float("threshold", 0.8, 3.0)},
        "ScalpRSI": lambda t: {"period": t.suggest_int("period", 3, 21), "oversold": t.suggest_int("oversold", 15, 40), "overbought": t.suggest_int("overbought", 60, 85)},
        "SmoothScalp": lambda t: {"fast": t.suggest_int("fast", 3, 10), "slow": t.suggest_int("slow", 10, 30)},
        "StochRSI": lambda t: {"rsi_period": t.suggest_int("rsi_period", 7, 21), "stoch_period": t.suggest_int("stoch_period", 7, 21)},
        "VWAP": lambda t: {"std_mult": t.suggest_float("std_mult", 1.0, 4.0)},
    }
    
    strat_fn = STRATEGIES[best_strat]
    param_fn = PARAM_SPACES.get(best_strat)
    
    def objective(trial):
        params = param_fn(trial) if callable(param_fn) else {}
        
        # Macro threshold for gate/filter
        if best_variant == "macro_gate":
            macro_thresh = trial.suggest_int("macro_thresh", 1, 5)
        
        sharpes = []
        for ticker in TICKERS:
            df = data[ticker]
            try:
                signals = strat_fn(df, **params)
            except:
                signals = strat_fn(df)
            
            if best_variant == "macro_gate":
                macro_s = macro_scores.get(ticker)
                if macro_s is not None:
                    ma = macro_s.reindex(df.index, method='ffill').fillna(3)
                    signals[ma < macro_thresh] = 0
            elif best_variant == "regime_filter":
                regime = regimes.get(ticker)
                if regime is not None:
                    ra = regime.reindex(df.index, method='ffill').fillna("NEUTRAL")
                    signals[~ra.isin(["BULL", "MILD_BULL", "ACCUMULATION"])] = 0
            elif best_variant == "adaptive_sizing":
                confl = confluences.get(ticker)
                if confl is not None:
                    ca = confl.reindex(df.index, method='ffill').fillna(0)
                    sizing = ca.map(DEFAULT_LEVERAGE_MAP).fillna(0.3)
                    m = backtest_signals(df, signals, position_sizing=sizing)
                    sharpes.append(m["sharpe"])
                    continue
            
            m = backtest_signals(df, signals)
            sharpes.append(m["sharpe"])
        
        return np.mean(sharpes) if sharpes else 0
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print(f"\n  Best Optuna Sharpe: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")
    
    # Run final backtest with best params
    final_results = {}
    for ticker in TICKERS:
        df = data[ticker]
        params = {k: v for k, v in study.best_params.items() if k not in ["macro_thresh"]}
        try:
            signals = strat_fn(df, **params)
        except:
            signals = strat_fn(df)
        
        if best_variant == "macro_gate":
            thresh = study.best_params.get("macro_thresh", 3)
            macro_s = macro_scores.get(ticker)
            if macro_s is not None:
                ma = macro_s.reindex(df.index, method='ffill').fillna(3)
                signals[ma < thresh] = 0
            m = backtest_signals(df, signals)
        elif best_variant == "regime_filter":
            regime = regimes.get(ticker)
            if regime is not None:
                ra = regime.reindex(df.index, method='ffill').fillna("NEUTRAL")
                signals[~ra.isin(["BULL", "MILD_BULL", "ACCUMULATION"])] = 0
            m = backtest_signals(df, signals)
        elif best_variant == "adaptive_sizing":
            confl = confluences.get(ticker)
            sizing = None
            if confl is not None:
                ca = confl.reindex(df.index, method='ffill').fillna(0)
                sizing = ca.map(DEFAULT_LEVERAGE_MAP).fillna(0.3)
            m = backtest_signals(df, signals, position_sizing=sizing)
        else:
            m = backtest_signals(df, signals)
        
        final_results[ticker] = m
        print(f"  Optimized {ticker}: Sharpe={m['sharpe']:+.3f} | Ret={m['total_return']:+.1f}% | DD={m['max_dd']:.1f}%")
    
    return {
        "best_strategy": best_strat,
        "best_variant": best_variant,
        "optimized": True,
        "best_params": study.best_params,
        "best_sharpe": round(study.best_value, 3),
        "final_results": final_results,
    }


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    print("🎯 SCALPING STRATEGY FULL EXPLORATION")
    print("="*70)
    
    # Load data
    print("\nLoading price data...")
    data = {}
    for ticker in TICKERS:
        print(f"  Loading {ticker}...")
        data[ticker] = load_crypto_data(ticker)
        print(f"    → {len(data[ticker])} bars from {data[ticker].index[0].date()} to {data[ticker].index[-1].date()}")
    
    # Load macro data
    print("\nLoading macro data...")
    fred_df = load_fred_data()
    cross_asset = load_cross_asset_data()
    
    # Compute macro scores and confluence for each ticker
    macro_scores = {}
    confluences = {}
    regimes = {}
    
    for ticker in TICKERS:
        df = data[ticker]
        
        # Macro score from FRED
        if fred_df is not None:
            ms = compute_macro_score_from_df(fred_df)
            macro_scores[ticker] = ms
        else:
            # Synthetic: use confluence score as proxy for macro score
            macro_scores[ticker] = None
        
        # Confluence & regime from V3
        try:
            confl, _ = compute_confluence(df['close'], fred_df, cross_asset)
            confluences[ticker] = confl
            regimes[ticker] = detect_regime(confl)
            print(f"  {ticker}: confluence computed, regime distribution: {regimes[ticker].value_counts().to_dict()}")
        except Exception as e:
            print(f"  {ticker}: confluence failed ({e}), using synthetic")
            # Synthetic regime based on SMA
            sma200 = df['close'].rolling(200).mean()
            regime = pd.Series("NEUTRAL", index=df.index)
            regime[df['close'] > sma200 * 1.1] = "BULL"
            regime[(df['close'] > sma200) & (df['close'] <= sma200 * 1.1)] = "MILD_BULL"
            regime[df['close'] < sma200 * 0.9] = "BEAR"
            regimes[ticker] = regime
            
            # Synthetic confluence: 0-5 based on multiple SMAs
            sma50 = df['close'].rolling(50).mean()
            sma100 = df['close'].rolling(100).mean()
            confl = pd.Series(0, index=df.index)
            confl += (df['close'] > sma50).astype(int)
            confl += (df['close'] > sma100).astype(int)
            confl += (df['close'] > sma200).astype(int)
            confl += (df['close'].pct_change(30) > 0).astype(int)
            confl += (df['close'].pct_change(90) > 0).astype(int)
            confluences[ticker] = confl
            
            # Synthetic macro score
            macro_scores[ticker] = confl.clip(0, 6)
    
    # Phase 1
    baseline_df = phase1_baseline(data)
    
    # Phase 2
    hybrid_df = phase2_hybrids(data, baseline_df, macro_scores, confluences, regimes)
    
    # Phase 3
    all_df = phase3_compare(baseline_df, hybrid_df)
    
    # Phase 4
    optuna_results = phase4_optimize(data, all_df, macro_scores, confluences, regimes)
    
    # Save results
    output_dir = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scalping_exploration_results.json")
    
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "tickers": TICKERS,
        "tx_cost": TX_COST,
        "baseline_results": baseline_df.to_dict(orient="records"),
        "hybrid_results": hybrid_df.to_dict(orient="records") if len(hybrid_df) > 0 else [],
        "all_results": all_df.to_dict(orient="records"),
        "rankings": all_df.groupby(["strategy", "variant"])["sharpe"].mean().sort_values(ascending=False).to_dict(),
        "optuna_optimization": optuna_results,
    }
    
    # Convert tuple keys to strings for JSON
    rankings = {}
    for k, v in save_data["rankings"].items():
        rankings[f"{k[0]}_{k[1]}"] = round(v, 3)
    save_data["rankings"] = rankings
    
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n\n✅ Results saved to {output_path}")
    print("="*70)
    print("DONE!")


if __name__ == "__main__":
    main()
