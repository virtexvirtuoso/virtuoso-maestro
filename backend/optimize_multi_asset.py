"""
Multi-Asset Optuna Optimization — Find optimal params per asset for winning filter combo.
100 trials each for BTC, ETH, SOL.
"""
import sys, os, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2017-01-01"
END = "2026-02-01"
TX_COST = 0.001
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
N_TRIALS = 100


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bb(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s


def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_with_params(df, params, use_sma=True, use_momentum=True):
    """Run strategy and return Sharpe ratio."""
    close = df["close"]
    n = len(close)
    
    sma_slow = int(params["sma_slow"])
    if n < sma_slow + 50:
        return -10.0
    
    sma = close.rolling(sma_slow).mean()
    roc = close.pct_change(int(params["momentum_period"]))
    rsi = compute_rsi(close, 14)
    ema = close.ewm(span=int(params["ema_period"]), adjust=False).mean()
    bb_lower, _, bb_upper = compute_bb(close, 20, params["bb_std"])
    atr = compute_atr(df)
    
    regime = pd.Series(True, index=df.index)
    if use_sma:
        regime &= (close > sma)
    if use_momentum:
        regime &= (roc > 0)
    
    rsi_dip = rsi < params["rsi_entry"]
    ema_dip = close < ema
    bb_dip = close <= bb_lower
    any_dip = rsi_dip | ema_dip | bb_dip
    
    regime = regime.shift(1).fillna(False)
    any_dip = any_dip.shift(1).fillna(False)
    rsi_hot = (rsi > params["rsi_exit"]).shift(1).fillna(False)
    bb_hot = (close > bb_upper).shift(1).fillna(False)
    atr_s = atr.shift(1).fillna(0)
    
    pos = 0.0
    avg_entry = 0.0
    equity = 1.0
    peak = 1.0
    total_cost = 0.0
    equities = np.ones(n)
    
    warmup = sma_slow + 5
    
    for i in range(warmup, n):
        price = close.iloc[i]
        prev = close.iloc[i-1]
        
        if pos > 0 and prev > 0:
            equity *= (1 + (price - prev) / prev * pos)
        peak = max(peak, equity)
        equities[i] = equity
        
        if pos > 0 and (1 - equity/peak) >= params["trail_stop"]:
            equity *= (1 - pos * TX_COST)
            pos = 0; avg_entry = 0; total_cost = 0
            continue
        
        if pos > 0:
            trim = 0
            if rsi_hot.iloc[i]: trim = pos * 0.25
            elif bb_hot.iloc[i]: trim = pos * 0.25
            elif avg_entry > 0 and atr_s.iloc[i] > 0 and (price - avg_entry)/atr_s.iloc[i] > params["atr_exit_mult"]:
                trim = pos * 0.25
            if trim > 0:
                equity *= (1 - trim * TX_COST)
                pos -= trim
                if pos < 0.01: pos = 0; avg_entry = 0; total_cost = 0
        
        if regime.iloc[i] and any_dip.iloc[i]:
            if pos == 0:
                add = min(params["initial_size"], params["max_position"])
                equity *= (1 - add * TX_COST)
                pos = add; avg_entry = price; total_cost = price * add
            elif pos < params["max_position"]:
                add = min(params["pyramid_size"], params["max_position"] - pos)
                if add > 0.01:
                    equity *= (1 - add * TX_COST)
                    total_cost += price * add; pos += add
                    avg_entry = total_cost / pos
    
    eq_series = pd.Series(equities, index=df.index)
    daily_rets = eq_series.pct_change().fillna(0)
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 1e-8 else 0
    
    # Penalize if too few trades or extreme drawdown
    max_dd = (eq_series / eq_series.cummax() - 1).min()
    if max_dd < -0.7:
        sharpe *= 0.5
    
    return sharpe


def optimize_asset(asset, df):
    """Run Optuna optimization for an asset."""
    
    def objective(trial):
        params = {
            "sma_slow": trial.suggest_int("sma_slow", 50, 150, step=10),
            "momentum_period": trial.suggest_int("momentum_period", 15, 40, step=5),
            "rsi_entry": trial.suggest_int("rsi_entry", 30, 55),
            "rsi_exit": trial.suggest_int("rsi_exit", 65, 85),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "bb_std": trial.suggest_float("bb_std", 1.5, 2.5, step=0.25),
            "trail_stop": trial.suggest_float("trail_stop", 0.10, 0.30, step=0.02),
            "atr_exit_mult": trial.suggest_float("atr_exit_mult", 1.5, 3.0, step=0.25),
            "max_position": trial.suggest_float("max_position", 0.8, 2.0, step=0.1),
            "initial_size": trial.suggest_float("initial_size", 0.3, 0.7, step=0.1),
            "pyramid_size": trial.suggest_float("pyramid_size", 0.1, 0.5, step=0.1),
        }
        return backtest_with_params(df, params)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    return study.best_params, study.best_value


def main():
    print("=" * 100)
    print("MULTI-ASSET OPTUNA OPTIMIZATION")
    print(f"Assets: {ASSETS} | Trials: {N_TRIALS} each")
    print("Filter: SMA + Momentum (no M2) — likely winner from exploration")
    print("=" * 100)
    
    loader = StockDataLoader()
    all_results = {}
    
    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"Optimizing {asset}...")
        print(f"{'='*60}")
        
        df = loader.get_ohlcv(asset, "1d", start_date=START, end_date=END)
        print(f"  Data: {len(df)} days")
        
        best_params, best_sharpe = optimize_asset(asset, df)
        all_results[asset] = {"params": best_params, "sharpe": round(best_sharpe, 3)}
        
        print(f"\n  Best Sharpe: {best_sharpe:.3f}")
        print(f"  Best Params:")
        for k, v in sorted(best_params.items()):
            print(f"    {k:>20}: {v}")
    
    # Compare params across assets
    print("\n" + "=" * 100)
    print("PARAMETER COMPARISON ACROSS ASSETS")
    print("=" * 100)
    
    param_names = sorted(all_results[ASSETS[0]]["params"].keys())
    header = f"{'Param':<22}"
    for asset in ASSETS:
        header += f" {asset.replace('-USD',''):>10}"
    print(header)
    print("-" * 100)
    
    for p in param_names:
        row = f"{p:<22}"
        values = []
        for asset in ASSETS:
            v = all_results[asset]["params"][p]
            values.append(v)
            if isinstance(v, float):
                row += f" {v:>10.2f}"
            else:
                row += f" {v:>10}"
        # Check similarity
        if all(isinstance(v, (int, float)) for v in values):
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            similar = "✓ similar" if cv < 0.2 else "✗ differs"
            row += f"  {similar}"
        print(row)
    
    print(f"\n{'Sharpe':<22}", end="")
    for asset in ASSETS:
        print(f" {all_results[asset]['sharpe']:>10.3f}", end="")
    print()
    
    # Save
    save_data = {k: v for k, v in all_results.items()}
    for k in save_data:
        save_data[k]["params"] = {pk: float(pv) if isinstance(pv, (np.floating, np.integer)) else pv 
                                   for pk, pv in save_data[k]["params"].items()}
    
    with open(RESULTS_DIR / "optimize_multi_asset_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {RESULTS_DIR / 'optimize_multi_asset_results.json'}")


if __name__ == "__main__":
    main()
