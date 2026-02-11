import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from data_loader import DataLoader
from strategies.hybrids import MomentumTrendConfirm

def calculate_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], abs(df["high"]-df["close"].shift(1)), abs(df["low"]-df["close"].shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def backtest(df, signals, tp_mult, sl_mult, trailing_mult, commission=0.001):
    atr = calculate_atr(df)
    position, entry_price, highest, lowest = 0, 0.0, 0.0, float("inf")
    trades = []
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else price * 0.02
        signal = signals.iloc[i] if i < len(signals) else 0
        if position == 1:
            tp, sl = entry_price + tp_mult * current_atr, entry_price - sl_mult * current_atr
            highest = max(highest, price)
            trail = highest - trailing_mult * current_atr
            if price >= tp: trades.append((tp - entry_price) / entry_price - commission * 2); position = 0
            elif price <= sl: trades.append((sl - entry_price) / entry_price - commission * 2); position = 0
            elif price <= trail and highest > entry_price * 1.01: trades.append((trail - entry_price) / entry_price - commission * 2); position = 0
            elif signal == -1: trades.append((price - entry_price) / entry_price - commission * 2); position, entry_price, lowest = -1, price, price
        elif position == -1:
            tp, sl = entry_price - tp_mult * current_atr, entry_price + sl_mult * current_atr
            lowest = min(lowest, price)
            trail = lowest + trailing_mult * current_atr
            if price <= tp: trades.append((entry_price - tp) / entry_price - commission * 2); position = 0
            elif price >= sl: trades.append((entry_price - sl) / entry_price - commission * 2); position = 0
            elif price >= trail and lowest < entry_price * 0.99: trades.append((entry_price - trail) / entry_price - commission * 2); position = 0
            elif signal == 1: trades.append((entry_price - price) / entry_price - commission * 2); position, entry_price, highest = 1, price, price
        if position == 0:
            if signal == 1: position, entry_price, highest = 1, price, price
            elif signal == -1: position, entry_price, lowest = -1, price, price
    if position != 0:
        price = df["close"].iloc[-1]
        trades.append((price - entry_price) / entry_price - commission * 2 if position == 1 else (entry_price - price) / entry_price - commission * 2)
    if not trades: return {"return": 0, "sharpe": 0, "trades": 0, "win_rate": 0, "max_dd": 0}
    total, peak, max_dd = 10000, 10000, 0
    for t in trades:
        total *= (1 + t)
        peak = max(peak, total)
        max_dd = max(max_dd, (peak - total) / peak * 100)
    ret = (total - 10000) / 10000 * 100
    sharpe = np.mean(trades) / (np.std(trades) + 0.0001) * np.sqrt(len(trades))
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
    return {"return": ret, "sharpe": sharpe, "trades": len(trades), "win_rate": win_rate, "max_dd": max_dd}

def optimize(train_df, n_trials=50):
    def objective(trial):
        tp = trial.suggest_float("tp_mult", 2.0, 8.0)
        sl = trial.suggest_float("sl_mult", 1.0, 4.0)
        tr = trial.suggest_float("trailing_mult", 1.0, 4.0)
        mom_period = trial.suggest_int("mom_period", 10, 40)
        mom_thresh = trial.suggest_float("mom_threshold", 2.0, 15.0)
        atr_period = trial.suggest_int("atr_period", 7, 21)
        atr_mult = trial.suggest_float("atr_mult", 1.5, 5.0)
        trend_period = trial.suggest_int("trend_period", 20, 100)
        signals = MomentumTrendConfirm.generate_signals(train_df, mom_period, mom_thresh, atr_period, atr_mult, trend_period)
        return backtest(train_df, signals, tp, sl, tr)["sharpe"]
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

loader = DataLoader()
ASSETS = ["ZEC", "CRV", "SEI", "RENDER", "DYDX", "XRP", "1000BONK", "WLD", "ETH", "BTC"]

print("=" * 95)
print("MomentumTrendConfirm: BACKTEST vs OOS (Top 10 Assets)")
print("=" * 95)
print("Asset        Backtest     OOS          BT_Sharpe    OOS_Sharpe   OOS_MaxDD    Trades")
print("-" * 95)

results = []
for asset in ASSETS:
    try:
        df = loader.get_ohlcv_full(asset, "1d").to_pandas()
        df.set_index("timestamp", inplace=True)
    except:
        print(f"{asset:<12} [NO DATA]")
        continue
    
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    
    params = optimize(train_df, n_trials=50)
    
    train_signals = MomentumTrendConfirm.generate_signals(
        train_df, params["mom_period"], params["mom_threshold"],
        params["atr_period"], params["atr_mult"], params["trend_period"])
    bt_result = backtest(train_df, train_signals, params["tp_mult"], params["sl_mult"], params["trailing_mult"])
    
    test_signals = MomentumTrendConfirm.generate_signals(
        test_df, params["mom_period"], params["mom_threshold"],
        params["atr_period"], params["atr_mult"], params["trend_period"])
    oos_result = backtest(test_df, test_signals, params["tp_mult"], params["sl_mult"], params["trailing_mult"])
    
    bt_ret = bt_result["return"]
    oos_ret = oos_result["return"]
    bt_sh = bt_result["sharpe"]
    oos_sh = oos_result["sharpe"]
    oos_dd = oos_result["max_dd"]
    oos_tr = oos_result["trades"]
    
    print(f"{asset:<12} {bt_ret:>+9.1f}%  {oos_ret:>+9.1f}%  {bt_sh:>10.2f}  {oos_sh:>10.2f}  {oos_dd:>10.1f}%  {oos_tr:>6}")
    
    results.append({"asset": asset, "backtest": bt_ret, "oos": oos_ret,
                    "bt_sharpe": bt_sh, "oos_sharpe": oos_sh,
                    "oos_maxdd": oos_dd, "oos_trades": oos_tr, "params": params})

loader.close()

print("-" * 95)
profitable = len([r for r in results if r["oos"] > 0])
avg_bt = np.mean([r["backtest"] for r in results])
avg_oos = np.mean([r["oos"] for r in results])
print(f"{'AVERAGE':<12} {avg_bt:>+9.1f}%  {avg_oos:>+9.1f}%")
print(f"\nOOS Win Rate: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
if avg_oos > 0:
    print(f"Overfitting Ratio: {avg_bt/avg_oos:.1f}x (lower is better)")
