"""Optuna hyperparameter optimization for 6 price structure strategies with walk-forward validation."""
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from strategies.technical.market_structure import generate_signals as market_structure_signals
from strategies.technical.range_sfp import generate_signals as range_sfp_signals
from strategies.technical.fair_value_gaps import generate_signals as fvg_signals
from strategies.technical.order_blocks import generate_signals as ob_signals
from strategies.technical.volume_profile import generate_signals as vp_signals
from strategies.technical.sr_levels import generate_signals as sr_signals


def backtest(df, signal_func, **params):
    try:
        signals = signal_func(df, **params)
        daily_returns = df['close'].pct_change().shift(-1)
        position_changes = signals.diff().abs().fillna(0)
        strategy_returns = (signals * daily_returns - position_changes * 0.001).fillna(0)
        if strategy_returns.std() == 0:
            return 0.0, strategy_returns
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)
        return sharpe, strategy_returns
    except Exception:
        return -999.0, pd.Series(0, index=df.index)


def calc_metrics(returns_series):
    """Calculate CAGR, MaxDD from a returns series."""
    equity = (1 + returns_series).cumprod()
    total_days = len(returns_series)
    if total_days == 0 or equity.iloc[-1] <= 0:
        return 0.0, 0.0
    cagr = (equity.iloc[-1] ** (365.0 / max(total_days, 1))) - 1
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return cagr, max_dd


STRATEGIES = {
    'MarketStructure': {
        'func': market_structure_signals,
        'space': lambda trial: {
            'swing_window': trial.suggest_int('swing_window', 3, 15),
            'min_swings': trial.suggest_int('min_swings', 2, 5),
        }
    },
    'RangeSFP': {
        'func': range_sfp_signals,
        'space': lambda trial: {
            'lookback': trial.suggest_int('lookback', 20, 100),
            'sfp_threshold': trial.suggest_float('sfp_threshold', 0.002, 0.02),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
        }
    },
    'FairValueGaps': {
        'func': fvg_signals,
        'space': lambda trial: {
            'lookback': trial.suggest_int('lookback', 20, 100),
            'proximity_pct': trial.suggest_float('proximity_pct', 0.005, 0.03),
            'max_gap_age': trial.suggest_int('max_gap_age', 10, 50),
        }
    },
    'OrderBlocks': {
        'func': ob_signals,
        'space': lambda trial: {
            'body_threshold': trial.suggest_float('body_threshold', 0.002, 0.015),
            'vol_threshold': trial.suggest_float('vol_threshold', 1.2, 2.5),
            'expansion_factor': trial.suggest_float('expansion_factor', 1.2, 2.5),
            'max_blocks': trial.suggest_int('max_blocks', 3, 10),
        }
    },
    'VolumeProfile': {
        'func': vp_signals,
        'space': lambda trial: {
            'bins': trial.suggest_int('bins', 20, 100),
            'value_area_pct': trial.suggest_float('value_area_pct', 0.5, 0.85),
            'lookback': trial.suggest_int('lookback', 15, 60),
        }
    },
    'SRLevels': {
        'func': sr_signals,
        'space': lambda trial: {
            'swing_window': trial.suggest_int('swing_window', 5, 20),
            'group_threshold': trial.suggest_float('group_threshold', 0.002, 0.015),
            'proximity_pct': trial.suggest_float('proximity_pct', 0.005, 0.03),
            'min_touches': trial.suggest_int('min_touches', 1, 4),
        }
    },
}


def optimize_strategy(name, config, df_is, df_oos, n_trials=200, slow=False):
    """Run Optuna on IS, evaluate on OOS."""
    func = config['func']
    space_fn = config['space']

    def objective(trial):
        params = space_fn(trial)
        sharpe, _ = backtest(df_is, func, **params)
        return sharpe

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    actual_trials = 80 if slow else n_trials
    study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)

    best_params = study.best_params
    is_sharpe = study.best_value
    oos_sharpe, oos_returns = backtest(df_oos, func, **best_params)
    oos_cagr, oos_maxdd = calc_metrics(oos_returns)

    # Statistical significance: bootstrap p-value
    n_boot = 1000
    rng = np.random.RandomState(42)
    oos_ret_arr = oos_returns.values
    count_better = 0
    for _ in range(n_boot):
        shuffled = rng.permutation(oos_ret_arr)
        shuf_sharpe = shuffled.mean() / (shuffled.std() + 1e-10) * np.sqrt(365)
        if shuf_sharpe >= oos_sharpe:
            count_better += 1
    p_value = count_better / n_boot

    return {
        'is_sharpe': round(is_sharpe, 3),
        'oos_sharpe': round(oos_sharpe, 3),
        'oos_cagr': round(oos_cagr * 100, 2),
        'oos_maxdd': round(oos_maxdd * 100, 2),
        'best_params': best_params,
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
    }


def walk_forward(name, config, df, n_folds=10):
    """Expanding walk-forward: for each fold, optimize on expanding IS, test on next OOS chunk."""
    func = config['func']
    space_fn = config['space']
    n = len(df)
    fold_size = n // (n_folds + 1)
    oos_sharpes = []
    # Fewer trials for slow strategies
    wf_trials = 20 if name == 'VolumeProfile' else 50

    for fold in range(n_folds):
        is_end = fold_size * (fold + 2)
        oos_end = min(is_end + fold_size, n)
        if oos_end <= is_end:
            break
        df_is = df.iloc[:is_end]
        df_oos = df.iloc[is_end:oos_end]

        def objective(trial):
            params = space_fn(trial)
            sharpe, _ = backtest(df_is, func, **params)
            return sharpe

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42 + fold))
        study.optimize(objective, n_trials=wf_trials, show_progress_bar=False)
        oos_sharpe, _ = backtest(df_oos, func, **study.best_params)
        oos_sharpes.append(oos_sharpe)

    return {
        'wf_mean_sharpe': round(np.mean(oos_sharpes), 3) if oos_sharpes else 0,
        'wf_std_sharpe': round(np.std(oos_sharpes), 3) if oos_sharpes else 0,
        'wf_sharpes': [round(s, 3) for s in oos_sharpes],
    }


def main():
    # Load data
    data_path = os.path.expanduser('~/Desktop/maestro/data/ohlcv/binance_btc_usdt_1d.csv')
    df = pd.read_csv(data_path)
    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    print(f"Data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # Split 70/30
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()
    print(f"IS: {len(df_is)} rows | OOS: {len(df_oos)} rows\n")

    results = {}
    for name, config in STRATEGIES.items():
        t0 = time.time()
        print(f"⚙️  Optimizing {name}...", end=' ', flush=True)
        res = optimize_strategy(name, config, df_is, df_oos, slow=(name == 'VolumeProfile'))
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) | IS={res['is_sharpe']} OOS={res['oos_sharpe']}")

        print(f"   Walk-forward...", end=' ', flush=True)
        t0 = time.time()
        wf = walk_forward(name, config, df)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) | WF mean={wf['wf_mean_sharpe']}")

        res.update(wf)
        results[name] = res

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Strategy':<18} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS CAGR':>10} {'OOS MaxDD':>10} {'WF Mean':>8} {'p-value':>8} {'Sig':>4}")
    print("-" * 120)
    for name, r in results.items():
        sig = "✅" if r['significant'] else "❌"
        print(f"{name:<18} {r['is_sharpe']:>10.3f} {r['oos_sharpe']:>11.3f} {r['oos_cagr']:>9.2f}% {r['oos_maxdd']:>9.2f}% {r['wf_mean_sharpe']:>8.3f} {r['p_value']:>8.4f} {sig:>4}")
    print("=" * 120)

    # Print best params
    print("\n📋 Best Parameters:")
    for name, r in results.items():
        print(f"  {name}: {r['best_params']}")

    # Save JSON
    out_dir = os.path.expanduser('~/Desktop/maestro/data/optimization')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'price_structure_optuna.json')
    # Convert for JSON serialization
    for k, v in results.items():
        for pk, pv in v['best_params'].items():
            if isinstance(pv, (np.integer,)):
                v['best_params'][pk] = int(pv)
            elif isinstance(pv, (np.floating,)):
                v['best_params'][pk] = float(pv)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved to {out_path}")


if __name__ == '__main__':
    main()
