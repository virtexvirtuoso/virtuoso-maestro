"""
Price Structure Ensemble — Top combos across 4 timeframes + 8 assets.
Tests the most promising ensembles from the 1D results on all TFs and assets.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))

from strategies.technical.market_structure import generate_signals as ms_signals
from strategies.technical.range_sfp import generate_signals as sfp_signals
from strategies.technical.fair_value_gaps import generate_signals as fvg_signals
from strategies.technical.order_blocks import generate_signals as ob_signals
from strategies.technical.volume_profile import generate_signals as vp_signals
from strategies.technical.sr_levels import generate_signals as sr_signals

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
COMMISSION = 0.001

STRATS = {'MS': ms_signals, 'SFP': sfp_signals, 'FVG': fvg_signals,
          'OB': ob_signals, 'VP': vp_signals, 'SR': sr_signals}

# Timeframes with annualization
TFS = {
    '1d':  365,
    '4h':  2190,
    '1h':  8760,
    '15m': 35040,
}

ASSETS = ['btc', 'eth', 'sol', 'sui', 'link', 'render', 'avax', 'inj', 'op']


def load(asset, tf):
    path = os.path.join(DATA_DIR, f'binance_{asset}_usdt_{tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    # Cap for speed
    if tf in ('1h', '15m') and len(df) > 10000:
        df = df.iloc[-10000:]
    return df


def get_all_signals(df):
    sigs = {}
    for name, func in STRATS.items():
        try:
            sigs[name] = func(df)
        except:
            sigs[name] = pd.Series(0, index=df.index)
    return pd.DataFrame(sigs, index=df.index)


def backtest(df, signals, bpy):
    daily_ret = df['close'].pct_change().shift(-1)
    pos_changes = signals.diff().abs().fillna(0)
    return (signals * daily_ret - pos_changes * COMMISSION).fillna(0)


def metrics(rets, bpy):
    if len(rets) == 0 or rets.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    total = (1 + rets).prod() - 1
    n_yr = len(rets) / bpy
    sharpe = rets.mean() / rets.std() * np.sqrt(bpy)
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr * 100, 1), 'max_dd': round(dd * 100, 1)}


def walk_forward(df, ens_func, bpy, n_folds=10):
    n = len(df)
    min_is = max(200, n // (n_folds + 2))
    fold_size = (n - min_is) // n_folds
    oos_sharpes = []
    for fold in range(n_folds):
        is_end = min_is + (fold + 1) * fold_size
        oos_end = min(is_end + fold_size, n)
        if is_end >= n or oos_end <= is_end or (oos_end - is_end) < 10:
            break
        df_oos = df.iloc[is_end:oos_end]
        sigs = ens_func(df_oos)
        rets = backtest(df_oos, sigs, bpy)
        m = metrics(rets, bpy)
        oos_sharpes.append(m['sharpe'])
    if len(oos_sharpes) < 3:
        return 0, 1.0
    arr = np.array(oos_sharpes)
    if arr.std() > 0:
        _, p = stats.ttest_1samp(arr, 0)
        p = p / 2
    else:
        p = 1.0
    return round(arr.mean(), 3), round(p, 4)


# === ENSEMBLE DEFINITIONS ===

def ms_sfp_both(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['MS'] == 1) & (s['SFP'] == 1)] = 1
    r[(s['MS'] == -1) & (s['SFP'] == -1)] = -1
    return r

def ms_filter_multi(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['MS'] == 1) & ((s['SFP'] == 1) | (s['FVG'] == 1) | (s['OB'] == 1))] = 1
    r[(s['MS'] == -1) & ((s['SFP'] == -1) | (s['FVG'] == -1) | (s['OB'] == -1))] = -1
    return r

def contrarian_3(df):
    s = get_all_signals(df)
    vote = s.sum(axis=1)
    r = pd.Series(0, index=df.index)
    r[vote >= 3] = -1
    r[vote <= -3] = 1
    return r

def majority_3(df):
    s = get_all_signals(df)
    vote = s.sum(axis=1)
    r = pd.Series(0, index=df.index)
    r[vote >= 3] = 1
    r[vote <= -3] = -1
    return r

def top3_vote(df):
    s = get_all_signals(df)
    vote = s[['MS', 'SFP', 'OB']].sum(axis=1)
    r = pd.Series(0, index=df.index)
    r[vote >= 2] = 1
    r[vote <= -2] = -1
    return r

def sfp_vp(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['SFP'] == 1) & (s['VP'] == 1)] = 1
    r[(s['SFP'] == -1) & (s['VP'] == -1)] = -1
    return r

def ms_ob(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['MS'] == 1) & (s['OB'] == 1)] = 1
    r[(s['MS'] == -1) & (s['OB'] == -1)] = -1
    return r

def sfp_fvg(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['SFP'] == 1) & (s['FVG'] == 1)] = 1
    r[(s['SFP'] == -1) & (s['FVG'] == -1)] = -1
    return r

def ms_fvg(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['MS'] == 1) & (s['FVG'] == 1)] = 1
    r[(s['MS'] == -1) & (s['FVG'] == -1)] = -1
    return r

def ob_fvg(df):
    s = get_all_signals(df)
    r = pd.Series(0, index=df.index)
    r[(s['OB'] == 1) & (s['FVG'] == 1)] = 1
    r[(s['OB'] == -1) & (s['FVG'] == -1)] = -1
    return r


ENSEMBLES = {
    'MS+SFP': ms_sfp_both,
    'MS_filter_multi': ms_filter_multi,
    'Contrarian(3)': contrarian_3,
    'Majority(3/6)': majority_3,
    'Top3(MS+SFP+OB)': top3_vote,
    'SFP+VP': sfp_vp,
    'MS+OB': ms_ob,
    'SFP+FVG': sfp_fvg,
    'MS+FVG': ms_fvg,
    'OB+FVG': ob_fvg,
}


def main():
    print("=" * 90)
    print("  PRICE STRUCTURE ENSEMBLE — Multi-TF × Multi-Asset")
    print("  10 ensembles × 4 timeframes × 9 assets = 360 tests")
    print("=" * 90)

    all_results = []

    for asset in ASSETS:
        print(f"\n{'='*90}")
        print(f"  {asset.upper()}")
        print(f"{'='*90}")

        for tf, bpy in TFS.items():
            df = load(asset, tf)
            if df is None or len(df) < 250:
                continue

            print(f"\n  {tf}: {len(df)} bars", end='')

            # B&H
            bh_rets = df['close'].pct_change().fillna(0)
            bh = metrics(bh_rets, bpy)
            print(f" | B&H Sharpe: {bh['sharpe']}")

            for ens_name, ens_func in ENSEMBLES.items():
                sigs = ens_func(df)
                rets = backtest(df, sigs, bpy)
                m = metrics(rets, bpy)
                wf_sh, wf_p = walk_forward(df, ens_func, bpy)
                pct = round((sigs != 0).mean() * 100, 1)

                all_results.append({
                    'asset': asset.upper(), 'tf': tf, 'ensemble': ens_name,
                    'full_sharpe': m['sharpe'], 'full_cagr': m['cagr'], 'full_dd': m['max_dd'],
                    'oos_sharpe': wf_sh, 'p_value': wf_p, 'sig': wf_p < 0.05,
                    'bh_sharpe': bh['sharpe'], 'pct_invested': pct,
                })

    # === SUMMARIES ===
    print(f"\n\n{'='*90}")
    print(f"  SIGNIFICANT RESULTS (p < 0.05, OOS > 0)")
    print(f"{'='*90}")
    sig_pos = [r for r in all_results if r['sig'] and r['oos_sharpe'] > 0]
    sig_pos.sort(key=lambda x: x['oos_sharpe'], reverse=True)
    if sig_pos:
        print(f"  {'Asset':<6} {'TF':>3} {'Ensemble':<20} {'Full Sh':>7} {'OOS Sh':>7} {'CAGR':>7} {'p-val':>7} {'B&H':>5} {'Inv%':>5}")
        print(f"  {'-'*75}")
        for r in sig_pos:
            print(f"  {r['asset']:<6} {r['tf']:>3} {r['ensemble']:<20} {r['full_sharpe']:>7.3f} {r['oos_sharpe']:>7.3f} {r['full_cagr']:>6.1f}% {r['p_value']:>7.4f} {r['bh_sharpe']:>5.2f} {r['pct_invested']:>4.1f}%")
    else:
        print("  None found.")

    # Near-misses
    print(f"\n  NEAR-MISSES (0.05 < p < 0.10, OOS > 0)")
    near = [r for r in all_results if 0.05 < r['p_value'] < 0.10 and r['oos_sharpe'] > 0]
    near.sort(key=lambda x: x['oos_sharpe'], reverse=True)
    if near:
        for r in near[:15]:
            print(f"  {r['asset']:<6} {r['tf']:>3} {r['ensemble']:<20} OOS {r['oos_sharpe']:>6.3f} p={r['p_value']:.4f}")

    # Best ensemble across all assets
    print(f"\n{'='*90}")
    print(f"  ENSEMBLE RANKING — Mean OOS Sharpe Across All Assets/TFs")
    print(f"{'='*90}")
    ens_scores = {}
    for r in all_results:
        key = r['ensemble']
        if key not in ens_scores:
            ens_scores[key] = []
        ens_scores[key].append(r['oos_sharpe'])

    ranked = sorted(ens_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"  {'Ensemble':<20} {'Mean OOS':>8} {'Median':>8} {'Pos%':>6} {'Count':>5}")
    print(f"  {'-'*50}")
    for name, scores in ranked:
        arr = np.array(scores)
        pos_pct = (arr > 0).mean() * 100
        print(f"  {name:<20} {arr.mean():>8.3f} {np.median(arr):>8.3f} {pos_pct:>5.1f}% {len(arr):>5}")

    # Best TF per ensemble
    print(f"\n  BEST TIMEFRAME PER ENSEMBLE")
    for ens_name in ENSEMBLES:
        tf_scores = {}
        for r in all_results:
            if r['ensemble'] == ens_name:
                tf = r['tf']
                if tf not in tf_scores:
                    tf_scores[tf] = []
                tf_scores[tf].append(r['oos_sharpe'])
        if tf_scores:
            best_tf = max(tf_scores.items(), key=lambda x: np.mean(x[1]))
            print(f"  {ens_name:<20} → {best_tf[0]} (mean OOS {np.mean(best_tf[1]):.3f})")

    # Save
    out = os.path.expanduser('~/Desktop/maestro/data/backtest_results/ps_ensemble_mtf.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
    print(f"Total combos tested: {len(all_results)}")


if __name__ == '__main__':
    main()
