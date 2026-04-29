"""
Price Structure Ensemble — Test combined signals across multiple methods.
Tests: majority vote, unanimous, weighted score, best-3 combo, and all pairwise combos.
BTC daily + 4H, walk-forward validated.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
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

STRATS = {
    'MS': ms_signals,
    'SFP': sfp_signals,
    'FVG': fvg_signals,
    'OB': ob_signals,
    'VP': vp_signals,
    'SR': sr_signals,
}


def load(tf):
    files = {'1d': 'binance_btc_usdt_1d.csv', '4h': 'binance_btc_usdt_4h.csv'}
    df = pd.read_csv(os.path.join(DATA_DIR, files[tf]), parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_all_signals(df):
    """Generate signals from all 6 strategies, return DataFrame."""
    sigs = {}
    for name, func in STRATS.items():
        try:
            s = func(df)
            sigs[name] = s
        except:
            sigs[name] = pd.Series(0, index=df.index)
    return pd.DataFrame(sigs, index=df.index)


def backtest(df, signals, bpy):
    """Backtest a signal series."""
    daily_ret = df['close'].pct_change().shift(-1)
    pos_changes = signals.diff().abs().fillna(0)
    rets = (signals * daily_ret - pos_changes * COMMISSION).fillna(0)
    return rets


def metrics(rets, bpy):
    if len(rets) == 0 or rets.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'total_ret': 0, 'win_rate': 0}
    total = (1 + rets).prod() - 1
    n_yr = len(rets) / bpy
    cagr = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(bpy)
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    active = rets[rets != 0]
    wr = (active > 0).mean() if len(active) > 0 else 0
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr * 100, 1), 'max_dd': round(dd * 100, 1),
            'total_ret': round(total * 100, 1), 'win_rate': round(wr * 100, 1)}


def walk_forward(df, ensemble_func, bpy, n_folds=10):
    """Walk-forward on ensemble (no optimization needed — ensemble methods are fixed rules)."""
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
        sigs = ensemble_func(df_oos)
        rets = backtest(df_oos, sigs, bpy)
        m = metrics(rets, bpy)
        oos_sharpes.append(m['sharpe'])

    if len(oos_sharpes) < 3:
        return 0, 1.0, []
    arr = np.array(oos_sharpes)
    mean_sh = arr.mean()
    if arr.std() > 0:
        _, p = stats.ttest_1samp(arr, 0)
        p = p / 2  # one-tailed
    else:
        p = 1.0
    return round(mean_sh, 3), round(p, 4), oos_sharpes


# === ENSEMBLE METHODS ===

def majority_vote(df):
    """Signal = sign of sum of all 6 strategies. Need 4+ agreement."""
    all_sigs = get_all_signals(df)
    vote = all_sigs.sum(axis=1)
    result = pd.Series(0, index=df.index)
    result[vote >= 4] = 1
    result[vote <= -4] = -1
    return result


def majority_3(df):
    """Signal = sign of sum, need 3+ agreement (weaker filter)."""
    all_sigs = get_all_signals(df)
    vote = all_sigs.sum(axis=1)
    result = pd.Series(0, index=df.index)
    result[vote >= 3] = 1
    result[vote <= -3] = -1
    return result


def any_2_agree(df):
    """Signal = 1 if any 2+ strategies agree, -1 if any 2+ agree on short."""
    all_sigs = get_all_signals(df)
    vote = all_sigs.sum(axis=1)
    result = pd.Series(0, index=df.index)
    result[vote >= 2] = 1
    result[vote <= -2] = -1
    return result


def weighted_score(df):
    """Weight by prior multi-asset OOS performance: SFP > MS > OB > FVG > VP > SR."""
    weights = {'MS': 0.23, 'SFP': 0.52, 'OB': 0.08, 'FVG': -0.03, 'VP': -0.03, 'SR': 0.0}
    all_sigs = get_all_signals(df)
    score = sum(all_sigs[name] * w for name, w in weights.items())
    result = pd.Series(0, index=df.index)
    result[score > 0.3] = 1
    result[score < -0.3] = -1
    return result


def top3_vote(df):
    """Only use the 3 best individual strategies: MS, SFP, OB. Need 2/3."""
    all_sigs = get_all_signals(df)
    top3 = all_sigs[['MS', 'SFP', 'OB']].sum(axis=1)
    result = pd.Series(0, index=df.index)
    result[top3 >= 2] = 1
    result[top3 <= -2] = -1
    return result


def ms_plus_sfp(df):
    """MarketStructure + RangeSFP: both must agree."""
    all_sigs = get_all_signals(df)
    result = pd.Series(0, index=df.index)
    both_long = (all_sigs['MS'] == 1) & (all_sigs['SFP'] == 1)
    both_short = (all_sigs['MS'] == -1) & (all_sigs['SFP'] == -1)
    result[both_long] = 1
    result[both_short] = -1
    return result


def ms_filter_sfp(df):
    """Use MS for trend direction, SFP for entry timing."""
    all_sigs = get_all_signals(df)
    result = pd.Series(0, index=df.index)
    # Long: MS says uptrend AND SFP gives bullish signal
    long_cond = (all_sigs['MS'] == 1) & (all_sigs['SFP'] == 1)
    short_cond = (all_sigs['MS'] == -1) & (all_sigs['SFP'] == -1)
    # Also: MS uptrend + any of FVG/OB/SR giving long
    long_support = (all_sigs['MS'] == 1) & ((all_sigs['FVG'] == 1) | (all_sigs['OB'] == 1))
    short_support = (all_sigs['MS'] == -1) & ((all_sigs['FVG'] == -1) | (all_sigs['OB'] == -1))
    result[long_cond | long_support] = 1
    result[short_cond | short_support] = -1
    return result


def contrarian_ensemble(df):
    """If all strategies are wrong individually, maybe the inverse works.
    Take inverse of majority vote."""
    all_sigs = get_all_signals(df)
    vote = all_sigs.sum(axis=1)
    result = pd.Series(0, index=df.index)
    result[vote >= 3] = -1  # Inverse!
    result[vote <= -3] = 1
    return result


# All pairwise combinations
def make_pair_func(name1, name2):
    def pair_func(df):
        all_sigs = get_all_signals(df)
        result = pd.Series(0, index=df.index)
        both_long = (all_sigs[name1] == 1) & (all_sigs[name2] == 1)
        both_short = (all_sigs[name1] == -1) & (all_sigs[name2] == -1)
        result[both_long] = 1
        result[both_short] = -1
        return result
    return pair_func


ENSEMBLES = {
    'Majority(4/6)': majority_vote,
    'Majority(3/6)': majority_3,
    'Any2Agree': any_2_agree,
    'WeightedScore': weighted_score,
    'Top3Vote(2/3)': top3_vote,
    'MS+SFP(both)': ms_plus_sfp,
    'MS_filter_SFP+': ms_filter_sfp,
    'Contrarian(inv3)': contrarian_ensemble,
}

# Add all 15 pairwise combinations
strat_names = list(STRATS.keys())
for i, n1 in enumerate(strat_names):
    for n2 in strat_names[i+1:]:
        ENSEMBLES[f'{n1}+{n2}'] = make_pair_func(n1, n2)


def run_all(tf, bpy):
    df = load(tf)
    print(f"\n{'='*80}")
    print(f"  {tf.upper()} — {len(df)} bars — ENSEMBLE TESTS")
    print(f"{'='*80}")

    # Buy & hold
    bh_rets = df['close'].pct_change().fillna(0)
    bh = metrics(bh_rets, bpy)
    print(f"  Buy & Hold: Sharpe {bh['sharpe']}, CAGR {bh['cagr']}%, MaxDD {bh['max_dd']}%\n")

    results = []
    for name, func in ENSEMBLES.items():
        # Full sample
        sigs = func(df)
        rets = backtest(df, sigs, bpy)
        m = metrics(rets, bpy)

        # Walk-forward
        wf_mean, wf_p, wf_folds = walk_forward(df, func, bpy)

        n_trades = (sigs.diff().abs() > 0).sum()
        pct_invested = round((sigs != 0).mean() * 100, 1)

        results.append({
            'name': name, 'tf': tf,
            'full_sharpe': m['sharpe'], 'full_cagr': m['cagr'], 'full_dd': m['max_dd'],
            'win_rate': m['win_rate'],
            'oos_sharpe': wf_mean, 'p_value': wf_p, 'sig': wf_p < 0.05,
            'trades': int(n_trades), 'pct_invested': pct_invested,
        })

    # Sort by OOS Sharpe
    results.sort(key=lambda x: x['oos_sharpe'], reverse=True)

    print(f"  {'Ensemble':<22} {'Full Sh':>7} {'CAGR':>7} {'MaxDD':>7} {'WinR':>5} {'OOS Sh':>7} {'p-val':>7} {'Sig':>3} {'Trades':>6} {'Inv%':>5}")
    print(f"  {'-'*85}")
    for r in results:
        sig = '✅' if r['sig'] else '❌'
        print(f"  {r['name']:<22} {r['full_sharpe']:>7.3f} {r['full_cagr']:>6.1f}% {r['full_dd']:>6.1f}% {r['win_rate']:>4.1f}% {r['oos_sharpe']:>7.3f} {r['p_value']:>7.4f} {sig:>3} {r['trades']:>6} {r['pct_invested']:>4.1f}%")

    return results


def main():
    print("=" * 80)
    print("  PRICE STRUCTURE ENSEMBLE — Combination Testing")
    print("  BTC Daily + 4H | Walk-Forward Validated")
    print("=" * 80)

    all_results = []
    all_results.extend(run_all('1d', 365))
    all_results.extend(run_all('4h', 2190))

    # Grand ranking
    all_results.sort(key=lambda x: x['oos_sharpe'], reverse=True)
    print(f"\n{'='*80}")
    print(f"  GRAND RANKING — Top 15 Ensembles by OOS Sharpe")
    print(f"{'='*80}")
    print(f"  {'Rank':>4} {'Ensemble':<22} {'TF':>3} {'Full Sh':>7} {'OOS Sh':>7} {'p-val':>7} {'Sig':>3}")
    print(f"  {'-'*58}")
    for i, r in enumerate(all_results[:15]):
        sig = '✅' if r['sig'] else '❌'
        print(f"  {i+1:>4} {r['name']:<22} {r['tf']:>3} {r['full_sharpe']:>7.3f} {r['oos_sharpe']:>7.3f} {r['p_value']:>7.4f} {sig:>3}")

    sig_results = [r for r in all_results if r['sig']]
    print(f"\n  SIGNIFICANT (p<0.05): {len(sig_results)} of {len(all_results)}")
    if sig_results:
        for r in sig_results:
            print(f"    ✅ {r['name']} ({r['tf']}): OOS {r['oos_sharpe']}, p={r['p_value']}")

    # Save
    out = os.path.expanduser('~/Desktop/maestro/data/backtest_results/ps_ensemble_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
