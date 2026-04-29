#!/usr/bin/env python3
"""Momentum strategy stress test gauntlet."""

import os, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
p = print

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

TOKENS = ['arb','avax','btc','eth','fet','inj','link','op','sol','sui']
N_FOLDS = 10

def load_ohlcv():
    frames = {}
    for t in TOKENS:
        df = pd.read_csv(f"{DATA_DIR}/binance_{t}_usdt_1d.csv", parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        frames[t] = df
    return frames

def build_returns_matrix(ohlcv):
    closes = pd.DataFrame({t: ohlcv[t]['close'] for t in TOKENS})
    closes = closes.sort_index().dropna(how='all')
    returns = closes.pct_change()
    return closes, returns

def calc_sharpe(returns):
    if len(returns) < 30 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(365)

def calc_cagr(returns):
    cum = (1 + returns).prod()
    n_years = len(returns) / 365
    if n_years <= 0 or cum <= 0:
        return 0.0
    return cum ** (1/n_years) - 1

def calc_maxdd(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def walk_forward_expanding(returns, n_folds=N_FOLDS):
    n = len(returns)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos_returns = []
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, n)
        if oos_start >= n:
            break
        oos_returns.append(returns.iloc[oos_start:oos_end])
    return oos_returns

def wf_stats(returns):
    folds = walk_forward_expanding(returns)
    if not folds:
        return 0.0, 1.0
    fold_sharpes = [calc_sharpe(f) for f in folds]
    oos_all = pd.concat(folds)
    oos_sharpe = calc_sharpe(oos_all)
    if len(fold_sharpes) >= 3:
        t_stat, p_val = stats.ttest_1samp(fold_sharpes, 0)
        p_val = p_val / 2 if t_stat > 0 else 1.0
    else:
        p_val = 1.0
    return oos_sharpe, p_val, fold_sharpes


def run_momentum(closes, returns, lb, reb, commission_bps=20, realistic_commission=True,
                 shuffle_rankings=False, rng=None):
    """
    Run cross-asset momentum strategy. Returns (series, metadata_dict).
    If realistic_commission: commission = turnover_pct * commission_bps/10000
    If not realistic (original): commission = commission_bps/10000 flat on every rebalance
    """
    n_tokens = len(TOKENS)
    q_size = max(n_tokens // 5, 1)
    trail_ret = closes.pct_change(lb)
    dates = trail_ret.dropna(how='all').index[lb:]

    strat_rets = []
    last_reb = -999
    long_tokens = []
    short_tokens = []
    turnovers = []
    token_flips = []
    holding_log = defaultdict(list)  # token -> list of consecutive hold counts
    current_holdings = {}  # token -> start_index

    for i, dt in enumerate(dates):
        row = trail_ret.loc[dt].dropna()
        daily_ret = returns.loc[dt].dropna()

        if len(row) < 5:
            strat_rets.append(0.0)
            continue

        if i - last_reb >= reb:
            ranked = row.sort_values(ascending=False)
            if shuffle_rankings and rng is not None:
                idx = list(ranked.index)
                rng.shuffle(idx)
                ranked = ranked.reindex(idx)

            new_long = set(ranked.index[:q_size].tolist())
            new_short = set(ranked.index[-q_size:].tolist())

            old_long = set(long_tokens)
            old_short = set(short_tokens)
            old_all = old_long | old_short
            new_all = new_long | new_short

            if old_all:
                changed = len((new_long - old_long) | (old_long - new_long) |
                              (new_short - old_short) | (old_short - new_short))
                total_positions = max(len(old_all | new_all), 1)
                turnover_pct = changed / total_positions
            else:
                turnover_pct = 1.0  # initial

            turnovers.append(turnover_pct)
            n_flipped = len((new_long - old_long) | (new_short - old_short))
            token_flips.append(n_flipped)

            # Track holding periods
            for t in old_all - new_all:
                if t in current_holdings:
                    holding_log[t].append(i - current_holdings[t])
                    del current_holdings[t]
            for t in new_all - old_all:
                current_holdings[t] = i

            long_tokens = list(new_long)
            short_tokens = list(new_short)
            last_reb = i

            if realistic_commission:
                comm = turnover_pct * commission_bps / 10000
            else:
                comm = commission_bps / 10000
        else:
            comm = 0.0

        long_ret = daily_ret.reindex(long_tokens).mean() if long_tokens else 0.0
        short_ret = daily_ret.reindex(short_tokens).mean() if short_tokens else 0.0
        spread = long_ret - short_ret - comm
        strat_rets.append(spread if not np.isnan(spread) else 0.0)

    # Close remaining holdings
    for t, start in current_holdings.items():
        holding_log[t].append(len(dates) - start)

    sr = pd.Series(strat_rets, index=dates)
    all_holds = [h for holds in holding_log.values() for h in holds]
    meta = {
        'turnovers': turnovers,
        'token_flips': token_flips,
        'holding_periods': all_holds,
    }
    return sr, meta


# ══════════════════════════════════════════════════════════════════════
# TEST 1: Realistic Commission Model
# ══════════════════════════════════════════════════════════════════════
def test1_commissions(closes, returns):
    p("\n" + "="*80)
    p("TEST 1: REALISTIC COMMISSION MODEL")
    p("="*80)
    p(f"{'Params':<16} | {'Orig Sharpe':>11} | {'Corr Sharpe':>11} | {'Corr30 Sharpe':>13} | {'Avg Turn%':>9} | {'OOS p-val':>9}")
    p("-"*85)

    results = []
    for lb in [7, 14, 30, 60]:
        for reb in [1, 7, 14]:
            # Original (flat commission)
            sr_orig, _ = run_momentum(closes, returns, lb, reb, 20, realistic_commission=False)
            orig_sharpe, orig_p, _ = wf_stats(sr_orig)

            # Corrected 20bps
            sr_corr, meta = run_momentum(closes, returns, lb, reb, 20, realistic_commission=True)
            corr_sharpe, corr_p, _ = wf_stats(sr_corr)

            # Corrected 30bps (conservative)
            sr_corr30, _ = run_momentum(closes, returns, lb, reb, 30, realistic_commission=True)
            corr30_sharpe, corr30_p, _ = wf_stats(sr_corr30)

            avg_turn = np.mean(meta['turnovers']) * 100 if meta['turnovers'] else 0

            label = f"lb={lb},reb={reb}"
            p(f"{label:<16} | {orig_sharpe:>11.3f} | {corr_sharpe:>11.3f} | {corr30_sharpe:>13.3f} | {avg_turn:>8.1f}% | {corr_p:>9.4f}")
            results.append({
                'lb': lb, 'reb': reb,
                'orig_oos_sharpe': round(orig_sharpe, 4),
                'corrected_20bps_sharpe': round(corr_sharpe, 4),
                'corrected_30bps_sharpe': round(corr30_sharpe, 4),
                'avg_turnover_pct': round(avg_turn, 2),
                'p_value': round(corr_p, 4),
            })
    return results


# ══════════════════════════════════════════════════════════════════════
# TEST 2: Permutation Test
# ══════════════════════════════════════════════════════════════════════
def test2_permutation(closes, returns, lb=14, reb=7, n_perms=500):
    p("\n" + "="*80)
    p(f"TEST 2: PERMUTATION TEST (lb={lb}, reb={reb}, {n_perms} shuffles)")
    p("="*80)

    sr_real, _ = run_momentum(closes, returns, lb, reb, 20, realistic_commission=True)
    real_sharpe, _, _ = wf_stats(sr_real)
    p(f"Real strategy OOS Sharpe: {real_sharpe:.3f}")

    rng = np.random.default_rng(42)
    shuffle_sharpes = []
    for i in range(n_perms):
        sr_shuf, _ = run_momentum(closes, returns, lb, reb, 20, realistic_commission=True,
                                   shuffle_rankings=True, rng=rng)
        sh, _, _ = wf_stats(sr_shuf)
        shuffle_sharpes.append(sh)
        if (i+1) % 100 == 0:
            p(f"  ...completed {i+1}/{n_perms} permutations")

    shuffle_sharpes = np.array(shuffle_sharpes)
    p_random = np.mean(shuffle_sharpes >= real_sharpe)
    p(f"\nResults:")
    p(f"  Real OOS Sharpe:     {real_sharpe:.3f}")
    p(f"  Random mean Sharpe:  {np.mean(shuffle_sharpes):.3f}")
    p(f"  Random median:       {np.median(shuffle_sharpes):.3f}")
    p(f"  Random 95th pctl:    {np.percentile(shuffle_sharpes, 95):.3f}")
    p(f"  p_random:            {p_random:.4f}  {'✅ Signal is real' if p_random < 0.05 else '❌ Likely noise'}")

    return {
        'lb': lb, 'reb': reb,
        'real_sharpe': round(real_sharpe, 4),
        'random_mean': round(float(np.mean(shuffle_sharpes)), 4),
        'random_median': round(float(np.median(shuffle_sharpes)), 4),
        'random_95th': round(float(np.percentile(shuffle_sharpes, 95)), 4),
        'p_random': round(float(p_random), 4),
        'significant': p_random < 0.05,
    }


# ══════════════════════════════════════════════════════════════════════
# TEST 3: Regime Dependence
# ══════════════════════════════════════════════════════════════════════
def test3_regime(closes, returns, lb=14, reb=7):
    p("\n" + "="*80)
    p("TEST 3: REGIME DEPENDENCE (BTC 200d SMA)")
    p("="*80)

    btc_close = closes['btc']
    btc_sma200 = btc_close.rolling(200).mean()
    bull_mask = btc_close > btc_sma200

    sr, _ = run_momentum(closes, returns, lb, reb, 20, realistic_commission=True)

    bull_rets = sr[bull_mask.reindex(sr.index).fillna(False)]
    bear_rets = sr[~bull_mask.reindex(sr.index).fillna(True)]

    bull_sharpe = calc_sharpe(bull_rets) if len(bull_rets) > 30 else 0.0
    bear_sharpe = calc_sharpe(bear_rets) if len(bear_rets) > 30 else 0.0
    full_sharpe = calc_sharpe(sr)

    p(f"  Full period:  Sharpe={full_sharpe:.3f}  ({len(sr)} days)")
    p(f"  Bull regime:  Sharpe={bull_sharpe:.3f}  ({len(bull_rets)} days, {len(bull_rets)/len(sr)*100:.0f}%)")
    p(f"  Bear regime:  Sharpe={bear_sharpe:.3f}  ({len(bear_rets)} days, {len(bear_rets)/len(sr)*100:.0f}%)")

    if bear_sharpe > 0.5:
        p("  ✅ Works in both regimes — genuine alpha")
    elif bear_sharpe > 0:
        p("  ⚠️  Weak in bear — partially beta-driven")
    else:
        p("  ❌ Negative in bear — this is disguised beta")

    return {
        'full_sharpe': round(full_sharpe, 4),
        'bull_sharpe': round(bull_sharpe, 4),
        'bear_sharpe': round(bear_sharpe, 4),
        'bull_days': len(bull_rets),
        'bear_days': len(bear_rets),
    }


# ══════════════════════════════════════════════════════════════════════
# TEST 4: Capacity / Concentration
# ══════════════════════════════════════════════════════════════════════
def test4_capacity(closes, returns, lb=14, reb=7):
    p("\n" + "="*80)
    p("TEST 4: CAPACITY / CONCENTRATION")
    p("="*80)

    sr, meta = run_momentum(closes, returns, lb, reb, 20, realistic_commission=True)

    flips = meta['token_flips']
    holds = meta['holding_periods']

    p(f"  Tokens flipped per rebalance:")
    p(f"    Mean:   {np.mean(flips):.1f}")
    p(f"    Median: {np.median(flips):.1f}")
    p(f"    Max:    {np.max(flips)}")

    if holds:
        p(f"  Holding period (days, assuming reb={reb}d):")
        hold_days = [h * reb for h in holds]  # approximate
        p(f"    Mean:   {np.mean(hold_days):.1f} days")
        p(f"    Median: {np.median(hold_days):.1f} days")

    # Correlation with equal-weight basket
    eq_basket = returns[TOKENS].mean(axis=1)
    common = sr.index.intersection(eq_basket.index)
    corr = sr.loc[common].corr(eq_basket.loc[common])
    p(f"  Correlation with equal-weight basket: {corr:.3f}")
    if abs(corr) > 0.5:
        p("  ❌ High market correlation — mostly beta")
    elif abs(corr) > 0.3:
        p("  ⚠️  Moderate market correlation")
    else:
        p("  ✅ Low market correlation — genuine alpha")

    return {
        'flips_mean': round(float(np.mean(flips)), 2),
        'flips_median': round(float(np.median(flips)), 2),
        'flips_max': int(np.max(flips)),
        'hold_mean_days': round(float(np.mean(hold_days)), 1) if holds else None,
        'hold_median_days': round(float(np.median(hold_days)), 1) if holds else None,
        'basket_correlation': round(float(corr), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# TEST 5: Slippage Sensitivity
# ══════════════════════════════════════════════════════════════════════
def test5_slippage(closes, returns, lb=14, reb=7):
    p("\n" + "="*80)
    p("TEST 5: SLIPPAGE SENSITIVITY")
    p("="*80)
    p(f"{'Cost (bps)':<12} | {'OOS Sharpe':>10} | {'CAGR':>8} | {'p-value':>8} | Edge?")
    p("-"*60)

    results = []
    edge_gone = None
    for bps in [0, 10, 20, 30, 50, 100]:
        sr, _ = run_momentum(closes, returns, lb, reb, bps, realistic_commission=True)
        sharpe, pv, _ = wf_stats(sr)
        cagr = calc_cagr(sr)
        has_edge = sharpe > 0.5 and pv < 0.1
        if not has_edge and edge_gone is None:
            edge_gone = bps
        p(f"{bps:<12} | {sharpe:>10.3f} | {cagr:>7.1%} | {pv:>8.4f} | {'✅' if has_edge else '❌'}")
        results.append({
            'cost_bps': bps,
            'oos_sharpe': round(sharpe, 4),
            'cagr': round(cagr, 4),
            'p_value': round(pv, 4),
        })

    if edge_gone is not None:
        p(f"\n  Edge disappears at ~{edge_gone}bps")
    else:
        p(f"\n  Edge survives even at 100bps! 🔥")

    return results, edge_gone


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    p("Loading data...")
    ohlcv = load_ohlcv()
    closes, returns = build_returns_matrix(ohlcv)
    p(f"Loaded {len(TOKENS)} tokens, {len(closes)} days ({closes.index[0].date()} to {closes.index[-1].date()})")

    all_results = {}

    p("\n" + "█"*80)
    p("  MOMENTUM STRATEGY GAUNTLET")
    p("█"*80)

    all_results['test1_commissions'] = test1_commissions(closes, returns)
    all_results['test2_permutation'] = test2_permutation(closes, returns)
    all_results['test3_regime'] = test3_regime(closes, returns)
    all_results['test4_capacity'] = test4_capacity(closes, returns)
    slippage_results, edge_break = test5_slippage(closes, returns)
    all_results['test5_slippage'] = {'results': slippage_results, 'edge_breaks_at_bps': edge_break}

    # ── Summary ──
    p("\n" + "█"*80)
    p("  VERDICT")
    p("█"*80)

    # Find best corrected combo from test1
    best = max(all_results['test1_commissions'], key=lambda x: x['corrected_20bps_sharpe'])
    p(f"  Best combo: lb={best['lb']}, reb={best['reb']}")
    p(f"  Original Sharpe:  {best['orig_oos_sharpe']:.3f}")
    p(f"  Corrected Sharpe: {best['corrected_20bps_sharpe']:.3f}  (avg turnover {best['avg_turnover_pct']:.1f}%)")
    perm = all_results['test2_permutation']
    p(f"  Permutation p:    {perm['p_random']:.4f}  ({'Signal real' if perm['significant'] else 'Likely noise'})")
    regime = all_results['test3_regime']
    p(f"  Bull Sharpe:      {regime['bull_sharpe']:.3f}")
    p(f"  Bear Sharpe:      {regime['bear_sharpe']:.3f}")
    cap = all_results['test4_capacity']
    p(f"  Basket corr:      {cap['basket_correlation']:.3f}")
    if edge_break:
        p(f"  Edge dies at:     {edge_break}bps")
    else:
        p(f"  Edge survives:    100bps+")

    elapsed = time.time() - t0
    p(f"\n  Completed in {elapsed:.1f}s")

    out_path = os.path.join(OUT_DIR, "momentum_gauntlet.json")
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    p(f"  Results saved to {out_path}")
