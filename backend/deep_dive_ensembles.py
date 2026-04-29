#!/usr/bin/env python3
"""Deep-dive analysis of 3 significant ensemble results from price structure research."""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
OUT_DIR = os.path.expanduser('~/Desktop/maestro/data/research')
os.makedirs(OUT_DIR, exist_ok=True)

from strategies.technical.market_structure import generate_signals as ms_signals
from strategies.technical.range_sfp import generate_signals as sfp_signals
from strategies.technical.fair_value_gaps import generate_signals as fvg_signals
from strategies.technical.mean_reversion import generate_signals as mr_signals

# ─── Helpers ────────────────────────────────────────────────────────
def load_ohlcv(asset, tf):
    path = os.path.join(DATA_DIR, f'binance_{asset.lower()}_usdt_{tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def sharpe(returns, periods=365):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))

def rolling_sharpe(returns, window=90, periods=365):
    m = returns.rolling(window).mean()
    s = returns.rolling(window).std().replace(0, np.nan)
    return (m / s * np.sqrt(periods))

def commission_adjusted_returns(signals, log_returns, commission_pct):
    trades = signals.diff().abs().fillna(0)
    cost = trades * commission_pct
    return signals.shift(1).fillna(0) * log_returns - cost

def walk_forward(signals_func, df, n_folds=10):
    """Walk-forward: train not needed (fixed signals), just split OOS."""
    n = len(df)
    fold_size = n // n_folds
    oos_sharpes = []
    log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    sig = signals_func(df)
    for i in range(n_folds):
        start = i * fold_size
        end = min((i + 1) * fold_size, n)
        fold_ret = (sig.shift(1).fillna(0) * log_ret).iloc[start:end]
        oos_sharpes.append(sharpe(fold_ret))
    return oos_sharpes

P = lambda *a, **kw: print(*a, flush=True, **kw)

# ─── Ensemble Definitions ──────────────────────────────────────────
def ensemble_ms_sfp(df):
    """OP 1D: MarketStructure + SFP ensemble (majority vote)"""
    s1 = ms_signals(df)
    s2 = sfp_signals(df)
    combined = s1 + s2
    return combined.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

def ensemble_sfp_fvg(df):
    """SOL 4H: SFP + FVG ensemble"""
    s1 = sfp_signals(df)
    s2 = fvg_signals(df)
    combined = s1 + s2
    return combined.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

def ensemble_contrarian(df):
    """AVAX 1D: Contrarian — mean reversion + inverted market structure"""
    s1 = mr_signals(df)
    s2 = -ms_signals(df)  # inverted
    combined = s1 + s2
    return combined.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

WINNERS = [
    {'name': 'OP 1D MS+SFP', 'asset': 'op', 'tf': '1d', 'func': ensemble_ms_sfp,
     'oos_sharpe': 1.347, 'pval': 0.037, 'periods': 365},
    {'name': 'SOL 4H SFP+FVG', 'asset': 'sol', 'tf': '4h', 'func': ensemble_sfp_fvg,
     'oos_sharpe': 0.974, 'pval': 0.042, 'periods': 365*6},
    {'name': 'AVAX 1D Contrarian', 'asset': 'avax', 'tf': '1d', 'func': ensemble_contrarian,
     'oos_sharpe': 0.960, 'pval': 0.041, 'periods': 365},
]

all_results = {}

# Load BTC daily for regime analysis
btc_daily = load_ohlcv('btc', '1d')

for w in WINNERS:
    name = w['name']
    P(f"\n{'='*80}")
    P(f"  {name}")
    P(f"  Reported OOS Sharpe: {w['oos_sharpe']:.3f}, p-value: {w['pval']:.3f}")
    P(f"{'='*80}")

    df = load_ohlcv(w['asset'], w['tf'])
    if df is None:
        P(f"  ❌ Data not found for {w['asset']} {w['tf']}")
        continue

    signals = w['func'](df)
    log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    strat_ret = signals.shift(1).fillna(0) * log_ret
    bh_ret = log_ret
    periods = w['periods']

    result = {'name': name, 'asset': w['asset'], 'tf': w['tf'], 'n_bars': len(df)}

    # ── 1. EQUITY CURVE ANALYSIS ──────────────────────────────────
    P(f"\n  ── 1. Equity Curve Analysis ──")
    cum_strat = strat_ret.cumsum()
    cum_bh = bh_ret.cumsum()
    total_strat = float(np.exp(cum_strat.iloc[-1]) - 1) * 100
    total_bh = float(np.exp(cum_bh.iloc[-1]) - 1) * 100
    P(f"  Total return: Strategy {total_strat:+.1f}% vs Buy&Hold {total_bh:+.1f}%")
    P(f"  Full-sample Sharpe: {sharpe(strat_ret, periods):.3f}")

    # Rolling Sharpe
    rs = rolling_sharpe(strat_ret, window=90, periods=periods)
    rs_valid = rs.dropna()
    if len(rs_valid) > 0:
        pct_positive = (rs_valid > 0).mean() * 100
        P(f"  Rolling 90-day Sharpe: mean={rs_valid.mean():.2f}, median={rs_valid.median():.2f}, >0 {pct_positive:.0f}% of time")
        P(f"    max={rs_valid.max():.2f}, min={rs_valid.min():.2f}")

    # Annual breakdown
    P(f"\n  Annual breakdown:")
    annual_rets = strat_ret.groupby(strat_ret.index.year)
    P(f"  {'Year':<6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    annual_data = {}
    for yr, grp in annual_rets:
        if len(grp) < 20:
            continue
        yr_ret = float(np.exp(grp.sum()) - 1) * 100
        yr_sharpe = sharpe(grp, periods)
        cum = grp.cumsum()
        dd = (cum - cum.cummax()).min()
        yr_dd = float(np.exp(dd) - 1) * 100
        P(f"  {yr:<6} {yr_ret:>+9.1f}% {yr_sharpe:>8.2f} {yr_dd:>+7.1f}%")
        annual_data[str(yr)] = {'return_pct': round(yr_ret, 2), 'sharpe': round(yr_sharpe, 2)}
    result['annual'] = annual_data
    result['total_return_pct'] = round(total_strat, 2)
    result['bh_return_pct'] = round(total_bh, 2)
    result['full_sharpe'] = round(sharpe(strat_ret, periods), 3)

    # ── 2. TRADE ANALYSIS ─────────────────────────────────────────
    P(f"\n  ── 2. Trade Analysis ──")
    pos = signals.shift(1).fillna(0)
    trades_mask = pos.diff().fillna(0) != 0
    n_trades = int(trades_mask.sum())

    # Compute per-trade returns
    trade_starts = pos.index[trades_mask]
    trade_returns = []
    for i in range(len(trade_starts) - 1):
        s, e = trade_starts[i], trade_starts[i + 1]
        mask = (strat_ret.index >= s) & (strat_ret.index < e)
        tr = strat_ret[mask].sum()
        trade_returns.append(float(tr))
    # last trade
    if len(trade_starts) > 0:
        mask = strat_ret.index >= trade_starts[-1]
        trade_returns.append(float(strat_ret[mask].sum()))

    trade_returns = np.array(trade_returns)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    P(f"  Total trades: {n_trades}")
    if len(df) > 0 and n_trades > 0:
        P(f"  Avg holding period: {len(df)/n_trades:.1f} bars")
    if len(trade_returns) > 0:
        wr = len(wins) / len(trade_returns) * 100
        P(f"  Win rate: {wr:.1f}%")
        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(np.abs(losses).mean()) if len(losses) > 0 else 0
        P(f"  Avg win: {avg_win*100:+.2f}%, Avg loss: {-avg_loss*100:.2f}%")
        pf = float(wins.sum() / np.abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')
        P(f"  Profit factor: {pf:.2f}")
        P(f"  Best trade: {trade_returns.max()*100:+.2f}%, Worst: {trade_returns.min()*100:+.2f}%")

        # Streaks
        signs = np.sign(trade_returns)
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for s in signs:
            if s > 0:
                cur_win += 1; cur_loss = 0
            elif s < 0:
                cur_loss += 1; cur_win = 0
            else:
                cur_win = cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
            max_loss_streak = max(max_loss_streak, cur_loss)
        P(f"  Longest win streak: {max_win_streak}, Longest loss streak: {max_loss_streak}")

        result['trades'] = {
            'n_trades': n_trades, 'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2), 'avg_win': round(avg_win*100, 2),
            'avg_loss': round(avg_loss*100, 2), 'best': round(float(trade_returns.max()*100), 2),
            'worst': round(float(trade_returns.min()*100), 2)
        }

    # ── 3. SIGNAL QUALITY ─────────────────────────────────────────
    P(f"\n  ── 3. Signal Quality ──")
    long_pct = (signals == 1).mean() * 100
    short_pct = (signals == -1).mean() * 100
    flat_pct = (signals == 0).mean() * 100
    P(f"  Long: {long_pct:.1f}%, Short: {short_pct:.1f}%, Flat: {flat_pct:.1f}%")

    # Signal clustering: avg consecutive same-signal bars
    changes = (signals.diff() != 0).cumsum()
    avg_cluster = signals.groupby(changes).count().mean()
    P(f"  Avg signal cluster length: {avg_cluster:.1f} bars")

    # Correlation with buy-and-hold
    corr = strat_ret.corr(bh_ret)
    P(f"  Correlation with B&H returns: {corr:.3f}")

    # Vol regime trading
    vol_30 = log_ret.rolling(30).std()
    vol_median = vol_30.median()
    high_vol_mask = vol_30 > vol_median
    sig_changes = signals.diff().abs().fillna(0) > 0
    if high_vol_mask.sum() > 0:
        trade_in_highvol = (sig_changes & high_vol_mask).sum() / sig_changes.sum() * 100 if sig_changes.sum() > 0 else 0
        P(f"  Trades in high-vol periods: {trade_in_highvol:.0f}%")

    result['signal'] = {'long_pct': round(long_pct, 1), 'short_pct': round(short_pct, 1),
                        'flat_pct': round(flat_pct, 1), 'bh_corr': round(corr, 3)}

    # ── 4. REGIME ANALYSIS ────────────────────────────────────────
    P(f"\n  ── 4. Regime Analysis ──")

    # BTC regime (only for daily strategies, approximate for 4h)
    if btc_daily is not None:
        btc_sma200 = btc_daily['close'].rolling(200).mean()
        btc_bull = btc_daily['close'] > btc_sma200

        # Align to strategy timeframe
        if w['tf'] == '4h':
            btc_bull_resampled = btc_bull.reindex(df.index, method='ffill')
        else:
            btc_bull_resampled = btc_bull.reindex(df.index, method='ffill')

        bull_mask = btc_bull_resampled.reindex(strat_ret.index).fillna(False)
        bear_mask = ~bull_mask

        if bull_mask.sum() > 30 and bear_mask.sum() > 30:
            bull_sharpe = sharpe(strat_ret[bull_mask], periods)
            bear_sharpe = sharpe(strat_ret[bear_mask], periods)
            bull_ret = float(np.exp(strat_ret[bull_mask].sum()) - 1) * 100
            bear_ret = float(np.exp(strat_ret[bear_mask].sum()) - 1) * 100
            P(f"  BTC Bull regime: Sharpe {bull_sharpe:.2f}, Return {bull_ret:+.1f}%")
            P(f"  BTC Bear regime: Sharpe {bear_sharpe:.2f}, Return {bear_ret:+.1f}%")
            result['regime_btc'] = {'bull_sharpe': round(bull_sharpe, 2), 'bear_sharpe': round(bear_sharpe, 2)}

    # Vol regime
    hv_ret = strat_ret[high_vol_mask.reindex(strat_ret.index).fillna(False)]
    lv_ret = strat_ret[~high_vol_mask.reindex(strat_ret.index).fillna(True)]
    if len(hv_ret) > 30 and len(lv_ret) > 30:
        P(f"  High-vol Sharpe: {sharpe(hv_ret, periods):.2f}, Low-vol Sharpe: {sharpe(lv_ret, periods):.2f}")

    # Monthly heatmap
    monthly = strat_ret.groupby(strat_ret.index.month).apply(lambda x: float(np.exp(x.sum()) - 1) * 100)
    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()
    P(f"  Best month: {best_month} ({monthly[best_month]:+.1f}%), Worst: {worst_month} ({monthly[worst_month]:+.1f}%)")

    # ── 5. ROBUSTNESS CHECKS ─────────────────────────────────────
    P(f"\n  ── 5. Robustness Checks ──")

    # Commission sensitivity
    P(f"  Commission sensitivity:")
    comm_levels = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]
    breakeven_comm = None
    for c in comm_levels:
        adj_ret = commission_adjusted_returns(signals, log_ret, c)
        s = sharpe(adj_ret, periods)
        P(f"    {c*100:.2f}%: Sharpe {s:.3f}")
        if s <= 0 and breakeven_comm is None:
            breakeven_comm = c
    if breakeven_comm:
        P(f"  ⚠️  Edge disappears at ~{breakeven_comm*100:.2f}% commission")
        result['breakeven_commission'] = round(breakeven_comm * 100, 3)
    else:
        P(f"  ✅ Edge survives up to 0.30% commission")
        result['breakeven_commission'] = '>0.30%'

    # Signal shift sensitivity
    P(f"\n  Signal shift sensitivity:")
    for shift_val in [-1, 0, 1]:
        shifted = signals.shift(shift_val).fillna(0)
        shift_ret = shifted.shift(1).fillna(0) * log_ret
        s = sharpe(shift_ret, periods)
        label = {-1: 'backward 1', 0: 'original', 1: 'forward 1'}[shift_val]
        P(f"    {label}: Sharpe {s:.3f}")

    # Random baseline
    P(f"\n  Random signal baseline (1000 trials):")
    invested_pct = (signals != 0).mean()
    n_bars = len(log_ret)
    random_sharpes = []
    np.random.seed(42)
    for _ in range(1000):
        rand_sig = np.random.choice([-1, 0, 1], size=n_bars,
                                     p=[invested_pct*0.5, 1-invested_pct, invested_pct*0.5])
        rand_ret = pd.Series(rand_sig).shift(1).fillna(0).values * log_ret.values
        random_sharpes.append(sharpe(pd.Series(rand_ret), periods))
    actual_sharpe = sharpe(strat_ret, periods)
    random_sharpes = np.array(random_sharpes)
    pval_random = (random_sharpes >= actual_sharpe).mean()
    P(f"    Actual Sharpe: {actual_sharpe:.3f}")
    P(f"    Random mean: {random_sharpes.mean():.3f} ± {random_sharpes.std():.3f}")
    P(f"    p-value vs random: {pval_random:.4f}")
    result['random_pval'] = round(float(pval_random), 4)

    # Walk-forward 14 folds
    P(f"\n  Walk-forward (14 folds):")
    wf_sharpes = walk_forward(w['func'], df, n_folds=14)
    wf_mean = np.mean(wf_sharpes)
    wf_pos = sum(1 for s in wf_sharpes if s > 0)
    P(f"    Mean fold Sharpe: {wf_mean:.3f}")
    P(f"    Positive folds: {wf_pos}/14")
    P(f"    Worst fold: {min(wf_sharpes):.3f}, Best fold: {max(wf_sharpes):.3f}")
    result['wf14'] = {'mean_sharpe': round(wf_mean, 3), 'positive_folds': wf_pos}

    # ── 6. CROSS-ASSET PORTABILITY ────────────────────────────────
    P(f"\n  ── 6. Cross-Asset Portability ──")
    # Get all daily assets
    daily_assets = []
    for f in os.listdir(DATA_DIR):
        if f.startswith('binance_') and f.endswith('_usdt_1d.csv'):
            a = f.replace('binance_', '').replace('_usdt_1d.csv', '')
            daily_assets.append(a)
    daily_assets.sort()

    portable = []
    P(f"  {'Asset':<8} {'Sharpe':>8} {'Return':>10} {'Trades':>7}")
    for asset in daily_assets:
        adf = load_ohlcv(asset, '1d')
        if adf is None or len(adf) < 100:
            continue
        try:
            asig = w['func'](adf)
            alr = np.log(adf['close'] / adf['close'].shift(1)).fillna(0)
            aret = asig.shift(1).fillna(0) * alr
            s = sharpe(aret, 365)
            ret = float(np.exp(aret.sum()) - 1) * 100
            nt = int((asig.diff().fillna(0) != 0).sum())
            P(f"  {asset:<8} {s:>8.3f} {ret:>+9.1f}% {nt:>7}")
            if s > 0.3:
                portable.append({'asset': asset, 'sharpe': round(s, 3), 'return_pct': round(ret, 1)})
        except Exception as e:
            P(f"  {asset:<8} ERROR: {e}")
    result['portable_assets'] = portable

    all_results[name] = result
    P("")

# ─── PRODUCTION RECOMMENDATION ────────────────────────────────────
P(f"\n{'='*80}")
P(f"  PRODUCTION RECOMMENDATION")
P(f"{'='*80}")

for w in WINNERS:
    name = w['name']
    if name not in all_results:
        continue
    r = all_results[name]
    P(f"\n  ── {name} ──")

    # Verdict
    random_pval = r.get('random_pval', 1.0)
    wf_pos = r.get('wf14', {}).get('positive_folds', 0)
    full_sharpe = r.get('full_sharpe', 0)
    bh_corr = r.get('signal', {}).get('bh_corr', 1.0)

    is_real = random_pval < 0.05 and wf_pos >= 8 and abs(bh_corr) < 0.7
    live_sharpe = full_sharpe * 0.5

    if is_real:
        P(f"  ✅ LIKELY REAL EDGE")
    else:
        reasons = []
        if random_pval >= 0.05:
            reasons.append(f"random p-val {random_pval:.3f} >= 0.05")
        if wf_pos < 8:
            reasons.append(f"only {wf_pos}/14 positive WF folds")
        if abs(bh_corr) >= 0.7:
            reasons.append(f"high B&H correlation {bh_corr:.2f}")
        P(f"  ⚠️  QUESTIONABLE — {'; '.join(reasons)}")

    P(f"  Estimated live Sharpe: {live_sharpe:.3f} (50% haircut from {full_sharpe:.3f})")

    # Position sizing (Kelly fraction with half-Kelly)
    if r.get('trades'):
        wr_dec = r['trades']['win_rate'] / 100
        avg_w = r['trades']['avg_win'] / 100
        avg_l = r['trades']['avg_loss'] / 100
        if avg_l > 0:
            kelly = wr_dec / avg_l - (1 - wr_dec) / avg_w if avg_w > 0 else 0
            half_kelly = max(0, kelly * 0.5)
            P(f"  Kelly fraction: {kelly:.2f}, Half-Kelly: {half_kelly:.2f}")
            P(f"  Suggested allocation: {min(half_kelly * 100, 20):.0f}% of portfolio (capped at 20%)")

    # Risk warnings
    P(f"  Risk warnings:")
    if abs(bh_corr) > 0.5:
        P(f"    ⚠️  Moderate B&H correlation ({bh_corr:.2f}) — partly directional beta")
    be = r.get('breakeven_commission', '')
    if isinstance(be, str) and '>' not in be:
        P(f"    ⚠️  Thin edge — breaks at {be}% commission")
    n_portable = len(r.get('portable_assets', []))
    if n_portable <= 1:
        P(f"    ⚠️  Low portability — may be asset-specific overfitting")
    P(f"    ⚠️  Price structure strategies are inherently backward-looking")
    P(f"    ⚠️  Crypto regime shifts can invalidate structure-based edges")

# Save results
out_path = os.path.join(OUT_DIR, 'ensemble_deep_dive.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
P(f"\n✅ Results saved to {out_path}")
P(f"\nAnalysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
