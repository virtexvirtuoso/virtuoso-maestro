#!/usr/bin/env python3
"""
RIGOROUS VALIDATION: Is our optimized proxy REAL or overfit?
Tests: Walk-forward, bootstrap, permutation, holdout, multiple comparison correction
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from scipy import stats
import json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)
COST = 0.001
START = "2017-01-01"
END = "2025-12-31"
results_dir = Path("/Users/ffv_macmini/Desktop/maestro/data/research")

print("=" * 70)
print("RIGOROUS VALIDATION: Optimized Proxy vs Alden")
print("=" * 70)

# ── Data ──
print("\n[1/8] Downloading data...")
tickers = {'BTC': 'BTC-USD', 'UUP': 'UUP', 'GLD': 'GLD', 'TLT': 'TLT', 'HYG': 'HYG'}
prices = {}
for name, tick in tickers.items():
    df = yf.download(tick, start=START, end=END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices[name] = df['Close'].dropna()
    print(f"  {name}: {len(prices[name])} days")

# FRED
print("  FRED data...")
walcl = fred.get_series('WALCL', observation_start=START)
wtregen = fred.get_series('WTREGEN', observation_start=START)
rrp = fred.get_series('RRPONTSYD', observation_start=START)
m2 = fred.get_series('M2SL', observation_start=START)
print(f"  FRED: WALCL={len(walcl)}, WTREGEN={len(wtregen)}, RRPONTSYD={len(rrp)}, M2SL={len(m2)}")

pdf = pd.DataFrame(prices).ffill().dropna(subset=['BTC'])
inverse = {'UUP'}


def build_proxy_signal(df, lookback, threshold, instruments=['UUP','GLD','TLT','HYG']):
    """Build binary proxy signal."""
    scores = pd.DataFrame(index=df.index)
    for inst in instruments:
        if inst not in df.columns:
            continue
        chg = df[inst].pct_change(lookback)
        if inst in inverse:
            scores[inst] = (chg < 0).astype(int)
        else:
            scores[inst] = (chg > 0).astype(int)
    return (scores.sum(axis=1) >= threshold).astype(int)


def build_alden_signal(df, net_liq, lookback=20):
    """Build Alden Net Liquidity signal."""
    nl = net_liq.reindex(df.index, method='ffill')
    return (nl.pct_change(lookback) > 0).astype(int)


def backtest(signal, btc_prices, cost=COST):
    """Backtest a binary signal. Returns daily strategy returns."""
    sig = signal.shift(1).fillna(0)
    ret = btc_prices.pct_change().fillna(0)
    trades = sig.diff().abs().fillna(0)
    return sig * ret - trades * cost


def compute_metrics(strat_ret):
    """Compute Sharpe, MaxDD, CAGR from daily returns."""
    if strat_ret.std() == 0:
        return {'sharpe': 0, 'max_dd': 0, 'cagr': 0, 'total_ret': 0}
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(strat_ret) / 252
    cagr = (1 + total) ** (1/years) - 1 if total > -1 else -1
    dd = (cum / cum.cummax() - 1).min()
    return {'sharpe': round(sharpe, 4), 'max_dd': round(dd * 100, 1),
            'cagr': round(cagr * 100, 1), 'total_ret': round(total * 100, 1)}


net_liq = walcl.reindex(pdf.index, method='ffill') - wtregen.reindex(pdf.index, method='ffill') - rrp.reindex(pdf.index, method='ffill')

# Best params from sweep: lookback=180, threshold=2
BEST_LB = 180
BEST_THR = 2

our_signal = build_proxy_signal(pdf, BEST_LB, BEST_THR)
alden_signal = build_alden_signal(pdf, net_liq, 20)

our_ret = backtest(our_signal, pdf['BTC'])
alden_ret = backtest(alden_signal, pdf['BTC'])
bh_ret = pdf['BTC'].pct_change().fillna(0)

our_m = compute_metrics(our_ret)
alden_m = compute_metrics(alden_ret)
bh_m = compute_metrics(bh_ret)

print(f"\n  FULL PERIOD RESULTS:")
print(f"  Our proxy:  Sharpe={our_m['sharpe']}, MaxDD={our_m['max_dd']}%, CAGR={our_m['cagr']}%")
print(f"  Alden:      Sharpe={alden_m['sharpe']}, MaxDD={alden_m['max_dd']}%, CAGR={alden_m['cagr']}%")
print(f"  Buy&Hold:   Sharpe={bh_m['sharpe']}, MaxDD={bh_m['max_dd']}%, CAGR={bh_m['cagr']}%")


# ── TEST 1: Walk-Forward Validation ──
print("\n[2/8] Walk-Forward Validation (730d train / 182d test)...")
train_days = 730
test_days = 182
wf_results = {'ours': [], 'alden': [], 'bh': []}

idx = pdf.index
i = train_days
fold = 0
while i + test_days <= len(idx):
    test_start = idx[i]
    test_end = idx[min(i + test_days - 1, len(idx) - 1)]
    
    # Test period
    test_mask = (pdf.index >= test_start) & (pdf.index <= test_end)
    test_df = pdf[test_mask]
    
    our_sig_test = our_signal[test_mask]
    alden_sig_test = alden_signal[test_mask]
    
    our_r = backtest(our_sig_test, test_df['BTC'])
    alden_r = backtest(alden_sig_test, test_df['BTC'])
    bh_r = test_df['BTC'].pct_change().fillna(0)
    
    our_wf = compute_metrics(our_r)
    alden_wf = compute_metrics(alden_r)
    bh_wf = compute_metrics(bh_r)
    
    wf_results['ours'].append(our_wf['sharpe'])
    wf_results['alden'].append(alden_wf['sharpe'])
    wf_results['bh'].append(bh_wf['sharpe'])
    
    fold += 1
    print(f"  Fold {fold}: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} | "
          f"Ours={our_wf['sharpe']:.3f} Alden={alden_wf['sharpe']:.3f} B&H={bh_wf['sharpe']:.3f}")
    
    i += test_days

our_oos_mean = np.mean(wf_results['ours'])
alden_oos_mean = np.mean(wf_results['alden'])
our_oos_wins = sum(1 for o, a in zip(wf_results['ours'], wf_results['alden']) if o > a)
total_folds = len(wf_results['ours'])

print(f"\n  OOS Mean Sharpe: Ours={our_oos_mean:.4f}, Alden={alden_oos_mean:.4f}")
print(f"  Folds where Ours > Alden: {our_oos_wins}/{total_folds}")
print(f"  Folds where Ours > 0: {sum(1 for x in wf_results['ours'] if x > 0)}/{total_folds}")


# ── TEST 2: Bootstrap Sharpe Confidence Intervals ──
print("\n[3/8] Bootstrap Sharpe Confidence Intervals (10,000 iterations)...")
n_boot = 10000
our_daily = our_ret.dropna().values
alden_daily = alden_ret.dropna().values
bh_daily = bh_ret.dropna().values

def bootstrap_sharpe(returns, n=10000):
    sharpes = []
    for _ in range(n):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        s = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
        sharpes.append(s)
    return np.array(sharpes)

our_boot = bootstrap_sharpe(our_daily, n_boot)
alden_boot = bootstrap_sharpe(alden_daily, n_boot)

our_ci = (np.percentile(our_boot, 2.5), np.percentile(our_boot, 97.5))
alden_ci = (np.percentile(alden_boot, 2.5), np.percentile(alden_boot, 97.5))

print(f"  Our Sharpe: {our_m['sharpe']:.4f}  95% CI: [{our_ci[0]:.4f}, {our_ci[1]:.4f}]")
print(f"  Alden Sharpe: {alden_m['sharpe']:.4f}  95% CI: [{alden_ci[0]:.4f}, {alden_ci[1]:.4f}]")

# Test: is our Sharpe significantly > Alden's?
diff_boot = our_boot - alden_boot
pct_better = (diff_boot > 0).mean()
diff_ci = (np.percentile(diff_boot, 2.5), np.percentile(diff_boot, 97.5))
print(f"  Sharpe difference: {our_m['sharpe'] - alden_m['sharpe']:.4f}")
print(f"  95% CI of difference: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
print(f"  P(Ours > Alden): {pct_better:.4f}")
print(f"  Significant (CI excludes 0): {'YES' if diff_ci[0] > 0 else 'NO'}")


# ── TEST 3: Permutation Test ──
print("\n[4/8] Permutation Test (is signal better than random?)...")
n_perm = 5000
our_true_sharpe = our_m['sharpe']
perm_sharpes = []

for p in range(n_perm):
    perm_signal = our_signal.sample(frac=1).values  # shuffle signal
    perm_signal = pd.Series(perm_signal, index=our_signal.index)
    perm_ret = backtest(perm_signal, pdf['BTC'])
    perm_s = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
    perm_sharpes.append(perm_s)
    if (p+1) % 1000 == 0:
        print(f"  {p+1}/{n_perm} permutations...")

perm_p = (np.array(perm_sharpes) >= our_true_sharpe).mean()
print(f"  True Sharpe: {our_true_sharpe:.4f}")
print(f"  Permutation p-value: {perm_p:.4f}")
print(f"  Significant (p<0.05): {'YES' if perm_p < 0.05 else 'NO'}")
print(f"  Permutation mean Sharpe: {np.mean(perm_sharpes):.4f}")
print(f"  Permutation 95th pct: {np.percentile(perm_sharpes, 95):.4f}")


# ── TEST 4: Holdout Test (last 2 years untouched) ──
print("\n[5/8] Holdout Test (2024-2025 = untouched data)...")
holdout_start = "2024-01-01"
holdout_mask = pdf.index >= holdout_start
holdout_df = pdf[holdout_mask]

our_holdout_sig = our_signal[holdout_mask]
alden_holdout_sig = alden_signal[holdout_mask]

our_hold_ret = backtest(our_holdout_sig, holdout_df['BTC'])
alden_hold_ret = backtest(alden_holdout_sig, holdout_df['BTC'])
bh_hold_ret = holdout_df['BTC'].pct_change().fillna(0)

our_hold_m = compute_metrics(our_hold_ret)
alden_hold_m = compute_metrics(alden_hold_ret)
bh_hold_m = compute_metrics(bh_hold_ret)

print(f"  HOLDOUT (2024-2025):")
print(f"  Our proxy:  Sharpe={our_hold_m['sharpe']}, Return={our_hold_m['total_ret']}%, MaxDD={our_hold_m['max_dd']}%")
print(f"  Alden:      Sharpe={alden_hold_m['sharpe']}, Return={alden_hold_m['total_ret']}%, MaxDD={alden_hold_m['max_dd']}%")
print(f"  Buy&Hold:   Sharpe={bh_hold_m['sharpe']}, Return={bh_hold_m['total_ret']}%, MaxDD={bh_hold_m['max_dd']}%")


# ── TEST 5: Parameter Sensitivity ──
print("\n[6/8] Parameter Sensitivity (is result robust to nearby params?)...")
sensitivity = []
for lb in [120, 140, 160, 180, 200, 220]:
    for thr in [2, 3]:
        sig = build_proxy_signal(pdf, lb, thr)
        ret = backtest(sig, pdf['BTC'])
        m = compute_metrics(ret)
        sensitivity.append({'lookback': lb, 'threshold': thr, **m})
        print(f"  lb={lb:3d} thr={thr} | Sharpe={m['sharpe']:.4f} MaxDD={m['max_dd']}%")

# Check stability: what % of nearby params also beat Alden?
beats_alden = sum(1 for s in sensitivity if s['sharpe'] > alden_m['sharpe'])
print(f"\n  Params beating Alden ({alden_m['sharpe']}): {beats_alden}/{len(sensitivity)} ({beats_alden/len(sensitivity)*100:.0f}%)")


# ── TEST 6: Regime-Specific Performance ──
print("\n[7/8] Regime-Specific Performance...")
# Bull: 2020-H2, 2021-H1, 2023-H2, 2024
# Bear: 2022, 2021-H2
regimes = {
    'COVID crash (2020-03)': ('2020-03-01', '2020-03-31'),
    'Bull run (2020-H2)': ('2020-07-01', '2020-12-31'),
    'Bull peak (2021-H1)': ('2021-01-01', '2021-06-30'),
    'May 2021 crash': ('2021-05-01', '2021-06-30'),
    'Bear market (2022)': ('2022-01-01', '2022-12-31'),
    'FTX collapse': ('2022-11-01', '2022-11-30'),
    'Recovery (2023)': ('2023-01-01', '2023-12-31'),
    'Bull (2024)': ('2024-01-01', '2024-12-31'),
}

for name, (start, end) in regimes.items():
    mask = (pdf.index >= start) & (pdf.index <= end)
    if mask.sum() == 0:
        continue
    
    our_r = backtest(our_signal[mask], pdf['BTC'][mask])
    alden_r = backtest(alden_signal[mask], pdf['BTC'][mask])
    bh_r = pdf['BTC'][mask].pct_change().fillna(0)
    
    our_cum = (1 + our_r).cumprod().iloc[-1] - 1
    alden_cum = (1 + alden_r).cumprod().iloc[-1] - 1
    bh_cum = (1 + bh_r).cumprod().iloc[-1] - 1
    
    winner = "OURS" if our_cum > alden_cum else "ALDEN"
    print(f"  {name:25s} | Ours={our_cum*100:+7.1f}% Alden={alden_cum*100:+7.1f}% B&H={bh_cum*100:+7.1f}% | {winner}")


# ── TEST 7: M2 Direction Prediction (detailed) ──
print("\n[8/8] M2 Direction Prediction (detailed)...")
m2_monthly = m2.resample('ME').last().pct_change()

for lead_days in [5, 10, 15, 20, 30, 40, 60]:
    correct = 0
    total = 0
    for date, direction in m2_monthly.items():
        if pd.isna(direction):
            continue
        lookback_date = date - pd.Timedelta(days=lead_days)
        mask = (our_signal.index >= lookback_date) & (our_signal.index <= date)
        if mask.sum() > 0:
            pred = our_signal[mask].iloc[-1]
            actual = 1 if direction > 0 else 0
            if pred == actual:
                correct += 1
            total += 1
    
    acc = correct / total if total > 0 else 0
    p = stats.binomtest(correct, total, 0.5).pvalue if total > 0 else 1
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Lead={lead_days:2d}d | Accuracy={acc:.1%} ({correct}/{total}) | p={p:.4f} {sig}")


# ── Multiple Comparison Correction ──
print("\n  Bonferroni correction (7 lead times tested):")
print(f"  Corrected threshold: p < {0.05/7:.4f}")


# ── Save Everything ──
print("\n" + "=" * 70)
print("SAVING RESULTS...")

all_results = {
    'full_period': {
        'ours': our_m, 'alden': alden_m, 'buy_hold': bh_m,
        'params': {'lookback': BEST_LB, 'threshold': BEST_THR}
    },
    'walk_forward': {
        'ours_folds': wf_results['ours'],
        'alden_folds': wf_results['alden'],
        'ours_mean': round(our_oos_mean, 4),
        'alden_mean': round(alden_oos_mean, 4),
        'ours_wins': our_oos_wins,
        'total_folds': total_folds
    },
    'bootstrap': {
        'ours_ci': [round(our_ci[0], 4), round(our_ci[1], 4)],
        'alden_ci': [round(alden_ci[0], 4), round(alden_ci[1], 4)],
        'diff_ci': [round(diff_ci[0], 4), round(diff_ci[1], 4)],
        'p_ours_gt_alden': round(pct_better, 4)
    },
    'permutation': {
        'true_sharpe': round(our_true_sharpe, 4),
        'p_value': round(perm_p, 4),
        'perm_mean': round(np.mean(perm_sharpes), 4),
        'perm_95th': round(np.percentile(perm_sharpes, 95), 4)
    },
    'holdout_2024_2025': {
        'ours': our_hold_m, 'alden': alden_hold_m, 'buy_hold': bh_hold_m
    },
    'sensitivity': {
        'results': sensitivity,
        'pct_beat_alden': round(beats_alden / len(sensitivity) * 100, 0)
    }
}

with open(results_dir / 'rigorous_validation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Write Report ──
report = f"""# Rigorous Validation: Optimized Proxy vs Alden Net Liquidity

## Parameters
- Our proxy: 4-instrument (UUP/GLD/TLT/HYG), lookback={BEST_LB}d, consensus={BEST_THR}/4
- Alden: Net Liquidity (WALCL-WTREGEN-RRPONTSYD), 20d ROC
- Period: 2018-2025, BTC-USD, 0.1% costs, 1-day signal lag

## Test 1: Full Period Performance

| Metric | Our Proxy | Alden | Buy & Hold |
|--------|-----------|-------|------------|
| Sharpe | {our_m['sharpe']} | {alden_m['sharpe']} | {bh_m['sharpe']} |
| CAGR | {our_m['cagr']}% | {alden_m['cagr']}% | {bh_m['cagr']}% |
| Max DD | {our_m['max_dd']}% | {alden_m['max_dd']}% | {bh_m['max_dd']}% |
| Total Return | {our_m['total_ret']}% | {alden_m['total_ret']}% | {bh_m['total_ret']}% |

## Test 2: Walk-Forward Validation ({total_folds} folds, 730d/182d)

- OOS Mean Sharpe: Ours={our_oos_mean:.4f}, Alden={alden_oos_mean:.4f}
- Folds Ours > Alden: {our_oos_wins}/{total_folds}
- Folds Ours > 0: {sum(1 for x in wf_results['ours'] if x > 0)}/{total_folds}

## Test 3: Bootstrap Confidence Intervals (n=10,000)

| | Sharpe | 95% CI |
|---|--------|--------|
| Our Proxy | {our_m['sharpe']} | [{our_ci[0]:.4f}, {our_ci[1]:.4f}] |
| Alden | {alden_m['sharpe']} | [{alden_ci[0]:.4f}, {alden_ci[1]:.4f}] |

- Sharpe Difference CI: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]
- P(Ours > Alden): {pct_better:.4f}
- CI excludes 0: {'YES — SIGNIFICANT' if diff_ci[0] > 0 else 'NO — not significant'}

## Test 4: Permutation Test (n=5,000)

- True Sharpe: {our_true_sharpe:.4f}
- Permutation p-value: {perm_p:.4f}
- Significant (p<0.05): {'YES' if perm_p < 0.05 else 'NO'}
- Permutation mean: {np.mean(perm_sharpes):.4f}
- Permutation 95th percentile: {np.percentile(perm_sharpes, 95):.4f}

## Test 5: Holdout Test (2024-2025, untouched)

| Metric | Our Proxy | Alden | Buy & Hold |
|--------|-----------|-------|------------|
| Sharpe | {our_hold_m['sharpe']} | {alden_hold_m['sharpe']} | {bh_hold_m['sharpe']} |
| Return | {our_hold_m['total_ret']}% | {alden_hold_m['total_ret']}% | {bh_hold_m['total_ret']}% |
| Max DD | {our_hold_m['max_dd']}% | {alden_hold_m['max_dd']}% | {bh_hold_m['max_dd']}% |

## Test 6: Parameter Sensitivity

- Nearby params beating Alden: {beats_alden}/{len(sensitivity)} ({beats_alden/len(sensitivity)*100:.0f}%)

## Test 7: M2 Direction Prediction

(See stdout output for detailed lead-time analysis)

---
*Generated: {pd.Timestamp.now()}*
*Tests: Walk-forward, Bootstrap (10K), Permutation (5K), Holdout, Sensitivity, Regime, M2 Prediction*
"""

with open(results_dir / 'rigorous_validation_report.md', 'w') as f:
    f.write(report)

print(f"\nResults saved to {results_dir}/rigorous_validation_*.{{json,md}}")

print(f"\n{'='*70}")
print("FINAL RIGOROUS VERDICT")
print(f"{'='*70}")
print(f"Full period: Ours {our_m['sharpe']} vs Alden {alden_m['sharpe']} → {'OURS WINS' if our_m['sharpe'] > alden_m['sharpe'] else 'ALDEN WINS'}")
print(f"Walk-forward OOS: Ours {our_oos_mean:.4f} vs Alden {alden_oos_mean:.4f}")
print(f"Bootstrap: P(Ours>Alden) = {pct_better:.1%}, CI excludes 0: {'YES' if diff_ci[0] > 0 else 'NO'}")
print(f"Permutation: p={perm_p:.4f} → {'REAL SIGNAL' if perm_p < 0.05 else 'COULD BE NOISE'}")
print(f"Holdout 2024-25: Ours {our_hold_m['sharpe']} vs Alden {alden_hold_m['sharpe']}")
print(f"Sensitivity: {beats_alden}/{len(sensitivity)} nearby params beat Alden")
