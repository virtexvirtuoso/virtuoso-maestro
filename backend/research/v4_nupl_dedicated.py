"""
Dedicated V4 + NUPL Capitulation Test
5000 permutations, 14-fold WF, bootstrap CI
Optimized: permutation uses SMA50 signal (no trailing stop loop per perm)
"""

import json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
np.random.seed(42)

def load_data():
    btc = pd.read_csv(DATA_DIR / "spot/BTC_spot_daily.csv", parse_dates=['Date'])
    btc = btc.rename(columns={'Date':'date','Close':'close'}).set_index('date').sort_index()
    btc['returns'] = btc['close'].pct_change()
    
    nupl = pd.DataFrame(json.load(open(DATA_DIR / "onchain/nupl.json")))
    nupl['date'] = pd.to_datetime(nupl['d'])
    nupl = nupl.set_index('date')
    nupl['nupl'] = nupl['nupl'].astype(float)
    
    btc['nupl'] = nupl['nupl']
    btc['nupl'] = btc['nupl'].ffill()
    return btc.dropna(subset=['returns','nupl'])

def sma50_signal(close):
    return (close > close.rolling(50).mean()).astype(int)

def trailing_stop(sig, close, trail_pct=0.12):
    """Apply trailing stop — only used for main test, not permutations."""
    pos = sig.shift(1).fillna(0).values.copy().astype(float)
    c = close.values
    in_pos = False
    peak = 0.0
    for i in range(1, len(c)):
        if pos[i] == 1:
            if not in_pos:
                in_pos = True
                peak = c[i]
            else:
                peak = max(peak, c[i])
            if c[i] < peak * (1 - trail_pct):
                pos[i] = 0
                in_pos = False
        else:
            in_pos = False
            peak = 0.0
    return pd.Series(pos, index=close.index)

def calc_metrics(r):
    r = r.dropna()
    if len(r) < 30:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'n_days': len(r)}
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_yr = len(r) / 252
    cagr = (1 + total_ret) ** (1/n_yr) - 1 if n_yr > 0 and total_ret > -1 else -1
    maxdd = float((cum / cum.cummax() - 1).min())
    return {'sharpe': round(sharpe, 4), 'cagr': round(cagr, 4), 'maxdd': round(maxdd, 4), 'n_days': len(r)}

def walk_forward(position, returns, n_folds=14, min_train=252):
    strat = position * returns
    total = len(returns)
    fold_size = (total - min_train) // n_folds
    oos_parts = []
    fold_results = []
    for i in range(n_folds):
        s = min_train + i * fold_size
        e = min(s + fold_size, total)
        if e <= s: break
        oos = strat.iloc[s:e]
        oos_parts.append(oos)
        sr = oos.mean() / oos.std() * np.sqrt(252) if oos.std() > 0 else 0
        fold_results.append({'fold': i, 'sharpe': round(sr, 4), 'ret': round(float(oos.sum()), 4)})
    
    all_oos = pd.concat(oos_parts)
    m = calc_metrics(all_oos)
    m['positive_folds'] = sum(1 for f in fold_results if f['ret'] > 0)
    m['total_folds'] = len(fold_results)
    m['folds'] = fold_results
    return m

def main():
    print("V4 + NUPL Capitulation — Dedicated Test")
    print("=" * 50)
    
    data = load_data()
    print(f"Data: {data.index[0].date()} to {data.index[-1].date()}, {len(data)} days")
    print(f"NUPL < -0.1 days: {(data['nupl'] < -0.1).sum()}")
    print(f"NUPL < 0 days: {(data['nupl'] < 0).sum()}")
    
    close = data['close']
    returns = data['returns']
    nupl = data['nupl']
    
    # V4 base: SMA50 + trailing stop
    sma_sig = sma50_signal(close)
    pos_base = trailing_stop(sma_sig, close)
    
    # V4 + NUPL: force long during capitulation
    sma_nupl = sma_sig.copy()
    sma_nupl[nupl < -0.1] = 1
    pos_nupl = trailing_stop(sma_nupl, close)
    
    diff = (pos_nupl != pos_base)
    print(f"Days where NUPL changes V4: {diff.sum()} ({diff.mean():.1%})")
    
    # Walk-forward
    print("\n--- Walk-Forward (14 folds) ---")
    wf_base = walk_forward(pos_base, returns)
    wf_nupl = walk_forward(pos_nupl, returns)
    
    print(f"V4 base:  Sharpe={wf_base['sharpe']}, CAGR={wf_base['cagr']:.1%}, MaxDD={wf_base['maxdd']:.1%}, {wf_base['positive_folds']}/{wf_base['total_folds']} pos folds")
    print(f"V4+NUPL:  Sharpe={wf_nupl['sharpe']}, CAGR={wf_nupl['cagr']:.1%}, MaxDD={wf_nupl['maxdd']:.1%}, {wf_nupl['positive_folds']}/{wf_nupl['total_folds']} pos folds")
    print(f"Delta:    Sharpe {wf_nupl['sharpe'] - wf_base['sharpe']:+.4f}")
    
    # Permutation test (5000x) — use SMA50 signal (no trailing stop per perm for speed)
    # This tests if the signal itself has edge, not the trailing stop implementation
    print("\n--- Permutation Test (5000x, signal-level) ---")
    pos_base_sig = sma_sig.shift(1).fillna(0)  # Signal without trailing stop
    pos_nupl_sig = sma_nupl.shift(1).fillna(0)
    
    actual_base = float((pos_base_sig * returns).dropna().mean() / (pos_base_sig * returns).dropna().std() * np.sqrt(252))
    actual_nupl = float((pos_nupl_sig * returns).dropna().mean() / (pos_nupl_sig * returns).dropna().std() * np.sqrt(252))
    
    ret_vals = returns.values
    base_pos = pos_base_sig.values
    nupl_pos = pos_nupl_sig.values
    
    count_base = 0
    count_nupl = 0
    perm_sharpes_base = []
    perm_sharpes_nupl = []
    
    for i in range(5000):
        shuf = np.random.permutation(ret_vals)
        pb = base_pos * shuf
        pn = nupl_pos * shuf
        sb = pb.mean() / pb.std() * np.sqrt(252) if pb.std() > 0 else 0
        sn = pn.mean() / pn.std() * np.sqrt(252) if pn.std() > 0 else 0
        perm_sharpes_base.append(sb)
        perm_sharpes_nupl.append(sn)
        if sb >= actual_base: count_base += 1
        if sn >= actual_nupl: count_nupl += 1
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/5000 permutations done...")
    
    p_base = count_base / 5000
    p_nupl = count_nupl / 5000
    print(f"V4 base p-value:  {p_base:.4f}")
    print(f"V4+NUPL p-value:  {p_nupl:.4f}")
    
    # Paired permutation: is the NUPL ADDITION significant?
    print("\n--- Paired Permutation (2000x, NUPL addition only) ---")
    actual_delta = actual_nupl - actual_base
    count_delta = 0
    for i in range(2000):
        # Shuffle NUPL values, recompute signal
        fake_nupl = np.random.permutation(nupl.values)
        fake_sig = sma_sig.values.copy()
        fake_sig[fake_nupl < -0.1] = 1
        fake_pos = np.roll(fake_sig, 1)  # shift(1)
        fake_pos[0] = 0
        
        fake_ret = fake_pos * ret_vals
        fake_sharpe = fake_ret.mean() / fake_ret.std() * np.sqrt(252) if fake_ret.std() > 0 else 0
        fake_delta = fake_sharpe - actual_base
        if fake_delta >= actual_delta:
            count_delta += 1
        if (i+1) % 500 == 0:
            print(f"  {i+1}/2000 paired perms done...")
    
    paired_p = count_delta / 2000
    print(f"Paired permutation p-value: {paired_p:.4f}")
    
    # Block permutation
    print("\n--- Block Permutation (1000x) ---")
    block_results = {}
    for bs in [20, 40, 60]:
        count_bp = 0
        n_blocks = len(ret_vals) // bs + 1
        blocks = [ret_vals[i*bs:(i+1)*bs] for i in range(n_blocks) if len(ret_vals[i*bs:(i+1)*bs]) > 0]
        for _ in range(1000):
            perm_blocks = [blocks[j] for j in np.random.permutation(len(blocks))]
            shuf = np.concatenate(perm_blocks)[:len(ret_vals)]
            pr = nupl_pos * shuf
            ps = pr.mean() / pr.std() * np.sqrt(252) if pr.std() > 0 else 0
            if ps >= actual_nupl:
                count_bp += 1
        block_results[bs] = round(count_bp / 1000, 4)
        print(f"  Block {bs}d: p={block_results[bs]}")
    
    # Bootstrap CI
    print("\n--- Bootstrap CI (5000x) ---")
    strat_base = (pos_base.shift(1) * returns).dropna().values
    strat_nupl = (pos_nupl.shift(1) * returns).dropna().values
    
    def boot_ci(r, n=5000):
        sharpes = []
        for _ in range(n):
            s = np.random.choice(r, size=len(r), replace=True)
            sharpes.append(s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0)
        return [round(np.percentile(sharpes, 2.5), 4), round(np.percentile(sharpes, 97.5), 4)]
    
    ci_base = boot_ci(strat_base)
    ci_nupl = boot_ci(strat_nupl)
    
    # Delta CI
    min_len = min(len(strat_base), len(strat_nupl))
    delta_rets = strat_nupl[:min_len] - strat_base[:min_len]
    ci_delta = boot_ci(delta_rets)
    
    print(f"V4 base CI:    {ci_base}")
    print(f"V4+NUPL CI:    {ci_nupl}")
    print(f"Delta CI:      {ci_delta}")
    
    # Year-by-year
    print("\n--- Year-by-Year ---")
    yearly = {}
    for yr in range(2019, 2027):
        mask = data.index.year == yr
        if mask.sum() < 30: continue
        r_base = (pos_base[mask].shift(1) * returns[mask]).dropna()
        r_nupl = (pos_nupl[mask].shift(1) * returns[mask]).dropna()
        cum_b = float((1 + r_base).cumprod().iloc[-1] - 1) if len(r_base) > 0 else 0
        cum_n = float((1 + r_nupl).cumprod().iloc[-1] - 1) if len(r_nupl) > 0 else 0
        yearly[yr] = {'base': round(cum_b, 4), 'nupl': round(cum_n, 4), 'delta': round(cum_n - cum_b, 4)}
        print(f"  {yr}: V4 base {cum_b:+.1%}, V4+NUPL {cum_n:+.1%}, Δ {cum_n-cum_b:+.1%}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'V4 + NUPL Capitulation Dedicated Test',
        'data_range': f"{data.index[0].date()} to {data.index[-1].date()}",
        'n_days': len(data),
        'nupl_capitulation_days': int((nupl < -0.1).sum()),
        'signal_change_days': int(diff.sum()),
        'v4_base_wf': wf_base,
        'v4_nupl_wf': wf_nupl,
        'delta_sharpe_wf': round(wf_nupl['sharpe'] - wf_base['sharpe'], 4),
        'permutation_5000': {
            'p_base': p_base,
            'p_nupl': p_nupl,
            'actual_sharpe_base': round(actual_base, 4),
            'actual_sharpe_nupl': round(actual_nupl, 4)
        },
        'paired_permutation_2000': {'p_value': paired_p, 'actual_delta': round(actual_delta, 4)},
        'block_permutation_1000': block_results,
        'bootstrap_ci': {'base': ci_base, 'nupl': ci_nupl, 'delta': ci_delta},
        'yearly': yearly,
        'verdict': ''
    }
    
    if paired_p < 0.05 and ci_delta[0] > 0:
        output['verdict'] = f'NUPL addition SIGNIFICANT: paired p={paired_p:.4f}, delta CI {ci_delta} excludes zero. Integrate.'
    elif paired_p < 0.05:
        output['verdict'] = f'NUPL addition marginally significant: paired p={paired_p:.4f}, but delta CI {ci_delta} includes zero. Weak evidence.'
    else:
        output['verdict'] = f'NUPL addition NOT significant: paired p={paired_p:.4f}. Improvement is noise.'
    
    print(f"\n{'='*50}")
    print(f"VERDICT: {output['verdict']}")
    
    with open(DATA_DIR / "backtest_results/v4_nupl_dedicated.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("Saved to v4_nupl_dedicated.json")

if __name__ == '__main__':
    main()
