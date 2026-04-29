"""Lean ensemble test — top combos, skip VP/SR, all assets × all TFs."""
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

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
COMMISSION = 0.001
STRATS = {'MS': ms_signals, 'SFP': sfp_signals, 'FVG': fvg_signals, 'OB': ob_signals}
TFS = {'1d': 365, '4h': 2190, '1h': 8760, '15m': 35040}
ASSETS = ['btc', 'eth', 'sol', 'sui', 'link', 'render', 'avax', 'inj', 'op']

def load(asset, tf):
    path = os.path.join(DATA_DIR, f'binance_{asset}_usdt_{tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    if tf in ('1h', '15m') and len(df) > 6000:
        df = df.iloc[-6000:]
    return df

def get_sigs(df):
    s = {}
    for n, f in STRATS.items():
        try: s[n] = f(df)
        except: s[n] = pd.Series(0, index=df.index)
    return pd.DataFrame(s, index=df.index)

def backtest(df, signals, bpy):
    r = df['close'].pct_change().shift(-1)
    c = signals.diff().abs().fillna(0)
    return (signals * r - c * COMMISSION).fillna(0)

def metrics(rets, bpy):
    if len(rets) == 0 or rets.std() == 0:
        return 0, 0, 0
    s = rets.mean() / rets.std() * np.sqrt(bpy)
    t = (1 + rets).prod() - 1
    cum = (1 + rets).cumprod(); dd = ((cum - cum.cummax()) / cum.cummax()).min()
    n = len(rets) / bpy
    cagr = (1 + t) ** (1 / max(n, 0.01)) - 1
    return round(s, 3), round(cagr * 100, 1), round(dd * 100, 1)

def wf(df, efunc, bpy, nf=10):
    n = len(df); mis = max(150, n // (nf + 2)); fs = (n - mis) // nf
    oos = []
    for fold in range(nf):
        ie = mis + (fold + 1) * fs; oe = min(ie + fs, n)
        if ie >= n or oe <= ie or (oe - ie) < 10: break
        s = efunc(df.iloc[ie:oe])
        r = backtest(df.iloc[ie:oe], s, bpy)
        sh, _, _ = metrics(r, bpy)
        oos.append(sh)
    if len(oos) < 3: return 0, 1.0
    a = np.array(oos)
    if a.std() > 0:
        _, p = stats.ttest_1samp(a, 0); p = p / 2
    else: p = 1.0
    return round(a.mean(), 3), round(p, 4)

# Ensembles
def ms_sfp(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['MS']==1)&(s['SFP']==1)] = 1; r[(s['MS']==-1)&(s['SFP']==-1)] = -1; return r
def ms_filter(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['MS']==1)&((s['SFP']==1)|(s['FVG']==1)|(s['OB']==1))] = 1
    r[(s['MS']==-1)&((s['SFP']==-1)|(s['FVG']==-1)|(s['OB']==-1))] = -1; return r
def contra3(df):
    s = get_sigs(df); v = s.sum(axis=1); r = pd.Series(0, index=df.index)
    r[v>=3] = -1; r[v<=-3] = 1; return r
def maj3(df):
    s = get_sigs(df); v = s.sum(axis=1); r = pd.Series(0, index=df.index)
    r[v>=3] = 1; r[v<=-3] = -1; return r
def top3(df):
    s = get_sigs(df); v = s[['MS','SFP','OB']].sum(axis=1); r = pd.Series(0, index=df.index)
    r[v>=2] = 1; r[v<=-2] = -1; return r
def ms_ob(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['MS']==1)&(s['OB']==1)] = 1; r[(s['MS']==-1)&(s['OB']==-1)] = -1; return r
def sfp_fvg(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['SFP']==1)&(s['FVG']==1)] = 1; r[(s['SFP']==-1)&(s['FVG']==-1)] = -1; return r
def ms_fvg(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['MS']==1)&(s['FVG']==1)] = 1; r[(s['MS']==-1)&(s['FVG']==-1)] = -1; return r
def sfp_ob(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['SFP']==1)&(s['OB']==1)] = 1; r[(s['SFP']==-1)&(s['OB']==-1)] = -1; return r
def ob_fvg(df):
    s = get_sigs(df); r = pd.Series(0, index=df.index)
    r[(s['OB']==1)&(s['FVG']==1)] = 1; r[(s['OB']==-1)&(s['FVG']==-1)] = -1; return r

ENS = {'MS+SFP': ms_sfp, 'MS_filter': ms_filter, 'Contrarian': contra3, 'Majority3': maj3,
       'Top3': top3, 'MS+OB': ms_ob, 'SFP+FVG': sfp_fvg, 'MS+FVG': ms_fvg, 'SFP+OB': sfp_ob, 'OB+FVG': ob_fvg}

all_r = []
for asset in ASSETS:
    for tf, bpy in TFS.items():
        df = load(asset, tf)
        if df is None or len(df) < 250: continue
        bh_sh, bh_cagr, bh_dd = metrics(df['close'].pct_change().fillna(0), bpy)
        for en, ef in ENS.items():
            s = ef(df); r = backtest(df, s, bpy)
            sh, cagr, dd = metrics(r, bpy)
            wsh, wp = wf(df, ef, bpy)
            inv = round((s != 0).mean() * 100, 1)
            all_r.append({'asset': asset.upper(), 'tf': tf, 'ens': en, 'sh': sh, 'cagr': cagr, 'dd': dd,
                          'oos': wsh, 'p': wp, 'sig': wp < 0.05, 'bh': bh_sh, 'inv': inv})
        print(f"  {asset.upper()} {tf}: done", flush=True)

# Significant positive
sig_pos = sorted([r for r in all_r if r['sig'] and r['oos'] > 0], key=lambda x: x['oos'], reverse=True)
print(f"\n{'='*90}\n  SIGNIFICANT POSITIVE RESULTS (p < 0.05, OOS > 0)\n{'='*90}")
if sig_pos:
    print(f"  {'Asset':<6} {'TF':>3} {'Ensemble':<12} {'Full':>6} {'OOS':>6} {'CAGR':>7} {'DD':>6} {'p':>7} {'B&H':>5} {'Inv':>5}")
    print(f"  {'-'*70}")
    for r in sig_pos:
        print(f"  {r['asset']:<6} {r['tf']:>3} {r['ens']:<12} {r['sh']:>6.3f} {r['oos']:>6.3f} {r['cagr']:>6.1f}% {r['dd']:>5.1f}% {r['p']:>7.4f} {r['bh']:>5.2f} {r['inv']:>4.1f}%")
else:
    print("  None.")

# Near misses
near = sorted([r for r in all_r if 0.05 < r['p'] < 0.10 and r['oos'] > 0], key=lambda x: x['oos'], reverse=True)
print(f"\n  NEAR MISSES (p < 0.10, OOS > 0): {len(near)}")
for r in near[:20]:
    print(f"  {r['asset']:<6} {r['tf']:>3} {r['ens']:<12} OOS {r['oos']:>6.3f} p={r['p']:.4f}")

# Ensemble ranking
print(f"\n{'='*90}\n  ENSEMBLE RANKING (mean OOS across all)\n{'='*90}")
esc = {}
for r in all_r:
    esc.setdefault(r['ens'], []).append(r['oos'])
ranked = sorted(esc.items(), key=lambda x: np.mean(x[1]), reverse=True)
print(f"  {'Ensemble':<12} {'Mean':>7} {'Med':>7} {'Pos%':>6}")
for n, s in ranked:
    a = np.array(s); print(f"  {n:<12} {a.mean():>7.3f} {np.median(a):>7.3f} {(a>0).mean()*100:>5.1f}%")

# Best TF
print(f"\n  BEST TIMEFRAME (mean OOS)")
tsc = {}
for r in all_r:
    tsc.setdefault(r['tf'], []).append(r['oos'])
for tf in ['1d','4h','1h','15m']:
    if tf in tsc:
        a = np.array(tsc[tf]); print(f"  {tf}: mean {a.mean():.3f}, pos {(a>0).mean()*100:.0f}%")

out = os.path.expanduser('~/Desktop/maestro/data/backtest_results/ps_ensemble_mtf.json')
with open(out, 'w') as f:
    json.dump(all_r, f, indent=2, default=str)
print(f"\nSaved to {out} ({len(all_r)} results)")
