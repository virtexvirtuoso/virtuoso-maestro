"""
V4 Ultrathink: 5 signal-level improvement directions for V3.1-H2
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import compute_confluence, detect_regime, ASSET_CONFIGS, TX_COST
from strategies.composite.mega_strategy_v31 import (
    LEVERAGE_MAP, SCORE4_CRYPTO_MOM_OVERRIDE, VOL_CEILING, VOL_LOOKBACK,
    BEAR_FILTER_DAYS, PORTFOLIO_TRAIL_STOP, TRAIL_REDUCE_FACTOR,
    TRAIL_RECOVERY_DAYS, TRAIL_RECOVERY_THRESHOLD, WEIGHTS,
)

# ── Data Loading (matching backtest_mega_v31.py exactly) ──────
CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}

print("Loading data...")
stock = StockDataLoader()
fred = MacroDataLoader()

assets_data = {}
for name, ticker in CRYPTO_TICKERS.items():
    df = stock.get_ohlcv(ticker, "1d", "2017-01-01")
    df.index = pd.to_datetime(df.index)
    assets_data[name] = df
    print(f"  {name}: {len(df)} days")

cross_asset_df = pd.DataFrame()
for col, ticker in CROSS_ASSET_MAP.items():
    try:
        df = stock.get_ohlcv(ticker, "1d", "2017-01-01")
        df.index = pd.to_datetime(df.index)
        cross_asset_df[col] = df["close"]
        print(f"  {col}: {len(df)} days")
    except Exception as e:
        print(f"  {col}: FAILED - {e}")

macro_df = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
macro_df.index = pd.to_datetime(macro_df.index)
print(f"  Macro: {macro_df.shape}, cols: {list(macro_df.columns)}")

# Also load raw cross-asset for risk-off signals
raw_cross = {}
for ticker in ["SPY", "GLD", "TLT"]:
    df = stock.get_ohlcv(ticker, "1d", "2017-01-01")
    df.index = pd.to_datetime(df.index)
    raw_cross[ticker] = df["close"]

# Load raw FRED for risk-off
dxy_raw = fred.get_series("DTWEXBGS", "2017-01-01")
if isinstance(dxy_raw, pd.DataFrame):
    dxy_raw = dxy_raw.iloc[:, 0]
dxy_raw.index = pd.to_datetime(dxy_raw.index)

hy_raw = fred.get_series("BAMLH0A0HYM2", "2017-01-01")
if isinstance(hy_raw, pd.DataFrame):
    hy_raw = hy_raw.iloc[:, 0]
hy_raw.index = pd.to_datetime(hy_raw.index)

# Precompute ADX for BTC
def _adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period).mean()

btc_adx = _adx(assets_data["BTC"]["high"], assets_data["BTC"]["low"], assets_data["BTC"]["close"], 14)

print("Data loaded.\n")

# ── Verify confluence works ───────────────────────────────────
test_conf, test_bd = compute_confluence(assets_data["BTC"]["close"], macro_df, cross_asset_df)
vc = test_conf.value_counts().sort_index()
print(f"Confluence check: {vc.to_dict()}")
assert test_conf.max() > 1, "Confluence broken — only 0/1 values"

# ── V3.1 Simulation Engine ───────────────────────────────────
def simulate_v31(asset_data, macro_data, cross_data, start_date=None, end_date=None,
                 leverage_modifier=None, weight_override_fn=None):
    """Full V3.1-H2 simulation with optional leverage modifier."""
    assets = list(asset_data.keys())
    
    common_idx = asset_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(asset_data[a].index)
    if start_date:
        common_idx = common_idx[common_idx >= pd.Timestamp(start_date)]
    if end_date:
        common_idx = common_idx[common_idx <= pd.Timestamp(end_date)]
    
    if len(common_idx) < 30:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Pre-compute per-asset signals
    per_asset_lev = {}
    per_asset_ret = {}
    per_asset_close = {}
    per_asset_bd = {}
    
    for asset in assets:
        df = asset_data[asset].reindex(common_idx)
        close = df["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        
        # Need full history for confluence, then slice
        full_close = asset_data[asset]["close"]
        conf, bd = compute_confluence(full_close, macro_data, cross_data,
                                       sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"])
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)
        
        base_lev = conf.map(lambda c: LEVERAGE_MAP.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), SCORE4_CRYPTO_MOM_OVERRIDE)
        base_lev = base_lev.shift(1).fillna(0)
        
        per_asset_lev[asset] = base_lev
        per_asset_ret[asset] = close.pct_change().fillna(0)
        per_asset_close[asset] = close
        per_asset_bd[asset] = bd
    
    btc_close = per_asset_close.get("BTC", per_asset_close[assets[0]])
    btc_conf_full = compute_confluence(asset_data["BTC"]["close"], macro_data, cross_data)[0]
    btc_conf = btc_conf_full.reindex(common_idx).fillna(0)
    
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    total_leverage = np.zeros(n)
    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0
    prev_exposure = 0.0
    
    for i in range(1, n):
        # Dynamic weights
        if weight_override_fn is not None:
            cur_w = weight_override_fn(i, per_asset_close, common_idx)
        else:
            cur_w = WEIGHTS
        
        # Base exposure
        per_asset_exp = {}
        for asset in assets:
            per_asset_exp[asset] = cur_w.get(asset, 0.25) * per_asset_lev[asset].iloc[i]
        target_exposure = sum(per_asset_exp.values())
        
        # Vol ceiling
        if i >= VOL_LOOKBACK:
            window = btc_close.iloc[max(0, i-VOL_LOOKBACK):i]
            rvol = window.pct_change().std() * np.sqrt(252)
            if rvol > VOL_CEILING:
                target_exposure *= 0.5
                per_asset_exp = {a: v*0.5 for a,v in per_asset_exp.items()}
        
        # Bear filter
        btc_score = int(btc_conf.iloc[i-1])
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0
        if consecutive_low >= BEAR_FILTER_DAYS:
            target_exposure = 0.0
            per_asset_exp = {a: 0.0 for a in assets}
        
        # Trail stop
        if PORTFOLIO_TRAIL_STOP > 0:
            dd = 1 - equity / peak_eq
            if dd > PORTFOLIO_TRAIL_STOP:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start_idx = i
                target_exposure *= TRAIL_REDUCE_FACTOR
                per_asset_exp = {a: v*TRAIL_REDUCE_FACTOR for a,v in per_asset_exp.items()}
            elif in_drawdown:
                if (i - dd_start_idx) > TRAIL_RECOVERY_DAYS and equity > peak_eq * TRAIL_RECOVERY_THRESHOLD:
                    in_drawdown = False
        
        # Apply direction modifier
        if leverage_modifier is not None:
            target_exposure, per_asset_exp = leverage_modifier(
                i, target_exposure, per_asset_exp,
                btc_close=btc_close, idx=common_idx, equity=equity,
                peak_eq=peak_eq, btc_conf=btc_conf
            )
        
        # PnL
        port_ret = sum(per_asset_exp.get(a, 0) * per_asset_ret[a].iloc[i] for a in assets)
        port_ret -= abs(target_exposure - prev_exposure) * TX_COST
        prev_exposure = target_exposure
        
        daily_pnl[i] = port_ret
        total_leverage[i] = target_exposure
        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)
    
    return pd.Series(daily_pnl, index=common_idx), pd.Series(total_leverage, index=common_idx)


def compute_metrics(pnl):
    if len(pnl) < 10 or pnl.std() == 0:
        return {k: 0.0 for k in ["sharpe","cagr","max_dd","sortino","calmar","total_return","win_rate"]}
    eq = (1 + pnl).cumprod()
    years = len(pnl) / 252
    total_ret = (eq.iloc[-1] - 1) * 100
    cagr = (eq.iloc[-1] ** (1/max(years, 0.1)) - 1) * 100
    ann_ret = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = pnl[pnl < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (pnl > 0).sum() / (pnl != 0).sum() * 100 if (pnl != 0).sum() > 0 else 0
    return {
        'sharpe': round(sharpe, 3), 'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1),
        'sortino': round(sortino, 3), 'calmar': round(calmar, 3),
        'total_return': round(total_ret, 1), 'win_rate': round(win_rate, 1),
    }


# ── Walk-Forward (13 folds, non-overlapping, matching v31 backtest) ──
WF_FOLDS = [
    ("2021-04-10", "2021-08-13"),
    ("2021-08-14", "2021-12-17"),
    ("2021-12-18", "2022-04-22"),
    ("2022-04-23", "2022-08-26"),
    ("2022-08-27", "2022-12-30"),
    ("2022-12-31", "2023-05-05"),
    ("2023-05-06", "2023-09-08"),
    ("2023-09-09", "2024-01-12"),
    ("2024-01-13", "2024-05-17"),
    ("2024-05-18", "2024-09-20"),
    ("2024-09-21", "2025-01-24"),
    ("2025-01-25", "2025-05-30"),
    ("2025-05-31", "2025-10-03"),
]

def walk_forward(leverage_modifier=None, weight_override_fn=None):
    folds = []
    for fold_num, (ts, te) in enumerate(WF_FOLDS):
        pnl, lev = simulate_v31(assets_data, macro_df, cross_asset_df,
                                 start_date=ts, end_date=te,
                                 leverage_modifier=leverage_modifier,
                                 weight_override_fn=weight_override_fn)
        if len(pnl) < 20:
            continue
        m = compute_metrics(pnl)
        folds.append({'fold': fold_num+1, 'test_start': ts, 'test_end': te,
                       'sharpe': m['sharpe'], 'return_pct': m['total_return']})
    
    sharpes = [f['sharpe'] for f in folds]
    n = len(sharpes)
    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes, ddof=1) if n > 1 else 0
    t_stat = mean_s / (std_s / np.sqrt(n)) if std_s > 0 else 0
    p_val = 1 - stats.t.cdf(t_stat, df=n-1) if t_stat > 0 and n > 1 else 1.0
    return {
        'folds': folds, 'n_folds': n,
        'mean_oos_sharpe': round(mean_s, 3), 'median_oos_sharpe': round(float(np.median(sharpes)), 3),
        'positive_folds': sum(1 for s in sharpes if s > 0),
        't_stat': round(t_stat, 3), 'p_value': round(p_val, 4),
    }


def run_test(name, leverage_modifier=None, weight_override_fn=None):
    print(f"\n{'='*60}\nTesting: {name}\n{'='*60}")
    pnl, lev = simulate_v31(assets_data, macro_df, cross_asset_df,
                             start_date="2020-01-01", end_date="2026-02-01",
                             leverage_modifier=leverage_modifier,
                             weight_override_fn=weight_override_fn)
    is_m = compute_metrics(pnl)
    is_m['mean_leverage'] = round(float(lev.mean()), 3)
    is_m['flat_pct'] = round(float((lev == 0).mean() * 100), 1)
    
    wf = walk_forward(leverage_modifier=leverage_modifier, weight_override_fn=weight_override_fn)
    
    print(f"  IS:  Sharpe {is_m['sharpe']}, CAGR {is_m['cagr']}%, MaxDD {is_m['max_dd']}%")
    print(f"  OOS: Sharpe {wf['mean_oos_sharpe']}, p={wf['p_value']}, Pos {wf['positive_folds']}/{wf['n_folds']}")
    return {'is': is_m, 'oos': wf}


# ── Direction Modifiers ───────────────────────────────────────

def d1_vol_regime(i, target_exp, per_asset_exp, btc_close=None, idx=None, **kw):
    if i < 30: return target_exp, per_asset_exp
    window = btc_close.iloc[max(0,i-30):i]
    rvol = window.pct_change().std() * np.sqrt(252)
    ret_20d = (btc_close.iloc[i-1] / btc_close.iloc[max(0,i-20)] - 1) if i >= 20 else 0
    if rvol > 0.60 and ret_20d > 0:
        mult = 1.3
    elif rvol > 0.60 and ret_20d <= 0:
        mult = 0.3
    elif rvol < 0.40:
        mult = 0.8
    else:
        mult = 1.0
    return target_exp * mult, {a: v*mult for a,v in per_asset_exp.items()}


def d2_risk_off(i, target_exp, per_asset_exp, idx=None, **kw):
    if i < 25: return target_exp, per_asset_exp
    date = idx[i]
    
    # DXY momentum
    dxy = dxy_raw.reindex(idx).ffill()
    dxy_risk = False
    if i >= 20:
        v1, v2 = dxy.iloc[i-1], dxy.iloc[max(0,i-20)]
        if not np.isnan(v1) and not np.isnan(v2) and v2 > 0:
            dxy_risk = (v1/v2 - 1) > 0.02
    
    # HY spread
    hy = hy_raw.reindex(idx).ffill()
    hy_risk = False
    if i >= 20:
        v1, v2 = hy.iloc[i-1], hy.iloc[max(0,i-20)]
        if not np.isnan(v1) and not np.isnan(v2):
            hy_risk = (v1 - v2) > 0.3
    
    # Gold outperformance
    gld = raw_cross["GLD"].reindex(idx).ffill()
    spy = raw_cross["SPY"].reindex(idx).ffill()
    gold_risk = False
    if i >= 20:
        g1, g2 = gld.iloc[i-1], gld.iloc[max(0,i-20)]
        s1, s2 = spy.iloc[i-1], spy.iloc[max(0,i-20)]
        if not any(np.isnan(x) for x in [g1,g2,s1,s2]) and g2 > 0 and s2 > 0:
            gold_risk = (g1/g2 - s1/s2) > 0.03
    
    if sum([dxy_risk, hy_risk, gold_risk]) >= 2:
        return target_exp * 0.5, {a: v*0.5 for a,v in per_asset_exp.items()}
    return target_exp, per_asset_exp


def d3_adx_quality(i, target_exp, per_asset_exp, idx=None, btc_conf=None, **kw):
    if i < 20: return target_exp, per_asset_exp
    adx_vals = btc_adx.reindex(idx).ffill()
    adx_val = adx_vals.iloc[i-1] if not np.isnan(adx_vals.iloc[i-1]) else 25
    conf = int(btc_conf.iloc[i-1]) if i < len(btc_conf) else 0
    
    if adx_val < 20 and conf >= 3:
        mult = 0.5
    elif adx_val > 30 and conf >= 3:
        mult = 1.2
    else:
        mult = 1.0
    return target_exp * mult, {a: v*mult for a,v in per_asset_exp.items()}


def d4_sentiment(i, target_exp, per_asset_exp, btc_close=None, idx=None, btc_conf=None, **kw):
    if i < 25: return target_exp, per_asset_exp
    spy = raw_cross["SPY"].reindex(idx).ffill()
    
    btc_r = btc_close.iloc[i-1] / btc_close.iloc[max(0,i-20)] - 1 if i >= 20 else 0
    spy_v = spy.iloc[max(0,i-20)]
    spy_r = (spy.iloc[i-1] / spy_v - 1) if not np.isnan(spy_v) and spy_v > 0 else 0
    ratio_mom = btc_r - spy_r
    conf = int(btc_conf.iloc[i-1]) if i < len(btc_conf) else 0
    
    if ratio_mom < -0.05 and conf >= 3:
        mult = 0.7
    elif ratio_mom > 0.10:
        mult = 1.2
    else:
        mult = 1.0
    return target_exp * mult, {a: v*mult for a,v in per_asset_exp.items()}


_dw_cache = {}
def d5_dynamic_weights(i, per_asset_close, common_idx):
    if i < 65: return WEIGHTS.copy()
    period = i // 21
    if period in _dw_cache: return _dw_cache[period]
    
    new_w = {}
    for asset in per_asset_close:
        close = per_asset_close[asset]
        rets = close.iloc[max(0,i-60):i].pct_change().dropna()
        s = rets.mean() / rets.std() * np.sqrt(252) if len(rets) > 10 and rets.std() > 0 else 0
        base = WEIGHTS.get(asset, 0.25)
        if s > 0.5:
            new_w[asset] = base * min(2.0, 1 + s/2)
        elif s < -0.5:
            new_w[asset] = base * max(0.0, 1 + s/2)
        else:
            new_w[asset] = base
    
    total = sum(new_w.values())
    if total > 0:
        new_w = {a: v/total for a,v in new_w.items()}
    else:
        new_w = WEIGHTS.copy()
    _dw_cache[period] = new_w
    return new_w


# ── Run All ───────────────────────────────────────────────────
results = {}
results['baseline'] = run_test("V3.1-H2 Baseline")
results['D1_vol_regime'] = run_test("D1: Vol Regime Overlay", leverage_modifier=d1_vol_regime)
results['D2_risk_off'] = run_test("D2: Cross-Asset Risk-Off Veto", leverage_modifier=d2_risk_off)
results['D3_adx_quality'] = run_test("D3: ADX Momentum Quality", leverage_modifier=d3_adx_quality)
results['D4_sentiment'] = run_test("D4: BTC/SPY Sentiment", leverage_modifier=d4_sentiment)
_dw_cache.clear()
results['D5_dynamic_weights'] = run_test("D5: Dynamic Weights", weight_override_fn=d5_dynamic_weights)

# Rank individual directions
dir_scores = [(k, v['oos']['mean_oos_sharpe']) for k,v in results.items() if k.startswith('D')]
dir_scores.sort(key=lambda x: x[1], reverse=True)
print(f"\n{'='*60}\nIndividual Rankings:\n{'='*60}")
for name, score in dir_scores:
    print(f"  {name}: OOS Sharpe {score:.3f}")

# Combos of top performers
def combine_mods(mods):
    def combined(i, target_exp, per_asset_exp, **kw):
        for m in mods:
            target_exp, per_asset_exp = m(i, target_exp, per_asset_exp, **kw)
        return target_exp, per_asset_exp
    return combined

mod_map = {'D1_vol_regime': d1_vol_regime, 'D2_risk_off': d2_risk_off,
           'D3_adx_quality': d3_adx_quality, 'D4_sentiment': d4_sentiment}

top_dirs = [d for d in dir_scores if d[0] in mod_map]
if len(top_dirs) >= 2:
    t1, t2 = top_dirs[0][0], top_dirs[1][0]
    results['combo_top2'] = run_test(f"Combo: {t1} + {t2}",
                                      leverage_modifier=combine_mods([mod_map[t1], mod_map[t2]]))
if len(top_dirs) >= 3:
    t3 = top_dirs[2][0]
    results['combo_top3'] = run_test(f"Combo: {t1} + {t2} + {t3}",
                                      leverage_modifier=combine_mods([mod_map[t1], mod_map[t2], mod_map[t3]]))

# Also test best modifier + dynamic weights
if len(top_dirs) >= 1:
    best_mod = top_dirs[0][0]
    _dw_cache.clear()
    results['combo_best_plus_dw'] = run_test(f"Combo: {best_mod} + DynamicWeights",
                                              leverage_modifier=mod_map[best_mod],
                                              weight_override_fn=d5_dynamic_weights)

# ── Save ──────────────────────────────────────────────────────
output = {'timestamp': datetime.now().isoformat(), 'results': results}
ranking = [(k, v['oos']['mean_oos_sharpe'], v['is']['sharpe']) for k,v in results.items()]
ranking.sort(key=lambda x: x[1], reverse=True)
output['ranking'] = ranking

with open('../data/research/v4_ultrathink_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

baseline_oos = results['baseline']['oos']['mean_oos_sharpe']
print(f"\n{'='*60}\nEXECUTIVE SUMMARY\n{'='*60}")
print(f"\n{'Variant':<50} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'CAGR':>8} {'MaxDD':>8} {'p-val':>7}")
print("-" * 100)
for name, oos_s, is_s in ranking:
    r = results[name]
    print(f"{name:<50} {r['is']['sharpe']:>10.3f} {r['oos']['mean_oos_sharpe']:>11.3f} {r['is']['cagr']:>7.1f}% {r['is']['max_dd']:>7.1f}% {r['oos']['p_value']:>7.4f}")

print(f"\nBaseline OOS: {baseline_oos:.3f}")
for name, oos_s, _ in ranking:
    if name != 'baseline':
        delta = oos_s - baseline_oos
        print(f"  {name}: {delta:+.3f} {'BETTER' if delta > 0 else 'WORSE'}")
