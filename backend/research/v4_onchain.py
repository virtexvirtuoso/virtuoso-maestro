"""
V4 On-Chain: Test btc_wiz on-chain signals as 6th confluence component for V3.1-H2
Hypothesis: Macro (M2/yields) captures LIQUIDITY, on-chain captures POSITIONING. Orthogonal edge.
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
from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS, TX_COST
from strategies.composite.mega_strategy_v31 import (
    LEVERAGE_MAP, SCORE4_CRYPTO_MOM_OVERRIDE, VOL_CEILING, VOL_LOOKBACK,
    BEAR_FILTER_DAYS, PORTFOLIO_TRAIL_STOP, TRAIL_REDUCE_FACTOR,
    TRAIL_RECOVERY_DAYS, TRAIL_RECOVERY_THRESHOLD, WEIGHTS,
)

# ── Inline btc_wiz on-chain functions (avoiding dependency hell) ──

def estimate_realized_cap(price, volume=None, decay_factor=0.999):
    weights = pd.Series(0.0, index=price.index)
    weights.iloc[0] = 1.0
    for i in range(1, len(price)):
        weights.iloc[i] = weights.iloc[i-1] * decay_factor + 1.0
    cost_basis = (price * weights).cumsum() / weights.cumsum()
    return cost_basis * 19_500_000  # approx supply

def calculate_mvrv(market_cap, realized_cap):
    return market_cap / realized_cap.replace(0, np.nan)

def estimate_sopr_from_price(price, lookback_window=155):
    cost_basis = price.ewm(span=lookback_window, adjust=False).mean()
    return price / cost_basis.replace(0, np.nan)

def calculate_nupl(market_cap, realized_cap):
    return (market_cap - realized_cap) / market_cap.replace(0, np.nan)

def calculate_sth_cost_basis(price, window=155):
    return price.rolling(window).mean()

def calculate_pi_cycle_indicator(price, short_period=111, long_period=350, multiplier=2):
    short_ma = price.rolling(short_period, min_periods=int(short_period*0.8)).mean()
    long_ma = price.rolling(long_period, min_periods=int(long_period*0.8)).mean()
    long_adj = long_ma * multiplier
    top_signal = (short_ma > long_adj) & (short_ma.shift(1) <= long_adj.shift(1))
    return short_ma, long_adj, top_signal

def calculate_two_year_ma_multiplier(price, period=730, multiplier=5):
    ma_2y = price.rolling(period, min_periods=int(period*0.5)).mean()
    ma_2y_mult = ma_2y * multiplier
    above_mult = (price > ma_2y_mult).astype(int)
    below_ma = (price < ma_2y).astype(int)
    return ma_2y, ma_2y_mult, above_mult, below_ma

# ── Data Loading ──────────────────────────────────────────────
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
    except:
        pass

macro_df = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
macro_df.index = pd.to_datetime(macro_df.index)

# Verify confluence
test_conf, _ = compute_confluence(assets_data["BTC"]["close"], macro_df, cross_asset_df)
print(f"Confluence check: {test_conf.value_counts().sort_index().to_dict()}")
assert test_conf.max() > 1, "Confluence broken"

# ── Compute On-Chain Signals from BTC Price ───────────────────
print("\nComputing on-chain signals from btc_wiz...")
btc = assets_data["BTC"]
btc_close = btc["close"]
btc_volume = btc["volume"]
btc_high = btc["high"]
btc_low = btc["low"]

# 1. Estimated MVRV
est_realized_cap = estimate_realized_cap(btc_close, btc_volume)
market_cap = btc_close * 19_500_000  # approx circulating supply
est_mvrv = calculate_mvrv(market_cap, est_realized_cap)
print(f"  MVRV: mean {est_mvrv.mean():.2f}, current {est_mvrv.iloc[-1]:.2f}")

# 2. Estimated SOPR
est_sopr = estimate_sopr_from_price(btc_close, lookback_window=155)
print(f"  SOPR: mean {est_sopr.mean():.3f}, current {est_sopr.iloc[-1]:.3f}")

# 3. NUPL (Net Unrealized Profit/Loss)
est_nupl = calculate_nupl(market_cap, est_realized_cap)
print(f"  NUPL: mean {est_nupl.mean():.3f}, current {est_nupl.iloc[-1]:.3f}")

# 4. STH Cost Basis
sth_cb = calculate_sth_cost_basis(btc_close, window=155)
print(f"  STH Cost Basis: current ${sth_cb.iloc[-1]:,.0f}")

# 5. Pi Cycle
pi_short, pi_long_adj, pi_top = calculate_pi_cycle_indicator(btc_close)
pi_distance = (pi_long_adj - pi_short) / pi_long_adj  # distance to cross
print(f"  Pi Cycle distance to top: {pi_distance.iloc[-1]:.3f}")

# 6. 2-Year MA Multiplier
ma_2y, ma_2y_mult, above_mult, below_ma = calculate_two_year_ma_multiplier(btc_close)
print(f"  2Y MA: below_ma={below_ma.iloc[-1]}, above_mult={above_mult.iloc[-1]}")

# 7. Reserve Risk
# Need to estimate hodl_bank
hodl_days = btc_close.expanding().apply(lambda x: len(x), raw=True) * btc_close
reserve_risk = btc_close / hodl_days.rolling(365).mean().replace(0, np.nan)
print(f"  Reserve Risk proxy: {reserve_risk.iloc[-1]:.6f}")

# 8. VDD Multiple (Value Days Destroyed)
est_cdd = btc_volume * btc_close  # rough proxy
vdd = est_cdd / est_cdd.rolling(365).mean().replace(0, np.nan)
print(f"  VDD Multiple: {vdd.iloc[-1]:.2f}")

# ── Build On-Chain Signal Variants ────────────────────────────
# Each variant adds an on-chain signal as a LEVERAGE MODIFIER on top of V3.1-H2
# NOT adding to confluence score (that's 0-5, well-tuned), but as an independent overlay

idx = btc_close.index

def build_onchain_signals():
    """Build daily on-chain signal DataFrame"""
    signals = pd.DataFrame(index=idx)
    
    # MVRV Zone: bullish when < 2 (undervalued), bearish when > 3.5 (overheated)
    signals['mvrv'] = est_mvrv.reindex(idx).ffill()
    signals['mvrv_bull'] = (signals['mvrv'] < 2.0).astype(int)
    signals['mvrv_bear'] = (signals['mvrv'] > 3.5).astype(int)
    
    # SOPR: bullish when > 1 (profit-taking healthy), bearish when < 0.95 (capitulation)
    signals['sopr'] = est_sopr.reindex(idx).ffill()
    signals['sopr_bull'] = (signals['sopr'] > 1.0).astype(int)
    signals['sopr_capitulation'] = (signals['sopr'] < 0.95).astype(int)
    
    # NUPL: bullish in accumulation (0-0.25), bearish in euphoria (>0.75)
    signals['nupl'] = est_nupl.reindex(idx).ffill()
    signals['nupl_accum'] = (signals['nupl'].between(0, 0.5)).astype(int)
    signals['nupl_euphoria'] = (signals['nupl'] > 0.75).astype(int)
    
    # Price vs STH Cost Basis: bullish when above (STH in profit = support)
    signals['sth_cb'] = sth_cb.reindex(idx).ffill()
    signals['above_sth'] = (btc_close > signals['sth_cb']).astype(int)
    
    # Pi Cycle distance: bearish when close to cross (< 5%)
    signals['pi_dist'] = pi_distance.reindex(idx).ffill()
    signals['pi_danger'] = (signals['pi_dist'] < 0.05).astype(int)
    
    # 2Y MA: bullish when below MA (deep value), bearish when above multiplier
    signals['below_2y_ma'] = below_ma.reindex(idx).ffill().astype(int)
    signals['above_2y_mult'] = above_mult.reindex(idx).ffill().astype(int)
    
    # VDD: bearish when > 2 (old coins moving = distribution)
    signals['vdd'] = vdd.reindex(idx).ffill()
    signals['vdd_distribution'] = (signals['vdd'] > 2.0).astype(int)
    
    return signals

onchain = build_onchain_signals()
print(f"\nOn-chain signals built: {onchain.shape}")
print(f"MVRV bull days: {onchain['mvrv_bull'].sum()}/{len(onchain)}")
print(f"SOPR bull days: {onchain['sopr_bull'].sum()}/{len(onchain)}")
print(f"NUPL accum days: {onchain['nupl_accum'].sum()}/{len(onchain)}")
print(f"Above STH CB: {onchain['above_sth'].sum()}/{len(onchain)}")
print(f"Pi danger: {onchain['pi_danger'].sum()}/{len(onchain)}")

# ── Simulation Engine ─────────────────────────────────────────
def simulate_v31(asset_data, macro_data, cross_data, start_date=None, end_date=None,
                 leverage_modifier=None):
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
    
    per_asset_lev = {}
    per_asset_ret = {}
    per_asset_close = {}
    
    for asset in assets:
        close = asset_data[asset]["close"]
        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        conf, bd = compute_confluence(close, macro_data, cross_data,
                                       sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"])
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)
        base_lev = conf.map(lambda c: LEVERAGE_MAP.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), SCORE4_CRYPTO_MOM_OVERRIDE)
        base_lev = base_lev.shift(1).fillna(0)
        per_asset_lev[asset] = base_lev
        per_asset_ret[asset] = asset_data[asset]["close"].reindex(common_idx).pct_change().fillna(0)
        per_asset_close[asset] = asset_data[asset]["close"].reindex(common_idx)
    
    btc_conf = compute_confluence(asset_data["BTC"]["close"], macro_data, cross_data)[0].reindex(common_idx).fillna(0)
    btc_close = per_asset_close.get("BTC")
    
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
        per_asset_exp = {}
        for asset in assets:
            per_asset_exp[asset] = WEIGHTS.get(asset, 0.25) * per_asset_lev[asset].iloc[i]
        target_exposure = sum(per_asset_exp.values())
        
        # Vol ceiling
        if i >= VOL_LOOKBACK:
            window = btc_close.iloc[max(0,i-VOL_LOOKBACK):i]
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
        
        # On-chain modifier
        if leverage_modifier is not None:
            target_exposure, per_asset_exp = leverage_modifier(i, target_exposure, per_asset_exp, common_idx)
        
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


# Walk-forward folds (same as v31 backtest)
WF_FOLDS = [
    ("2021-04-10", "2021-08-13"), ("2021-08-14", "2021-12-17"),
    ("2021-12-18", "2022-04-22"), ("2022-04-23", "2022-08-26"),
    ("2022-08-27", "2022-12-30"), ("2022-12-31", "2023-05-05"),
    ("2023-05-06", "2023-09-08"), ("2023-09-09", "2024-01-12"),
    ("2024-01-13", "2024-05-17"), ("2024-05-18", "2024-09-20"),
    ("2024-09-21", "2025-01-24"), ("2025-01-25", "2025-05-30"),
    ("2025-05-31", "2025-10-03"),
]

def walk_forward(modifier=None):
    folds = []
    for fold_num, (ts, te) in enumerate(WF_FOLDS):
        pnl, lev = simulate_v31(assets_data, macro_df, cross_asset_df,
                                 start_date=ts, end_date=te, leverage_modifier=modifier)
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


def run_test(name, modifier=None):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    pnl, lev = simulate_v31(assets_data, macro_df, cross_asset_df,
                             start_date="2020-01-01", end_date="2026-02-01",
                             leverage_modifier=modifier)
    is_m = compute_metrics(pnl)
    is_m['mean_leverage'] = round(float(lev.mean()), 3)
    is_m['flat_pct'] = round(float((lev == 0).mean() * 100), 1)
    wf = walk_forward(modifier=modifier)
    print(f"  IS:  Sharpe {is_m['sharpe']}, CAGR {is_m['cagr']}%, MaxDD {is_m['max_dd']}%")
    print(f"  OOS: Sharpe {wf['mean_oos_sharpe']}, p={wf['p_value']}, Pos {wf['positive_folds']}/{wf['n_folds']}")
    return {'is': is_m, 'oos': wf}


# ── On-Chain Modifiers ────────────────────────────────────────

# OC1: MVRV Cycle Position — reduce when overheated, boost when undervalued
def oc1_mvrv(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    mvrv_val = oc['mvrv'].iloc[i-1]  # lagged
    if np.isnan(mvrv_val): return target, per_asset
    if mvrv_val > 3.5:  # overheated
        mult = 0.3
    elif mvrv_val > 2.5:  # getting warm
        mult = 0.7
    elif mvrv_val < 1.2:  # deep value
        mult = 1.3
    elif mvrv_val < 2.0:  # undervalued
        mult = 1.1
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC2: SOPR Momentum — cut when capitulating, lean in on healthy profit-taking
def oc2_sopr(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    sopr_val = oc['sopr'].iloc[i-1]
    if np.isnan(sopr_val): return target, per_asset
    if sopr_val < 0.95:  # capitulation
        mult = 0.5
    elif sopr_val > 1.05:  # healthy profit-taking
        mult = 1.1
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC3: STH Cost Basis — below STH CB = bearish (short-term holders underwater)
def oc3_sth(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    above = oc['above_sth'].iloc[i-1]
    if np.isnan(above): return target, per_asset
    if above == 0:  # below STH cost basis
        mult = 0.5
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC4: Pi Cycle Danger — reduce near cycle top
def oc4_pi(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    danger = oc['pi_danger'].iloc[i-1]
    if np.isnan(danger): return target, per_asset
    if danger == 1:  # Pi cycle close to crossing
        mult = 0.3
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC5: Composite On-Chain Health (MVRV + SOPR + NUPL + STH CB)
def oc5_composite(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    
    score = 0  # 0-4 health score
    row = oc.iloc[i-1]
    
    # MVRV: undervalued = +1
    if not np.isnan(row['mvrv']) and row['mvrv'] < 2.0:
        score += 1
    # SOPR: healthy = +1
    if not np.isnan(row['sopr']) and row['sopr'] > 1.0:
        score += 1
    # NUPL: not euphoric = +1
    if not np.isnan(row['nupl']) and row['nupl'] < 0.75:
        score += 1
    # Above STH CB = +1
    if not np.isnan(row.get('above_sth', np.nan)) and row['above_sth'] == 1:
        score += 1
    
    # Map score to multiplier
    mult_map = {0: 0.3, 1: 0.6, 2: 1.0, 3: 1.15, 4: 1.3}
    mult = mult_map.get(score, 1.0)
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC6: VDD Distribution Warning — reduce when old coins moving
def oc6_vdd(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    vdd_val = oc['vdd'].iloc[i-1]
    if np.isnan(vdd_val): return target, per_asset
    if vdd_val > 3.0:  # heavy distribution
        mult = 0.4
    elif vdd_val > 2.0:  # moderate distribution
        mult = 0.7
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}

# OC7: 2Y MA Value Zone — boost when below 2Y MA (deep value)
def oc7_2y_ma(i, target, per_asset, idx):
    oc = onchain.reindex(idx)
    if i < 1: return target, per_asset
    below = oc['below_2y_ma'].iloc[i-1]
    above = oc['above_2y_mult'].iloc[i-1]
    if not np.isnan(below) and below == 1:
        mult = 1.3  # deep value buy zone
    elif not np.isnan(above) and above == 1:
        mult = 0.5  # overbought
    else:
        mult = 1.0
    return target * mult, {a: v*mult for a,v in per_asset.items()}


# ── Run All Tests ─────────────────────────────────────────────
results = {}
results['baseline'] = run_test("V3.1-H2 Baseline")
results['OC1_mvrv'] = run_test("OC1: MVRV Cycle Position", modifier=oc1_mvrv)
results['OC2_sopr'] = run_test("OC2: SOPR Momentum", modifier=oc2_sopr)
results['OC3_sth_basis'] = run_test("OC3: STH Cost Basis", modifier=oc3_sth)
results['OC4_pi_cycle'] = run_test("OC4: Pi Cycle Danger", modifier=oc4_pi)
results['OC5_composite'] = run_test("OC5: Composite On-Chain", modifier=oc5_composite)
results['OC6_vdd'] = run_test("OC6: VDD Distribution", modifier=oc6_vdd)
results['OC7_2y_ma'] = run_test("OC7: 2Y MA Value Zone", modifier=oc7_2y_ma)

# Combos of top performers
dir_scores = [(k, v['oos']['mean_oos_sharpe']) for k,v in results.items() if k.startswith('OC')]
dir_scores.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*60}\nIndividual Rankings:\n{'='*60}")
for name, score in dir_scores:
    delta = score - results['baseline']['oos']['mean_oos_sharpe']
    print(f"  {name}: OOS {score:.3f} ({delta:+.3f} vs baseline)")

# Combine top 2
top1, top2 = dir_scores[0][0], dir_scores[1][0]
mod_map = {'OC1_mvrv': oc1_mvrv, 'OC2_sopr': oc2_sopr, 'OC3_sth_basis': oc3_sth,
           'OC4_pi_cycle': oc4_pi, 'OC5_composite': oc5_composite,
           'OC6_vdd': oc6_vdd, 'OC7_2y_ma': oc7_2y_ma}

def combo_mod(m1, m2):
    def combined(i, target, per_asset, idx):
        target, per_asset = m1(i, target, per_asset, idx)
        target, per_asset = m2(i, target, per_asset, idx)
        return target, per_asset
    return combined

results['combo_top2'] = run_test(f"Combo: {top1} + {top2}",
                                  modifier=combo_mod(mod_map[top1], mod_map[top2]))

# Triple combo
if len(dir_scores) >= 3:
    top3 = dir_scores[2][0]
    def combo3(m1, m2, m3):
        def combined(i, target, per_asset, idx):
            target, per_asset = m1(i, target, per_asset, idx)
            target, per_asset = m2(i, target, per_asset, idx)
            target, per_asset = m3(i, target, per_asset, idx)
            return target, per_asset
        return combined
    results['combo_top3'] = run_test(f"Combo: {top1} + {top2} + {top3}",
                                      modifier=combo3(mod_map[top1], mod_map[top2], mod_map[top3]))

# ── Save ──────────────────────────────────────────────────────
ranking = [(k, v['oos']['mean_oos_sharpe'], v['is']['sharpe']) for k,v in results.items()]
ranking.sort(key=lambda x: x[1], reverse=True)

output = {
    'timestamp': datetime.now().isoformat(),
    'description': 'V4 On-Chain: btc_wiz signals as leverage modifier on V3.1-H2',
    'results': results,
    'ranking': ranking,
    'onchain_stats': {
        'mvrv_current': float(est_mvrv.iloc[-1]),
        'sopr_current': float(est_sopr.iloc[-1]),
        'nupl_current': float(est_nupl.iloc[-1]),
        'sth_cb_current': float(sth_cb.iloc[-1]),
        'pi_distance': float(pi_distance.iloc[-1]),
    }
}

with open('../data/research/v4_onchain_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

baseline_oos = results['baseline']['oos']['mean_oos_sharpe']
print(f"\n{'='*60}\nEXECUTIVE SUMMARY\n{'='*60}")
print(f"\n{'Variant':<45} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'CAGR':>8} {'MaxDD':>8} {'p-val':>7}")
print("-" * 95)
for name, oos_s, is_s in ranking:
    r = results[name]
    print(f"{name:<45} {r['is']['sharpe']:>10.3f} {r['oos']['mean_oos_sharpe']:>11.3f} {r['is']['cagr']:>7.1f}% {r['is']['max_dd']:>7.1f}% {r['oos']['p_value']:>7.4f}")

print(f"\nBaseline OOS: {baseline_oos:.3f}")
for name, oos_s, _ in ranking:
    if name != 'baseline':
        delta = oos_s - baseline_oos
        print(f"  {name}: {delta:+.3f} {'** BETTER **' if delta > 0.05 else 'BETTER' if delta > 0 else 'WORSE'}")
