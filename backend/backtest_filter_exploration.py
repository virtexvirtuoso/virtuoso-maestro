"""
Filter Exploration: Test 10 filter combos × 6 assets for MacroMomentum V2.
Identifies which filters add edge vs just reduce frequency.
"""
import sys, os, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score, FRED_SERIES

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2017-01-01"
END = "2026-02-01"
TX_COST = 0.001

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD"]

# Optimized defaults from V2
PARAMS = dict(
    sma_slow=90, trail_stop=0.18, rsi_entry=44, rsi_exit=70,
    ema_period=21, bb_period=20, bb_std=2.0, atr_exit_mult=2.0,
    max_position=1.46, initial_size=0.5, pyramid_size=0.3,
    rsi_period=14, momentum_period=25,
)

FILTER_COMBOS = {
    "1_NoFilter": dict(use_sma=False, use_momentum=False, use_m2=False, use_macro=False, macro_thresh=0),
    "2_SMA_only": dict(use_sma=True, use_momentum=False, use_m2=False, use_macro=False, macro_thresh=0),
    "3_Momentum_only": dict(use_sma=False, use_momentum=True, use_m2=False, use_macro=False, macro_thresh=0),
    "4_M2_only": dict(use_sma=False, use_momentum=False, use_m2=True, use_macro=False, macro_thresh=0),
    "5_MacroScore_only": dict(use_sma=False, use_momentum=False, use_m2=False, use_macro=True, macro_thresh=3),
    "6_SMA+Mom": dict(use_sma=True, use_momentum=True, use_m2=False, use_macro=False, macro_thresh=0),
    "7_SMA+M2": dict(use_sma=True, use_momentum=False, use_m2=True, use_macro=False, macro_thresh=0),
    "8_SMA+Mom+M2(V2)": dict(use_sma=True, use_momentum=True, use_m2=True, use_macro=False, macro_thresh=0),
    "9_SMA+Mom+Macro>=2": dict(use_sma=True, use_momentum=True, use_m2=False, use_macro=True, macro_thresh=2),
    "10_SMA+Mom+AnyMacro": dict(use_sma=True, use_momentum=True, use_m2=False, use_macro=True, macro_thresh=1),
}


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bb(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s


def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_strategy(df, m2_acc, macro_score, filter_cfg):
    """Run dip-buying strategy with specified filter combo. Returns equity, returns, stats."""
    p = PARAMS
    close = df["close"].copy()
    n = len(close)
    
    if n < p["sma_slow"] + 50:
        return None
    
    # Indicators
    sma = close.rolling(p["sma_slow"]).mean()
    roc = close.pct_change(p["momentum_period"])
    rsi = compute_rsi(close, p["rsi_period"])
    ema = close.ewm(span=p["ema_period"], adjust=False).mean()
    bb_lower, bb_mid, bb_upper = compute_bb(close, p["bb_period"], p["bb_std"])
    atr = compute_atr(df)
    
    # Build regime filter
    regime = pd.Series(True, index=df.index)
    
    if filter_cfg["use_sma"]:
        regime &= (close > sma)
    if filter_cfg["use_momentum"]:
        regime &= (roc > 0)
    if filter_cfg["use_m2"]:
        m2 = m2_acc.reindex(df.index, method="ffill").fillna(False).astype(bool)
        regime &= m2
    if filter_cfg["use_macro"]:
        ms = macro_score.reindex(df.index, method="ffill").fillna(3)
        regime &= (ms >= filter_cfg["macro_thresh"])
    
    # Dip conditions
    rsi_dip = rsi < p["rsi_entry"]
    ema_dip = close < ema
    bb_dip = close <= bb_lower
    any_dip = rsi_dip | ema_dip | bb_dip
    
    # Shift to avoid lookahead
    regime = regime.shift(1).fillna(False)
    any_dip = any_dip.shift(1).fillna(False)
    rsi_hot = (rsi > p["rsi_exit"]).shift(1).fillna(False)
    bb_hot = (close > bb_upper).shift(1).fillna(False)
    atr_s = atr.shift(1).fillna(0)
    
    # Simulate
    pos = 0.0
    avg_entry = 0.0
    equity = 1.0
    peak = 1.0
    total_cost = 0.0
    
    equities = np.ones(n)
    positions = np.zeros(n)
    n_entries = 0
    n_pyramids = 0
    n_wins = 0
    n_losses = 0
    
    warmup = p["sma_slow"] + 5
    
    for i in range(warmup, n):
        price = close.iloc[i]
        prev = close.iloc[i-1]
        
        if pos > 0 and prev > 0:
            daily_ret = (price - prev) / prev
            equity *= (1 + daily_ret * pos)
        peak = max(peak, equity)
        equities[i] = equity
        
        # Trailing stop
        if pos > 0:
            dd = 1 - equity / peak
            if dd >= p["trail_stop"]:
                trade_ret = (price - avg_entry) / avg_entry if avg_entry > 0 else 0
                if trade_ret > 0: n_wins += 1
                else: n_losses += 1
                equity *= (1 - pos * TX_COST)  # exit cost
                pos = 0.0; avg_entry = 0.0; total_cost = 0.0
                positions[i] = 0
                continue
        
        # Trim
        if pos > 0:
            trim = 0
            if rsi_hot.iloc[i]:
                trim = pos * 0.25
            elif bb_hot.iloc[i]:
                trim = pos * 0.25
            elif avg_entry > 0 and atr_s.iloc[i] > 0:
                ext = (price - avg_entry) / atr_s.iloc[i]
                if ext > p["atr_exit_mult"]:
                    trim = pos * 0.25
            if trim > 0:
                equity *= (1 - trim * TX_COST)
                pos -= trim
                if pos < 0.01:
                    trade_ret = (price - avg_entry) / avg_entry if avg_entry > 0 else 0
                    if trade_ret > 0: n_wins += 1
                    else: n_losses += 1
                    pos = 0; avg_entry = 0; total_cost = 0
        
        # Entry / pyramid
        if regime.iloc[i] and any_dip.iloc[i]:
            if pos == 0:
                add = p["initial_size"]
                add = min(add, p["max_position"])
                equity *= (1 - add * TX_COST)
                pos = add
                avg_entry = price
                total_cost = price * add
                n_entries += 1
            elif pos < p["max_position"]:
                add = min(p["pyramid_size"], p["max_position"] - pos)
                if add > 0.01:
                    equity *= (1 - add * TX_COST)
                    total_cost += price * add
                    pos += add
                    avg_entry = total_cost / pos
                    n_pyramids += 1
        
        positions[i] = pos
    
    # Fill equities forward from warmup
    for i in range(1, warmup):
        equities[i] = 1.0
    
    eq_series = pd.Series(equities, index=df.index)
    pos_series = pd.Series(positions, index=df.index)
    
    # Metrics
    total_ret = equity - 1
    days = (df.index[-1] - df.index[warmup]).days
    years = max(days / 365.25, 0.1)
    cagr = (equity) ** (1/years) - 1
    
    daily_rets = eq_series.pct_change().fillna(0)
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0
    
    running_max = eq_series.cummax()
    drawdown = eq_series / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    
    time_in_mkt = (pos_series > 0).mean() * 100
    total_trades = n_entries + n_pyramids
    win_rate = n_wins / (n_wins + n_losses) * 100 if (n_wins + n_losses) > 0 else 0
    
    return {
        "sharpe": round(sharpe, 2),
        "total_return": round(total_ret * 100, 1),
        "cagr": round(cagr * 100, 1),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "entries": n_entries,
        "pyramids": n_pyramids,
        "time_in_mkt": round(time_in_mkt, 1),
        "win_rate": round(win_rate, 1),
        "equity": eq_series,
        "daily_returns": daily_rets,
        "positions": pos_series,
    }


def main():
    print("=" * 120)
    print("FILTER EXPLORATION — MacroMomentum V2")
    print("=" * 120)
    
    # Load macro data
    print("\nLoading macro data...")
    fred = MacroDataLoader()
    macro_score = compute_macro_score(fred, start_date="2015-01-01", end_date=END)
    
    m2 = fred.get_series("M2SL", start_date="2015-01-01", end_date=END)
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6).resample("D").ffill().fillna(False)
    
    # Load assets
    print("Loading asset data...")
    loader = StockDataLoader()
    asset_data = {}
    for asset in ASSETS:
        try:
            df = loader.get_ohlcv(asset, "1d", start_date=START, end_date=END)
            if len(df) > 200:
                asset_data[asset] = df
                print(f"  {asset}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            else:
                print(f"  {asset}: insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  {asset}: FAILED — {e}")
    
    # Run all combos
    print("\n" + "=" * 120)
    print("RUNNING 10 FILTER COMBOS × {} ASSETS".format(len(asset_data)))
    print("=" * 120)
    
    results = {}  # {filter_name: {asset: metrics}}
    all_equities = {}
    
    for fname, fcfg in FILTER_COMBOS.items():
        results[fname] = {}
        for asset, df in asset_data.items():
            res = run_strategy(df, m2_acc, macro_score, fcfg)
            if res is not None:
                eq = res.pop("equity")
                dr = res.pop("daily_returns")
                pos = res.pop("positions")
                results[fname][asset] = res
                all_equities[(fname, asset)] = {"equity": eq, "returns": dr, "positions": pos}
            else:
                results[fname][asset] = None
        print(f"  {fname} — done")
    
    # ==================== MAIN COMPARISON TABLE ====================
    print("\n" + "=" * 160)
    print("MAIN COMPARISON TABLE")
    print("=" * 160)
    
    # Header
    asset_short = {a: a.replace("-USD", "") for a in ASSETS}
    header = f"{'Filter':<25}"
    for asset in asset_data:
        s = asset_short[asset]
        header += f" | {s:>5} Shp {s:>5} Ret {s:>5} DD {s:>4} Ent {s:>4} InM"
    print(header)
    print("-" * 160)
    
    for fname in FILTER_COMBOS:
        row = f"{fname:<25}"
        for asset in asset_data:
            m = results[fname].get(asset)
            if m:
                row += f" | {m['sharpe']:>5.2f}   {m['total_return']:>7.1f} {m['max_dd']:>6.1f} {m['entries']:>4}  {m['time_in_mkt']:>5.1f}"
            else:
                row += f" |    --      --     --   --    --"
        print(row)
    
    # ==================== COMPACT SHARPE TABLE ====================
    print("\n" + "=" * 100)
    print("SHARPE RATIO COMPARISON (higher = better risk-adjusted)")
    print("=" * 100)
    
    header = f"{'Filter':<25}"
    for asset in asset_data:
        header += f" {asset_short[asset]:>8}"
    header += f" {'AVG':>8}"
    print(header)
    print("-" * 100)
    
    best_per_asset = {}
    for fname in FILTER_COMBOS:
        row = f"{fname:<25}"
        sharpes = []
        for asset in asset_data:
            m = results[fname].get(asset)
            if m:
                row += f" {m['sharpe']:>8.2f}"
                sharpes.append(m['sharpe'])
            else:
                row += f" {'--':>8}"
        avg = np.mean(sharpes) if sharpes else 0
        row += f" {avg:>8.2f}"
        print(row)
        
        for asset in asset_data:
            m = results[fname].get(asset)
            if m and (asset not in best_per_asset or m['sharpe'] > best_per_asset[asset][1]):
                best_per_asset[asset] = (fname, m['sharpe'])
    
    # ==================== MAX DD TABLE ====================
    print("\n" + "=" * 100)
    print("MAX DRAWDOWN % (closer to 0 = better)")
    print("=" * 100)
    
    header = f"{'Filter':<25}"
    for asset in asset_data:
        header += f" {asset_short[asset]:>8}"
    print(header)
    print("-" * 100)
    
    for fname in FILTER_COMBOS:
        row = f"{fname:<25}"
        for asset in asset_data:
            m = results[fname].get(asset)
            if m:
                row += f" {m['max_dd']:>8.1f}"
            else:
                row += f" {'--':>8}"
        print(row)
    
    # ==================== ENTRIES TABLE ====================
    print("\n" + "=" * 100)
    print("NUMBER OF ENTRIES (trade frequency)")
    print("=" * 100)
    
    header = f"{'Filter':<25}"
    for asset in asset_data:
        header += f" {asset_short[asset]:>8}"
    print(header)
    print("-" * 100)
    
    for fname in FILTER_COMBOS:
        row = f"{fname:<25}"
        for asset in asset_data:
            m = results[fname].get(asset)
            if m:
                row += f" {m['entries']:>8}"
            else:
                row += f" {'--':>8}"
        print(row)
    
    # ==================== OPTIMAL FILTER PER ASSET ====================
    print("\n" + "=" * 80)
    print("OPTIMAL FILTER PER ASSET (by Sharpe)")
    print("=" * 80)
    for asset, (fname, sharpe) in best_per_asset.items():
        m = results[fname][asset]
        print(f"  {asset_short[asset]:>6}: {fname:<25} Sharpe={sharpe:.2f} Ret={m['total_return']:.1f}% DD={m['max_dd']:.1f}% Entries={m['entries']}")
    
    # ==================== FILTER COST ANALYSIS ====================
    print("\n" + "=" * 100)
    print("FILTER COST ANALYSIS (Return sacrificed for DD protection)")
    print("=" * 100)
    
    header = f"{'Filter':<25} {'BTC RetDiff':>12} {'BTC DDDiff':>12} {'Efficiency':>12}"
    print(header)
    print("-" * 100)
    
    no_filter = results.get("1_NoFilter", {})
    for fname in FILTER_COMBOS:
        btc_nf = no_filter.get("BTC-USD")
        btc_f = results[fname].get("BTC-USD")
        if btc_nf and btc_f:
            ret_diff = btc_f['total_return'] - btc_nf['total_return']
            dd_diff = btc_f['max_dd'] - btc_nf['max_dd']  # less negative = better
            # Efficiency: Sharpe per unit of time in market
            eff = btc_f['sharpe'] / (btc_f['time_in_mkt'] / 100) if btc_f['time_in_mkt'] > 0 else 0
            print(f"{fname:<25} {ret_diff:>+12.1f}% {dd_diff:>+12.1f}% {eff:>12.2f}")
    
    # ==================== CORRELATION OF RETURNS ====================
    print("\n" + "=" * 80)
    print("STRATEGY RETURN CORRELATIONS (best filter per asset, diversification potential)")
    print("=" * 80)
    
    # Use filter combo "6_SMA+Mom" as a common one for correlation
    common_filter = "6_SMA+Mom"
    ret_df = pd.DataFrame()
    for asset in asset_data:
        key = (common_filter, asset)
        if key in all_equities:
            ret_df[asset_short[asset]] = all_equities[key]["returns"]
    
    if not ret_df.empty:
        corr = ret_df.corr()
        print(corr.round(2).to_string())
    
    # ==================== YEARLY RETURNS ====================
    print("\n" + "=" * 120)
    print("YEARLY RETURNS — Top 2 Filters vs Buy & Hold per Asset")
    print("=" * 120)
    
    for asset in asset_data:
        df = asset_data[asset]
        # Buy and hold yearly
        bh_yearly = df["close"].resample("YE").last().pct_change().dropna() * 100
        
        # Top 2 by Sharpe
        asset_results = [(fn, results[fn].get(asset)) for fn in FILTER_COMBOS if results[fn].get(asset)]
        asset_results.sort(key=lambda x: x[1]['sharpe'], reverse=True)
        top2 = asset_results[:2]
        
        print(f"\n  {asset_short[asset]}:")
        header = f"    {'Year':>6} {'B&H':>10}"
        for fn, _ in top2:
            header += f" {fn[:20]:>22}"
        print(header)
        
        for fn, _ in top2:
            key = (fn, asset)
            if key in all_equities:
                eq = all_equities[key]["equity"]
                yearly = eq.resample("YE").last().pct_change().dropna() * 100
                # merge
                for year in sorted(set(bh_yearly.index.year) | set(yearly.index.year)):
                    pass  # handled below
        
        # Collect yearly data
        all_yearly = {"B&H": bh_yearly}
        for fn, _ in top2:
            key = (fn, asset)
            if key in all_equities:
                eq = all_equities[key]["equity"]
                yr = eq.resample("YE").last().pct_change().dropna() * 100
                all_yearly[fn[:20]] = yr
        
        years = sorted(set().union(*[set(s.index.year) for s in all_yearly.values()]))
        for year in years:
            row = f"    {year:>6}"
            for label, series in all_yearly.items():
                val = series[series.index.year == year]
                if len(val) > 0:
                    row += f" {val.iloc[0]:>10.1f}%"
                else:
                    row += f" {'--':>10}"
            print(row)
    
    # ==================== PORTFOLIO TEST ====================
    print("\n" + "=" * 100)
    print("PORTFOLIO TEST — Equal-weight top 3 assets with best filter combo")
    print("=" * 100)
    
    # Find best overall filter (avg Sharpe across all assets)
    avg_sharpes = {}
    for fname in FILTER_COMBOS:
        sharpes = [results[fname][a]['sharpe'] for a in asset_data if results[fname].get(a)]
        avg_sharpes[fname] = np.mean(sharpes) if sharpes else -999
    
    best_filter = max(avg_sharpes, key=avg_sharpes.get)
    print(f"Best overall filter: {best_filter} (avg Sharpe = {avg_sharpes[best_filter]:.2f})")
    
    # Top 3 assets by Sharpe with this filter
    asset_sharpes = [(a, results[best_filter][a]['sharpe']) for a in asset_data if results[best_filter].get(a)]
    asset_sharpes.sort(key=lambda x: x[1], reverse=True)
    top3 = [a for a, _ in asset_sharpes[:3]]
    print(f"Top 3 assets: {[asset_short[a] for a in top3]}")
    
    # Equal weight portfolio
    port_rets = pd.DataFrame()
    for asset in top3:
        key = (best_filter, asset)
        if key in all_equities:
            port_rets[asset] = all_equities[key]["returns"]
    
    if not port_rets.empty:
        port_rets = port_rets.fillna(0)
        avg_ret = port_rets.mean(axis=1)
        port_eq = (1 + avg_ret).cumprod()
        
        total_ret = port_eq.iloc[-1] - 1
        days = (port_eq.index[-1] - port_eq.index[0]).days
        years = max(days / 365.25, 0.1)
        cagr = (port_eq.iloc[-1]) ** (1/years) - 1
        ann_vol = avg_ret.std() * np.sqrt(252)
        sharpe = (avg_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
        max_dd = (port_eq / port_eq.cummax() - 1).min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0
        
        print(f"\nEqual-weight portfolio ({', '.join(asset_short[a] for a in top3)}):")
        print(f"  Total Return: {total_ret*100:.1f}%")
        print(f"  CAGR: {cagr*100:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Max DD: {max_dd*100:.1f}%")
        print(f"  Calmar: {calmar:.2f}")
    
    # ==================== WHICH ASSETS NEED MACRO FILTERS? ====================
    print("\n" + "=" * 80)
    print("MACRO FILTER BENEFIT BY ASSET")
    print("=" * 80)
    print(f"{'Asset':>8} {'NoFilter Shp':>14} {'SMA+Mom Shp':>14} {'Full V2 Shp':>14} {'Benefit':>10}")
    print("-" * 80)
    
    for asset in asset_data:
        nf = results["1_NoFilter"].get(asset, {})
        sm = results["6_SMA+Mom"].get(asset, {})
        v2 = results["8_SMA+Mom+M2(V2)"].get(asset, {})
        
        nf_s = nf.get('sharpe', 0) if nf else 0
        sm_s = sm.get('sharpe', 0) if sm else 0
        v2_s = v2.get('sharpe', 0) if v2 else 0
        
        benefit = "YES" if v2_s > nf_s + 0.1 else "MARGINAL" if v2_s > nf_s else "NO"
        print(f"{asset_short[asset]:>8} {nf_s:>14.2f} {sm_s:>14.2f} {v2_s:>14.2f} {benefit:>10}")
    
    # Save all results
    save_results = {}
    for fname in FILTER_COMBOS:
        save_results[fname] = {}
        for asset in asset_data:
            m = results[fname].get(asset)
            if m:
                save_results[fname][asset] = m
    
    with open(RESULTS_DIR / "filter_exploration_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {RESULTS_DIR / 'filter_exploration_results.json'}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Best overall filter: {best_filter}")
    print(f"Avg Sharpe rankings:")
    for fname, avg in sorted(avg_sharpes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fname:<25} {avg:.2f}")


if __name__ == "__main__":
    main()
