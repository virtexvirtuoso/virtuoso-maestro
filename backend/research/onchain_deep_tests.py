"""
On-Chain Deep Test Battery — 6 experiments using REAL on-chain data + full BTC history.
Tests: (1) Longer history, (2) Regime-conditional, (3) Composite scoring,
       (4) Alternative signals, (5) Weekly timeframe, (6) Event-based overlays.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ONCHAIN_DIR = DATA_DIR / "onchain"
OUTPUT_PATH = DATA_DIR / "research" / "onchain_deep_tests.json"

COMMISSION_BPS = 20
N_FOLDS = 10
N_PERMUTATIONS = 200
SQRT_365 = np.sqrt(365)

# ── Data Loading ────────────────────────────────────────────────────────────

def load_real_onchain():
    """Load real on-chain JSON data (SOPR, NUPL, NVT, Reserve Risk)."""
    signals = {}
    for fname, col in [("sopr.json", "sopr"), ("nupl.json", "nupl"),
                        ("nvt.json", "nvt"), ("reserve_risk.json", "reserveRisk")]:
        path = ONCHAIN_DIR / fname
        if not path.exists():
            print(f"  SKIP: {fname} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["d"])
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("date")[[col]].sort_index()
        signals[col.lower()] = df[col]
        print(f"  {col}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    return signals


def load_btc_full_history():
    """Load BTC-USD full history via yfinance (cached in data/stocks/)."""
    from datasource.yfinance_loader import StockDataLoader
    loader = StockDataLoader()
    df = loader.get_ohlcv("BTC-USD", "1d", "2014-01-01")
    df.index = pd.to_datetime(df.index)
    print(f"BTC full history: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    return df


def load_btc_binance():
    """Load BTC from Binance OHLCV (shorter, higher quality)."""
    path = DATA_DIR / "ohlcv" / "binance_btc_usdt_1d.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── Proxy On-Chain Metrics (for signals we don't have real data for) ────────

def compute_proxy_metrics(df):
    """Compute proxy on-chain metrics from price data."""
    price = df["close"]
    volume = df.get("volume", pd.Series(0, index=price.index))
    metrics = pd.DataFrame(index=df.index)

    # MVRV proxy
    days = np.arange(len(price))
    supply = pd.Series(21_000_000 * (1 - np.exp(-days / 1000)), index=price.index).clip(upper=21_000_000)
    market_cap = price * supply
    weighted = price * volume
    total_w = volume.expanding().sum()
    realized_price = weighted.expanding().sum() / total_w.replace(0, np.nan)
    realized_cap = realized_price * supply
    diff = market_cap - realized_cap
    std = market_cap.rolling(365, min_periods=30).std()
    metrics["mvrv_zscore"] = diff / std.replace(0, np.nan)
    metrics["mvrv_ratio"] = market_cap / realized_cap.replace(0, np.nan)

    # SOPR proxy
    cost_basis = price.ewm(span=155, adjust=False).mean()
    metrics["sopr_proxy"] = price / cost_basis.replace(0, np.nan)

    # NUPL proxy
    metrics["nupl_proxy"] = (market_cap - realized_cap) / market_cap.replace(0, np.nan)

    # Puell Multiple proxy
    daily_rev = (price * volume).fillna(0)
    rev_ma = daily_rev.rolling(365, min_periods=30).mean()
    metrics["puell_multiple"] = daily_rev / rev_ma.replace(0, np.nan)

    # Pi Cycle
    sma_111 = price.rolling(111, min_periods=80).mean()
    sma_350x2 = price.rolling(350, min_periods=280).mean() * 2
    metrics["pi_cycle_ratio"] = sma_111 / sma_350x2.replace(0, np.nan)
    metrics["pi_cycle_top"] = (sma_111 > sma_350x2).astype(int)

    # 2-Year MA position
    ma_730 = price.rolling(730, min_periods=365).mean()
    ma_730_upper = ma_730 * 5
    metrics["two_year_ma_pos"] = ((price - ma_730) / (ma_730_upper - ma_730).replace(0, np.nan)).clip(0, 1)

    # STH Realized Price (155-day MA as proxy)
    metrics["sth_realized"] = price.rolling(155, min_periods=50).mean()

    # LTH Realized Price (365-day EWM as proxy)
    metrics["lth_realized"] = price.ewm(span=365, adjust=False).mean()

    # RHODL ratio proxy (short-term vs long-term holder cost basis ratio)
    sth_cost = price.rolling(30, min_periods=10).mean()
    lth_cost = price.ewm(span=365, adjust=False).mean()
    metrics["rhodl_proxy"] = sth_cost / lth_cost.replace(0, np.nan)

    # Reserve Risk proxy
    vol = price.pct_change().rolling(365, min_periods=30).std()
    confidence = (1 / vol.replace(0, np.nan)).clip(lower=0)
    hodl_bank = confidence.cumsum()
    hodl_bank_norm = hodl_bank / hodl_bank.expanding().max()
    rr = price / (hodl_bank_norm * price.rolling(365, min_periods=30).mean()).replace(0, np.nan)
    metrics["reserve_risk_proxy"] = rr / rr.expanding().median().replace(0, np.nan) * 0.005

    # Exchange flow proxy (volume spike = potential exchange flow)
    vol_ma = volume.rolling(30, min_periods=10).mean()
    metrics["exchange_flow_proxy"] = volume / vol_ma.replace(0, np.nan)

    return metrics


# ── Backtesting Engine ──────────────────────────────────────────────────────

def apply_commission(returns, positions, bps=COMMISSION_BPS):
    trades = positions.diff().abs().fillna(0)
    cost = trades * (bps / 10_000)
    return returns - cost


def backtest(price, positions, label=""):
    pos = positions.shift(1).fillna(0)
    daily_ret = price.pct_change().fillna(0)
    strat_ret = daily_ret * pos
    strat_ret = apply_commission(strat_ret, pos)
    cum = (1 + strat_ret).cumprod()
    bnh = (1 + daily_ret).cumprod()
    total = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    bnh_ret = bnh.iloc[-1] - 1 if len(bnh) > 0 else 0
    years = len(strat_ret) / 365
    cagr = (cum.iloc[-1] ** (1 / max(years, 0.1)) - 1) if len(cum) > 0 else 0
    sharpe = strat_ret.mean() / strat_ret.std() * SQRT_365 if strat_ret.std() > 0 else 0
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    trading_days = strat_ret[pos.shift(1).fillna(0) > 0]
    win_rate = (trading_days > 0).mean() if len(trading_days) > 0 else 0
    exposure = (pos > 0).mean()

    yearly = {}
    for year in sorted(strat_ret.index.year.unique()):
        yr = strat_ret[strat_ret.index.year == year]
        yearly[str(year)] = round(float((1 + yr).prod() - 1) * 100, 2)

    return {
        "label": label,
        "total_return_pct": round(float(total) * 100, 2),
        "buy_hold_pct": round(float(bnh_ret) * 100, 2),
        "cagr_pct": round(float(cagr) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "win_rate": round(float(win_rate), 4),
        "exposure_pct": round(float(exposure) * 100, 1),
        "num_days": len(strat_ret),
        "yearly_returns": yearly,
    }


def walk_forward(price, strategy_fn, n_folds=N_FOLDS, **kw):
    n = len(price)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos_results = []
    all_oos_rets = []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        test_price = price.iloc[train_end:test_end]
        if len(test_price) < 10:
            continue
        positions = strategy_fn(train_end, test_end, **kw)
        if positions is None or len(positions) < 10:
            continue
        res = backtest(test_price, positions.reindex(test_price.index).fillna(0), label=f"fold_{fold}")
        res["fold"] = fold
        res["test_start"] = str(test_price.index[0].date())
        res["test_end"] = str(test_price.index[-1].date())
        oos_results.append(res)

        pos = positions.reindex(test_price.index).fillna(0).shift(1).fillna(0)
        sr = test_price.pct_change().fillna(0) * pos
        sr = apply_commission(sr, pos)
        all_oos_rets.append(sr)

    if not oos_results:
        return {"error": "No valid folds", "avg_oos_sharpe": 0, "avg_oos_return_pct": 0}

    sharpes = [r["sharpe"] for r in oos_results]
    n_s = len(sharpes)
    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes, ddof=1) if n_s > 1 else 0
    t_stat = mean_s / (std_s / np.sqrt(n_s)) if std_s > 0 else 0
    p_val = 1 - stats.t.cdf(t_stat, df=n_s - 1) if t_stat > 0 and n_s > 1 else 1.0

    return {
        "folds": oos_results,
        "avg_oos_return_pct": round(float(np.mean([r["total_return_pct"] for r in oos_results])), 2),
        "avg_oos_sharpe": round(float(mean_s), 3),
        "median_oos_sharpe": round(float(np.median(sharpes)), 3),
        "positive_folds": sum(1 for s in sharpes if s > 0),
        "total_folds": n_s,
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
    }


def permutation_test(price, positions, observed_sharpe, n_perms=N_PERMUTATIONS):
    daily_ret = price.pct_change().fillna(0)
    pos = positions.shift(1).fillna(0)
    rng = np.random.default_rng(42)
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = pos.copy()
        shuffled[:] = rng.permutation(pos.values)
        sr = daily_ret * shuffled
        sr = apply_commission(sr, shuffled)
        s = sr.mean() / sr.std() * SQRT_365 if sr.std() > 0 else 0
        perm_sharpes.append(s)
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).mean()
    return {
        "observed_sharpe": round(float(observed_sharpe), 3),
        "p_value": round(float(p_value), 4),
        "significant_5pct": bool(p_value < 0.05),
        "perm_mean": round(float(perm_sharpes.mean()), 3),
        "perm_95th": round(float(np.percentile(perm_sharpes, 95)), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: LONGER HISTORY (Full-cycle spot from 2014+)
# ══════════════════════════════════════════════════════════════════════════════

def test1_longer_history(btc_full, real_onchain):
    print("\n" + "=" * 70)
    print("TEST 1: LONGER HISTORY — Full BTC cycle (2014+) with real on-chain")
    print("=" * 70)

    price = btc_full["close"]
    # Merge real on-chain into a single df
    oc = pd.DataFrame(index=price.index)
    for name, series in real_onchain.items():
        oc[name] = series.reindex(price.index, method="ffill")
    oc = oc.ffill()
    # Also compute proxy metrics for full history
    proxies = compute_proxy_metrics(btc_full)

    results = {}

    # Strategy A: SOPR regime (real data)
    def sopr_regime(pos_series):
        sopr = oc["sopr"].reindex(pos_series.index).ffill()
        # SOPR > 1 = profit-taking, healthy = long
        # SOPR < 0.97 = capitulation = reduce
        position = pd.Series(0.0, index=pos_series.index)
        position[sopr > 1.0] = 1.0
        position[sopr.between(0.97, 1.0)] = 0.5
        position[sopr < 0.97] = 0.0
        return position

    pos_a = sopr_regime(price)
    full_a = backtest(price, pos_a, "SOPR_regime_real")

    def sopr_wf_fn(train_end, test_end):
        return sopr_regime(price.iloc[train_end:test_end])
    wf_a = walk_forward(price, sopr_wf_fn)
    perm_a = permutation_test(price, pos_a, full_a["sharpe"])
    results["sopr_regime_real"] = {"full": full_a, "wf": wf_a, "perm": perm_a}
    print(f"\n  SOPR Regime (real): Full Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")
    print(f"    Perm: p={perm_a['p_value']} {'*' if perm_a['significant_5pct'] else ''}")

    # Strategy B: NUPL regime (real data)
    def nupl_regime(idx):
        nupl = oc["nupl"].reindex(idx).ffill()
        position = pd.Series(0.5, index=idx)
        position[nupl < 0] = 1.0         # capitulation = buy
        position[nupl.between(0, 0.25)] = 0.8  # hope = lean in
        position[nupl.between(0.25, 0.5)] = 0.6  # optimism
        position[nupl.between(0.5, 0.75)] = 0.3  # belief = start reducing
        position[nupl > 0.75] = 0.0       # euphoria = flat
        return position

    pos_b = nupl_regime(price.index)
    full_b = backtest(price, pos_b, "NUPL_regime_real")

    def nupl_wf_fn(train_end, test_end):
        return nupl_regime(price.iloc[train_end:test_end].index)
    wf_b = walk_forward(price, nupl_wf_fn)
    perm_b = permutation_test(price, pos_b, full_b["sharpe"])
    results["nupl_regime_real"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  NUPL Regime (real): Full Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")
    print(f"    Perm: p={perm_b['p_value']} {'*' if perm_b['significant_5pct'] else ''}")

    # Strategy C: NVT signal (real data) — high NVT = overvalued
    def nvt_signal(idx):
        nvt = oc["nvt"].reindex(idx).ffill()
        nvt_ma = nvt.rolling(90, min_periods=30).mean()
        nvt_z = (nvt - nvt_ma) / nvt.rolling(90, min_periods=30).std().replace(0, np.nan)
        position = pd.Series(0.5, index=idx)
        position[nvt_z < -1] = 1.0  # undervalued
        position[nvt_z < 0] = 0.7
        position[nvt_z > 1] = 0.0   # overvalued
        position[nvt_z > 2] = 0.0
        return position.fillna(0.5)

    pos_c = nvt_signal(price.index)
    full_c = backtest(price, pos_c, "NVT_signal_real")

    def nvt_wf_fn(train_end, test_end):
        return nvt_signal(price.iloc[train_end:test_end].index)
    wf_c = walk_forward(price, nvt_wf_fn)
    perm_c = permutation_test(price, pos_c, full_c["sharpe"])
    results["nvt_signal_real"] = {"full": full_c, "wf": wf_c, "perm": perm_c}
    print(f"\n  NVT Signal (real): Full Sharpe={full_c['sharpe']}, CAGR={full_c['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_c['avg_oos_sharpe']}, p={wf_c.get('p_value','N/A')}")
    print(f"    Perm: p={perm_c['p_value']} {'*' if perm_c['significant_5pct'] else ''}")

    # Strategy D: Reserve Risk (real data)
    def rr_signal(idx):
        rr = oc["reserverisk"].reindex(idx).ffill()
        rr_pctile = rr.rolling(365, min_periods=60).rank(pct=True)
        position = pd.Series(0.5, index=idx)
        position[rr_pctile < 0.2] = 1.0   # low RR = accumulate
        position[rr_pctile < 0.4] = 0.7
        position[rr_pctile > 0.8] = 0.0   # high RR = distribute
        position[rr_pctile > 0.6] = 0.3
        return position.fillna(0.5)

    pos_d = rr_signal(price.index)
    full_d = backtest(price, pos_d, "ReserveRisk_real")

    def rr_wf_fn(train_end, test_end):
        return rr_signal(price.iloc[train_end:test_end].index)
    wf_d = walk_forward(price, rr_wf_fn)
    perm_d = permutation_test(price, pos_d, full_d["sharpe"])
    results["reserve_risk_real"] = {"full": full_d, "wf": wf_d, "perm": perm_d}
    print(f"\n  Reserve Risk (real): Full Sharpe={full_d['sharpe']}, CAGR={full_d['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_d['avg_oos_sharpe']}, p={wf_d.get('p_value','N/A')}")
    print(f"    Perm: p={perm_d['p_value']} {'*' if perm_d['significant_5pct'] else ''}")

    # Buy & hold baseline
    bnh = backtest(price, pd.Series(1.0, index=price.index), "BuyHold")
    results["buy_hold"] = {"full": bnh}
    print(f"\n  Buy & Hold: CAGR={bnh['cagr_pct']}%, MaxDD={bnh['max_drawdown_pct']}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: REGIME-CONDITIONAL (on-chain improves V4 only during specific regimes)
# ══════════════════════════════════════════════════════════════════════════════

def test2_regime_conditional(btc, real_onchain, proxies):
    print("\n" + "=" * 70)
    print("TEST 2: REGIME-CONDITIONAL — On-chain as exit-only filter near cycle tops")
    print("=" * 70)

    price = btc["close"]
    oc = pd.DataFrame(index=price.index)
    for name, series in real_onchain.items():
        oc[name] = series.reindex(price.index, method="ffill")
    oc = oc.ffill()

    # Baseline: simple momentum strategy (price > 200-day SMA)
    sma200 = price.rolling(200, min_periods=100).mean()
    momentum_pos = (price > sma200).astype(float)

    results = {}

    # Baseline momentum
    full_base = backtest(price, momentum_pos, "Momentum_200SMA")
    results["baseline_momentum"] = {"full": full_base}
    print(f"\n  Baseline (200 SMA): Sharpe={full_base['sharpe']}, CAGR={full_base['cagr_pct']}%")

    # Strategy A: NUPL exit-only filter — kill position when NUPL > 0.7 (euphoria)
    def nupl_exit_filter(idx):
        base = momentum_pos.reindex(idx).fillna(0)
        nupl = oc["nupl"].reindex(idx).ffill()
        # Only override to flat when NUPL signals extreme
        pos = base.copy()
        pos[nupl > 0.7] = 0.0   # euphoria exit
        pos[nupl > 0.6] = base[nupl > 0.6] * 0.5  # reduce in belief phase
        return pos

    pos_a = nupl_exit_filter(price.index)
    full_a = backtest(price, pos_a, "Momentum+NUPL_exit")

    def nupl_exit_wf(train_end, test_end):
        return nupl_exit_filter(price.iloc[train_end:test_end].index)
    wf_a = walk_forward(price, nupl_exit_wf)
    perm_a = permutation_test(price, pos_a, full_a["sharpe"])
    results["nupl_exit_filter"] = {"full": full_a, "wf": wf_a, "perm": perm_a}
    print(f"\n  NUPL exit-only: Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")
    print(f"    vs Baseline: {full_a['sharpe'] - full_base['sharpe']:+.3f} Sharpe")

    # Strategy B: SOPR exit-only — kill when SOPR drops below 0.95 (capitulation risk)
    def sopr_exit_filter(idx):
        base = momentum_pos.reindex(idx).fillna(0)
        sopr = oc["sopr"].reindex(idx).ffill()
        pos = base.copy()
        pos[sopr < 0.95] = 0.0  # capitulation exit
        return pos

    pos_b = sopr_exit_filter(price.index)
    full_b = backtest(price, pos_b, "Momentum+SOPR_exit")

    def sopr_exit_wf(train_end, test_end):
        return sopr_exit_filter(price.iloc[train_end:test_end].index)
    wf_b = walk_forward(price, sopr_exit_wf)
    perm_b = permutation_test(price, pos_b, full_b["sharpe"])
    results["sopr_exit_filter"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  SOPR exit-only: Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")
    print(f"    vs Baseline: {full_b['sharpe'] - full_base['sharpe']:+.3f} Sharpe")

    # Strategy C: NVT exit — reduce when NVT z-score > 1.5
    def nvt_exit_filter(idx):
        base = momentum_pos.reindex(idx).fillna(0)
        nvt = oc["nvt"].reindex(idx).ffill()
        nvt_z = (nvt - nvt.rolling(180, min_periods=60).mean()) / nvt.rolling(180, min_periods=60).std().replace(0, np.nan)
        pos = base.copy()
        pos[nvt_z > 1.5] = 0.0
        pos[(nvt_z > 1.0) & (nvt_z <= 1.5)] = base[(nvt_z > 1.0) & (nvt_z <= 1.5)] * 0.5
        return pos.fillna(0)

    pos_c = nvt_exit_filter(price.index)
    full_c = backtest(price, pos_c, "Momentum+NVT_exit")

    def nvt_exit_wf(train_end, test_end):
        return nvt_exit_filter(price.iloc[train_end:test_end].index)
    wf_c = walk_forward(price, nvt_exit_wf)
    perm_c = permutation_test(price, pos_c, full_c["sharpe"])
    results["nvt_exit_filter"] = {"full": full_c, "wf": wf_c, "perm": perm_c}
    print(f"\n  NVT exit-only: Sharpe={full_c['sharpe']}, CAGR={full_c['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_c['avg_oos_sharpe']}, p={wf_c.get('p_value','N/A')}")
    print(f"    vs Baseline: {full_c['sharpe'] - full_base['sharpe']:+.3f} Sharpe")

    # Strategy D: Pi Cycle proxy exit (SMA111 > SMA350*2 = cycle top)
    def pi_exit_filter(idx):
        base = momentum_pos.reindex(idx).fillna(0)
        p = price.reindex(idx)
        sma111 = p.rolling(111, min_periods=80).mean()
        sma350x2 = p.rolling(350, min_periods=280).mean() * 2
        pi_dist = (sma350x2 - sma111) / sma350x2.replace(0, np.nan)
        pos = base.copy()
        pos[pi_dist < 0] = 0.0      # crossed = flat
        pos[pi_dist < 0.05] = base[pi_dist < 0.05] * 0.3  # close to cross = reduce
        return pos.fillna(0)

    pos_d = pi_exit_filter(price.index)
    full_d = backtest(price, pos_d, "Momentum+PiCycle_exit")

    def pi_exit_wf(train_end, test_end):
        return pi_exit_filter(price.iloc[train_end:test_end].index)
    wf_d = walk_forward(price, pi_exit_wf)
    perm_d = permutation_test(price, pos_d, full_d["sharpe"])
    results["pi_cycle_exit"] = {"full": full_d, "wf": wf_d, "perm": perm_d}
    print(f"\n  Pi Cycle exit: Sharpe={full_d['sharpe']}, CAGR={full_d['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_d['avg_oos_sharpe']}, p={wf_d.get('p_value','N/A')}")
    print(f"    vs Baseline: {full_d['sharpe'] - full_base['sharpe']:+.3f} Sharpe")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: COMPOSITE SCORING — Combine multiple on-chain into single score
# ══════════════════════════════════════════════════════════════════════════════

def test3_composite_scoring(btc, real_onchain, proxies):
    print("\n" + "=" * 70)
    print("TEST 3: COMPOSITE SCORING — Multi-signal on-chain score (like V3 confluence)")
    print("=" * 70)

    price = btc["close"]
    oc = pd.DataFrame(index=price.index)
    for name, series in real_onchain.items():
        oc[name] = series.reindex(price.index, method="ffill")
    oc = oc.ffill()
    px = proxies.reindex(price.index)

    results = {}

    # Score 0-5 from binary signals (mirroring V3 confluence pattern)
    def onchain_confluence(idx):
        score = pd.Series(0, index=idx, dtype=float)

        # S1: SOPR healthy (> 1.0)
        sopr = oc["sopr"].reindex(idx).ffill()
        score += (sopr > 1.0).astype(float)

        # S2: NUPL not euphoric (< 0.7)
        nupl = oc["nupl"].reindex(idx).ffill()
        score += (nupl < 0.7).astype(float)

        # S3: NVT not overvalued (z-score < 1.0)
        nvt = oc["nvt"].reindex(idx).ffill()
        nvt_z = (nvt - nvt.rolling(180, min_periods=60).mean()) / nvt.rolling(180, min_periods=60).std().replace(0, np.nan)
        score += (nvt_z < 1.0).fillna(True).astype(float)

        # S4: Reserve Risk low (< 50th percentile)
        rr = oc["reserverisk"].reindex(idx).ffill()
        rr_pctile = rr.rolling(365, min_periods=60).rank(pct=True)
        score += (rr_pctile < 0.5).fillna(True).astype(float)

        # S5: Price above STH realized (proxy: 155-day MA)
        sth = px["sth_realized"].reindex(idx).ffill()
        p = price.reindex(idx)
        score += (p > sth).astype(float)

        return score

    def composite_strategy(idx, score_to_pos_map=None):
        score = onchain_confluence(idx)
        if score_to_pos_map is None:
            score_to_pos_map = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.7, 4: 1.0, 5: 1.0}
        pos = score.map(lambda s: score_to_pos_map.get(min(int(s), 5), 0.5))
        return pos

    # Strategy A: Binary composite (0-5 → position)
    pos_a = composite_strategy(price.index)
    full_a = backtest(price, pos_a, "OnChain_Confluence_0to5")

    def comp_wf(train_end, test_end):
        return composite_strategy(price.iloc[train_end:test_end].index)
    wf_a = walk_forward(price, comp_wf)
    perm_a = permutation_test(price, pos_a, full_a["sharpe"])
    results["confluence_5signal"] = {"full": full_a, "wf": wf_a, "perm": perm_a}

    score_dist = onchain_confluence(price.index).value_counts().sort_index()
    print(f"\n  Confluence Score Distribution: {score_dist.to_dict()}")
    print(f"  Confluence (0-5): Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")
    print(f"    Perm: p={perm_a['p_value']} {'*' if perm_a['significant_5pct'] else ''}")

    # Strategy B: Continuous composite (weighted average of z-scored signals)
    def continuous_composite(idx):
        sopr = oc["sopr"].reindex(idx).ffill()
        nupl = oc["nupl"].reindex(idx).ffill()
        nvt = oc["nvt"].reindex(idx).ffill()
        rr = oc["reserverisk"].reindex(idx).ffill()

        # Normalize each to z-score
        def zscore(s, w=365):
            return (s - s.rolling(w, min_periods=60).mean()) / s.rolling(w, min_periods=60).std().replace(0, np.nan)

        # SOPR: higher is bullish → positive z = bullish
        z_sopr = zscore(sopr)
        # NUPL: lower is bullish → negate
        z_nupl = -zscore(nupl)
        # NVT: lower is bullish → negate
        z_nvt = -zscore(nvt)
        # Reserve Risk: lower is bullish → negate
        z_rr = -zscore(rr)

        composite = (z_sopr.fillna(0) + z_nupl.fillna(0) + z_nvt.fillna(0) + z_rr.fillna(0)) / 4
        # Map to position: clip to [-2, 2], scale to [0, 1]
        pos = ((composite + 2) / 4).clip(0, 1)
        return pos

    pos_b = continuous_composite(price.index)
    full_b = backtest(price, pos_b, "OnChain_Continuous_Composite")

    def cont_wf(train_end, test_end):
        return continuous_composite(price.iloc[train_end:test_end].index)
    wf_b = walk_forward(price, cont_wf)
    perm_b = permutation_test(price, pos_b, full_b["sharpe"])
    results["continuous_composite"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  Continuous Composite: Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")
    print(f"    Perm: p={perm_b['p_value']} {'*' if perm_b['significant_5pct'] else ''}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: DIFFERENT SIGNALS — RHODL, SOPR, Puell, Exchange Flows, STH/LTH RP
# ══════════════════════════════════════════════════════════════════════════════

def test4_alternative_signals(btc, proxies):
    print("\n" + "=" * 70)
    print("TEST 4: ALTERNATIVE SIGNALS — Less popular on-chain metrics")
    print("=" * 70)

    price = btc["close"]
    px = proxies.reindex(price.index)

    results = {}

    # Strategy A: RHODL Ratio — high = new money entering, low = old hands accumulating
    def rhodl_strategy(idx):
        rhodl = px["rhodl_proxy"].reindex(idx).ffill()
        rhodl_ma = rhodl.rolling(30, min_periods=10).mean()
        rhodl_slow = rhodl.rolling(90, min_periods=30).mean()
        # Rising RHODL (new money > old money) = bullish early, bearish late
        # Use trend: buy when RHODL starts rising from low, sell when peaks
        rhodl_pctile = rhodl.rolling(365, min_periods=60).rank(pct=True)
        pos = pd.Series(0.5, index=idx)
        pos[rhodl_pctile < 0.3] = 1.0   # old hands accumulating
        pos[rhodl_pctile > 0.8] = 0.0   # new money frenzy
        pos[(rhodl_ma < rhodl_slow) & (rhodl_pctile < 0.5)] = 0.8  # declining = accumulation
        return pos.fillna(0.5)

    pos_a = rhodl_strategy(price.index)
    full_a = backtest(price, pos_a, "RHODL_proxy")

    def rhodl_wf(train_end, test_end):
        return rhodl_strategy(price.iloc[train_end:test_end].index)
    wf_a = walk_forward(price, rhodl_wf)
    perm_a = permutation_test(price, pos_a, full_a["sharpe"])
    results["rhodl"] = {"full": full_a, "wf": wf_a, "perm": perm_a}
    print(f"\n  RHODL Ratio: Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")

    # Strategy B: Puell Multiple — high = miners selling, low = miner capitulation (buy)
    def puell_strategy(idx):
        puell = px["puell_multiple"].reindex(idx).ffill()
        pos = pd.Series(0.5, index=idx)
        pos[puell < 0.5] = 1.0   # miner capitulation = buy
        pos[puell < 0.8] = 0.7
        pos[puell > 4.0] = 0.0   # miners dumping = sell
        pos[puell > 2.0] = 0.3
        return pos.fillna(0.5)

    pos_b = puell_strategy(price.index)
    full_b = backtest(price, pos_b, "Puell_Multiple")

    def puell_wf(train_end, test_end):
        return puell_strategy(price.iloc[train_end:test_end].index)
    wf_b = walk_forward(price, puell_wf)
    perm_b = permutation_test(price, pos_b, full_b["sharpe"])
    results["puell"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  Puell Multiple: Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")

    # Strategy C: Exchange Flow — volume spike = potential outflow/inflow
    def exchange_flow_strategy(idx):
        flow = px["exchange_flow_proxy"].reindex(idx).ffill()
        flow_ma = flow.rolling(7, min_periods=3).mean()
        p = price.reindex(idx)
        ret = p.pct_change(5)  # 5-day return
        # High volume + price up = exchange outflow (bullish, accumulation)
        # High volume + price down = exchange inflow (bearish, distribution)
        pos = pd.Series(0.5, index=idx)
        pos[(flow_ma > 1.5) & (ret > 0)] = 1.0  # volume spike + up = accumulation
        pos[(flow_ma > 1.5) & (ret < 0)] = 0.0  # volume spike + down = distribution
        return pos.fillna(0.5)

    pos_c = exchange_flow_strategy(price.index)
    full_c = backtest(price, pos_c, "ExchangeFlow_proxy")

    def flow_wf(train_end, test_end):
        return exchange_flow_strategy(price.iloc[train_end:test_end].index)
    wf_c = walk_forward(price, flow_wf)
    perm_c = permutation_test(price, pos_c, full_c["sharpe"])
    results["exchange_flow"] = {"full": full_c, "wf": wf_c, "perm": perm_c}
    print(f"\n  Exchange Flow: Sharpe={full_c['sharpe']}, CAGR={full_c['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_c['avg_oos_sharpe']}, p={wf_c.get('p_value','N/A')}")

    # Strategy D: STH/LTH Realized Price — price vs short-term and long-term holder cost
    def sth_lth_strategy(idx):
        p = price.reindex(idx)
        sth = px["sth_realized"].reindex(idx).ffill()
        lth = px["lth_realized"].reindex(idx).ffill()
        pos = pd.Series(0.5, index=idx)
        # Bullish: price above both STH and LTH realized
        pos[(p > sth) & (p > lth)] = 1.0
        # Bearish: price below both
        pos[(p < sth) & (p < lth)] = 0.0
        # Mixed: price between STH and LTH
        pos[(p > lth) & (p < sth)] = 0.3  # below STH = underwater short-term holders
        return pos.fillna(0.5)

    pos_d = sth_lth_strategy(price.index)
    full_d = backtest(price, pos_d, "STH_LTH_Realized")

    def sth_lth_wf(train_end, test_end):
        return sth_lth_strategy(price.iloc[train_end:test_end].index)
    wf_d = walk_forward(price, sth_lth_wf)
    perm_d = permutation_test(price, pos_d, full_d["sharpe"])
    results["sth_lth"] = {"full": full_d, "wf": wf_d, "perm": perm_d}
    print(f"\n  STH/LTH Realized: Sharpe={full_d['sharpe']}, CAGR={full_d['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_d['avg_oos_sharpe']}, p={wf_d.get('p_value','N/A')}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: NON-DAILY TIMEFRAME — Weekly on-chain signals
# ══════════════════════════════════════════════════════════════════════════════

def test5_weekly_signals(btc, real_onchain, proxies):
    print("\n" + "=" * 70)
    print("TEST 5: WEEKLY TIMEFRAME — On-chain signals resampled to weekly")
    print("=" * 70)

    price = btc["close"]
    # Resample everything to weekly (Friday close)
    price_w = price.resample("W-FRI").last().dropna()

    oc_w = pd.DataFrame(index=price_w.index)
    for name, series in real_onchain.items():
        oc_w[name] = series.resample("W-FRI").last().reindex(price_w.index, method="ffill")
    oc_w = oc_w.ffill()

    results = {}

    # Strategy A: Weekly SOPR regime
    def weekly_sopr(idx):
        sopr = oc_w["sopr"].reindex(idx).ffill()
        sopr_4w = sopr.rolling(4, min_periods=2).mean()
        pos = pd.Series(0.5, index=idx)
        pos[sopr_4w > 1.02] = 1.0
        pos[sopr_4w.between(1.0, 1.02)] = 0.7
        pos[sopr_4w < 0.98] = 0.0
        return pos.fillna(0.5)

    pos_a = weekly_sopr(price_w.index)
    full_a = backtest(price_w, pos_a, "Weekly_SOPR")

    def w_sopr_wf(train_end, test_end):
        return weekly_sopr(price_w.iloc[train_end:test_end].index)
    wf_a = walk_forward(price_w, w_sopr_wf, n_folds=8)
    perm_a = permutation_test(price_w, pos_a, full_a["sharpe"])
    results["weekly_sopr"] = {"full": full_a, "wf": wf_a, "perm": perm_a}
    print(f"\n  Weekly SOPR: Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")

    # Strategy B: Weekly NUPL regime
    def weekly_nupl(idx):
        nupl = oc_w["nupl"].reindex(idx).ffill()
        nupl_4w = nupl.rolling(4, min_periods=2).mean()
        pos = pd.Series(0.5, index=idx)
        pos[nupl_4w < 0] = 1.0
        pos[nupl_4w.between(0, 0.25)] = 0.8
        pos[nupl_4w.between(0.5, 0.7)] = 0.3
        pos[nupl_4w > 0.7] = 0.0
        return pos.fillna(0.5)

    pos_b = weekly_nupl(price_w.index)
    full_b = backtest(price_w, pos_b, "Weekly_NUPL")

    def w_nupl_wf(train_end, test_end):
        return weekly_nupl(price_w.iloc[train_end:test_end].index)
    wf_b = walk_forward(price_w, w_nupl_wf, n_folds=8)
    perm_b = permutation_test(price_w, pos_b, full_b["sharpe"])
    results["weekly_nupl"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  Weekly NUPL: Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")

    # Strategy C: Weekly multi-signal composite
    def weekly_composite(idx):
        sopr = oc_w["sopr"].reindex(idx).ffill()
        nupl = oc_w["nupl"].reindex(idx).ffill()
        nvt = oc_w["nvt"].reindex(idx).ffill()
        rr = oc_w["reserverisk"].reindex(idx).ffill()

        # 4-week smoothed
        sopr_s = sopr.rolling(4, min_periods=2).mean()
        nupl_s = nupl.rolling(4, min_periods=2).mean()

        score = pd.Series(0.0, index=idx)
        score += (sopr_s > 1.0).astype(float)
        score += (nupl_s < 0.5).astype(float)
        score += (nupl_s < 0).astype(float)  # extra point for deep value

        nvt_z = (nvt - nvt.rolling(52, min_periods=20).mean()) / nvt.rolling(52, min_periods=20).std().replace(0, np.nan)
        score += (nvt_z < 1.0).fillna(True).astype(float)

        rr_pctile = rr.rolling(52, min_periods=20).rank(pct=True)
        score += (rr_pctile < 0.5).fillna(True).astype(float)

        # Map 0-5 to position
        pos_map = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.7, 4: 1.0, 5: 1.0}
        pos = score.map(lambda s: pos_map.get(min(int(s), 5), 0.5))
        return pos

    pos_c = weekly_composite(price_w.index)
    full_c = backtest(price_w, pos_c, "Weekly_Composite")

    def w_comp_wf(train_end, test_end):
        return weekly_composite(price_w.iloc[train_end:test_end].index)
    wf_c = walk_forward(price_w, w_comp_wf, n_folds=8)
    perm_c = permutation_test(price_w, pos_c, full_c["sharpe"])
    results["weekly_composite"] = {"full": full_c, "wf": wf_c, "perm": perm_c}
    print(f"\n  Weekly Composite: Sharpe={full_c['sharpe']}, CAGR={full_c['cagr_pct']}%")
    print(f"    WF: Sharpe={wf_c['avg_oos_sharpe']}, p={wf_c.get('p_value','N/A')}")

    # Buy & Hold weekly
    bnh = backtest(price_w, pd.Series(1.0, index=price_w.index), "BuyHold_weekly")
    results["buy_hold_weekly"] = {"full": bnh}
    print(f"\n  Buy & Hold (weekly): CAGR={bnh['cagr_pct']}%, MaxDD={bnh['max_drawdown_pct']}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: EVENT-BASED — Extreme readings as risk-on/risk-off overlays
# ══════════════════════════════════════════════════════════════════════════════

def test6_event_based(btc, real_onchain, proxies):
    print("\n" + "=" * 70)
    print("TEST 6: EVENT-BASED — Extreme readings as risk-on/risk-off overlays")
    print("=" * 70)

    price = btc["close"]
    oc = pd.DataFrame(index=price.index)
    for name, series in real_onchain.items():
        oc[name] = series.reindex(price.index, method="ffill")
    oc = oc.ffill()

    # Baseline: always in (100% long)
    baseline_pos = pd.Series(1.0, index=price.index)
    base_result = backtest(price, baseline_pos, "Always_In")

    results = {}
    results["baseline_always_in"] = {"full": base_result}
    print(f"\n  Baseline (always in): Sharpe={base_result['sharpe']}, CAGR={base_result['cagr_pct']}%")

    # Strategy A: SOPR extreme events — go flat for 14 days after extreme capitulation (<0.9)
    #             then re-enter after 14-day cooldown
    def sopr_extreme_events(idx):
        sopr = oc["sopr"].reindex(idx).ffill()
        pos = pd.Series(1.0, index=idx)
        cooldown = 0
        for i in range(len(idx)):
            s = sopr.iloc[i]
            if pd.isna(s):
                continue
            if s < 0.90:  # extreme capitulation — risk off
                cooldown = 14
            if cooldown > 0:
                pos.iloc[i] = 0.0
                cooldown -= 1
        return pos

    pos_a = sopr_extreme_events(price.index)
    full_a = backtest(price, pos_a, "SOPR_extreme_events")
    events_triggered = (pos_a == 0).sum()

    def sopr_ev_wf(train_end, test_end):
        return sopr_extreme_events(price.iloc[train_end:test_end].index)
    wf_a = walk_forward(price, sopr_ev_wf)
    perm_a = permutation_test(price, pos_a, full_a["sharpe"])
    results["sopr_extreme"] = {"full": full_a, "wf": wf_a, "perm": perm_a}
    print(f"\n  SOPR Extreme Events: Sharpe={full_a['sharpe']}, CAGR={full_a['cagr_pct']}%")
    print(f"    Risk-off days: {events_triggered} ({events_triggered/len(price)*100:.1f}%)")
    print(f"    WF: Sharpe={wf_a['avg_oos_sharpe']}, p={wf_a.get('p_value','N/A')}")
    print(f"    vs Always-In: {full_a['sharpe'] - base_result['sharpe']:+.3f} Sharpe")

    # Strategy B: NUPL extreme events — go flat for 21 days when NUPL > 0.75 (euphoria)
    def nupl_extreme_events(idx):
        nupl = oc["nupl"].reindex(idx).ffill()
        pos = pd.Series(1.0, index=idx)
        cooldown = 0
        for i in range(len(idx)):
            n = nupl.iloc[i]
            if pd.isna(n):
                continue
            if n > 0.75:  # euphoria — risk off
                cooldown = 21
            if cooldown > 0:
                pos.iloc[i] = 0.0
                cooldown -= 1
        return pos

    pos_b = nupl_extreme_events(price.index)
    full_b = backtest(price, pos_b, "NUPL_extreme_events")
    events_b = (pos_b == 0).sum()

    def nupl_ev_wf(train_end, test_end):
        return nupl_extreme_events(price.iloc[train_end:test_end].index)
    wf_b = walk_forward(price, nupl_ev_wf)
    perm_b = permutation_test(price, pos_b, full_b["sharpe"])
    results["nupl_extreme"] = {"full": full_b, "wf": wf_b, "perm": perm_b}
    print(f"\n  NUPL Extreme Events: Sharpe={full_b['sharpe']}, CAGR={full_b['cagr_pct']}%")
    print(f"    Risk-off days: {events_b} ({events_b/len(price)*100:.1f}%)")
    print(f"    WF: Sharpe={wf_b['avg_oos_sharpe']}, p={wf_b.get('p_value','N/A')}")
    print(f"    vs Always-In: {full_b['sharpe'] - base_result['sharpe']:+.3f} Sharpe")

    # Strategy C: NVT spike events — reduce when NVT z-score > 2 for 30 days
    def nvt_spike_events(idx):
        nvt = oc["nvt"].reindex(idx).ffill()
        nvt_z = (nvt - nvt.rolling(180, min_periods=60).mean()) / nvt.rolling(180, min_periods=60).std().replace(0, np.nan)
        pos = pd.Series(1.0, index=idx)
        cooldown = 0
        for i in range(len(idx)):
            z = nvt_z.iloc[i]
            if pd.isna(z):
                continue
            if z > 2.0:  # extreme NVT = overvalued
                cooldown = 30
            if cooldown > 0:
                pos.iloc[i] = 0.3
                cooldown -= 1
        return pos

    pos_c = nvt_spike_events(price.index)
    full_c = backtest(price, pos_c, "NVT_spike_events")
    events_c = (pos_c < 1.0).sum()

    def nvt_ev_wf(train_end, test_end):
        return nvt_spike_events(price.iloc[train_end:test_end].index)
    wf_c = walk_forward(price, nvt_ev_wf)
    perm_c = permutation_test(price, pos_c, full_c["sharpe"])
    results["nvt_spike"] = {"full": full_c, "wf": wf_c, "perm": perm_c}
    print(f"\n  NVT Spike Events: Sharpe={full_c['sharpe']}, CAGR={full_c['cagr_pct']}%")
    print(f"    Reduced days: {events_c} ({events_c/len(price)*100:.1f}%)")
    print(f"    WF: Sharpe={wf_c['avg_oos_sharpe']}, p={wf_c.get('p_value','N/A')}")
    print(f"    vs Always-In: {full_c['sharpe'] - base_result['sharpe']:+.3f} Sharpe")

    # Strategy D: Multi-signal risk-off — go flat when ANY 2+ signals are extreme
    def multi_extreme_events(idx):
        sopr = oc["sopr"].reindex(idx).ffill()
        nupl = oc["nupl"].reindex(idx).ffill()
        nvt = oc["nvt"].reindex(idx).ffill()
        nvt_z = (nvt - nvt.rolling(180, min_periods=60).mean()) / nvt.rolling(180, min_periods=60).std().replace(0, np.nan)
        rr = oc["reserverisk"].reindex(idx).ffill()
        rr_pctile = rr.rolling(365, min_periods=60).rank(pct=True)

        pos = pd.Series(1.0, index=idx)
        cooldown = 0
        for i in range(len(idx)):
            extremes = 0
            # Bearish extremes (cycle top signals)
            if not pd.isna(nupl.iloc[i]) and nupl.iloc[i] > 0.7:
                extremes += 1
            if not pd.isna(nvt_z.iloc[i]) and nvt_z.iloc[i] > 1.5:
                extremes += 1
            if not pd.isna(rr_pctile.iloc[i]) and rr_pctile.iloc[i] > 0.8:
                extremes += 1
            if not pd.isna(sopr.iloc[i]) and sopr.iloc[i] < 0.92:
                extremes += 1  # capitulation

            if extremes >= 2:
                cooldown = 21

            if cooldown > 0:
                pos.iloc[i] = 0.0
                cooldown -= 1
        return pos

    pos_d = multi_extreme_events(price.index)
    full_d = backtest(price, pos_d, "Multi_extreme_events")
    events_d = (pos_d == 0).sum()

    def multi_ev_wf(train_end, test_end):
        return multi_extreme_events(price.iloc[train_end:test_end].index)
    wf_d = walk_forward(price, multi_ev_wf)
    perm_d = permutation_test(price, pos_d, full_d["sharpe"])
    results["multi_extreme"] = {"full": full_d, "wf": wf_d, "perm": perm_d}
    print(f"\n  Multi-Signal Extreme: Sharpe={full_d['sharpe']}, CAGR={full_d['cagr_pct']}%")
    print(f"    Risk-off days: {events_d} ({events_d/len(price)*100:.1f}%)")
    print(f"    WF: Sharpe={wf_d['avg_oos_sharpe']}, p={wf_d.get('p_value','N/A')}")
    print(f"    vs Always-In: {full_d['sharpe'] - base_result['sharpe']:+.3f} Sharpe")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ON-CHAIN DEEP TEST BATTERY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    real_onchain = load_real_onchain()
    btc_full = load_btc_full_history()
    proxies = compute_proxy_metrics(btc_full)

    # Determine overlap
    oc_start = max(s.index[0] for s in real_onchain.values())
    btc_start = btc_full.index[0]
    overlap_start = max(oc_start, btc_start)
    print(f"\nData overlap: {overlap_start.date()} to {btc_full.index[-1].date()}")
    print(f"Full history: {len(btc_full)} days, On-chain overlap: {len(btc_full[btc_full.index >= overlap_start])} days")

    # Trim to overlap
    btc_overlap = btc_full[btc_full.index >= overlap_start].copy()

    all_results = {}

    # Run all 6 tests
    all_results["test1_longer_history"] = test1_longer_history(btc_overlap, real_onchain)
    all_results["test2_regime_conditional"] = test2_regime_conditional(btc_overlap, real_onchain, proxies)
    all_results["test3_composite_scoring"] = test3_composite_scoring(btc_overlap, real_onchain, proxies)
    all_results["test4_alternative_signals"] = test4_alternative_signals(btc_overlap, proxies)
    all_results["test5_weekly_signals"] = test5_weekly_signals(btc_overlap, real_onchain, proxies)
    all_results["test6_event_based"] = test6_event_based(btc_overlap, real_onchain, proxies)

    # ── Executive Summary ──
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    print(f"\n{'Test':<12} {'Strategy':<30} {'Full Sharpe':>11} {'WF Sharpe':>10} {'WF p-val':>9} {'CAGR':>8} {'MaxDD':>8}")
    print("-" * 95)

    for test_name, test_results in all_results.items():
        test_num = test_name.split("_")[0].replace("test", "T")
        for strat_name, strat_data in test_results.items():
            full = strat_data.get("full", {})
            wf = strat_data.get("wf", {})
            sharpe = full.get("sharpe", "N/A")
            wf_sharpe = wf.get("avg_oos_sharpe", "N/A") if wf else "N/A"
            wf_p = wf.get("p_value", "N/A") if wf else "N/A"
            cagr = full.get("cagr_pct", "N/A")
            maxdd = full.get("max_drawdown_pct", "N/A")

            sharpe_str = f"{sharpe:>11.3f}" if isinstance(sharpe, (int, float)) else f"{sharpe:>11}"
            wf_str = f"{wf_sharpe:>10.3f}" if isinstance(wf_sharpe, (int, float)) else f"{wf_sharpe:>10}"
            p_str = f"{wf_p:>9.4f}" if isinstance(wf_p, (int, float)) else f"{wf_p:>9}"
            cagr_str = f"{cagr:>7.1f}%" if isinstance(cagr, (int, float)) else f"{cagr:>8}"
            dd_str = f"{maxdd:>7.1f}%" if isinstance(maxdd, (int, float)) else f"{maxdd:>8}"

            print(f"{test_num:<12} {strat_name:<30} {sharpe_str} {wf_str} {p_str} {cagr_str} {dd_str}")

    # Find best performers (WF Sharpe > 0 and p < 0.1)
    print("\n" + "-" * 70)
    print("TOP PERFORMERS (WF p < 0.10):")
    print("-" * 70)
    candidates = []
    for test_name, test_results in all_results.items():
        for strat_name, strat_data in test_results.items():
            wf = strat_data.get("wf", {})
            if wf and isinstance(wf.get("p_value"), (int, float)) and wf["p_value"] < 0.10:
                candidates.append((strat_name, strat_data["full"]["sharpe"],
                                   wf["avg_oos_sharpe"], wf["p_value"],
                                   strat_data["full"]["cagr_pct"]))

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        for name, fs, ws, p, cagr in candidates:
            sig = "**" if p < 0.05 else "*"
            print(f"  {sig} {name:<30} Full={fs:.3f}  WF={ws:.3f}  p={p:.4f}  CAGR={cagr:.1f}%")
    else:
        print("  No strategies survived WF validation at p < 0.10")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "btc_data_range": f"{btc_overlap.index[0].date()} to {btc_overlap.index[-1].date()}",
            "num_days": len(btc_overlap),
            "real_onchain_signals": list(real_onchain.keys()),
            "commission_bps": COMMISSION_BPS,
            "walk_forward_folds": N_FOLDS,
            "permutation_tests": N_PERMUTATIONS,
        },
        "results": all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
