"""
Comprehensive Signal Hunt -- Short, Alt, and Cross-Asset Strategies
===================================================================
Tests EVERY plausible signal using derivatives data from DuckDB.
Prevents lookahead with shift(1), includes 10bps TX costs.
Outputs results to data/backtest_results/comprehensive_signal_hunt.json

Run: python scripts/comprehensive_signal_hunt.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
TX_COST = 0.001  # 10 bps per trade (round trip = 20bps)
ANNUAL_FACTOR = np.sqrt(365)  # crypto trades 365 days
MIN_TRADES = 30  # minimum trades for a signal to be considered
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "maestro.duckdb"
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "backtest_results"
    / "comprehensive_signal_hunt.json"
)

# Symbols with full derivatives coverage (price + LSR + funding + liq + taker)
FULL_SYMBOLS = [
    "BTC", "ETH", "SOL", "LINK", "AVAX", "DOGE", "DOT",
    "NEAR", "ATOM", "BNB", "XRP", "UNI", "ADA", "FIL",
]

# Top-tier alts for focused testing
TOP_ALTS = ["ETH", "SOL", "LINK", "AVAX", "DOGE", "BNB", "XRP"]


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_all_data(con: duckdb.DuckDBPyConnection) -> dict:
    """Load and merge all data for each symbol into a dict of DataFrames."""
    data = {}

    for sym in FULL_SYMBOLS:
        # Price
        price = con.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM perps_daily WHERE symbol=? ORDER BY date",
            [sym],
        ).fetchdf()
        if price.empty:
            continue
        price["date"] = pd.to_datetime(price["date"])
        price = price.set_index("date").sort_index()

        # LSR
        lsr = con.execute(
            "SELECT date, global_account_long_short_ratio as lsr, "
            "global_account_long_percent as long_pct "
            "FROM cg_lsr_global WHERE symbol=? ORDER BY date",
            [sym],
        ).fetchdf()
        lsr["date"] = pd.to_datetime(lsr["date"])
        lsr = lsr.set_index("date").sort_index()
        # Deduplicate LSR (some symbols have multiple readings per day)
        lsr = lsr.groupby(level=0).last()

        # Funding rate
        fr = con.execute(
            "SELECT date, close as funding_rate "
            "FROM cg_funding_rate WHERE symbol=? ORDER BY date",
            [sym],
        ).fetchdf()
        fr["date"] = pd.to_datetime(fr["date"])
        fr = fr.set_index("date").sort_index()
        fr = fr.groupby(level=0).last()

        # Liquidations
        liq = con.execute(
            "SELECT date, aggregated_long_liquidation_usd as long_liq, "
            "aggregated_short_liquidation_usd as short_liq "
            "FROM cg_liquidations WHERE symbol=? ORDER BY date",
            [sym],
        ).fetchdf()
        liq["date"] = pd.to_datetime(liq["date"])
        liq = liq.set_index("date").sort_index()
        liq = liq.groupby(level=0).last()

        # Taker volume
        tv = con.execute(
            "SELECT date, taker_buy_volume_usd as taker_buy, "
            "taker_sell_volume_usd as taker_sell "
            "FROM cg_taker_volume WHERE symbol=? ORDER BY date",
            [sym],
        ).fetchdf()
        tv["date"] = pd.to_datetime(tv["date"])
        tv = tv.set_index("date").sort_index()
        tv = tv.groupby(level=0).last()

        # Merge all
        df = price.copy()
        df = df.join(lsr, how="left")
        df = df.join(fr, how="left")
        df = df.join(liq, how="left")
        df = df.join(tv, how="left")

        # Derived columns
        total_taker = df["taker_buy"] + df["taker_sell"]
        df["taker_ratio"] = df["taker_buy"] / total_taker.replace(0, np.nan)
        df["total_liq"] = df["long_liq"].fillna(0) + df["short_liq"].fillna(0)
        df["liq_ratio"] = df["long_liq"] / df["total_liq"].replace(0, np.nan)

        # Forward returns (next-day close-to-close) -- this IS what we're predicting
        df["fwd_ret"] = df["close"].pct_change().shift(-1)

        # Daily returns (for position entry cost calc)
        df["ret"] = df["close"].pct_change()

        data[sym] = df

    return data


# ── Metrics Calculator ───────────────────────────────────────────────────────
def compute_metrics(
    returns: pd.Series,
    signal: pd.Series,
    direction: str = "long",
    label: str = "",
) -> dict:
    """
    Compute strategy metrics from a signal and returns series.

    signal: 1 = position ON, 0 = position OFF (for long)
            -1 = short, 0 = flat (for short)
    returns: forward returns (already shifted so no lookahead)
    direction: 'long', 'short', or 'longshort'
    """
    # Align
    aligned = pd.concat([signal, returns], axis=1).dropna()
    if len(aligned) < 50:
        return {"label": label, "error": "insufficient_data", "n_obs": len(aligned)}

    aligned.columns = ["signal", "fwd_ret"]

    # Strategy returns with transaction costs
    position = aligned["signal"]
    position_changes = position.diff().abs().fillna(0)
    # Each position change incurs TX_COST
    strat_ret = position * aligned["fwd_ret"] - position_changes * TX_COST

    # Only count non-zero position days
    active_mask = position != 0
    active_returns = strat_ret[active_mask]

    n_trades = int(position_changes[position_changes > 0].count())
    exposure = float(active_mask.mean())

    if n_trades < MIN_TRADES:
        return {
            "label": label,
            "direction": direction,
            "error": "too_few_trades",
            "n_trades": n_trades,
            "exposure": round(exposure, 4),
        }

    # Cumulative returns
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = float(cum_ret.iloc[-1] - 1)
    n_years = len(strat_ret) / 365
    if n_years > 0 and (1 + total_ret) > 0:
        cagr = float((1 + total_ret) ** (1 / n_years) - 1)
    else:
        cagr = -1.0  # total loss

    # Sharpe (annualized, 365-day)
    mean_daily = float(strat_ret.mean())
    std_daily = float(strat_ret.std())
    sharpe = (mean_daily / std_daily * ANNUAL_FACTOR) if std_daily > 0 else 0.0

    # Max drawdown
    peak = cum_ret.expanding().max()
    dd = (cum_ret - peak) / peak
    max_dd = float(dd.min())

    # Win rate
    win_rate = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

    # Avg return when signal ON vs OFF
    ret_on = float(aligned.loc[active_mask, "fwd_ret"].mean()) if active_mask.any() else 0.0
    ret_off = (
        float(aligned.loc[~active_mask, "fwd_ret"].mean())
        if (~active_mask).any()
        else 0.0
    )

    # Buy-and-hold benchmark
    bh_ret = aligned["fwd_ret"]
    bh_cum = (1 + bh_ret).cumprod()
    bh_sharpe = (
        float(bh_ret.mean() / bh_ret.std() * ANNUAL_FACTOR) if bh_ret.std() > 0 else 0.0
    )

    # Profit factor
    gross_profit = float(active_returns[active_returns > 0].sum())
    gross_loss = float(active_returns[active_returns < 0].sum())
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf

    # Calmar ratio
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    # Flag suspiciously good results
    too_good = sharpe > 2.0 or cagr > 1.5
    flag = "SUSPICIOUS - verify not overfit" if too_good else ""
    worth_wf = sharpe > 0.5 and n_trades >= MIN_TRADES

    return {
        "label": label,
        "direction": direction,
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "exposure": round(exposure, 4),
        "n_trades": n_trades,
        "n_days": len(strat_ret),
        "avg_ret_on": round(ret_on * 100, 4),  # in bps
        "avg_ret_off": round(ret_off * 100, 4),
        "profit_factor": round(min(profit_factor, 99.9), 3),
        "calmar": round(calmar, 3),
        "bh_sharpe": round(bh_sharpe, 3),
        "worth_wf_validation": worth_wf,
        "flag": flag,
    }


# ── SECTION 1: SHORT SIGNALS ────────────────────────────────────────────────
def test_short_signals(data: dict) -> list:
    """Test all short signal hypotheses on BTC and top alts."""
    results = []

    for sym in ["BTC"] + TOP_ALTS:
        if sym not in data:
            continue
        df = data[sym]
        fwd = df["fwd_ret"]

        print(f"\n  Testing SHORT signals on {sym}...")

        # ── 1a. High Funding Rate Short ──
        # When funding is high (crowded longs), short
        for threshold_name, threshold in [
            ("p75", df["funding_rate"].quantile(0.75)),
            ("p80", df["funding_rate"].quantile(0.80)),
            ("p90", df["funding_rate"].quantile(0.90)),
            ("abs_0.05pct", 0.0005),  # 0.05% per 8h
            ("abs_0.03pct", 0.0003),  # 0.03% per 8h
        ]:
            # shift(1): use yesterday's funding rate to decide today's position
            signal = (df["funding_rate"].shift(1) > threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_high_funding_{threshold_name}",
            )
            results.append(metrics)

        # ── 1b. LSR Extreme Short ──
        # When too many longs (LSR high), short
        for pct in [0.75, 0.80, 0.85, 0.90]:
            threshold = df["lsr"].quantile(pct)
            signal = (df["lsr"].shift(1) > threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_lsr_extreme_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 1c. Liquidation Cascade Short ──
        # When long liquidations spike, short the follow-through
        for pct in [0.85, 0.90, 0.95]:
            threshold = df["long_liq"].quantile(pct)
            signal = (df["long_liq"].shift(1) > threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_long_liq_cascade_p{int(pct*100)}",
            )
            results.append(metrics)

        # Also test: short when liq ratio (long/total) is extreme (longs getting rekt)
        for pct in [0.80, 0.90]:
            threshold = df["liq_ratio"].quantile(pct)
            signal = (df["liq_ratio"].shift(1) > threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_liq_ratio_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 1d. Taker Sell Dominance Short ──
        # When sellers dominating (taker_ratio < threshold), short
        for threshold in [0.48, 0.47, 0.45]:
            signal = (df["taker_ratio"].shift(1) < threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_taker_sell_dom_{threshold}",
            )
            results.append(metrics)

        # Percentile-based taker ratio
        for pct in [0.15, 0.20, 0.25]:
            threshold = df["taker_ratio"].quantile(pct)
            signal = (df["taker_ratio"].shift(1) < threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_taker_ratio_below_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 1e. Regime OFF Short ──
        # When our weighted regime score is bearish, short instead of flat
        regime_score = compute_regime_score(df)
        for thresh in [0.25, 0.30, 0.35]:
            signal = (regime_score.shift(1) < thresh).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_regime_off_{thresh}",
            )
            results.append(metrics)

        # ── 1f. Combined Short: 2+ signals agree ──
        # Build individual binary short signals (all shifted)
        funding_short = (
            df["funding_rate"].shift(1) > df["funding_rate"].quantile(0.80)
        ).astype(int)
        lsr_short = (df["lsr"].shift(1) > df["lsr"].quantile(0.80)).astype(int)
        liq_short = (
            df["long_liq"].shift(1) > df["long_liq"].quantile(0.90)
        ).astype(int)
        taker_short = (
            df["taker_ratio"].shift(1) < df["taker_ratio"].quantile(0.20)
        ).astype(int)

        combined_count = funding_short + lsr_short + liq_short + taker_short

        for min_agree in [2, 3]:
            signal = (combined_count >= min_agree).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_combined_{min_agree}of4",
            )
            results.append(metrics)

        # ── 1g. Momentum-based Short ──
        # When 20d returns are strongly negative, short (momentum following)
        mom_20 = df["close"].pct_change(20)
        for pct in [0.10, 0.20]:
            threshold = mom_20.quantile(pct)
            signal = (mom_20.shift(1) < threshold).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_neg_momentum_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 1h. Contrarian Long after Short Signals ──
        # Flip: when everyone is bearish, go LONG (contrarian)
        # Low funding + high short liquidations = bearish sentiment exhaustion
        low_funding = (
            df["funding_rate"].shift(1) < df["funding_rate"].quantile(0.10)
        ).astype(int)
        short_liq_spike = (
            df["short_liq"].shift(1) > df["short_liq"].quantile(0.90)
        ).astype(int)

        signal = ((low_funding + short_liq_spike) >= 2).astype(int)
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_contrarian_bearish_exhaustion",
        )
        results.append(metrics)

        # Low funding contrarian long
        signal = low_funding
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_contrarian_low_funding",
        )
        results.append(metrics)

    return results


def compute_regime_score(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Compute the weighted regime score matching our proven BTC formula.
    LSR x 0.35 + Funding x 0.35 + Liq x 0.15 + Taker x 0.15
    Each component is a rolling percentile rank (0-1).
    """
    # Rolling percentile rank for each component
    lsr_rank = df["lsr"].rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    fr_rank = df["funding_rate"].rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    # For liquidations, we want the liq_ratio (proportion of longs being liquidated)
    # LOWER liq ratio = fewer longs liquidated = bullish
    # So we INVERT: high rank = low liq ratio = bullish
    liq_rank = (
        1
        - df["liq_ratio"].rolling(window, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    )
    taker_rank = df["taker_ratio"].rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    score = lsr_rank * 0.35 + fr_rank * 0.35 + liq_rank * 0.15 + taker_rank * 0.15
    return score


# ── SECTION 2: ALT-SPECIFIC SIGNALS ─────────────────────────────────────────
def test_alt_signals(data: dict) -> list:
    """Test alt-specific regime and relative signals."""
    results = []
    btc = data.get("BTC")
    if btc is None:
        return results

    for sym in TOP_ALTS:
        if sym not in data:
            continue
        df = data[sym]
        fwd = df["fwd_ret"]

        print(f"\n  Testing ALT signals on {sym}...")

        # ── 2a. Alt's Own Regime Signal ──
        # Apply the same weighted regime formula using that alt's OWN data
        regime_score = compute_regime_score(df)
        for thresh in [0.40, 0.50, 0.60]:
            signal = (regime_score.shift(1) > thresh).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_own_regime_{thresh}",
            )
            results.append(metrics)

        # ── 2b. BTC-Relative Momentum ──
        # When alt outperforms BTC over N days, go long alt
        btc_close = btc["close"].reindex(df.index)
        for lookback in [10, 20, 40]:
            alt_mom = df["close"].pct_change(lookback)
            btc_mom = btc_close.pct_change(lookback)
            rel_strength = alt_mom - btc_mom

            signal = (rel_strength.shift(1) > 0).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_btc_relative_mom_{lookback}d",
            )
            results.append(metrics)

            # Also test: only go long when BOTH alt AND BTC momentum positive
            signal = ((rel_strength.shift(1) > 0) & (btc_mom.shift(1) > 0)).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_rel_mom_plus_btc_bull_{lookback}d",
            )
            results.append(metrics)

        # ── 2c. Alt Funding Divergence ──
        # When alt funding << BTC funding, capital rotation to alt
        btc_fr = btc["funding_rate"].reindex(df.index)
        funding_spread = df["funding_rate"] - btc_fr

        for pct in [0.15, 0.25]:
            threshold = funding_spread.quantile(pct)
            signal = (funding_spread.shift(1) < threshold).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_funding_divergence_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 2d. Alt Liquidation Recovery ──
        # After a big liquidation cascade in the alt, go long (bounce)
        for delay in [1, 2, 3]:
            for pct in [0.90, 0.95]:
                threshold = df["long_liq"].quantile(pct)
                # Signal fires N days AFTER the cascade
                liq_spike = (df["long_liq"] > threshold).astype(int)
                signal = liq_spike.shift(delay)
                signal = signal.fillna(0).astype(int)
                metrics = compute_metrics(
                    fwd, signal, "long",
                    f"{sym}_LONG_liq_recovery_delay{delay}_p{int(pct*100)}",
                )
                results.append(metrics)

        # ── 2e. Alt LSR Contrarian ──
        # When alt LSR is extremely low (too many shorts), go long
        for pct in [0.10, 0.20]:
            threshold = df["lsr"].quantile(pct)
            signal = (df["lsr"].shift(1) < threshold).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_lsr_contrarian_below_p{int(pct*100)}",
            )
            results.append(metrics)

        # ── 2f. BTC Regime Filter for Alt ──
        # Only trade alt when BTC regime is ON
        btc_regime = compute_regime_score(btc).reindex(df.index)
        alt_regime = compute_regime_score(df)

        # Alt regime + BTC regime both bullish
        signal = ((alt_regime.shift(1) > 0.5) & (btc_regime.shift(1) > 0.5)).astype(int)
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_alt_AND_btc_regime",
        )
        results.append(metrics)

        # Alt regime only (no BTC filter)
        signal = (alt_regime.shift(1) > 0.5).astype(int)
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_alt_regime_only",
        )
        results.append(metrics)

    return results


# ── SECTION 3: CROSS-ASSET / PORTFOLIO STRATEGIES ───────────────────────────
def test_cross_asset_signals(data: dict) -> list:
    """Test cross-asset, rotation, and relative value strategies."""
    results = []

    # Build a multi-asset close price panel
    closes = pd.DataFrame({sym: data[sym]["close"] for sym in FULL_SYMBOLS if sym in data})
    returns = closes.pct_change()
    fwd_returns = returns.shift(-1)  # next-day returns

    # Build derivative panels
    funding_panel = pd.DataFrame(
        {sym: data[sym]["funding_rate"] for sym in FULL_SYMBOLS if sym in data}
    )
    lsr_panel = pd.DataFrame(
        {sym: data[sym]["lsr"] for sym in FULL_SYMBOLS if sym in data}
    )

    print("\n  Testing CROSS-ASSET signals...")

    # ── 3a. Funding Rate Carry ──
    # Long lowest-funding, short highest-funding (market neutral)
    for n_long, n_short in [(3, 3), (5, 5), (3, 0)]:
        label = f"CARRY_long{n_long}_short{n_short}"

        # Rank assets by funding rate each day
        fr_rank = funding_panel.shift(1).rank(axis=1)
        n_assets = fr_rank.count(axis=1)

        # Equal weight: +1/n_long for lowest, -1/n_short for highest
        strat_ret = pd.Series(0.0, index=closes.index)
        prev_positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        for sym in closes.columns:
            long_signal = (fr_rank[sym] <= n_long).astype(int) if n_long > 0 else 0
            short_signal = (
                (fr_rank[sym] > (n_assets - n_short)).astype(int) if n_short > 0 else 0
            )

            position = long_signal / max(n_long, 1) - short_signal / max(n_short, 1)
            pos_change = position.diff().abs().fillna(0)

            asset_ret = fwd_returns.get(sym, pd.Series(0.0, index=closes.index))
            strat_ret += position * asset_ret - pos_change * TX_COST

        # Compute metrics manually for portfolio strategies
        metrics = compute_portfolio_metrics(strat_ret, label, "longshort" if n_short > 0 else "long")
        results.append(metrics)

    # ── 3b. LSR Divergence Pairs ──
    # When one asset LSR is extreme high and another extreme low, pair trade
    lsr_ranked = lsr_panel.shift(1).rank(axis=1, pct=True)

    strat_ret = pd.Series(0.0, index=closes.index)
    n_pairs = 0
    for i, sym_long in enumerate(closes.columns):
        for sym_short in list(closes.columns)[i + 1 :]:
            # Long the low-LSR asset, short the high-LSR asset
            # Only when divergence is extreme
            lsr_diff = lsr_ranked.get(sym_long, 0) - lsr_ranked.get(sym_short, 0)
            pair_signal = (lsr_diff < -0.5).astype(int)  # sym_long has much lower LSR rank

            pair_ret = (
                pair_signal * fwd_returns.get(sym_long, 0)
                - pair_signal * fwd_returns.get(sym_short, 0)
            )
            strat_ret += pair_ret
            n_pairs += 1

    if n_pairs > 0:
        strat_ret = strat_ret / max(n_pairs, 1)  # normalize
    metrics = compute_portfolio_metrics(strat_ret, "LSR_divergence_pairs", "longshort")
    results.append(metrics)

    # ── 3c. Momentum Rotation ──
    # Each day, long top N by 20d returns, flat/short bottom N
    for lookback in [10, 20, 40]:
        for n_top in [3, 5]:
            for go_short in [False, True]:
                label = f"MOM_ROTATION_{lookback}d_top{n_top}{'_with_short' if go_short else ''}"

                mom = returns.rolling(lookback).sum().shift(1)
                mom_rank = mom.rank(axis=1, ascending=False)
                n_assets_available = mom.count(axis=1)

                strat_ret = pd.Series(0.0, index=closes.index)

                for sym in closes.columns:
                    long_sig = (mom_rank[sym] <= n_top).astype(int) / n_top
                    if go_short:
                        short_sig = (
                            (mom_rank[sym] > (n_assets_available - n_top)).astype(int) / n_top
                        )
                    else:
                        short_sig = 0

                    position = long_sig - short_sig
                    pos_change = position.diff().abs().fillna(0)
                    asset_ret = fwd_returns.get(sym, pd.Series(0.0, index=closes.index))
                    strat_ret += position * asset_ret - pos_change * TX_COST

                metrics = compute_portfolio_metrics(strat_ret, label, "longshort" if go_short else "long")
                results.append(metrics)

    # ── 3d. Regime Rotation ──
    # Only trade assets where their individual regime score > threshold
    regime_scores = {}
    for sym in FULL_SYMBOLS:
        if sym in data:
            regime_scores[sym] = compute_regime_score(data[sym])

    for thresh in [0.45, 0.50, 0.55]:
        label = f"REGIME_ROTATION_thresh_{thresh}"

        strat_ret = pd.Series(0.0, index=closes.index)

        for sym in closes.columns:
            if sym not in regime_scores:
                continue
            # Equal weight across all "on" assets
            regime_on = (regime_scores[sym].shift(1) > thresh).astype(int)
            position = regime_on  # will normalize below
            pos_change = position.diff().abs().fillna(0)
            asset_ret = fwd_returns.get(sym, pd.Series(0.0, index=closes.index))
            strat_ret += position * asset_ret - pos_change * TX_COST

        # Count how many assets are "on" each day for normalization
        active_count = pd.DataFrame(
            {
                sym: (regime_scores[sym].shift(1) > thresh).astype(int)
                for sym in regime_scores
            }
        ).sum(axis=1).replace(0, np.nan)

        strat_ret = strat_ret / active_count

        metrics = compute_portfolio_metrics(strat_ret, label, "long")
        results.append(metrics)

    # ── 3e. Volatility-Weighted Regime Rotation ──
    # Weight positions by inverse vol (risk parity within regime-on assets)
    vol_20 = returns.rolling(20).std().shift(1)

    for thresh in [0.50]:
        label = f"REGIME_ROTATION_vol_weighted_{thresh}"
        strat_ret = pd.Series(0.0, index=closes.index)
        total_inv_vol = pd.Series(0.0, index=closes.index)

        inv_vol_positions = {}
        for sym in closes.columns:
            if sym not in regime_scores:
                continue
            regime_on = (regime_scores[sym].shift(1) > thresh).astype(int)
            inv_vol = (1.0 / vol_20[sym]).replace([np.inf, -np.inf], np.nan).fillna(0)
            inv_vol_positions[sym] = regime_on * inv_vol
            total_inv_vol += inv_vol_positions[sym]

        total_inv_vol = total_inv_vol.replace(0, np.nan)

        for sym in inv_vol_positions:
            weight = inv_vol_positions[sym] / total_inv_vol
            weight = weight.fillna(0)
            pos_change = weight.diff().abs().fillna(0)
            asset_ret = fwd_returns.get(sym, pd.Series(0.0, index=closes.index))
            strat_ret += weight * asset_ret - pos_change * TX_COST

        metrics = compute_portfolio_metrics(strat_ret, label, "long")
        results.append(metrics)

    # ── 3f. BTC-Beta Hedge ──
    # Long alts, short BTC as beta hedge
    # Idea: capture alt alpha while hedging systematic risk
    btc_ret = fwd_returns.get("BTC", pd.Series(0.0, index=closes.index))

    for lookback in [20]:
        for n_top in [3, 5]:
            label = f"ALT_ALPHA_btc_hedge_top{n_top}"

            # Pick top N alts by momentum, short BTC as hedge
            alt_mom = returns.drop(columns=["BTC"], errors="ignore").rolling(lookback).sum().shift(1)
            alt_rank = alt_mom.rank(axis=1, ascending=False)

            strat_ret = pd.Series(0.0, index=closes.index)

            for sym in alt_mom.columns:
                long_sig = (alt_rank[sym] <= n_top).astype(int) / n_top
                pos_change = long_sig.diff().abs().fillna(0)
                asset_ret = fwd_returns.get(sym, pd.Series(0.0, index=closes.index))
                strat_ret += long_sig * asset_ret - pos_change * TX_COST

            # Short BTC with equal notional
            strat_ret -= btc_ret

            metrics = compute_portfolio_metrics(strat_ret, label, "longshort")
            results.append(metrics)

    return results


# ── SECTION 4: COMBINED SYSTEMS ──────────────────────────────────────────────
def test_combined_systems(data: dict) -> list:
    """Test combined long + short and multi-asset systems."""
    results = []
    btc = data.get("BTC")
    if btc is None:
        return results

    btc_fwd = btc["fwd_ret"]
    btc_regime = compute_regime_score(btc)

    print("\n  Testing COMBINED systems...")

    # ── 4a. BTC Long + Short Portfolio ──
    # BTC regime long PLUS short signal to reduce drawdowns
    # The idea: when regime is ON, we're long. When OFF, we try to be SHORT.

    # Best short candidates (will pick top performers from section 1)
    # For now, test the combined_2of4 short signal
    funding_short = (
        btc["funding_rate"].shift(1) > btc["funding_rate"].quantile(0.80)
    ).astype(int)
    lsr_short = (btc["lsr"].shift(1) > btc["lsr"].quantile(0.80)).astype(int)
    liq_short = (
        btc["long_liq"].shift(1) > btc["long_liq"].quantile(0.90)
    ).astype(int)
    taker_short = (
        btc["taker_ratio"].shift(1) < btc["taker_ratio"].quantile(0.20)
    ).astype(int)
    combined_short = (funding_short + lsr_short + liq_short + taker_short) >= 2

    for long_thresh, short_thresh in [
        (0.50, 0.35),
        (0.50, 0.30),
        (0.55, 0.30),
    ]:
        # Position: +1 when regime ON, -1 when regime OFF AND short signal
        long_signal = (btc_regime.shift(1) > long_thresh).astype(int)
        short_signal = (
            (btc_regime.shift(1) < short_thresh) & combined_short.shift(0)
        ).astype(int)

        position = long_signal - short_signal
        pos_change = position.diff().abs().fillna(0)
        strat_ret = position * btc_fwd - pos_change * TX_COST

        label = f"BTC_LONG_SHORT_regime_{long_thresh}_{short_thresh}"
        metrics = compute_portfolio_metrics(strat_ret, label, "longshort")
        results.append(metrics)

    # ── 4b. BTC Regime + Short (just the short complement) ──
    # When BTC regime is OFF, short if any 2 derivative signals agree
    for min_agree in [2, 3]:
        combined_count = funding_short + lsr_short + liq_short + taker_short
        position = pd.Series(0, index=btc.index, dtype=int)
        position[btc_regime.shift(1) > 0.5] = 1  # long when regime ON
        position[(btc_regime.shift(1) <= 0.35) & (combined_count >= min_agree)] = -1  # short when OFF + signals

        pos_change = position.diff().abs().fillna(0)
        strat_ret = position * btc_fwd - pos_change * TX_COST

        label = f"BTC_REGIME_LONG_plus_SHORT_{min_agree}of4"
        metrics = compute_portfolio_metrics(strat_ret, label, "longshort")
        results.append(metrics)

    # ── 4c. BTC Long + Alt Rotation During Regime ON ──
    # When BTC regime ON: 50% BTC + 50% top-N momentum alts
    # When regime OFF: flat
    for n_top in [3, 5]:
        label = f"BTC_PLUS_ALT_ROTATION_top{n_top}"

        alt_closes = pd.DataFrame(
            {sym: data[sym]["close"] for sym in TOP_ALTS if sym in data}
        )
        alt_returns = alt_closes.pct_change()
        alt_fwd = alt_returns.shift(-1)

        alt_mom = alt_returns.rolling(20).sum().shift(1)
        alt_rank = alt_mom.rank(axis=1, ascending=False)

        regime_on = (btc_regime.shift(1) > 0.5).astype(int)

        strat_ret = pd.Series(0.0, index=btc.index)

        # 50% in BTC
        btc_pos = regime_on * 0.5
        btc_pos_change = btc_pos.diff().abs().fillna(0)
        strat_ret += btc_pos * btc_fwd - btc_pos_change * TX_COST

        # 50% split among top-N alts
        for sym in alt_mom.columns:
            alt_top = (alt_rank[sym] <= n_top).astype(int) / n_top * 0.5
            alt_pos = regime_on * alt_top
            alt_pos = alt_pos.reindex(btc.index).fillna(0)
            alt_pos_change = alt_pos.diff().abs().fillna(0)
            sym_fwd = alt_fwd.get(sym, pd.Series(0.0, index=alt_fwd.index))
            sym_fwd = sym_fwd.reindex(btc.index).fillna(0)
            strat_ret += alt_pos * sym_fwd - alt_pos_change * TX_COST

        metrics = compute_portfolio_metrics(strat_ret, label, "long")
        results.append(metrics)

    # ── 4d. All-Weather: BTC regime long + Alt regime rotation + Short hedge ──
    label = "ALL_WEATHER_btc_alt_short"

    regime_scores = {}
    for sym in FULL_SYMBOLS:
        if sym in data:
            regime_scores[sym] = compute_regime_score(data[sym])

    strat_ret = pd.Series(0.0, index=btc.index)

    # Component 1: BTC regime long (40% weight)
    btc_on = (btc_regime.shift(1) > 0.5).astype(int)
    btc_pos = btc_on * 0.4
    btc_pos_change = btc_pos.diff().abs().fillna(0)
    strat_ret += btc_pos * btc_fwd - btc_pos_change * TX_COST

    # Component 2: Alt regime rotation (40% weight, equal among ON alts)
    alt_on_count = pd.Series(0.0, index=btc.index)
    alt_positions = {}
    for sym in TOP_ALTS:
        if sym in regime_scores:
            on = (regime_scores[sym].shift(1) > 0.5).astype(int).reindex(btc.index).fillna(0)
            alt_on_count += on
            alt_positions[sym] = on

    alt_on_count = alt_on_count.replace(0, np.nan)

    for sym in alt_positions:
        weight = (alt_positions[sym] / alt_on_count * 0.4).fillna(0)
        pos_change = weight.diff().abs().fillna(0)
        sym_fwd = data[sym]["fwd_ret"].reindex(btc.index).fillna(0)
        strat_ret += weight * sym_fwd - pos_change * TX_COST

    # Component 3: BTC short when regime OFF + 2-of-4 derivatives short (20% weight)
    combined_count = funding_short + lsr_short + liq_short + taker_short
    short_on = ((btc_regime.shift(1) < 0.35) & (combined_count >= 2)).astype(int)
    short_pos = short_on * -0.2
    short_pos_change = short_pos.diff().abs().fillna(0)
    strat_ret += short_pos * btc_fwd - short_pos_change * TX_COST

    metrics = compute_portfolio_metrics(strat_ret, label, "longshort")
    results.append(metrics)

    # ── 4e. Benchmark: BTC Regime Long Only (for comparison) ──
    signal = (btc_regime.shift(1) > 0.5).astype(int)
    metrics = compute_metrics(btc_fwd, signal, "long", "BENCHMARK_BTC_regime_long_only")
    results.append(metrics)

    # ── 4f. Benchmark: BTC Buy and Hold ──
    signal = pd.Series(1, index=btc.index)
    metrics = compute_metrics(btc_fwd, signal, "long", "BENCHMARK_BTC_buy_hold")
    results.append(metrics)

    return results


# ── SECTION 5: ADDITIONAL EXPLORATIONS ───────────────────────────────────────
def test_additional_signals(data: dict) -> list:
    """Test additional signal ideas: mean reversion, vol breakout, etc."""
    results = []
    btc = data.get("BTC")
    if btc is None:
        return results

    print("\n  Testing ADDITIONAL signals...")

    for sym in ["BTC"] + TOP_ALTS:
        if sym not in data:
            continue
        df = data[sym]
        fwd = df["fwd_ret"]

        # ── 5a. Funding Rate Mean Reversion ──
        # When funding spikes high, it tends to revert. Short the spike.
        fr_zscore = (
            (df["funding_rate"] - df["funding_rate"].rolling(30).mean())
            / df["funding_rate"].rolling(30).std()
        )
        for z_thresh in [1.5, 2.0, 2.5]:
            signal = (fr_zscore.shift(1) > z_thresh).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_funding_zscore_{z_thresh}",
            )
            results.append(metrics)

            # Contrarian: extreme negative z-score -> go long
            signal = (fr_zscore.shift(1) < -z_thresh).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_funding_neg_zscore_{z_thresh}",
            )
            results.append(metrics)

        # ── 5b. Vol Breakout ──
        # When realized vol spikes above its moving average, trade the direction
        vol_20 = df["ret"].rolling(20).std()
        vol_ratio = vol_20 / vol_20.rolling(60).mean()

        # High vol + negative returns = short (vol expansion in downtrend)
        mom_5 = df["close"].pct_change(5)
        signal = ((vol_ratio.shift(1) > 1.5) & (mom_5.shift(1) < 0)).astype(int) * -1
        metrics = compute_metrics(
            fwd, signal, "short",
            f"{sym}_SHORT_vol_breakout_downtrend",
        )
        results.append(metrics)

        # High vol + positive returns = long (vol expansion in uptrend)
        signal = ((vol_ratio.shift(1) > 1.5) & (mom_5.shift(1) > 0)).astype(int)
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_vol_breakout_uptrend",
        )
        results.append(metrics)

        # ── 5c. LSR Mean Reversion ──
        lsr_zscore = (
            (df["lsr"] - df["lsr"].rolling(30).mean()) / df["lsr"].rolling(30).std()
        )
        for z_thresh in [1.5, 2.0]:
            # Too many longs -> short
            signal = (lsr_zscore.shift(1) > z_thresh).astype(int) * -1
            metrics = compute_metrics(
                fwd, signal, "short",
                f"{sym}_SHORT_lsr_zscore_{z_thresh}",
            )
            results.append(metrics)

            # Too many shorts (low LSR) -> long (contrarian)
            signal = (lsr_zscore.shift(1) < -z_thresh).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_lsr_contrarian_zscore_{z_thresh}",
            )
            results.append(metrics)

        # ── 5d. Cross-Derivative Divergence ──
        # When funding is high but LSR is dropping = smart money exiting
        # (divergence between retail sentiment and cost of carry)
        fr_rank = df["funding_rate"].rolling(30).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        lsr_change_5d = df["lsr"].pct_change(5)

        signal = (
            (fr_rank.shift(1) > 0.8) & (lsr_change_5d.shift(1) < -0.1)
        ).astype(int) * -1
        metrics = compute_metrics(
            fwd, signal, "short",
            f"{sym}_SHORT_funding_lsr_divergence",
        )
        results.append(metrics)

        # ── 5e. Taker Volume Trend ──
        # When taker ratio is trending down over 5d (persistent selling)
        taker_ma5 = df["taker_ratio"].rolling(5).mean()
        taker_ma20 = df["taker_ratio"].rolling(20).mean()

        signal = (taker_ma5.shift(1) < taker_ma20.shift(1)).astype(int) * -1
        metrics = compute_metrics(
            fwd, signal, "short",
            f"{sym}_SHORT_taker_trend_down",
        )
        results.append(metrics)

        signal = (taker_ma5.shift(1) > taker_ma20.shift(1)).astype(int)
        metrics = compute_metrics(
            fwd, signal, "long",
            f"{sym}_LONG_taker_trend_up",
        )
        results.append(metrics)

        # ── 5f. Liquidation Asymmetry ──
        # When short liquidations >> long liquidations, shorts are getting squeezed -> long
        liq_asym = df["short_liq"] / (df["long_liq"] + 1)  # +1 to avoid div/0
        for pct in [0.85, 0.90]:
            threshold = liq_asym.quantile(pct)
            signal = (liq_asym.shift(1) > threshold).astype(int)
            metrics = compute_metrics(
                fwd, signal, "long",
                f"{sym}_LONG_short_squeeze_p{int(pct*100)}",
            )
            results.append(metrics)

    return results


def compute_portfolio_metrics(
    strat_ret: pd.Series, label: str, direction: str
) -> dict:
    """Compute metrics directly from strategy returns (for portfolio strategies)."""
    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 100:
        return {"label": label, "error": "insufficient_data", "n_obs": len(strat_ret)}

    cum_ret = (1 + strat_ret).cumprod()
    total_ret = float(cum_ret.iloc[-1] - 1)
    n_years = len(strat_ret) / 365
    if n_years > 0 and (1 + total_ret) > 0:
        cagr = float((1 + total_ret) ** (1 / n_years) - 1)
    else:
        cagr = -1.0

    mean_daily = float(strat_ret.mean())
    std_daily = float(strat_ret.std())
    sharpe = (mean_daily / std_daily * ANNUAL_FACTOR) if std_daily > 0 else 0.0

    peak = cum_ret.expanding().max()
    dd = (cum_ret - peak) / peak
    max_dd = float(dd.min())

    win_rate = float((strat_ret[strat_ret != 0] > 0).mean()) if (strat_ret != 0).any() else 0.0
    exposure = float((strat_ret != 0).mean())

    n_trades = int(strat_ret.diff().abs().gt(0.001).sum())  # rough trade count

    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    # Profit factor
    gross_profit = float(strat_ret[strat_ret > 0].sum())
    gross_loss = float(strat_ret[strat_ret < 0].sum())
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf

    too_good = sharpe > 2.0 or cagr > 1.5
    flag = "SUSPICIOUS - verify not overfit" if too_good else ""
    worth_wf = sharpe > 0.5 and n_trades >= MIN_TRADES

    return {
        "label": label,
        "direction": direction,
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "exposure": round(exposure, 4),
        "n_trades": n_trades,
        "n_days": len(strat_ret),
        "profit_factor": round(min(profit_factor, 99.9), 3),
        "calmar": round(calmar, 3),
        "worth_wf_validation": worth_wf,
        "flag": flag,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("COMPREHENSIVE SIGNAL HUNT")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"TX Cost: {TX_COST*100:.1f} bps per trade")
    print(f"Annualization: sqrt({365})")
    print(f"Min trades: {MIN_TRADES}")
    print("=" * 80)

    # Connect and load
    con = duckdb.connect(str(DB_PATH), read_only=True)
    print("\nLoading data from DuckDB...")
    data = load_all_data(con)
    con.close()

    print(f"Loaded {len(data)} symbols: {list(data.keys())}")
    for sym in data:
        n = len(data[sym])
        print(f"  {sym}: {n} days ({data[sym].index[0].date()} to {data[sym].index[-1].date()})")

    all_results = []

    # Section 1: Short signals
    print("\n" + "=" * 60)
    print("SECTION 1: SHORT SIGNALS")
    print("=" * 60)
    short_results = test_short_signals(data)
    all_results.extend(short_results)

    # Section 2: Alt-specific signals
    print("\n" + "=" * 60)
    print("SECTION 2: ALT-SPECIFIC SIGNALS")
    print("=" * 60)
    alt_results = test_alt_signals(data)
    all_results.extend(alt_results)

    # Section 3: Cross-asset strategies
    print("\n" + "=" * 60)
    print("SECTION 3: CROSS-ASSET / PORTFOLIO STRATEGIES")
    print("=" * 60)
    cross_results = test_cross_asset_signals(data)
    all_results.extend(cross_results)

    # Section 4: Combined systems
    print("\n" + "=" * 60)
    print("SECTION 4: COMBINED SYSTEMS")
    print("=" * 60)
    combined_results = test_combined_systems(data)
    all_results.extend(combined_results)

    # Section 5: Additional explorations
    print("\n" + "=" * 60)
    print("SECTION 5: ADDITIONAL EXPLORATIONS")
    print("=" * 60)
    additional_results = test_additional_signals(data)
    all_results.extend(additional_results)

    # ── Summary ──────────────────────────────────────────────────────────
    valid_results = [r for r in all_results if "error" not in r]
    error_results = [r for r in all_results if "error" in r]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total signals tested: {len(all_results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Insufficient data / too few trades: {len(error_results)}")

    # Sort by Sharpe
    valid_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    # Worth WF validation
    wf_candidates = [r for r in valid_results if r.get("worth_wf_validation")]
    suspicious = [r for r in valid_results if r.get("flag")]

    print(f"\nWorth WF validation (Sharpe > 0.5): {len(wf_candidates)}")
    print(f"Suspicious (Sharpe > 2.0 or CAGR > 150%): {len(suspicious)}")

    # Top 30 by Sharpe
    print("\n" + "-" * 100)
    print("TOP 30 SIGNALS BY SHARPE RATIO")
    print("-" * 100)
    header = f"{'Label':<55} {'Dir':<6} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'WinR':>6} {'Expo':>6} {'Trades':>7} {'PF':>6} {'Flag'}"
    print(header)
    print("-" * 100)

    for r in valid_results[:30]:
        line = (
            f"{r['label']:<55} "
            f"{r['direction']:<6} "
            f"{r['sharpe']:>7.3f} "
            f"{r['cagr']*100:>7.1f}% "
            f"{r['max_drawdown']*100:>7.1f}% "
            f"{r['win_rate']*100:>5.1f}% "
            f"{r['exposure']*100:>5.1f}% "
            f"{r['n_trades']:>7d} "
            f"{r['profit_factor']:>6.2f} "
            f"{r.get('flag', '')}"
        )
        print(line)

    # SHORT signals specifically
    short_valid = [r for r in valid_results if r.get("direction") == "short"]
    short_valid.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    print("\n" + "-" * 100)
    print("TOP 15 SHORT SIGNALS")
    print("-" * 100)
    print(header)
    print("-" * 100)

    for r in short_valid[:15]:
        line = (
            f"{r['label']:<55} "
            f"{r['direction']:<6} "
            f"{r['sharpe']:>7.3f} "
            f"{r['cagr']*100:>7.1f}% "
            f"{r['max_drawdown']*100:>7.1f}% "
            f"{r['win_rate']*100:>5.1f}% "
            f"{r['exposure']*100:>5.1f}% "
            f"{r['n_trades']:>7d} "
            f"{r['profit_factor']:>6.2f} "
            f"{r.get('flag', '')}"
        )
        print(line)

    # Cross-asset specifically
    cross_valid = [
        r for r in valid_results
        if any(
            tag in r.get("label", "")
            for tag in ["CARRY", "LSR_divergence", "MOM_ROTATION", "REGIME_ROTATION",
                        "ALT_ALPHA", "ALL_WEATHER", "BTC_PLUS", "BTC_LONG_SHORT", "BTC_REGIME_LONG_plus"]
        )
    ]
    cross_valid.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    print("\n" + "-" * 100)
    print("CROSS-ASSET & COMBINED STRATEGIES")
    print("-" * 100)
    print(header)
    print("-" * 100)

    for r in cross_valid:
        line = (
            f"{r['label']:<55} "
            f"{r['direction']:<6} "
            f"{r['sharpe']:>7.3f} "
            f"{r['cagr']*100:>7.1f}% "
            f"{r['max_drawdown']*100:>7.1f}% "
            f"{r['win_rate']*100:>5.1f}% "
            f"{r['exposure']*100:>5.1f}% "
            f"{r['n_trades']:>7d} "
            f"{r['profit_factor']:>6.02f} "
            f"{r.get('flag', '')}"
        )
        print(line)

    # Benchmarks
    benchmarks = [r for r in valid_results if "BENCHMARK" in r.get("label", "")]
    if benchmarks:
        print("\n" + "-" * 100)
        print("BENCHMARKS")
        print("-" * 100)
        for r in benchmarks:
            line = (
                f"{r['label']:<55} "
                f"{r['direction']:<6} "
                f"{r['sharpe']:>7.3f} "
                f"{r['cagr']*100:>7.1f}% "
                f"{r['max_drawdown']*100:>7.1f}% "
                f"{r['win_rate']*100:>5.1f}% "
                f"{r['exposure']*100:>5.1f}% "
                f"{r['n_trades']:>7d} "
                f"{r['profit_factor']:>6.02f} "
                f"{r.get('flag', '')}"
            )
            print(line)

    # ── Honest Assessment ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT")
    print("=" * 80)

    # Category analysis
    categories = {
        "SHORT signals (BTC)": [r for r in valid_results if "BTC_SHORT" in r.get("label", "")],
        "SHORT signals (Alts)": [r for r in valid_results if r.get("direction") == "short" and "BTC" not in r.get("label", "")],
        "Alt regime signals": [r for r in valid_results if "own_regime" in r.get("label", "") or "alt_regime" in r.get("label", "").lower()],
        "BTC-relative alt": [r for r in valid_results if "btc_relative" in r.get("label", "").lower() or "rel_mom" in r.get("label", "")],
        "Funding carry": [r for r in valid_results if "CARRY" in r.get("label", "")],
        "Momentum rotation": [r for r in valid_results if "MOM_ROTATION" in r.get("label", "")],
        "Regime rotation": [r for r in valid_results if "REGIME_ROTATION" in r.get("label", "")],
        "Combined systems": [r for r in valid_results if any(t in r.get("label", "") for t in ["BTC_LONG_SHORT", "BTC_REGIME_LONG_plus", "BTC_PLUS", "ALL_WEATHER"])],
        "Contrarian signals": [r for r in valid_results if "contrarian" in r.get("label", "").lower()],
        "Z-score signals": [r for r in valid_results if "zscore" in r.get("label", "").lower()],
    }

    for cat_name, cat_results in categories.items():
        if not cat_results:
            continue
        sharpes = [r["sharpe"] for r in cat_results]
        best = max(cat_results, key=lambda x: x["sharpe"])
        worst = min(cat_results, key=lambda x: x["sharpe"])
        n_positive = sum(1 for s in sharpes if s > 0)
        n_wf = sum(1 for r in cat_results if r.get("worth_wf_validation"))

        print(f"\n  {cat_name} ({len(cat_results)} signals)")
        print(f"    Avg Sharpe: {np.mean(sharpes):.3f}")
        print(f"    Best: {best['label']} (Sharpe={best['sharpe']:.3f})")
        print(f"    Worst: {worst['label']} (Sharpe={worst['sharpe']:.3f})")
        print(f"    Positive Sharpe: {n_positive}/{len(cat_results)}")
        print(f"    Worth WF: {n_wf}")

    # ── Key Findings ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 80)

    # Find the single best signal in each category
    print("\n  BEST IN CLASS:")
    for cat_name, cat_results in categories.items():
        if not cat_results:
            continue
        best = max(cat_results, key=lambda x: x["sharpe"])
        verdict = "WORTH WF" if best.get("worth_wf_validation") else "SKIP"
        print(f"    {cat_name}: {best['label']}")
        print(f"      Sharpe={best['sharpe']:.3f}, CAGR={best['cagr']*100:.1f}%, MaxDD={best['max_drawdown']*100:.1f}%, Trades={best['n_trades']} -> [{verdict}]")

    # Overall warnings
    print("\n  WARNINGS:")
    print("    - These are FULL-SAMPLE results. OOS will be worse.")
    print("    - Short signals in crypto are inherently dangerous (long-term upward bias).")
    print("    - High Sharpe on low-exposure signals may not be practically tradeable.")
    print("    - Cross-asset strategies with many parameters are overfitting candidates.")
    print("    - Any signal with Sharpe > 2.0 should be assumed overfit until proven otherwise.")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "tx_cost_bps": TX_COST * 10000,
            "annualization": "sqrt(365)",
            "min_trades": MIN_TRADES,
            "symbols_tested": list(data.keys()),
            "total_signals": len(all_results),
            "valid_signals": len(valid_results),
            "wf_candidates": len(wf_candidates),
        },
        "results": all_results,
        "wf_candidates": [r for r in wf_candidates],
        "top_30": valid_results[:30],
        "category_summary": {
            cat: {
                "count": len(cat_results),
                "avg_sharpe": round(np.mean([r["sharpe"] for r in cat_results]), 3) if cat_results else 0,
                "best_label": max(cat_results, key=lambda x: x["sharpe"])["label"] if cat_results else "",
                "best_sharpe": max(cat_results, key=lambda x: x["sharpe"])["sharpe"] if cat_results else 0,
                "n_wf_worthy": sum(1 for r in cat_results if r.get("worth_wf_validation")),
            }
            for cat, cat_results in categories.items()
            if cat_results
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {OUT_PATH}")
    print(f"\n{'=' * 80}")
    print(f"DONE. {len(all_results)} signals tested. {len(wf_candidates)} worth walk-forward validation.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
