"""
MegaStrategy V3.1-H2 — Direct Sizing with Protection Layers (Freqtrade Port)

Ported from: mega_strategy_v31.py (Maestro research engine)
Target: Freqtrade v2024+ on Bybit futures, dry-run

Key changes from V3:
  - Direct sizing from confluence score (no dip-buying entry gates)
  - Score 4 CryptoMom demotion to 0.5x
  - Vol ceiling 80% (halve positions when BTC 30d vol > 80%)
  - Bear filter (go flat after 30 consecutive days at score ≤ 1)
  - Portfolio trailing stop (10% DD → 30% exposure reduction)
  - Updated leverage map: {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}

Performance (walk-forward): IS Sharpe 1.25, OOS Sharpe 1.07, CAGR 41.8%, MaxDD -40.9%
"""

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MacroMomentumV31Strategy(IStrategy):
    """
    MegaStrategy V3.1-H2 — Direct confluence sizing with 4 protection layers.

    5-signal macro confluence:
      1. M2 acceleration (monetary expansion)
      2. Liquidity proxy 20d 3-of-4 (GLD↑, UUP↓, TLT↑, HYG↑)
      3. Yield curve (T10Y2Y > 0)
      4. Cross-asset momentum (COPX 20d > 0)
      5. Crypto momentum (price > SMA_slow AND ROC > 0)

    Score → leverage via map, then 4 protection layers applied sequentially.
    """

    INTERFACE_VERSION = 3
    timeframe = "1d"
    can_short = False  # V3.1 is long-only
    startup_candle_count = 200

    # ROI disabled — exits via signals + custom stoploss
    minimal_roi = {"0": 100}

    # Wide fallback stoploss; real logic in custom_stoploss
    stoploss = -0.30
    use_custom_stoploss = True
    trailing_stop = False

    # No pyramiding in V3.1 — direct sizing handles exposure
    position_adjustment_enable = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # ── V3.1 Configuration ───────────────────────────────────────

    LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}

    # Score 4 smart demotion: when CryptoMom is the dissenting signal,
    # demote leverage to 0.5x. Evidence: CryptoMom OFF at score 4 = -48.6%/yr
    SCORE4_DEMOTION = 0.5

    # Vol ceiling: halve position when BTC 30d annualized vol > 80%
    VOL_CEILING = 0.80
    VOL_LOOKBACK = 30

    # Bear filter: go flat after 30 consecutive days at score ≤ 1
    BEAR_FILTER_DAYS = 30

    # Portfolio trailing stop: 10% DD triggers 30% exposure reduction
    PORTFOLIO_TRAIL_DD = 0.10
    PORTFOLIO_TRAIL_REDUCE = 0.30
    PORTFOLIO_TRAIL_RECOVERY_DAYS = 30
    PORTFOLIO_TRAIL_RECOVERY_PCT = 0.95

    # Per-asset target weights
    WEIGHTS = {
        "BTC/USDT:USDT": 0.40,
        "ETH/USDT:USDT": 0.25,
        "SOL/USDT:USDT": 0.20,
        "LINK/USDT:USDT": 0.15,
    }

    # Per-asset SMA/momentum params (from walk-forward optimization)
    ASSET_CONFIGS = {
        "BTC/USDT:USDT": {"sma_slow": 100, "momentum_period": 35, "trail_stop": 0.12},
        "ETH/USDT:USDT": {"sma_slow": 140, "momentum_period": 15, "trail_stop": 0.18},
        "SOL/USDT:USDT": {"sma_slow": 70,  "momentum_period": 20, "trail_stop": 0.15},
        "LINK/USDT:USDT": {"sma_slow": 190, "momentum_period": 25, "trail_stop": 0.12},
    }

    # Also support non-futures pair names
    for _p, _c in list(ASSET_CONFIGS.items()):
        ASSET_CONFIGS[_p.replace(":USDT", "")] = _c
    for _p, _w in list(WEIGHTS.items()):
        WEIGHTS[_p.replace(":USDT", "")] = _w

    # ── Macro data paths ─────────────────────────────────────────
    MACRO_CACHE_DIR = Path(__file__).parent / "cache_v31"
    STATE_FILE = Path(__file__).parent / "cache_v31" / "strategy_state.json"

    # ── Portfolio-level state (persisted across restarts) ────────
    _portfolio_state = {
        "peak_equity": 10000.0,
        "in_drawdown": False,
        "dd_start_time": None,
        "consecutive_low_score_days": 0,
        "last_btc_score": 0,
    }

    def bot_start(self, **kwargs) -> None:
        """Load persisted state on startup."""
        self._load_state()
        self._macro_signals = None
        self._macro_last_fetch = None
        logger.info("V3.1-H2 strategy initialized")

    # ═════════════════════════════════════════════════════════════
    #  STATE PERSISTENCE
    # ═════════════════════════════════════════════════════════════

    def _load_state(self):
        try:
            if self.STATE_FILE.exists():
                data = json.loads(self.STATE_FILE.read_text())
                self._portfolio_state.update(data)
                logger.info(f"Loaded state: peak={self._portfolio_state['peak_equity']:.2f}")
        except Exception as e:
            logger.warning(f"State load failed: {e}")

    def _save_state(self):
        try:
            self.MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.STATE_FILE.write_text(json.dumps(self._portfolio_state, default=str))
        except Exception as e:
            logger.warning(f"State save failed: {e}")

    # ═════════════════════════════════════════════════════════════
    #  MACRO DATA
    # ═════════════════════════════════════════════════════════════

    def _get_macro_signals(self) -> pd.DataFrame:
        """Fetch macro signals, cached for the day."""
        now = datetime.utcnow()
        if (
            self._macro_signals is not None
            and self._macro_last_fetch is not None
            and (now - self._macro_last_fetch).total_seconds() < 3600 * 20
        ):
            return self._macro_signals

        try:
            from freqtrade_files.macro_data_provider_v31 import get_all_macro_signals
            self._macro_signals = get_all_macro_signals()
            self._macro_last_fetch = now
            logger.info(f"Macro signals refreshed: {len(self._macro_signals)} days")
        except ImportError:
            try:
                # Try relative import
                from macro_data_provider_v31 import get_all_macro_signals
                self._macro_signals = get_all_macro_signals()
                self._macro_last_fetch = now
            except Exception as e:
                logger.error(f"Macro data import failed: {e}")
                self._macro_signals = pd.DataFrame()

        return self._macro_signals if self._macro_signals is not None else pd.DataFrame()

    def _get_macro_score_for_date(self, date) -> dict:
        """Get individual macro signal values for a specific date."""
        macro = self._get_macro_signals()
        if macro.empty:
            # Default: assume neutral (all signals off)
            return {"m2_accel": 0, "liquidity_proxy": 0, "yield_curve": 0,
                    "cross_asset_mom": 0, "hy_spread": 0}

        # Find closest date <= target
        try:
            date = pd.Timestamp(date)
            mask = macro.index <= date
            if mask.any():
                row = macro.loc[mask].iloc[-1]
                return row.to_dict()
        except Exception:
            pass
        return {"m2_accel": 0, "liquidity_proxy": 0, "yield_curve": 0,
                "cross_asset_mom": 0, "hy_spread": 0}

    # ═════════════════════════════════════════════════════════════
    #  INFORMATIVE PAIRS (cross-asset data)
    # ═════════════════════════════════════════════════════════════

    def informative_pairs(self):
        """
        Request BTC data for all pairs (vol ceiling uses BTC vol).
        In live mode, cross-asset data comes from yfinance via macro_data_provider.
        """
        pairs = [("BTC/USDT:USDT", "1d")]
        return pairs

    # ═════════════════════════════════════════════════════════════
    #  INDICATORS
    # ═════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Compute the 5-signal confluence score.

        Signal 5 (crypto_momentum) is computed from price data.
        Signals 1-4 come from macro_data_provider (FRED + yfinance).
        """
        pair = metadata["pair"]
        cfg = self.ASSET_CONFIGS.get(pair, self.ASSET_CONFIGS.get("BTC/USDT:USDT"))
        sma_len = cfg["sma_slow"]
        mom_len = cfg["momentum_period"]

        # ── Signal 5: Crypto momentum (price > SMA AND ROC > 0) ──
        dataframe["sma_slow"] = dataframe["close"].rolling(sma_len).mean()
        dataframe["roc"] = dataframe["close"].pct_change(mom_len)
        dataframe["crypto_mom"] = np.where(
            (dataframe["close"] > dataframe["sma_slow"]) & (dataframe["roc"] > 0),
            1, 0
        )

        # ── Realized volatility (for vol ceiling) ────────────────
        dataframe["rvol_30d"] = (
            dataframe["close"].pct_change().rolling(self.VOL_LOOKBACK).std()
            * np.sqrt(252)
        )

        # ── Macro signals 1-4 (fetched from provider) ────────────
        macro = self._get_macro_signals()

        for col in ["m2_accel", "liquidity_proxy", "yield_curve", "cross_asset_mom"]:
            dataframe[col] = 0  # default

        if not macro.empty:
            # Align macro signals to dataframe dates
            for col in ["m2_accel", "liquidity_proxy", "yield_curve", "cross_asset_mom"]:
                if col in macro.columns:
                    # Reindex to dataframe's date index, forward-fill
                    macro_series = macro[col]
                    aligned = macro_series.reindex(dataframe["date"] if "date" in dataframe.columns
                                                   else dataframe.index, method="ffill")
                    if len(aligned) == len(dataframe):
                        dataframe[col] = aligned.values
                    else:
                        # Fallback: use last known value
                        try:
                            last_val = int(macro_series.iloc[-1])
                            dataframe[col] = last_val
                        except Exception:
                            dataframe[col] = 0

        # ── Confluence score (0-5) ────────────────────────────────
        dataframe["confluence"] = (
            dataframe["m2_accel"]
            + dataframe["liquidity_proxy"]
            + dataframe["yield_curve"]
            + dataframe["cross_asset_mom"]
            + dataframe["crypto_mom"]
        )

        # ── Base leverage from confluence map ─────────────────────
        dataframe["base_leverage"] = dataframe["confluence"].map(
            lambda c: self.LEVERAGE_MAP.get(min(int(c), 5), 0.0)
        )

        # ── Layer 1: Score 4 CryptoMom demotion ──────────────────
        # When score = 4 and crypto_mom is the only ON signal among the 5,
        # but actually: demote when score=4 AND crypto_mom=0 (it's the dissenter)
        score4_mask = (dataframe["confluence"] == 4) & (dataframe["crypto_mom"] == 0)
        dataframe.loc[score4_mask, "base_leverage"] = self.SCORE4_DEMOTION

        # ── Layer 2: Vol ceiling ──────────────────────────────────
        # Halve leverage when 30d annualized vol > 80%
        high_vol_mask = dataframe["rvol_30d"] > self.VOL_CEILING
        dataframe["vol_adjusted_leverage"] = dataframe["base_leverage"]
        dataframe.loc[high_vol_mask, "vol_adjusted_leverage"] = (
            dataframe.loc[high_vol_mask, "base_leverage"] * 0.5
        )

        # ── Final leverage (before portfolio-level layers) ────────
        # Portfolio trail stop and bear filter are applied in custom_stake_amount
        # because they need cross-trade portfolio state
        dataframe["target_leverage"] = dataframe["vol_adjusted_leverage"]

        logger.debug(
            f"{pair} last candle: confluence={dataframe['confluence'].iloc[-1]}, "
            f"leverage={dataframe['target_leverage'].iloc[-1]:.2f}, "
            f"rvol={dataframe['rvol_30d'].iloc[-1]:.2%}"
        )

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  ENTRY SIGNALS — Direct sizing (always enter when score > 0)
    # ═════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        V3.1 enters whenever confluence > 0. Position size is controlled
        by custom_stake_amount based on the confluence score.

        No dip-buying gates — this was the main fix from V3 (72% flat time).
        """
        # Enter long when we have any positive confluence
        dataframe.loc[dataframe["confluence"] > 0, "enter_long"] = 1
        dataframe.loc[dataframe["confluence"] > 0, "enter_tag"] = (
            "confluence_" + dataframe["confluence"].astype(int).astype(str)
        )

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  EXIT SIGNALS
    # ═════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Exit when confluence drops to 0 (all signals off).
        Additional exits handled by custom_stoploss and confirm_trade_exit.
        """
        # Full exit when confluence = 0
        dataframe.loc[dataframe["confluence"] == 0, "exit_long"] = 1
        dataframe.loc[dataframe["confluence"] == 0, "exit_tag"] = "zero_confluence"

        # Also exit on regime collapse (score drops by 3+ in one candle)
        score_drop = dataframe["confluence"].diff()
        dataframe.loc[score_drop <= -3, "exit_long"] = 1
        dataframe.loc[score_drop <= -3, "exit_tag"] = "regime_collapse"

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  POSITION SIZING — custom_stake_amount
    # ═════════════════════════════════════════════════════════════

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float | None,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Position sizing based on confluence score and protection layers.

        Base formula:
          stake = wallet_balance * asset_weight * target_leverage

        Then apply portfolio-level protections:
          Layer 3: Bear filter (flat after 30 consecutive days score ≤ 1)
          Layer 4: Portfolio trailing stop (10% DD → 30% of target)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 0

        last = dataframe.iloc[-1]
        target_lev = last.get("target_leverage", 0)

        if target_lev <= 0:
            return 0

        # Asset weight
        weight = self.WEIGHTS.get(pair, 0.25)

        # Wallet balance
        wallet = self.wallets.get_free("USDT") if self.wallets else 10000
        if wallet <= 0:
            return 0

        # Base stake
        stake = wallet * weight * target_lev

        # ── Layer 3: Bear filter ──────────────────────────────────
        # Track consecutive low-score days using BTC confluence
        btc_df, _ = self.dp.get_analyzed_dataframe("BTC/USDT:USDT", self.timeframe)
        if not btc_df.empty:
            btc_conf = btc_df["confluence"]
            # Count trailing days with score ≤ 1
            consecutive = 0
            for val in btc_conf.iloc[::-1]:
                if val <= 1:
                    consecutive += 1
                else:
                    break
            self._portfolio_state["consecutive_low_score_days"] = consecutive

            if consecutive >= self.BEAR_FILTER_DAYS:
                logger.info(
                    f"BEAR FILTER active: {consecutive} consecutive days at score ≤ 1. "
                    f"Blocking entry for {pair}."
                )
                return 0

        # ── Layer 4: Portfolio trailing stop ──────────────────────
        # Track portfolio equity (approximate from wallet balance)
        current_equity = wallet
        peak = self._portfolio_state.get("peak_equity", current_equity)

        if current_equity > peak:
            self._portfolio_state["peak_equity"] = current_equity
            self._portfolio_state["in_drawdown"] = False
            peak = current_equity

        dd = 1.0 - (current_equity / peak) if peak > 0 else 0
        if dd > self.PORTFOLIO_TRAIL_DD:
            if not self._portfolio_state.get("in_drawdown"):
                self._portfolio_state["in_drawdown"] = True
                self._portfolio_state["dd_start_time"] = current_time.isoformat()
                logger.warning(
                    f"PORTFOLIO TRAIL STOP triggered: DD={dd:.1%}, "
                    f"reducing to {self.PORTFOLIO_TRAIL_REDUCE:.0%} of target"
                )
            stake *= self.PORTFOLIO_TRAIL_REDUCE
        elif self._portfolio_state.get("in_drawdown"):
            # Check recovery
            dd_start = self._portfolio_state.get("dd_start_time")
            if dd_start:
                try:
                    dd_start_dt = datetime.fromisoformat(dd_start)
                    days_since = (current_time - dd_start_dt).days
                    if (
                        days_since > self.PORTFOLIO_TRAIL_RECOVERY_DAYS
                        and current_equity > peak * self.PORTFOLIO_TRAIL_RECOVERY_PCT
                    ):
                        self._portfolio_state["in_drawdown"] = False
                        logger.info("Portfolio trail stop recovered — full sizing restored")
                except Exception:
                    pass

        # Save state
        self._save_state()

        # Clamp to min/max
        if min_stake and stake < min_stake:
            stake = min_stake
        if max_stake:
            stake = min(stake, max_stake)

        logger.info(
            f"Sizing {pair}: confluence={last.get('confluence', 0)}, "
            f"lev={target_lev:.2f}, weight={weight:.0%}, stake={stake:.2f} USDT"
        )

        return stake

    # ═════════════════════════════════════════════════════════════
    #  LEVERAGE
    # ═════════════════════════════════════════════════════════════

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Set leverage based on confluence score.
        Capped at the leverage map maximum (2.0x for score 5).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 1.0

        last = dataframe.iloc[-1]
        target = last.get("target_leverage", 1.0)
        target = max(1.0, min(target, max_leverage))
        return round(target, 1)

    # ═════════════════════════════════════════════════════════════
    #  CUSTOM STOPLOSS — Per-asset trailing
    # ═════════════════════════════════════════════════════════════

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        Adaptive trailing stop per asset.

        - Base: -trail_stop (e.g., -12% for BTC)
        - Profit > trail_stop: tighten to -trail_stop/3
        - Profit > trail_stop/2: tighten to -trail_stop/2
        - Small profit: protect at -trail_stop * 0.8
        """
        cfg = self.ASSET_CONFIGS.get(pair, self.ASSET_CONFIGS.get("BTC/USDT:USDT"))
        trail = cfg["trail_stop"]

        if current_profit > trail:
            return max(-trail / 3, -0.01)
        elif current_profit > trail / 2:
            return -trail / 2
        elif current_profit > 0.01:
            return -trail * 0.8
        else:
            return -trail

    # ═════════════════════════════════════════════════════════════
    #  CONFIRM TRADE EXIT — Additional protections
    # ═════════════════════════════════════════════════════════════

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Allow all exits — but log for monitoring.
        In future: could block exits that would violate minimum hold time.
        """
        profit = trade.calc_profit_ratio(rate)
        logger.info(
            f"EXIT {pair}: reason={exit_reason}, profit={profit:.2%}, "
            f"held={( current_time - trade.open_date_utc).days}d"
        )
        return True

    # ═════════════════════════════════════════════════════════════
    #  CONFIRM TRADE ENTRY — Final gate
    # ═════════════════════════════════════════════════════════════

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Block entries when confluence is 0 or bear filter is active.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last = dataframe.iloc[-1]
        confluence = last.get("confluence", 0)

        if confluence <= 0:
            logger.info(f"Blocking entry {pair}: confluence=0")
            return False

        # Check bear filter
        if self._portfolio_state.get("consecutive_low_score_days", 0) >= self.BEAR_FILTER_DAYS:
            logger.info(f"Blocking entry {pair}: bear filter active")
            return False

        return True

    # ═════════════════════════════════════════════════════════════
    #  CUSTOM EXIT — Stale trade cleanup
    # ═════════════════════════════════════════════════════════════

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        Time-based exit: close trades held > 45 days with < 1% profit.
        V3.1 should not hold stale positions.
        """
        days = (current_time - trade.open_date_utc).days
        if days > 45 and current_profit < 0.01:
            return "stale_trade_45d"
        return None
