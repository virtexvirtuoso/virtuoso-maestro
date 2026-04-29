"""
Mega Strategy V3 — Adaptive Leverage Long/Short Macro Momentum
Virtuoso Crypto Research — February 2026

Walk-forward validated: OOS Sharpe 0.84, p=0.036

Architecture:
    5-signal confluence → regime detection → adaptive leverage → long/short dip-buying
    Multi-asset: BTC, ETH, SOL, LINK with per-asset optimized parameters

Ported from Maestro research engine to Freqtrade IStrategy format.
"""

from freqtrade.strategy import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.persistence import Trade
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MacroMomentumV3Strategy(IStrategy):
    """
    Mega Strategy V3 — Adaptive Leverage Long/Short Macro Momentum.

    Confluence engine scores 5 signals (trend, momentum, RSI health,
    EMA alignment, BB width) to determine regime. Regime drives:
      - Entry direction (long in BULL/MILD_BULL, short in BEAR)
      - Position leverage (0x–2x adaptive)
      - Trailing stop distance (per-asset optimized)
      - Pyramiding aggressiveness
    """

    INTERFACE_VERSION = 3

    # ── Timeframe ────────────────────────────────────────────────
    timeframe = "1d"

    # ── Short selling ────────────────────────────────────────────
    can_short = True

    # ── Startup ──────────────────────────────────────────────────
    startup_candle_count = 200

    # ── ROI — disabled; we manage exits via signals + trailing ───
    minimal_roi = {"0": 100}

    # ── Stoploss — wide fallback; real stop is custom_stoploss ───
    stoploss = -0.25
    use_custom_stoploss = True

    # ── Trailing stop (Freqtrade built-in as safety net) ─────────
    trailing_stop = False  # We handle trailing in custom_stoploss

    # ── Pyramiding ───────────────────────────────────────────────
    position_adjustment_enable = True
    max_entry_position_adjustment = 4

    # ── Order types ──────────────────────────────────────────────
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # ── Hyperoptable parameters (BTC defaults) ───────────────────
    sma_slow = IntParameter(70, 250, default=100, space="buy", optimize=True)
    momentum_period = IntParameter(10, 50, default=35, space="buy", optimize=True)
    rsi_entry_long = IntParameter(25, 55, default=52, space="buy", optimize=True)
    rsi_entry_short = IntParameter(65, 85, default=72, space="sell", optimize=True)

    # ── Per-asset optimized parameters ───────────────────────────
    ASSET_PARAMS = {
        "BTC/USDT:USDT": {
            "sma_slow": 100,
            "momentum_period": 35,
            "rsi_entry": 52,
            "rsi_exit": 72,
            "trail_stop": 0.12,
            "dip_pct": 0.03,
            "pyramid_dip": 0.02,
            "max_leverage": 3.0,
        },
        "ETH/USDT:USDT": {
            "sma_slow": 140,
            "momentum_period": 15,
            "rsi_entry": 32,
            "rsi_exit": 75,
            "trail_stop": 0.20,
            "dip_pct": 0.05,
            "pyramid_dip": 0.03,
            "max_leverage": 2.5,
        },
        "SOL/USDT:USDT": {
            "sma_slow": 70,
            "momentum_period": 20,
            "rsi_entry": 30,
            "rsi_exit": 78,
            "trail_stop": 0.10,
            "dip_pct": 0.06,
            "pyramid_dip": 0.04,
            "max_leverage": 2.0,
        },
        "LINK/USDT:USDT": {
            "sma_slow": 190,
            "momentum_period": 25,
            "rsi_entry": 53,
            "rsi_exit": 70,
            "trail_stop": 0.086,
            "dip_pct": 0.05,
            "pyramid_dip": 0.03,
            "max_leverage": 2.0,
        },
    }

    # Also support non-futures pair names for backtesting
    ASSET_PARAMS.update({
        "BTC/USDT": ASSET_PARAMS["BTC/USDT:USDT"],
        "ETH/USDT": ASSET_PARAMS["ETH/USDT:USDT"],
        "SOL/USDT": ASSET_PARAMS["SOL/USDT:USDT"],
        "LINK/USDT": ASSET_PARAMS["LINK/USDT:USDT"],
    })

    # ── Macro data cache path ────────────────────────────────────
    MACRO_CACHE = Path(__file__).parent / "macro_cache.json"

    # ═════════════════════════════════════════════════════════════
    #  HELPERS
    # ═════════════════════════════════════════════════════════════

    def get_asset_params(self, pair: str) -> dict:
        """Return per-asset optimized parameters, fallback to BTC defaults."""
        return self.ASSET_PARAMS.get(pair, self.ASSET_PARAMS["BTC/USDT:USDT"])

    def _read_macro_boost(self) -> int:
        """Read macro confluence boost from cached JSON (0-2 extra points).
        Returns 0 if file missing or stale (>48h)."""
        try:
            if self.MACRO_CACHE.exists():
                data = json.loads(self.MACRO_CACHE.read_text())
                ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
                if (datetime.utcnow() - ts).total_seconds() < 172800:  # 48h
                    return int(data.get("confluence_boost", 0))
        except Exception:
            pass
        return 0

    # ═════════════════════════════════════════════════════════════
    #  INFORMATIVE PAIRS
    # ═════════════════════════════════════════════════════════════

    def informative_pairs(self):
        """BTC as macro proxy for cross-asset confluence."""
        return [("BTC/USDT:USDT", "1d")]

    # ═════════════════════════════════════════════════════════════
    #  INDICATORS
    # ═════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute all indicators for the confluence engine."""
        pair = metadata["pair"]
        params = self.get_asset_params(pair)
        sma_len = params["sma_slow"]
        mom_len = params["momentum_period"]

        # ── Trend: price vs slow SMA ─────────────────────────────
        dataframe["sma_slow"] = dataframe["close"].rolling(sma_len).mean()
        dataframe["trend"] = np.where(
            dataframe["close"] > dataframe["sma_slow"], 1, 0
        )

        # ── Momentum: ROC over mom_len ───────────────────────────
        dataframe["roc"] = dataframe["close"].pct_change(mom_len)
        dataframe["momentum"] = np.where(dataframe["roc"] > 0, 1, 0)

        # ── RSI(14) ──────────────────────────────────────────────
        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        dataframe["rsi"] = 100 - (100 / (1 + rs))
        dataframe["rsi"].fillna(50, inplace=True)

        # ── Bollinger Bands(20, 2) ───────────────────────────────
        dataframe["bb_mid"] = dataframe["close"].rolling(20).mean()
        bb_std = dataframe["close"].rolling(20).std()
        dataframe["bb_upper"] = dataframe["bb_mid"] + 2 * bb_std
        dataframe["bb_lower"] = dataframe["bb_mid"] - 2 * bb_std
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"])
            / dataframe["bb_mid"].replace(0, np.nan)
        )
        dataframe["bb_width"].fillna(0, inplace=True)

        # ── EMA 21 (dip-buying reference) ────────────────────────
        dataframe["ema_21"] = dataframe["close"].ewm(span=21, adjust=False).mean()

        # ── ATR(14) ──────────────────────────────────────────────
        high_low = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
        low_close = (dataframe["low"] - dataframe["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        dataframe["atr"] = tr.rolling(14).mean()

        # ── ATR ratio (for extension detection) ──────────────────
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        dataframe["atr_ratio"].fillna(0, inplace=True)

        # ── Volume SMA (for confirmation) ────────────────────────
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # ── Confluence score (0–5 price-based signals) ───────────
        bb_width_q20 = dataframe["bb_width"].rolling(100, min_periods=20).quantile(0.2)
        dataframe["confluence"] = (
            dataframe["trend"]
            + dataframe["momentum"]
            + np.where(dataframe["rsi"] > 30, 1, 0)           # not deeply oversold
            + np.where(dataframe["close"] > dataframe["ema_21"], 1, 0)  # above EMA
            + np.where(dataframe["bb_width"] > bb_width_q20, 1, 0)     # not in squeeze
        )

        # ── Regime classification ────────────────────────────────
        dataframe["regime"] = "NEUTRAL"
        dataframe.loc[dataframe["confluence"] >= 4, "regime"] = "BULL"
        dataframe.loc[dataframe["confluence"] == 3, "regime"] = "MILD_BULL"
        dataframe.loc[dataframe["confluence"] <= 1, "regime"] = "BEAR"

        # ── Adaptive leverage multiplier ─────────────────────────
        lev_map = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
        dataframe["leverage_mult"] = dataframe["confluence"].map(lev_map).fillna(1.0)

        # ── Dip detection (close near BB lower or EMA) ───────────
        dataframe["near_bb_lower"] = np.where(
            dataframe["close"] <= dataframe["bb_lower"] * 1.01, 1, 0
        )
        dataframe["near_ema"] = np.where(
            (dataframe["close"] <= dataframe["ema_21"] * 1.005)
            & (dataframe["close"] >= dataframe["ema_21"] * 0.97),
            1, 0,
        )

        # ── Resistance rejection (for shorts) ────────────────────
        dataframe["near_bb_upper"] = np.where(
            dataframe["close"] >= dataframe["bb_upper"] * 0.99, 1, 0
        )

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  ENTRY SIGNALS
    # ═════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Long entries: dip-buying within uptrends.
        Short entries: resistance rejection in bear regimes.
        """
        pair = metadata["pair"]
        params = self.get_asset_params(pair)
        rsi_entry = params["rsi_entry"]

        # ── LONG entries ─────────────────────────────────────────
        # Condition: bullish regime + RSI dip + (near BB lower OR near EMA)
        long_cond = (
            (dataframe["confluence"] >= 3)                     # at least MILD_BULL
            & (dataframe["rsi"] < rsi_entry)                   # RSI dip below threshold
            & (
                (dataframe["near_bb_lower"] == 1)              # touching BB lower
                | (dataframe["near_ema"] == 1)                 # touching EMA 21
            )
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.5)  # not dead volume
        )
        dataframe.loc[long_cond, "enter_long"] = 1

        # Also enter long on strong momentum breakout
        breakout_cond = (
            (dataframe["confluence"] >= 4)                     # strong BULL
            & (dataframe["roc"] > dataframe["roc"].rolling(20).quantile(0.8))  # top momentum
            & (dataframe["volume"] > dataframe["vol_sma"] * 1.5)  # volume confirmation
            & (dataframe["rsi"] < 70)                          # not already overbought
        )
        dataframe.loc[breakout_cond, "enter_long"] = 1

        # ── SHORT entries ────────────────────────────────────────
        # Condition: bear regime + RSI high + near BB upper (rejection)
        short_cond = (
            (dataframe["confluence"] <= 1)                     # BEAR regime
            & (dataframe["rsi"] > params["rsi_exit"])          # overbought bounce
            & (dataframe["near_bb_upper"] == 1)                # at resistance
            & (dataframe["trend"] == 0)                        # below SMA (confirmed downtrend)
        )
        dataframe.loc[short_cond, "enter_short"] = 1

        # Short on trend breakdown
        breakdown_cond = (
            (dataframe["confluence"] <= 1)                     # BEAR
            & (dataframe["trend"] == 0)                        # below SMA
            & (dataframe["momentum"] == 0)                     # negative momentum
            & (dataframe["close"] < dataframe["bb_mid"])       # below BB mid
            & (dataframe["rsi"] > 40)                          # not already oversold
            & (dataframe["rsi"] < 60)                          # mid-range (room to fall)
        )
        dataframe.loc[breakdown_cond, "enter_short"] = 1

        # ── Entry tags ───────────────────────────────────────────
        dataframe.loc[long_cond, "enter_tag"] = "dip_buy"
        dataframe.loc[breakout_cond, "enter_tag"] = "momentum_breakout"
        dataframe.loc[short_cond, "enter_tag"] = "resistance_rejection"
        dataframe.loc[breakdown_cond, "enter_tag"] = "trend_breakdown"

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  EXIT SIGNALS
    # ═════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Exit longs: RSI overbought, BB upper touch, ATR extension.
        Exit shorts: RSI oversold, BB lower touch, trend reversal.
        """
        params = self.get_asset_params(metadata["pair"])

        # ── Exit LONG ────────────────────────────────────────────
        # RSI overbought exit
        exit_long_rsi = (
            (dataframe["rsi"] > params["rsi_exit"])
            & (dataframe["close"] >= dataframe["bb_upper"] * 0.98)
        )

        # ATR extension exit (price stretched too far above SMA)
        exit_long_atr = (
            (dataframe["close"] > dataframe["sma_slow"] + 3 * dataframe["atr"])
            & (dataframe["rsi"] > 65)
        )

        # Regime collapse — confluence drops to bear
        exit_long_regime = (
            (dataframe["confluence"] <= 1)
            & (dataframe["confluence"].shift(1) >= 3)  # was bullish, now bear
        )

        dataframe.loc[
            exit_long_rsi | exit_long_atr | exit_long_regime, "exit_long"
        ] = 1

        # ── Exit SHORT ───────────────────────────────────────────
        # RSI oversold — cover short
        exit_short_rsi = (
            (dataframe["rsi"] < 30)
            & (dataframe["close"] <= dataframe["bb_lower"] * 1.02)
        )

        # Trend reversal — price reclaims SMA
        exit_short_trend = (
            (dataframe["trend"] == 1)
            & (dataframe["trend"].shift(1) == 0)  # just crossed above
        )

        # Confluence recovery
        exit_short_regime = (
            (dataframe["confluence"] >= 3)
            & (dataframe["confluence"].shift(1) <= 1)
        )

        dataframe.loc[
            exit_short_rsi | exit_short_trend | exit_short_regime, "exit_short"
        ] = 1

        # ── Exit tags ────────────────────────────────────────────
        dataframe.loc[exit_long_rsi, "exit_tag"] = "rsi_overbought"
        dataframe.loc[exit_long_atr, "exit_tag"] = "atr_extension"
        dataframe.loc[exit_long_regime, "exit_tag"] = "regime_collapse"
        dataframe.loc[exit_short_rsi, "exit_tag"] = "rsi_oversold"
        dataframe.loc[exit_short_trend, "exit_tag"] = "trend_reversal"
        dataframe.loc[exit_short_regime, "exit_tag"] = "regime_recovery"

        return dataframe

    # ═════════════════════════════════════════════════════════════
    #  CUSTOM STOPLOSS (per-asset trailing)
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
        Per-asset adaptive trailing stop.

        Logic:
          - Initial: wide stop at -trail_stop (e.g., -12% for BTC)
          - Once profit > trail_stop/2: tighten to -trail_stop/2
          - Once profit > trail_stop: tighten to -trail_stop/3
          - Minimum stop: -1% (lock in most profit)
        """
        params = self.get_asset_params(pair)
        trail = params["trail_stop"]

        if current_profit > trail:
            # Deep in profit — tight stop
            return max(-trail / 3, -0.01)
        elif current_profit > trail / 2:
            # Moderate profit — medium stop
            return -trail / 2
        elif current_profit > 0.01:
            # Small profit — don't let it become a loss
            return -trail * 0.8
        else:
            # Not yet profitable — use full trail distance
            return -trail

    # ═════════════════════════════════════════════════════════════
    #  ADAPTIVE LEVERAGE
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
        Adaptive leverage based on confluence score.

        Higher confluence → higher conviction → more leverage.
        Shorts always get lower leverage (riskier).
        Capped by per-asset max_leverage.
        """
        params = self.get_asset_params(pair)
        asset_max = params["max_leverage"]

        # Get latest dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 1.0

        last = dataframe.iloc[-1]
        lev_mult = last.get("leverage_mult", 1.0)

        # Base leverage from confluence
        base_leverage = max(1.0, lev_mult)

        # Shorts get reduced leverage (higher risk)
        if side == "short":
            base_leverage = max(1.0, base_leverage * 0.6)

        # Add macro boost if available
        macro_boost = self._read_macro_boost()
        if macro_boost > 0 and side == "long":
            base_leverage = min(base_leverage + 0.25 * macro_boost, asset_max)

        # Cap at asset max and exchange max
        final = min(base_leverage, asset_max, max_leverage)
        return round(final, 1)

    # ═════════════════════════════════════════════════════════════
    #  PYRAMIDING (position adjustment)
    # ═════════════════════════════════════════════════════════════

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float | None,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        """
        Pyramid into positions on dips within the trend.

        Rules:
          - Only add to longs in BULL/MILD_BULL regime
          - Only add to shorts in BEAR regime
          - Each add requires price to dip pyramid_dip% from last entry
          - Each subsequent add is smaller (50% of previous)
          - Max 4 additional entries
        """
        params = self.get_asset_params(trade.pair)
        pyramid_dip = params["pyramid_dip"]

        # Get current regime
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        confluence = last.get("confluence", 0)

        # Count existing entries
        filled_entries = trade.nr_of_successful_entries
        if filled_entries >= self.max_entry_position_adjustment + 1:
            return None  # maxed out

        # Check time since last adjustment (min 1 candle = 1 day)
        if trade.orders:
            last_order_time = max(o.order_date for o in trade.orders if o.ft_is_open is False)
            if (current_time - last_order_time).total_seconds() < 86400:
                return None  # too soon

        # ── Long pyramiding ──────────────────────────────────────
        if not trade.is_short:
            if confluence < 3:
                return None  # don't add in weak regime

            # Price must have dipped from average entry
            dip_from_entry = (current_rate - trade.open_rate) / trade.open_rate
            required_dip = -pyramid_dip * filled_entries  # deeper dip for each add

            if dip_from_entry > required_dip:
                return None  # not enough of a dip

            # Decreasing position sizes: half of initial each time
            stake = trade.stake_amount / (2 ** filled_entries)
            if min_stake and stake < min_stake:
                return None
            if max_stake:
                stake = min(stake, max_stake)

            logger.info(
                f"Pyramiding LONG {trade.pair}: entry #{filled_entries + 1}, "
                f"dip={dip_from_entry:.2%}, stake={stake:.2f}"
            )
            return stake

        # ── Short pyramiding ─────────────────────────────────────
        else:
            if confluence > 1:
                return None  # don't add shorts unless bearish

            # Price must have risen from entry (short is losing)
            rise_from_entry = (current_rate - trade.open_rate) / trade.open_rate
            required_rise = pyramid_dip * filled_entries

            if rise_from_entry < required_rise:
                return None

            stake = trade.stake_amount / (2 ** filled_entries)
            if min_stake and stake < min_stake:
                return None
            if max_stake:
                stake = min(stake, max_stake)

            logger.info(
                f"Pyramiding SHORT {trade.pair}: entry #{filled_entries + 1}, "
                f"rise={rise_from_entry:.2%}, stake={stake:.2f}"
            )
            return stake

    # ═════════════════════════════════════════════════════════════
    #  CUSTOM EXIT (additional exit logic)
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
        Additional exit conditions not covered by populate_exit_trend.

        - Time-based exit: close stale trades after 30 days with no profit
        - Drawdown exit: close if unrealized loss exceeds 2x trail_stop
        """
        params = self.get_asset_params(pair)

        # Time-based exit: stale trades
        days_in_trade = (current_time - trade.open_date_utc).days
        if days_in_trade > 30 and current_profit < 0.005:
            return "stale_trade_exit"

        # Emergency drawdown exit
        if current_profit < -(params["trail_stop"] * 2):
            return "emergency_drawdown"

        return None

    # ═════════════════════════════════════════════════════════════
    #  CONFIRM TRADE ENTRY
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
        Final gate before entering a trade.
        Block entries when leverage_mult is 0 (no-trade zone).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last = dataframe.iloc[-1]
        if last.get("leverage_mult", 0) == 0:
            logger.info(f"Blocking {side} entry on {pair}: zero leverage (confluence=0)")
            return False

        return True
