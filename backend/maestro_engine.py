"""
Maestro Engine — Daily Signal Generator
Runs as cron job, computes all signals, writes state to JSON for MCP/Freqtrade consumption.

Architecture:
    1. Fetch prices (yfinance) + macro (FRED) + cross-asset
    2. Compute confluence (5 signals) + regime
    3. Per-asset signals with entry/exit/pyramiding
    4. Adaptive leverage + portfolio allocation
    5. ML enhancement (graceful fallback)
    6. Write atomic JSON state file
"""

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    ASSET_CONFIGS,
    DEFAULT_LEVERAGE_MAP,
    DEFAULT_SAFETY_PARAMS,
    compute_confluence,
    detect_regime,
    _rsi,
    _bb,
    _atr,
    _realized_vol,
)
from strategies.composite.macro_score_builder import (
    FRED_SERIES as MACRO_FRED_SERIES,
    compute_macro_score_from_df,
)

logger = logging.getLogger("maestro.engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSETS = ["BTC", "ETH", "SOL", "LINK"]
CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_TICKERS = {
    "gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "CPER",
}
MACRO_SERIES = {
    "yield_curve": "T10Y2Y",
    "m2": "M2SL",
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "hy_spread": "BAMLH0A0HYM2",
}

STATE_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/live"))
STATE_PATH = STATE_DIR / "maestro_state.json"
CACHE_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/cache"))

# Base allocation weights (risk-parity inspired)
BASE_WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}


class MaestroEngine:
    """Core daily signal engine."""

    def __init__(self, config_path: Optional[str] = None):
        self.state: Dict[str, Any] = {}
        self.config = self._load_config(config_path)
        self._setup_logging()

        # Data loaders (lazy init to handle missing API keys gracefully)
        self._stock_loader: Optional[StockDataLoader] = None
        self._macro_loader: Optional[MacroDataLoader] = None

        # Cached data for the run
        self._crypto_data: Dict[str, pd.DataFrame] = {}
        self._cross_asset_data: Optional[pd.DataFrame] = None
        self._macro_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_config(self, path: Optional[str]) -> dict:
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return {
            "assets": ASSETS,
            "lookback_days": 730,
            "api_key": "maestro-default-key",
        }

    def _setup_logging(self):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    @property
    def stock_loader(self) -> StockDataLoader:
        if self._stock_loader is None:
            self._stock_loader = StockDataLoader()
        return self._stock_loader

    @property
    def macro_loader(self) -> Optional[MacroDataLoader]:
        if self._macro_loader is None:
            try:
                self._macro_loader = MacroDataLoader()
            except Exception as e:
                logger.warning(f"FRED loader unavailable: {e}")
        return self._macro_loader

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------

    def _fetch_crypto_prices(self) -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV for all crypto assets."""
        logger.info("Fetching crypto prices...")
        results = {}
        start = (datetime.now() - timedelta(days=self.config["lookback_days"])).strftime("%Y-%m-%d")
        for asset in self.config["assets"]:
            ticker = CRYPTO_TICKERS.get(asset, f"{asset}-USD")
            try:
                df = self.stock_loader.get_ohlcv(ticker, "1d", start_date=start)
                if not df.empty:
                    results[asset] = df
                    logger.info(f"  {asset}: {len(df)} bars, latest {df.index[-1].date()}")
                else:
                    logger.warning(f"  {asset}: no data returned")
            except Exception as e:
                logger.error(f"  {asset} fetch failed: {e}")
        self._crypto_data = results
        return results

    def _fetch_cross_asset(self) -> Optional[pd.DataFrame]:
        """Fetch cross-asset data (gold, DXY, bonds, HYG, copper)."""
        logger.info("Fetching cross-asset data...")
        start = (datetime.now() - timedelta(days=self.config["lookback_days"])).strftime("%Y-%m-%d")
        frames = {}
        for name, ticker in CROSS_ASSET_TICKERS.items():
            try:
                df = self.stock_loader.get_ohlcv(ticker, "1d", start_date=start)
                if not df.empty:
                    frames[name] = df["close"]
            except Exception as e:
                logger.warning(f"  Cross-asset {name} ({ticker}) failed: {e}")
        if frames:
            self._cross_asset_data = pd.DataFrame(frames).ffill()
            logger.info(f"  Cross-asset: {len(self._cross_asset_data)} rows, cols={list(frames.keys())}")
        else:
            self._cross_asset_data = None
            logger.warning("  No cross-asset data available")
        return self._cross_asset_data

    def _fetch_macro(self) -> Optional[pd.DataFrame]:
        """Fetch FRED macro data."""
        logger.info("Fetching macro data...")
        loader = self.macro_loader
        if loader is None:
            logger.warning("  Macro loader not available, skipping")
            self._macro_data = None
            return None
        try:
            start = (datetime.now() - timedelta(days=self.config["lookback_days"])).strftime("%Y-%m-%d")
            df = loader.get_multiple(MACRO_SERIES, start_date=start)
            if not df.empty:
                self._macro_data = df
                logger.info(f"  Macro: {len(df)} rows, cols={list(df.columns)}")
            else:
                self._macro_data = None
                logger.warning("  Macro data empty")
        except Exception as e:
            logger.error(f"  Macro fetch failed: {e}")
            self._macro_data = None
        return self._macro_data

    # ------------------------------------------------------------------
    # Signal Computation
    # ------------------------------------------------------------------

    def _compute_confluence(self, asset: str = "BTC") -> Dict[str, Any]:
        """Compute confluence score and breakdown for an asset."""
        df = self._crypto_data.get(asset)
        if df is None or df.empty:
            return {"score": 0, "signals": {k: False for k in [
                "m2_acceleration", "rt_liquidity", "yield_curve",
                "cross_asset_momentum", "crypto_momentum"
            ]}}

        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        confluence_series, breakdown = compute_confluence(
            crypto_close=df["close"],
            macro_data=self._macro_data,
            cross_asset_data=self._cross_asset_data,
            sma_slow=cfg["sma_slow"],
            momentum_period=cfg["momentum_period"],
        )

        latest = breakdown.iloc[-1] if not breakdown.empty else {}
        score = int(confluence_series.iloc[-1]) if not confluence_series.empty else 0

        return {
            "score": score,
            "signals": {
                "m2_acceleration": bool(latest.get("m2_accel", 0)),
                "rt_liquidity": bool(latest.get("liquidity_proxy", 0)),
                "yield_curve": bool(latest.get("yield_curve", 0)),
                "cross_asset_momentum": bool(latest.get("cross_asset_mom", 0)),
                "crypto_momentum": bool(latest.get("crypto_momentum", 0)),
            },
        }

    def _detect_regime(self, asset: str = "BTC") -> str:
        """Detect current regime from confluence."""
        df = self._crypto_data.get(asset)
        if df is None or df.empty:
            return "NEUTRAL"

        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        confluence_series, _ = compute_confluence(
            crypto_close=df["close"],
            macro_data=self._macro_data,
            cross_asset_data=self._cross_asset_data,
            sma_slow=cfg["sma_slow"],
            momentum_period=cfg["momentum_period"],
        )
        regime_series = detect_regime(confluence_series)
        return str(regime_series.iloc[-1]) if not regime_series.empty else "NEUTRAL"

    def _compute_asset_signal(self, asset: str, regime: str, confluence_score: int) -> Dict[str, Any]:
        """Compute trading signal for a single asset."""
        df = self._crypto_data.get(asset)
        if df is None or len(df) < 200:
            return {
                "signal": "FLAT", "confidence": 0.0, "position_size": 0.0,
                "entry_score": 0.0, "regime": regime,
                "indicators": {}, "dip_buy_active": False,
                "pyramid_level": 0, "trailing_stop_level": 0,
            }

        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Compute indicators
        rsi = _rsi(close, 14)
        bb_lower, bb_mid, bb_upper = _bb(close, cfg["bb_period"], cfg["bb_std"])
        atr = _atr(df, 14)
        sma = close.rolling(cfg["sma_slow"]).mean()
        ema = close.ewm(span=cfg["ema_period"], adjust=False).mean()
        roc = close.pct_change(cfg["momentum_period"])
        vol = _realized_vol(close, 20)

        latest_close = float(close.iloc[-1])
        latest_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
        latest_atr = float(atr.iloc[-1]) if not atr.empty else 0
        latest_sma = float(sma.iloc[-1]) if not sma.empty else latest_close
        latest_bb_lower = float(bb_lower.iloc[-1]) if not bb_lower.empty else latest_close
        latest_bb_upper = float(bb_upper.iloc[-1]) if not bb_upper.empty else latest_close
        latest_bb_mid = float(bb_mid.iloc[-1]) if not bb_mid.empty else latest_close
        latest_vol = float(vol.iloc[-1]) if not vol.empty else 0.5

        trend_up = latest_close > latest_sma
        momentum_pos = float(roc.iloc[-1]) > 0 if not roc.empty else False

        # BB position (0=lower, 1=upper)
        bb_range = latest_bb_upper - latest_bb_lower
        bb_position = (latest_close - latest_bb_lower) / bb_range if bb_range > 0 else 0.5

        # ATR as % of price
        atr_pct = (latest_atr / latest_close * 100) if latest_close > 0 else 0

        # Signal determination
        signal = "FLAT"
        confidence = 0.0
        dip_buy = False

        if regime in ("BULL", "MILD_BULL", "ACCUMULATION"):
            if trend_up and momentum_pos:
                signal = "LONG"
                confidence = min(0.5 + confluence_score * 0.1, 1.0)
            elif trend_up and latest_rsi < cfg["rsi_entry"]:
                signal = "LONG"
                confidence = 0.6
                dip_buy = True
            elif latest_close < latest_bb_lower and regime != "BEAR":
                signal = "LONG"
                confidence = 0.55
                dip_buy = True
        elif regime == "BEAR":
            if not trend_up and latest_rsi > 65:
                signal = "SHORT"
                confidence = min(0.4 + (5 - confluence_score) * 0.1, 0.8)
            elif latest_rsi > cfg["rsi_exit"]:
                signal = "SHORT"
                confidence = 0.5

        if signal == "FLAT" and regime == "NEUTRAL":
            confidence = 0.3

        # Position sizing based on confluence and regime
        leverage = DEFAULT_LEVERAGE_MAP.get(confluence_score, 0.5)
        base_weight = BASE_WEIGHTS.get(asset, 0.15)
        position_size = round(base_weight * leverage * confidence, 4)

        # Pyramid level (simple heuristic from trend duration)
        sma_above_count = int((close.tail(60) > sma.tail(60)).sum()) if len(close) >= 60 else 0
        pyramid_level = min(sma_above_count // 20, 3)  # 0-3

        # Trailing stop level
        recent_high = float(high.tail(20).max())
        trail_pct = cfg["trail_stop_pct"]
        trailing_stop = round(recent_high * (1 - trail_pct), 2)

        # Entry score (composite)
        entry_factors = []
        if dip_buy:
            entry_factors.append(0.8)
        if latest_rsi < 40:
            entry_factors.append(0.7)
        if bb_position < 0.3:
            entry_factors.append(0.75)
        if trend_up:
            entry_factors.append(0.6)
        entry_score = round(np.mean(entry_factors), 2) if entry_factors else 0.5

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "position_size": position_size,
            "entry_score": entry_score,
            "regime": regime,
            "indicators": {
                "rsi": round(latest_rsi, 1),
                "trend": trend_up,
                "momentum": bool(momentum_pos),
                "bb_position": round(bb_position, 2),
                "atr_pct": round(atr_pct, 1),
                "vol_20d": round(latest_vol, 3),
                "price": latest_close,
                "sma_slow": round(latest_sma, 2),
            },
            "dip_buy_active": dip_buy,
            "pyramid_level": pyramid_level,
            "trailing_stop_level": trailing_stop,
        }

    def _compute_leverage(self, confluence_score: int, portfolio_vol: float, dd: float) -> float:
        """Adaptive leverage with safety overrides."""
        base = DEFAULT_LEVERAGE_MAP.get(confluence_score, 0.5)
        safety = DEFAULT_SAFETY_PARAMS

        # Vol ceiling override
        if portfolio_vol > safety["vol_ceiling"]:
            base *= 0.5
            logger.warning(f"Vol ceiling hit ({portfolio_vol:.2f}), halving leverage")

        # Drawdown reduction
        if abs(dd) > safety["dd_reduction_threshold"]:
            base *= safety["dd_reduction_factor"]
            logger.warning(f"DD threshold hit ({dd:.2%}), reducing leverage")

        return round(min(base, safety["max_portfolio_leverage"]), 2)

    def _compute_portfolio(self, asset_signals: Dict[str, Dict], leverage: float) -> Dict[str, Any]:
        """Compute portfolio-level metrics."""
        weights = {}
        long_exp = 0.0
        short_exp = 0.0

        for asset, sig in asset_signals.items():
            w = sig["position_size"]
            weights[asset] = round(w, 4)
            if sig["signal"] == "LONG":
                long_exp += w
            elif sig["signal"] == "SHORT":
                short_exp += w

        total_exp = long_exp + short_exp

        # Portfolio vol (simple weighted average of per-asset vol)
        vols = []
        for asset, sig in asset_signals.items():
            v = sig.get("indicators", {}).get("vol_20d", 0.5)
            w = sig["position_size"]
            vols.append(v * w)
        portfolio_vol = sum(vols) / max(total_exp, 0.01)

        # Drawdown from peak (using BTC as proxy if available)
        dd = 0.0
        btc_data = self._crypto_data.get("BTC")
        if btc_data is not None and len(btc_data) > 0:
            close = btc_data["close"]
            peak = close.cummax()
            dd = float((close.iloc[-1] / peak.iloc[-1]) - 1)

        circuit_breaker = abs(dd) > 0.20

        return {
            "total_exposure": round(total_exp, 4),
            "long_exposure": round(long_exp, 4),
            "short_exposure": round(short_exp, 4),
            "weights": weights,
            "risk_metrics": {
                "portfolio_vol_20d": round(portfolio_vol, 3),
                "current_dd_from_peak": round(dd, 4),
                "circuit_breaker_active": circuit_breaker,
            },
        }

    def _compute_macro_score(self) -> Dict[str, Any]:
        """Compute macro score from FRED data."""
        if self._macro_data is None or self._macro_data.empty:
            return {"total": 0, "components": {}}

        try:
            score_series = compute_macro_score_from_df(self._macro_data)
            latest_score = int(score_series.iloc[-1]) if not score_series.empty else 0

            # Compute individual components for breakdown
            df = self._macro_data.ffill()
            components = {}

            if "yield_curve" in df.columns:
                components["yield_curve_positive"] = bool(df["yield_curve"].iloc[-1] > 0)

            if "m2" in df.columns:
                m2_3m = df["m2"].pct_change(90).iloc[-1]
                m2_6m = df["m2"].pct_change(180).iloc[-1]
                components["m2_expanding"] = bool(m2_3m > 0) if not pd.isna(m2_3m) else False
                components["m2_accelerating"] = bool(m2_3m > m2_6m) if not (pd.isna(m2_3m) or pd.isna(m2_6m)) else False

            if "cpi" in df.columns:
                cpi_yoy = df["cpi"].pct_change(365)
                if len(cpi_yoy.dropna()) >= 91:
                    cpi_chg = cpi_yoy.iloc[-1] - cpi_yoy.iloc[-91]
                    components["cpi_declining"] = bool(cpi_chg < 0) if not pd.isna(cpi_chg) else False
                else:
                    components["cpi_declining"] = False

            if "fed_funds" in df.columns:
                ff = df["fed_funds"]
                if len(ff.dropna()) >= 91:
                    components["fed_not_hiking"] = bool(ff.iloc[-1] <= ff.iloc[-91])
                else:
                    components["fed_not_hiking"] = False

            if "hy_spread" in df.columns:
                hy = df["hy_spread"]
                if len(hy.dropna()) >= 91:
                    components["hy_tightening"] = bool(hy.iloc[-1] <= hy.iloc[-91])
                else:
                    components["hy_tightening"] = False

            return {"total": latest_score, "components": components}
        except Exception as e:
            logger.error(f"Macro score computation failed: {e}")
            return {"total": 0, "components": {}}

    def _run_ml_enhancement(self, asset_signals: Dict, regime: str) -> Dict[str, Any]:
        """Run ML models if available, graceful fallback."""
        try:
            from ml.regime_classifier import RegimeClassifier
            from ml.signal_weighter import SignalWeighter

            # Try to load trained models
            rc = RegimeClassifier()
            sw = SignalWeighter()

            # Build feature vector from current state
            btc = self._crypto_data.get("BTC")
            if btc is not None and len(btc) > 100:
                from ml.feature_engine import build_features
                features = build_features(
                    crypto_data=self._crypto_data,
                    macro_data=self._macro_data,
                    cross_asset_data=self._cross_asset_data,
                )
                if features is not None and not features.empty:
                    latest = features.iloc[[-1]]
                    ml_regime = rc.predict(latest)
                    ml_conf = rc.predict_proba(latest)
                    weights = sw.predict(latest)
                    return {
                        "regime_ml": str(ml_regime),
                        "regime_ml_confidence": round(float(ml_conf), 2),
                        "signal_weights": weights,
                        "entry_score": round(float(np.mean(list(weights.values()))), 2),
                    }
        except Exception as e:
            logger.info(f"ML enhancement not available: {e}")

        # Fallback: use rule-based
        return {
            "regime_ml": regime,
            "regime_ml_confidence": 0.5,
            "signal_weights": {"m2": 0.25, "liquidity": 0.20, "yield_curve": 0.20,
                               "cross_asset": 0.15, "crypto_momentum": 0.20},
            "entry_score": 0.5,
        }

    def _generate_alerts(self, asset_signals: Dict, regime: str, portfolio: Dict) -> List[str]:
        """Generate actionable alerts."""
        alerts = []
        for asset, sig in asset_signals.items():
            rsi = sig["indicators"].get("rsi", 50)
            if rsi < 30:
                alerts.append(f"🟢 {asset} RSI oversold ({rsi:.0f}) — potential dip buy")
            elif rsi > 75:
                alerts.append(f"🔴 {asset} RSI overbought ({rsi:.0f}) — consider trimming")

            if sig["dip_buy_active"]:
                alerts.append(f"📉 {asset} dip buy signal active")

        if portfolio["risk_metrics"]["circuit_breaker_active"]:
            alerts.append("🚨 CIRCUIT BREAKER: Portfolio DD > 20% — reduce exposure")

        if portfolio["risk_metrics"]["portfolio_vol_20d"] > 0.8:
            alerts.append("⚠️ High portfolio volatility — consider hedging")

        return alerts

    def _generate_next_actions(self, asset_signals: Dict, regime: str) -> List[str]:
        """Generate suggested next actions."""
        actions = []
        for asset, sig in asset_signals.items():
            rsi = sig["indicators"].get("rsi", 50)
            price = sig["indicators"].get("price", 0)
            trail = sig["trailing_stop_level"]

            if sig["signal"] == "LONG" and rsi < 35:
                actions.append(f"Consider adding to {asset} position on RSI dip below {rsi:.0f}")
            if trail > 0 and sig["signal"] == "LONG":
                actions.append(f"{asset} trailing stop at ${trail:,.0f}")

        if regime == "ACCUMULATION":
            actions.append("Accumulation phase — scale in gradually on dips")
        elif regime == "BEAR":
            actions.append("Bear regime — maintain hedges, reduce long exposure")

        return actions

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def run_daily(self) -> Dict[str, Any]:
        """Full daily pipeline."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("MAESTRO ENGINE — Daily Run Starting")
        logger.info("=" * 60)

        # 1-2. Fetch data
        self._fetch_crypto_prices()
        self._fetch_cross_asset()
        self._fetch_macro()

        # 3. Confluence (use BTC as primary)
        confluence = self._compute_confluence("BTC")
        logger.info(f"Confluence score: {confluence['score']}/5")

        # 4. Regime
        regime = self._detect_regime("BTC")
        logger.info(f"Regime: {regime}")

        # 5. Per-asset signals
        asset_signals = {}
        for asset in self.config["assets"]:
            if asset in self._crypto_data:
                asset_signals[asset] = self._compute_asset_signal(
                    asset, regime, confluence["score"]
                )
                logger.info(f"  {asset}: {asset_signals[asset]['signal']} "
                            f"(conf={asset_signals[asset]['confidence']:.2f})")

        # 6. Macro score
        macro_score = self._compute_macro_score()
        logger.info(f"Macro score: {macro_score['total']}/6")

        # 7. Portfolio
        portfolio_vol = 0.5  # default
        dd = 0.0
        if asset_signals:
            portfolio = self._compute_portfolio(asset_signals, 1.0)
            portfolio_vol = portfolio["risk_metrics"]["portfolio_vol_20d"]
            dd = portfolio["risk_metrics"]["current_dd_from_peak"]

        # 6b. Adaptive leverage (needs portfolio metrics)
        leverage = self._compute_leverage(confluence["score"], portfolio_vol, dd)
        logger.info(f"Leverage recommendation: {leverage}x")

        # Recompute portfolio with leverage
        portfolio = self._compute_portfolio(asset_signals, leverage)

        # 8. ML enhancement
        ml_enhanced = self._run_ml_enhancement(asset_signals, regime)

        # 9. Alerts & actions
        alerts = self._generate_alerts(asset_signals, regime, portfolio)
        next_actions = self._generate_next_actions(asset_signals, regime)

        # Build state
        elapsed = time.time() - start_time
        self.state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "engine_version": "1.0.0",
            "run_duration_seconds": round(elapsed, 2),
            "regime": regime,
            "confluence": confluence,
            "leverage_recommendation": leverage,
            "macro_score": macro_score,
            "assets": asset_signals,
            "portfolio": portfolio,
            "ml_enhanced": ml_enhanced,
            "alerts": alerts,
            "next_actions": next_actions,
        }

        logger.info(f"Engine run complete in {elapsed:.1f}s")
        return self.state

    def get_state(self) -> Dict[str, Any]:
        """Return current state dict. Load from disk if not in memory."""
        if not self.state and STATE_PATH.exists():
            with open(STATE_PATH) as f:
                self.state = json.load(f)
        return self.state

    def write_state(self, path: Optional[str] = None):
        """Atomic write state to JSON."""
        target = Path(os.path.expanduser(path)) if path else STATE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)

        # Atomic: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
            os.replace(tmp_path, str(target))
            logger.info(f"State written to {target}")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def generate_report(self) -> str:
        """Generate human-readable daily report."""
        s = self.state
        if not s:
            return "No state available. Run engine first."

        lines = []
        lines.append("=" * 60)
        lines.append(f"  MAESTRO DAILY REPORT — {s.get('timestamp', 'N/A')}")
        lines.append("=" * 60)
        lines.append("")

        # Regime & Confluence
        lines.append(f"🏛️  REGIME: {s.get('regime', 'N/A')}")
        conf = s.get("confluence", {})
        lines.append(f"📊 CONFLUENCE: {conf.get('score', 0)}/5")
        for sig_name, active in conf.get("signals", {}).items():
            icon = "✅" if active else "❌"
            lines.append(f"   {icon} {sig_name}")
        lines.append("")

        # Macro Score
        macro = s.get("macro_score", {})
        lines.append(f"🌍 MACRO SCORE: {macro.get('total', 0)}/6")
        for comp, active in macro.get("components", {}).items():
            icon = "✅" if active else "❌"
            lines.append(f"   {icon} {comp}")
        lines.append("")

        # Leverage
        lines.append(f"⚡ LEVERAGE: {s.get('leverage_recommendation', 0)}x")
        lines.append("")

        # Asset Signals
        lines.append("📈 ASSET SIGNALS:")
        lines.append("-" * 50)
        for asset, asig in s.get("assets", {}).items():
            sig = asig.get("signal", "FLAT")
            conf_val = asig.get("confidence", 0)
            price = asig.get("indicators", {}).get("price", 0)
            rsi = asig.get("indicators", {}).get("rsi", 0)
            icon = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}.get(sig, "⚪")
            lines.append(f"  {icon} {asset:5s} | {sig:5s} | conf={conf_val:.2f} | "
                         f"RSI={rsi:.0f} | ${price:,.2f}")
            if asig.get("dip_buy_active"):
                lines.append(f"         📉 DIP BUY active | pyramid={asig.get('pyramid_level', 0)}")
            trail = asig.get("trailing_stop_level", 0)
            if trail > 0:
                lines.append(f"         🛑 Trail stop: ${trail:,.2f}")
        lines.append("")

        # Portfolio
        port = s.get("portfolio", {})
        lines.append("💼 PORTFOLIO:")
        lines.append(f"   Total exposure: {port.get('total_exposure', 0):.2f}x")
        lines.append(f"   Long:  {port.get('long_exposure', 0):.2f}x")
        lines.append(f"   Short: {port.get('short_exposure', 0):.2f}x")
        risk = port.get("risk_metrics", {})
        lines.append(f"   Vol(20d): {risk.get('portfolio_vol_20d', 0):.3f}")
        lines.append(f"   DD from peak: {risk.get('current_dd_from_peak', 0):.2%}")
        if risk.get("circuit_breaker_active"):
            lines.append("   🚨 CIRCUIT BREAKER ACTIVE")
        lines.append("")

        # Weights
        weights = port.get("weights", {})
        if weights:
            lines.append("   Weights: " + " | ".join(f"{a}={w:.1%}" for a, w in weights.items()))
            lines.append("")

        # Alerts
        alerts = s.get("alerts", [])
        if alerts:
            lines.append("🔔 ALERTS:")
            for a in alerts:
                lines.append(f"   {a}")
            lines.append("")

        # Next Actions
        actions = s.get("next_actions", [])
        if actions:
            lines.append("📋 NEXT ACTIONS:")
            for a in actions:
                lines.append(f"   → {a}")
            lines.append("")

        # ML
        ml = s.get("ml_enhanced", {})
        lines.append(f"🤖 ML: regime={ml.get('regime_ml', 'N/A')} "
                     f"(conf={ml.get('regime_ml_confidence', 0):.2f})")
        lines.append("")
        lines.append(f"⏱️  Run time: {s.get('run_duration_seconds', 0):.1f}s")
        lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = MaestroEngine()
    state = engine.run_daily()
    engine.write_state()
    print(engine.generate_report())
