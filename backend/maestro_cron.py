#!/usr/bin/env python3
"""
Maestro Daily Cron
Runs the engine, updates state, and optionally triggers alerts.
Designed to be called by systemd timer or crontab.

Usage:
    python maestro_cron.py              # full run
    python maestro_cron.py --dry-run    # compute but don't write
    
Crontab example (6 AM UTC daily):
    0 6 * * * cd ~/Desktop/maestro/backend && ./venv/bin/python maestro_cron.py >> ~/Desktop/maestro/logs/cron.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maestro_engine import MaestroEngine

LOG_DIR = Path(os.path.expanduser("~/Desktop/maestro/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"maestro_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger("maestro.cron")


def check_alerts(state: dict) -> list:
    """Check state for critical alert conditions."""
    alerts = state.get("alerts", [])
    critical = []

    # Circuit breaker
    risk = state.get("portfolio", {}).get("risk_metrics", {})
    if risk.get("circuit_breaker_active"):
        critical.append("🚨 CIRCUIT BREAKER ACTIVE — DD > 20%")

    # Regime change detection
    prev_state_path = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state_prev.json"))
    if prev_state_path.exists():
        with open(prev_state_path) as f:
            prev = json.load(f)
        if prev.get("regime") != state.get("regime"):
            critical.append(
                f"🔄 REGIME CHANGE: {prev.get('regime')} → {state.get('regime')}"
            )

    # High-confidence signals
    for asset, sig in state.get("assets", {}).items():
        if sig.get("confidence", 0) > 0.8 and sig.get("dip_buy_active"):
            critical.append(f"🟢 HIGH-CONF DIP BUY: {asset} (conf={sig['confidence']:.2f})")

    return critical + alerts


def send_alerts(alerts: list):
    """Log alerts. Integration with Telegram/OpenClaw can be added here."""
    logger.info("=" * 40)
    logger.info("ALERTS:")
    for a in alerts:
        logger.warning(f"  {a}")
    logger.info("=" * 40)


def save_prev_state():
    """Copy current state to prev for regime change detection."""
    state_path = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state.json"))
    prev_path = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state_prev.json"))
    if state_path.exists():
        import shutil
        shutil.copy2(state_path, prev_path)


def main():
    parser = argparse.ArgumentParser(description="Maestro Daily Cron")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write state")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Maestro Cron starting at {datetime.utcnow().isoformat()}Z")
    logger.info("=" * 60)

    try:
        # Save previous state for change detection
        save_prev_state()

        engine = MaestroEngine()
        state = engine.run_daily()

        if not args.dry_run:
            engine.write_state()
            logger.info("State written successfully")
        else:
            logger.info("DRY RUN — state not written")

        report = engine.generate_report()
        print(report)

        # Check alerts
        alerts = check_alerts(state)
        if alerts:
            send_alerts(alerts)

        logger.info("Maestro Cron completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Maestro Cron FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
