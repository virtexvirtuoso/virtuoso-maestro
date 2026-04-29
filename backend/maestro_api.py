"""
Maestro API — REST endpoints for MCP integration
Reads from maestro_state.json (updated by daily cron)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

app = FastAPI(
    title="Maestro Quant API",
    version="1.0",
    description="Unified quant signals for MCP/Freqtrade consumption",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("MAESTRO_API_KEY", "maestro-default-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ---------------------------------------------------------------------------
# State loader
# ---------------------------------------------------------------------------

STATE_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state.json"))

_state_cache = {"data": None, "mtime": 0.0}


def load_state() -> dict:
    """Load state with simple file-mtime caching."""
    if not STATE_PATH.exists():
        raise HTTPException(status_code=503, detail="State file not found. Run engine first.")
    mtime = STATE_PATH.stat().st_mtime
    if _state_cache["data"] is None or mtime > _state_cache["mtime"]:
        with open(STATE_PATH) as f:
            _state_cache["data"] = json.load(f)
        _state_cache["mtime"] = mtime
    return _state_cache["data"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    """System health check (no auth required)."""
    state_exists = STATE_PATH.exists()
    state_age = None
    if state_exists:
        state_age = (datetime.utcnow() - datetime.utcfromtimestamp(STATE_PATH.stat().st_mtime)).total_seconds()
    return {
        "status": "ok" if state_exists else "degraded",
        "state_file_exists": state_exists,
        "state_age_seconds": round(state_age, 0) if state_age else None,
        "state_stale": state_age > 86400 if state_age else True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/regime")
async def regime(_key: str = Depends(verify_api_key)):
    """Current regime + confluence."""
    state = load_state()
    return {
        "regime": state.get("regime"),
        "confluence": state.get("confluence"),
        "macro_score": state.get("macro_score"),
        "leverage_recommendation": state.get("leverage_recommendation"),
        "timestamp": state.get("timestamp"),
    }


@app.get("/api/signals")
async def signals(asset: Optional[str] = None, _key: str = Depends(verify_api_key)):
    """Per-asset signals with confidence."""
    state = load_state()
    assets = state.get("assets", {})
    if asset:
        asset = asset.upper()
        if asset not in assets:
            raise HTTPException(status_code=404, detail=f"Asset {asset} not found")
        return {"asset": asset, **assets[asset], "timestamp": state.get("timestamp")}
    return {
        "regime": state.get("regime"),
        "assets": assets,
        "timestamp": state.get("timestamp"),
    }


@app.get("/api/allocation")
async def allocation(_key: str = Depends(verify_api_key)):
    """Portfolio weights and exposure."""
    state = load_state()
    return {
        "portfolio": state.get("portfolio"),
        "leverage_recommendation": state.get("leverage_recommendation"),
        "timestamp": state.get("timestamp"),
    }


@app.get("/api/risk")
async def risk(_key: str = Depends(verify_api_key)):
    """Drawdown, vol, circuit breakers."""
    state = load_state()
    portfolio = state.get("portfolio", {})
    return {
        "risk_metrics": portfolio.get("risk_metrics"),
        "alerts": state.get("alerts"),
        "regime": state.get("regime"),
        "leverage_recommendation": state.get("leverage_recommendation"),
        "timestamp": state.get("timestamp"),
    }


@app.get("/api/state")
async def full_state(_key: str = Depends(verify_api_key)):
    """Full state dump."""
    return load_state()


@app.get("/api/report")
async def report(_key: str = Depends(verify_api_key)):
    """Human-readable daily report."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from maestro_engine import MaestroEngine

    engine = MaestroEngine()
    engine.state = load_state()
    return {"report": engine.generate_report(), "timestamp": engine.state.get("timestamp")}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
