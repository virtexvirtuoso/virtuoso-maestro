"""
MCP Bridge — Defines MCP tools that bridge Maestro signals into Virtuoso's MCP.

These tools can be registered with any MCP-compatible server.
Each tool reads from the Maestro API (or directly from state JSON).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

MAESTRO_API_BASE = os.environ.get("MAESTRO_API_URL", "http://localhost:8787")
MAESTRO_API_KEY = os.environ.get("MAESTRO_API_KEY", "maestro-default-key")
STATE_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state.json"))


def _load_state_direct() -> dict:
    """Load state directly from JSON (fallback if API is down)."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def _api_get(endpoint: str) -> dict:
    """Call Maestro API with fallback to direct file read."""
    try:
        r = requests.get(
            f"{MAESTRO_API_BASE}{endpoint}",
            headers={"X-API-Key": MAESTRO_API_KEY},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        # Fallback to direct state read
        return _load_state_direct()


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def fetch_regime(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get current market regime with confluence score and signal breakdown."""
    data = _api_get("/api/regime")
    return {
        "regime": data.get("regime"),
        "confluence_score": data.get("confluence", {}).get("score"),
        "confluence_signals": data.get("confluence", {}).get("signals"),
        "macro_score": data.get("macro_score", {}).get("total"),
        "leverage_recommendation": data.get("leverage_recommendation"),
        "timestamp": data.get("timestamp"),
    }


def fetch_signals(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get per-asset trading signals."""
    asset = (params or {}).get("asset")
    endpoint = f"/api/signals?asset={asset}" if asset else "/api/signals"
    return _api_get(endpoint)


def fetch_allocation(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get recommended portfolio allocation."""
    return _api_get("/api/allocation")


def fetch_risk(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get current risk metrics."""
    return _api_get("/api/risk")


def fetch_report(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get daily report."""
    return _api_get("/api/report")


# ---------------------------------------------------------------------------
# MCP Tool Definitions (for Virtuoso MCP registration)
# ---------------------------------------------------------------------------

HANDLER_MAP = {
    "fetch_regime": fetch_regime,
    "fetch_signals": fetch_signals,
    "fetch_allocation": fetch_allocation,
    "fetch_risk": fetch_risk,
    "fetch_report": fetch_report,
}

MCP_TOOLS = [
    {
        "name": "maestro.get_regime",
        "description": (
            "Get current market regime (BULL/MILD_BULL/NEUTRAL/BEAR/ACCUMULATION) "
            "with confluence score (0-5) and signal breakdown (M2 acceleration, "
            "real-time liquidity, yield curve, cross-asset momentum, crypto momentum)"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": "fetch_regime",
    },
    {
        "name": "maestro.get_signals",
        "description": (
            "Get per-asset trading signals (LONG/SHORT/FLAT) with confidence scores, "
            "entry/exit levels, RSI, trend, momentum, position sizing, and pyramid levels. "
            "Optionally filter by asset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset": {
                    "type": "string",
                    "description": "Optional asset filter: BTC, ETH, SOL, LINK",
                    "enum": ["BTC", "ETH", "SOL", "LINK"],
                },
            },
            "required": [],
        },
        "handler": "fetch_signals",
    },
    {
        "name": "maestro.get_allocation",
        "description": (
            "Get recommended portfolio allocation with per-asset weights, "
            "total/long/short exposure, and adaptive leverage recommendation"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": "fetch_allocation",
    },
    {
        "name": "maestro.get_risk_status",
        "description": (
            "Get current risk metrics: portfolio volatility, drawdown from peak, "
            "circuit breaker status, and active alerts"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": "fetch_risk",
    },
    {
        "name": "maestro.get_daily_report",
        "description": (
            "Get Maestro's daily market analysis and trading recommendations "
            "as a human-readable report with regime, signals, portfolio, and alerts"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": "fetch_report",
    },
]


def handle_mcp_call(tool_name: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Route an MCP tool call to its handler."""
    for tool in MCP_TOOLS:
        if tool["name"] == tool_name:
            handler_fn = HANDLER_MAP.get(tool["handler"])
            if handler_fn:
                return handler_fn(params)
            raise ValueError(f"Handler not found: {tool['handler']}")
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    print("MCP Tool Definitions:")
    print(json.dumps(MCP_TOOLS, indent=2))
    print(f"\nTotal tools: {len(MCP_TOOLS)}")
