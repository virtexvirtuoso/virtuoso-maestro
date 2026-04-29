#!/usr/bin/env python3
"""
Maestro Integration Test
Tests the full pipeline: engine → state → API → MCP bridge
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATE_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/live/maestro_state.json"))
API_BASE = "http://localhost:8787"
API_KEY = os.environ.get("MAESTRO_API_KEY", "maestro-default-key")

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


def test_engine():
    """Test 1: Run the engine with real data."""
    print("\n" + "=" * 50)
    print("TEST 1: Engine Run")
    print("=" * 50)

    from maestro_engine import MaestroEngine

    engine = MaestroEngine()
    state = engine.run_daily()

    check("State is dict", isinstance(state, dict))
    check("Has timestamp", "timestamp" in state)
    check("Has regime", state.get("regime") in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"])
    check("Has confluence", isinstance(state.get("confluence"), dict))
    check("Confluence score 0-5", 0 <= state.get("confluence", {}).get("score", -1) <= 5)
    check("Has assets", isinstance(state.get("assets"), dict) and len(state["assets"]) > 0)
    check("Has portfolio", isinstance(state.get("portfolio"), dict))
    check("Has leverage", isinstance(state.get("leverage_recommendation"), (int, float)))
    check("Has macro_score", isinstance(state.get("macro_score"), dict))
    check("Has ml_enhanced", isinstance(state.get("ml_enhanced"), dict))

    # Validate asset signals
    for asset, sig in state.get("assets", {}).items():
        check(f"{asset} has signal", sig.get("signal") in ["LONG", "SHORT", "FLAT"])
        check(f"{asset} has confidence", 0 <= sig.get("confidence", -1) <= 1)
        check(f"{asset} has indicators", isinstance(sig.get("indicators"), dict))

    # Write state
    engine.write_state()
    check("State file written", STATE_PATH.exists())

    # Validate JSON
    with open(STATE_PATH) as f:
        loaded = json.load(f)
    check("State JSON valid", loaded.get("timestamp") == state["timestamp"])

    # Generate report
    report = engine.generate_report()
    check("Report generated", len(report) > 100)
    check("Report has regime", "REGIME" in report)

    return state, report


def test_api():
    """Test 3: Start API and test endpoints."""
    print("\n" + "=" * 50)
    print("TEST 2: API Endpoints")
    print("=" * 50)

    import requests

    # Start API server in background
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "maestro_api:app", "--host", "0.0.0.0", "--port", "8787"],
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for startup
    time.sleep(3)

    headers = {"X-API-Key": API_KEY}

    try:
        # Health (no auth)
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        check("GET /api/health", r.status_code == 200)
        check("Health status ok", r.json().get("status") == "ok")

        # Auth required
        r = requests.get(f"{API_BASE}/api/regime", timeout=5)
        check("Auth enforced (no key)", r.status_code == 401)

        # Regime
        r = requests.get(f"{API_BASE}/api/regime", headers=headers, timeout=5)
        check("GET /api/regime", r.status_code == 200)
        data = r.json()
        check("Regime has regime field", "regime" in data)
        check("Regime has confluence", "confluence" in data)

        # Signals
        r = requests.get(f"{API_BASE}/api/signals", headers=headers, timeout=5)
        check("GET /api/signals", r.status_code == 200)
        check("Signals has assets", "assets" in r.json())

        # Signal for specific asset
        r = requests.get(f"{API_BASE}/api/signals?asset=BTC", headers=headers, timeout=5)
        check("GET /api/signals?asset=BTC", r.status_code == 200)
        check("BTC signal present", "signal" in r.json())

        # Allocation
        r = requests.get(f"{API_BASE}/api/allocation", headers=headers, timeout=5)
        check("GET /api/allocation", r.status_code == 200)
        check("Allocation has portfolio", "portfolio" in r.json())

        # Risk
        r = requests.get(f"{API_BASE}/api/risk", headers=headers, timeout=5)
        check("GET /api/risk", r.status_code == 200)
        check("Risk has risk_metrics", "risk_metrics" in r.json())

        # State
        r = requests.get(f"{API_BASE}/api/state", headers=headers, timeout=5)
        check("GET /api/state", r.status_code == 200)
        check("State has all keys", all(k in r.json() for k in ["regime", "assets", "portfolio"]))

        # Report
        r = requests.get(f"{API_BASE}/api/report", headers=headers, timeout=5)
        check("GET /api/report", r.status_code == 200)
        check("Report has content", len(r.json().get("report", "")) > 50)

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_bridge():
    """Test 4: MCP tool definitions."""
    print("\n" + "=" * 50)
    print("TEST 3: MCP Bridge")
    print("=" * 50)

    from maestro_mcp_bridge import MCP_TOOLS, HANDLER_MAP, handle_mcp_call

    check("5 MCP tools defined", len(MCP_TOOLS) == 5)

    for tool in MCP_TOOLS:
        check(f"Tool {tool['name']} has description", len(tool.get("description", "")) > 10)
        check(f"Tool {tool['name']} has inputSchema", "inputSchema" in tool)
        check(f"Tool {tool['name']} handler exists", tool["handler"] in HANDLER_MAP)

    # Test direct state read (API not running)
    result = handle_mcp_call("maestro.get_regime")
    check("MCP regime call works", "regime" in result or "regime" in str(result))


def main():
    global passed, failed

    print("🎼 MAESTRO INTEGRATION TEST")
    print("=" * 50)

    state, report = test_engine()

    try:
        import requests
        test_api()
    except ImportError:
        print("\n⚠️  requests not installed, skipping API tests")
        # Install and retry
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests
        test_api()

    test_mcp_bridge()

    # Summary
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("🎉 ALL TESTS PASSED")
    else:
        print(f"⚠️  {failed} test(s) failed")

    # Print the daily report
    print("\n" + report)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
