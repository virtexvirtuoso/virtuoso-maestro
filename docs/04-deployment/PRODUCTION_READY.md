# Production Deployment Checklist

**System:** Maestro Macro Momentum V3 + ML Regime + FF Bridge
**Date:** 2026-02-12

---

## Validated Components

| Component | Metric | Status |
|-----------|--------|--------|
| V3 Macro Momentum | OOS Sharpe 0.84, p=0.036 | VALIDATED |
| ML Regime Classifier | OOS Sharpe 1.33, Calmar 1.54 | VALIDATED |
| FF Bridge Overlay | Sharpe 1.06, 0.42 corr to V3 | VALIDATED |
| Integration Tests | 61/61 passing | VALIDATED |
| Engine Runtime | 1.2s end-to-end | VALIDATED |

---

## System Architecture

```
+------------------+     +------------------+     +------------------+
|  Data Sources    |     |  Maestro Engine   |     |  Output Layer    |
|                  |     |                  |     |                  |
|  yfinance  ------+---->|  Macro Score     |     |  FastAPI (7 ep) -+---> Dashboard
|  FRED      ------+---->|  Builder         |     |                  |
|  Fama-French ----+---->|       |          |     |  MCP Bridge  ----+---> Claude
|  Alpha Vantage --+---->|  Confluence      |     |                  |
|                  |     |  Engine          |     |  Freqtrade   ----+---> Exchange
+------------------+     |       |          |     |                  |
                         |  ML Regime       |     |  Cron Jobs   ----+---> Alerts
                         |  Classifier      |     |                  |
                         |       |          |     +------------------+
                         |  Signal Gen      |
                         |  + FF Bridge     |
                         |       |          |
                         |  Portfolio       |
                         |  Allocator       |
                         +------------------+
```

---

## Deployment Steps

### Step 1: API Server
```bash
cd ~/trading/maestro
source .venv/bin/activate
uvicorn maestro_api:app --host 127.0.0.1 --port 8090
```

### Step 2: Cron Scheduler
```bash
python maestro_cron.py  # Daily macro data refresh + signal generation
```

### Step 3: Freqtrade
```bash
cd ~/trading/freqtrade-maestro
freqtrade trade --config freqtrade_config.json \
  --strategy MacroMomentumV3Strategy
```

### Step 4: Whale Hunter Integration
- Connect Virtuoso Whale Hunter alerts to Maestro regime filter
- Only execute whale signals when regime is BULL or MILD_BULL

### Step 5: Go Live
- Start with 10% of target allocation
- Monitor for 2 weeks
- Scale to full allocation if metrics track expectations

---

## Files Needed on VPS

```
backend/
  maestro_engine.py
  maestro_api.py
  maestro_cron.py
  maestro_mcp_bridge.py
  datasource/
    __init__.py
    yfinance_loader.py
    fred_loader.py
    factor_loader.py
    alphavantage_loader.py
    providers.py
  strategies/composite/
    __init__.py
    mega_strategy_v3.py
    macro_score_builder.py
  ml/
    __init__.py
    feature_engine.py
    regime_classifier.py
  freqtrade/
    macro_momentum_v3_strategy.py
    macro_data_provider.py
    freqtrade_config.json
data/
  optimization/
    mega_v3_best_params.json
  live/
    maestro_state.json
```

---

## Environment Variables

```bash
FRED_API_KEY=<your-fred-key>
ALPHAVANTAGE_API_KEY=<your-av-key>
MAESTRO_PORT=8090
MAESTRO_ENV=production
MAESTRO_LOG_LEVEL=INFO
```

---

## Monitoring

| Check | Frequency | Alert Threshold |
|-------|-----------|-----------------|
| Engine execution | Every run | > 5s or failure |
| Regime change | Daily | Any regime transition |
| Drawdown | Daily | > -15% from peak |
| Data freshness | Daily | FRED data > 48h stale |
| API health | 5 min | Non-200 response |
| Position drift | Daily | > 5% from target weights |

---

## Rollback Plan

1. Kill Freqtrade: `systemctl stop freqtrade`
2. Close all positions manually on exchange
3. Revert to Virtuoso-only signals
4. Investigate, fix, re-validate before redeployment

---

*Deploy carefully. Scale gradually. Trust the walk-forward.*
