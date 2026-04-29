# Maestro -- Quantitative Crypto Research Platform

Systematic macro-momentum trading system for cryptocurrency perpetual futures. Built on the discovery that M2 money supply acceleration is the dominant predictor of crypto returns.

## Quick Start

```bash
cd ~/Desktop/maestro/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the engine
python maestro_engine.py

# Start the API
uvicorn maestro_api:app --port 8001

# Run integration tests
python maestro_integration_test.py
```

## Architecture

```
maestro/
  backend/
    datasource/       # 4 data providers (yfinance, FRED, FF, Alpha Vantage)
    strategies/
      composite/      # Macro momentum V1-V4 + 6 legacy composites
    ml/               # Regime classifier, feature engine, signal weighter
    freqtrade/        # IStrategy port for live trading
    research/         # Ultrathink scripts, validation studies
    maestro_engine.py # Core engine (1.2s execution)
    maestro_api.py    # FastAPI (7 endpoints, 61 tests)
    maestro_mcp_bridge.py  # Claude MCP integration
  data/
    backtest_results/ # JSON results for all strategies
    optimization/     # Optuna studies and best params
    research/         # Research notes and blueprints
    derivatives/      # Coinalyze data (65 tokens, 2yr)
  docs/               # Reports, architecture, guides
  frontend/           # React dashboard
```

## Strategies

**66 original strategies** across 6 categories (technical, scalping, momentum, composite, derivatives, hybrids), plus:

| Strategy | Sharpe | OOS Sharpe | Max DD | Status |
|----------|--------|------------|--------|--------|
| V3 Macro Momentum | 1.71 | 0.84 (p=0.036) | -5.5% | Production |
| V4 Multi-Module | 1.70 | 0.256 | -30.4% | Research only |
| ML Regime Classifier | -- | 1.33 | -18.2% | Production overlay |
| FF Bridge | 1.06 | +3.66% WF | -- | Production overlay |

## Data Sources

| Provider | Data | Update |
|----------|------|--------|
| yfinance | OHLCV, DXY, Gold, HYG, 10Y | Daily |
| FRED | M2, CPI, Unemployment, Fed Funds | Monthly |
| Fama-French | Mkt-RF, SMB, HML, RMW, CMA | Daily |
| Alpha Vantage | GDP, Treasury Yield, Inflation | Monthly |

## Key Results

- **M2 acceleration** is THE edge: BTC returns 102.8%/yr when accelerating vs 2.1%/yr when not
- **Walk-forward validated**: 14/14 folds pass, OOS Sharpe 0.84, p-value 0.036
- **Crash protection**: 0% loss during COVID, May 2021, FTX
- **Short alpha**: +23.1% during 2022 bear market

## Documentation

- [Mega Strategy Report](docs/MEGA_STRATEGY_REPORT.md) -- Full research report
- [File Inventory](docs/FILE_INVENTORY.md) -- Complete file listing
- [Production Ready](docs/PRODUCTION_READY.md) -- Deployment checklist
- [Integration Architecture](docs/INTEGRATION_ARCHITECTURE.md) -- System design
- [Research Findings](docs/RESEARCH_FINDINGS.md) -- Historical findings

## Dashboard

Live at https://virtuosocrypto.com/quant/

---

*Virtuoso Research Division -- 2026*
