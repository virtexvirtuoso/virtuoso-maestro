# Maestro Documentation

*The Master Conductor of Trading Strategies*

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Project Timeline](10-maintenance/project/TIMELINE.md) | Full history from Filos (2020) to Maestro (2026) |
| [Architecture Pipeline](03-developer-guide/architecture/PIPELINE.md) | Maestro → Jesse → Freqtrade flow |
| [Walk-Forward Guide](06-reference/guides/walk-forward-analysis.md) | WFA methodology and time series splits |
| [Branding](10-maintenance/project/BRANDING.md) | Logo, colors, naming conventions |

### Data & Research

| Document | Description |
|----------|-------------|
| [Data Inventory](06-reference/DATA_INVENTORY.md) | Available OHLCV, funding, OI data sources |
| [Strategies Roadmap](08-features/trading/STRATEGIES_ROADMAP.md) | 30 strategies across 3 phases |
| [Open Interest Data](08-features/trading/OPEN_INTEREST_DATA.md) | Free OI sources + why proxies don't work |
| [Synthetic Proxies](08-features/trading/SYNTHETIC_PROXIES.md) | Funding/liquidation proxy methods (validated) |
| [Quant Problem Summary](08-features/trading/QUANT_PROBLEM_SUMMARY.md) | Research problem + action plan |

---

## Directory Structure

```
docs/
├── README.md                       # You are here
│
├── 02-user-guide/                  # End-user documentation & UI
│   ├── UX_IMPROVEMENT_ROADMAP.md
│   └── UX_IMPROVEMENTS.md
│
├── 03-developer-guide/             # Architecture & development
│   ├── FRONTEND_IMPLEMENTATION_PLAN.md
│   ├── INTEGRATION_ARCHITECTURE.md
│   ├── PATENT_TECHNICAL_SPEC.md
│   └── architecture/
│       ├── PIPELINE.md
│       ├── MODERNIZATION.md
│       ├── CONSENSUS_MATRIX.md
│       ├── IMPLEMENTATION_PLAN.md
│       └── IMPLEMENTATION_PLAN_V2.md
│
├── 04-deployment/                  # Production & live trading
│   ├── PRODUCTION_READY.md
│   └── TIER1_LIVE_TRADING_PLAN.md
│
├── 05-operations/                  # Monitoring & services
│   ├── DISTRIBUTION_SHIFT_MONITOR.md
│   └── TICK_COLLECTOR.md
│
├── 06-reference/                   # Data, configuration, guides
│   ├── DATA_INVENTORY.md
│   ├── FILE_INVENTORY.md
│   └── guides/
│       └── walk-forward-analysis.md
│
├── 07-technical/                   # Fixes, incidents, investigations
│   └── investigations/
│       └── V3_DIAGNOSIS_AND_V3.1_PLAN.md
│
├── 08-features/                    # Trading strategies & capabilities
│   └── trading/
│       ├── STRATEGIES_ROADMAP.md
│       ├── STRATEGY_EVOLUTION_ROADMAP.md
│       ├── PROFITABILITY_ROADMAP.md
│       ├── OPEN_INTEREST_DATA.md
│       ├── SYNTHETIC_PROXIES.md
│       └── QUANT_PROBLEM_SUMMARY.md
│
├── 09-reports/                     # Research results & validation
│   ├── audits/
│   │   ├── ACADEMIC_PAPERS_DRAFT.md
│   │   ├── ULTRATHINK_REVIEW_V4.md
│   │   └── PRD_VALIDATION_REPORT.md
│   ├── optimization/
│   │   ├── BUILDING_THE_EDGE.md
│   │   ├── COINGLASS_RESEARCH_REPORT.md
│   │   ├── MEGA_STRATEGY_REPORT.md
│   │   └── optuna-optimization-report-2026-02-06.md
│   └── validation/
│       ├── BACKTEST_VALIDATION_REPORT.md
│       ├── ONCHAIN_DEEP_TESTS_2026-02-17.md
│       ├── RESEARCH_FINDINGS.md
│       ├── RESEARCH_FINDINGS_2026.md
│       ├── RESEARCH_SYNTHESIS.md
│       ├── THE_EDGE_RESEARCH.md
│       └── WHY_LONG_SHORT_FAILS_CRYPTO.md
│
├── 10-maintenance/                 # Branding, project history
│   └── project/
│       ├── BRANDING.md
│       └── TIMELINE.md
│
└── 11-archive/                     # Superseded docs & academic papers
    ├── CROSS_ASSET_MOMENTUM_REPORT.md
    └── research/
        ├── STRATEGY_IMPLEMENTATION_PLAN.md
        ├── crypto-trading/         # 9 PDFs
        ├── ml-strategies/          # 4 PDFs
        ├── risk-management/        # 3 PDFs
        └── walk-forward/           # WFA tutorials & papers
```

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    MAESTRO      │     │     JESSE       │     │   FREQTRADE     │
│   (Research)    │ ──▶ │  (Simulation)   │ ──▶ │  (Execution)    │
│                 │     │                 │     │                 │
│ • 30 strategies │     │ • Paper trading │     │ • Live trading  │
│ • VectorBT      │     │ • Validation    │     │ • Risk mgmt     │
│ • Optuna        │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Contributing

| Type | Location | Naming |
|------|----------|--------|
| User guides | `02-user-guide/` | `FEATURE_NAME.md` |
| Architecture | `03-developer-guide/architecture/` | `TOPIC.md` |
| Deployment | `04-deployment/` | `TOPIC.md` |
| Reference | `06-reference/` | `TOPIC.md` |
| Bug fixes/incidents | `07-technical/fixes/` | `YYYY-MM-DD-description.md` |
| Strategy features | `08-features/trading/` | `FEATURE_NAME.md` |
| Research reports | `09-reports/{audits,optimization,validation}/` | `REPORT_NAME.md` |
| Obsolete docs | `11-archive/` | Move with note at top |

---

*Last updated: February 17, 2026*
