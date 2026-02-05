# Maestro Documentation

*The Master Conductor of Trading Strategies*

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Project Timeline](project/TIMELINE.md) | Full history from Filos (2020) to Maestro (2026) |
| [Architecture Pipeline](architecture/PIPELINE.md) | Maestro → Jesse → Freqtrade flow |
| [Walk-Forward Guide](guides/walk-forward-analysis.md) | WFA methodology and time series splits |
| [Branding](project/BRANDING.md) | Logo, colors, naming conventions |

---

## Directory Structure

```
docs/
├── README.md                 # You are here
│
├── project/                  # Project-level documentation
│   ├── TIMELINE.md           # Complete project history
│   ├── BRANDING.md           # Brand guidelines & assets
│   └── ROADMAP.md            # Future plans (TODO)
│
├── architecture/             # Technical design & planning
│   ├── PIPELINE.md           # System pipeline overview
│   ├── MODERNIZATION.md      # V1 → V2 modernization notes
│   ├── CONSENSUS_MATRIX.md   # Multi-strategy consensus logic
│   ├── IMPLEMENTATION_PLAN.md
│   └── IMPLEMENTATION_PLAN_V2.md
│
├── guides/                   # How-to guides & methodology
│   └── walk-forward-analysis.md
│
├── research/                 # Academic papers & references
│   ├── crypto-trading/       # Trading strategy papers
│   ├── ml-strategies/        # ML/DL approach papers
│   └── risk-management/      # Risk & volatility papers
│
├── reports/                  # Generated reports & validation
│   └── PRD_VALIDATION_REPORT.md
│
└── legacy/                   # Filos (2020) archive [GITIGNORED - local only]
    ├── README.md             # Archive index
    ├── original-codebase/    # Complete Oct 2020 codebase
    ├── videos/               # 13 demo videos (~3.7 GB)
    ├── sketches/             # Workflow diagrams
    ├── outlines/             # Project specs
    ├── strategies/           # 22 Freqtrade strategies
    └── branding/             # Logo & assets
```

---

## Research Library

### Crypto Trading (9 papers)

| Paper | Year | Key Contribution |
|-------|------|------------------|
| **Glucksmann** (ETH Zurich) | 2019 | BBW + VLI → FernandoStrategy basis |
| Gort et al. | 2022 | Deep RL backtest overfitting prevention |
| Jabbar & Jalil | 2024 | 41 ML models comparison for BTC |
| Jiang et al. | 2017 | CNN/RNN/LSTM portfolio management |
| Tadi & Kortchmeski | 2021 | Dynamic cointegration pairs trading |
| Zhang et al. | 2020 | GA optimization (Sharpe/Sterling) |
| Chio | 2022 | MACD trading strategies |
| Sezer et al. | 2017 | ANN stock trading with TA |
| GT Score | 2026 | Reducing overfitting techniques |

### ML Strategies (4 papers)

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Jiang et al. | 2016 | Crypto portfolio Deep RL |
| Tumpa et al. | 2024 | RNN crypto prediction |
| Herremans et al. | 2022 | Bitcoin volatility & whale activity |
| Chen et al. | 2025 | Dynamic grid trading |

### Risk Management (3 papers)

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Letteri et al. | 2023 | VolTS volatility trading |
| Letteri et al. | 2022 | DNN forward testing |
| Matic et al. | 2021 | Hedging crypto with options |

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

1. **Guides** → Add to `guides/` with kebab-case filenames
2. **Papers** → Add to appropriate `research/` subfolder
3. **Reports** → Generated outputs go in `reports/`
4. **Architecture** → Major design docs in `architecture/`

---

*Last updated: February 5, 2026*
