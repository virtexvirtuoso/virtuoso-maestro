# Maestro Documentation

*The Master Conductor of Trading Strategies*

## Directory Structure

```
docs/
├── architecture/          # System design & implementation plans
│   ├── IMPLEMENTATION_PLAN.md
│   ├── IMPLEMENTATION_PLAN_V2.md
│   ├── MODERNIZATION.md
│   ├── PIPELINE.md
│   └── CONSENSUS_MATRIX.md
│
├── guides/                # How-to guides & methodology
│   └── Walk-Forward Analysis and Time Series.md
│
├── papers/                # Academic papers & research
│   ├── crypto-trading/    # Cryptocurrency trading strategies
│   ├── ml-strategies/     # Machine learning approaches
│   └── risk-management/   # Risk & volatility management
│
├── BRANDING.md            # Project branding guidelines
└── README.md
```

---

## Papers Library

### Crypto Trading Strategies

| Paper | Author(s) | Year | Key Topics |
|-------|-----------|------|------------|
| **Glucksmann_Bitcoin_Trading_Strategies_2019.pdf** | Alain Glucksmann (ETH Zurich) | 2019 | BBW, VLI, walk-forward optimization - **BASIS FOR FERNANDOSTRATEGY** |
| Gort_2022_Deep_RL_Backtest_Overfitting.pdf | Gort et al. | 2022 | Deep RL, backtest overfitting prevention |
| Jabbar_2024_ML_Models_Bitcoin_Trading.pdf | Jabbar & Jalil | 2024 | 41 ML models comparison for BTC |
| Jiang_2017_Deep_RL_Portfolio_Management.pdf | Jiang et al. | 2017 | CNN/RNN/LSTM portfolio management |
| Tadi_2021_Cointegration_Pairs_Trading.pdf | Tadi & Kortchmeski | 2021 | Dynamic cointegration pairs trading |
| Chio_2022_MACD_Trading_Strategies.pdf | Pat Tong Chio | 2022 | MACD strategy comparison |
| Sezer_2017_ANN_Stock_Trading_TA.pdf | Sezer et al. | 2017 | ANN with technical analysis |
| Zhang_2020_GA_Sharpe_Sterling_Optimization.pdf | Zhang & Khushi | 2020 | GA for Sharpe/Sterling ratio optimization |

### ML Strategies

| Paper | Author(s) | Year | Key Topics |
|-------|-----------|------|------------|
| Tumpa_2024_RNN_Crypto_Prediction.pdf | Tumpa & Maduranga | 2024 | RNN for real-time prediction |
| Jiang_2016_Crypto_Portfolio_Deep_RL.pdf | Jiang & Liang | 2016 | Deep RL crypto portfolio |
| Chen_2025_Dynamic_Grid_Trading.pdf | Chen et al. | 2025 | Dynamic grid trading |
| Herremans_2022_Bitcoin_Volatility_Whale.pdf | Herremans & Low | 2022 | Whale transactions, volatility forecasting |

### Risk Management

| Paper | Author(s) | Year | Key Topics |
|-------|-----------|------|------------|
| Letteri_2023_VolTS_Volatility_Trading.pdf | Letteri | 2023 | Volatility-based trading system |
| Letteri_2022_DNN_Forward_Testing.pdf | Letteri et al. | 2022 | DNN forward testing validation |
| Matic_2021_Hedging_Crypto_Options.pdf | Matic et al. | 2021 | Hedging crypto options |

---

## Key Concepts from Papers

### Walk-Forward Optimization (Glucksmann 2019)
The foundation of Maestro's approach to preventing overfitting:
1. Split data into N rolling periods
2. Train on period i, test on period i+1
3. Optimal params selected by Sharpe + VWR
4. Aggregate out-of-sample results

### Volatility Level Index (VLI)
Custom indicator from Glucksmann thesis:
- BBW = (Upper Band - Lower Band) / Middle Band
- VLI_fast = SMA(BBW, 20)
- VLI_slow = SMA(BBW, 100)
- Low volatility regime: VLI_fast < VLI_slow

### Backtest Overfitting Prevention (Gort 2022)
- Use combinatorially-symmetric cross-validation
- Multiple testing correction
- Out-of-sample validation mandatory

---

## Adding New Papers

1. Download PDF to appropriate subfolder
2. Use naming convention: `Author_Year_ShortTitle.pdf`
3. Update this README with paper details
4. Tag relevant Maestro strategies that could benefit

---

*Last updated: 2026-02-05*
