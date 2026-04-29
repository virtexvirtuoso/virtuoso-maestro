# File Inventory -- Maestro Research Project

**Generated:** 2026-02-12
**Base Path:** `~/Desktop/maestro/`

---

## Data Loaders (`backend/datasource/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `yfinance_loader.py` | OHLCV and cross-asset prices via yfinance |
| `fred_loader.py` | FRED macro series (M2, CPI, unemployment) |
| `factor_loader.py` | Fama-French 5-factor model loader |
| `alphavantage_loader.py` | Alpha Vantage economic indicators |
| `providers.py` | Unified data provider interface |
| `binance_batch_downloader.py` | Binance historical data downloader |
| `binance_batch_downloader_worker.py` | Binance download worker |
| `bitmex_batch_downloader.py` | BitMEX historical data downloader |
| `bitmex_batch_downloader_worker.py` | BitMEX download worker |
| `ccxt_batch_downloader.py` | CCXT multi-exchange downloader |
| `funding_rate_downloader.py` | Funding rate data collector |

## Strategies (`backend/strategies/composite/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `macro_momentum.py` | V1 Macro Momentum (Golden Cross) |
| `macro_momentum_v2.py` | V2 Dip-Buyer with adaptive leverage |
| `macro_momentum_portfolio.py` | V2 Portfolio multi-asset allocation |
| `macro_score_builder.py` | Macro score construction module |
| `mega_strategy_v3.py` | V3 Mega Strategy (production) |
| `mega_strategy_v4.py` | V4 Multi-Module expansion |
| `fernando.py` | Fernando composite strategy |
| `combined_bin_cluc.py` | CombinedBinCluc strategy |
| `smooth_operator.py` | SmoothOperator strategy |
| `ema_skip_pump.py` | EMA Skip Pump strategy |
| `low_bb.py` | Low Bollinger Band strategy |
| `reinforced_average.py` | Reinforced Average strategy |

## ML Layer (`backend/ml/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `feature_engine.py` | Feature construction from raw data |
| `regime_classifier.py` | ML regime classifier (OOS Sharpe 1.33) |
| `signal_weighter.py` | Dynamic signal weight optimization |
| `entry_timer.py` | Entry timing refinement |
| `ensemble_strategy_selector.py` | Multi-strategy ensemble selection |

## Backtests (`backend/`)

| File | Purpose |
|------|---------|
| `backtest_mega_v3.py` | V3 Mega Strategy backtest |
| `backtest_mega_v4.py` | V4 Multi-Module backtest |
| `backtest_macro_momentum_v2.py` | V2 backtest |
| `backtest_portfolio_final.py` | Portfolio allocation backtest |
| `backtest_ml_enhanced.py` | ML-enhanced backtest |
| `backtest_macro_crypto.py` | Macro-crypto overlay backtest |
| `backtest_macro_overlay.py` | Macro overlay backtest |
| `backtest_scalping_full.py` | Full scalping strategy suite |
| `backtest_hybrid_strategies.py` | Hybrid strategy evaluation |
| `backtest_gap_strategies.py` | Gap strategy research |
| `backtest_smart_funding.py` | Funding rate strategy |
| `backtest_vol_regime.py` | Volatility regime strategy |
| `backtest_saylor_signal.py` | Saylor Signal strategy |
| `backtest_cascade.py` | Cascade liquidation strategy |
| `backtest_conductor.py` | Conductor strategy |
| `backtest_combined_system.py` | Combined system backtest |
| `backtest_commodity_overlay.py` | Commodity overlay |
| `backtest_earnings_tremor.py` | Earnings tremor strategy |
| `backtest_filter_exploration.py` | Filter exploration |
| `backtest_asset_allocator.py` | Asset allocation backtest |
| `backtest_utils.py` | Shared backtest utilities |

## Optimization (`backend/`)

| File | Purpose |
|------|---------|
| `optimize_mega_v3.py` | V3 Optuna optimization |
| `optimize_mega_v4.py` | V4 Optuna optimization |
| `optimize_macro_momentum.py` | V1 optimization |
| `optimize_macro_momentum_v2.py` | V2 optimization |
| `optimize_multi_asset.py` | Multi-asset optimization |
| `optimize_portfolio.py` | Portfolio optimization |

## Walk-Forward (`backend/`)

| File | Purpose |
|------|---------|
| `walkforward_mega_v3.py` | V3 walk-forward (OOS 0.84, p=0.036) |
| `walkforward_mega_v4.py` | V4 walk-forward validation |
| `walkforward_macro_momentum.py` | V1 walk-forward |
| `walkforward_macro_momentum_v2.py` | V2 walk-forward |
| `walkforward_multi_asset.py` | Multi-asset walk-forward |
| `walkforward_portfolio.py` | Portfolio walk-forward |

## Research Scripts (`backend/research/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `README.md` | Research module documentation |
| `cli.py` | Research CLI |
| `orchestrator.py` | Research orchestrator |
| `ultrathink_01_realtime_m2_proxy.py` | Real-time M2 proxy research |
| `ultrathink_02_adaptive_leverage.py` | Adaptive leverage research |
| `ultrathink_03_yield_curve_uninversion.py` | Yield curve uninversion |
| `ultrathink_04_cross_asset_leadlag.py` | Cross-asset lead-lag |
| `ultrathink_05_funding_carry_short.py` | Funding carry short side |
| `validate_01_carry_momentum.py` | Carry momentum validation |
| `validate_02_cross_asset_lead.py` | Cross-asset lead validation |
| `validate_03_m2_liquidity_overlay.py` | M2 liquidity overlay validation |
| `calibrate_proxies.py` | Proxy calibration |
| `synthetic_proxies.py` | Synthetic proxy construction |
| `data_collectors.py` | Data collection utilities |
| `coinalyze_collector.py` | Coinalyze API collector |
| `grid_backtest.py` | Grid backtest runner |
| `pattern_analyzer.py` | Pattern analysis |
| `results_aggregator.py` | Results aggregation |
| `strategies_phase1.py` | Phase 1 strategy research |
| `strategy_combiner.py` | Strategy combination research |

## Integration (`backend/`)

| File | Purpose |
|------|---------|
| `maestro_engine.py` | Core engine (1.2s execution) |
| `maestro_api.py` | FastAPI server (7 endpoints) |
| `maestro_mcp_bridge.py` | MCP bridge (5 tools) |
| `maestro_cron.py` | Cron scheduler |
| `maestro_integration_test.py` | Integration tests (61/61) |

## Freqtrade (`backend/freqtrade/`)

| File | Purpose |
|------|---------|
| `macro_momentum_v3_strategy.py` | Full IStrategy port of V3 |
| `macro_data_provider.py` | Live macro data for Freqtrade |
| `freqtrade_config.json` | Freqtrade configuration |
| `README.md` | Freqtrade deployment guide |

## Backtest Results (`data/backtest_results/`)

| File | Description |
|------|-------------|
| `mega_v3_results.json` | V3 backtest results |
| `mega_v4_results.json` | V4 backtest results |
| `macro_momentum_v2_results.json` | V2 results |
| `portfolio_final_results.json` | Portfolio results |
| `ml_enhanced_results.json` | ML enhanced results |
| `dashboard_data.json` | Dashboard display data |
| `all_strategies_comprehensive.json` | All 66 strategies |
| `scalping_exploration_results.json` | Scalping results |
| `hybrid_strategies_results.json` | Hybrid results |
| `gap_strategies_results.json` | Gap strategies |
| `smart_funding_results.json` | Funding rate results |
| `vol_regime_results.json` | Vol regime results |
| `saylor_signal_results.json` | Saylor signal |
| `cascade_results.json` | Cascade results |
| `conductor_results.json` | Conductor results |
| `commodity_overlay_results.json` | Commodity overlay |
| `earnings_tremor_results.json` | Earnings tremor |
| `filter_exploration_results.json` | Filter exploration |
| `asset_allocator_results.json` | Asset allocator |
| `macro_crypto_results.json` | Macro-crypto |
| `macro_overlay_results.json` | Macro overlay |
| `optimize_multi_asset_results.json` | Multi-asset optimization |
| `STRATEGY_OPTIMIZATION_SUMMARY.md` | Summary report |
| `apex_*.json` | Apex strategy variants (3 files) |
| `btc_*.json` | BTC-specific backtests (8 files) |

## Optimization Studies (`data/optimization/`)

| File | Description |
|------|-------------|
| `mega_v3_study.pkl` | V3 Optuna study |
| `mega_v3_best_params.json` | V3 best parameters |
| `mega_v4_study.pkl` | V4 Optuna study |
| `mega_v4_best_params.json` | V4 best parameters |
| `macro_momentum_study.pkl` | V1 study |
| `macro_momentum_v2_study.pkl` | V2 study |
| `portfolio_study.pkl` | Portfolio study |
| `walkforward_mega_v3_results.json` | V3 WF results |
| `walkforward_mega_v4_results.json` | V4 WF results (as `walkforward_v4_results.json`) |
| `walkforward_portfolio_results.json` | Portfolio WF results |
| `walkforward_results.json` | V1 WF results |
| `walkforward_v2_results.json` | V2 WF results |
| `optimize_multi_asset_results.json` | Multi-asset results |

## Research Docs (`data/research/`)

| File | Description |
|------|-------------|
| `mega_strategy_blueprint.md` | Strategy blueprint |
| `ultrathink_strategies.md` | Ultrathink strategy notes |
| `ultrathink_v2_deep_edges.md` | V2 deep edges research |
| `brand_naming_research.md` | Branding research |
| `ml_feature_importance.json` | SHAP feature importances |

## Documentation (`docs/`)

| File | Description |
|------|-------------|
| `MEGA_STRATEGY_REPORT.md` | This session's full report |
| `FILE_INVENTORY.md` | This file |
| `PRODUCTION_READY.md` | Deployment checklist |
| `RESEARCH_FINDINGS.md` | Research findings summary |
| `INTEGRATION_ARCHITECTURE.md` | Integration architecture |
| `DATA_INVENTORY.md` | Data inventory |
| `STRATEGIES_ROADMAP.md` | Strategy roadmap |
| `SYNTHETIC_PROXIES.md` | Synthetic proxy documentation |
| `OPEN_INTEREST_DATA.md` | OI data documentation |
| `QUANT_PROBLEM_SUMMARY.md` | Problem summary |
| `FRONTEND_IMPLEMENTATION_PLAN.md` | Frontend plan |
| `UX_IMPROVEMENTS.md` | UX improvements |
| `UX_IMPROVEMENT_ROADMAP.md` | UX roadmap |
| `README.md` | Docs readme |
| `optuna-optimization-report-2026-02-06.md` | Optuna report |
| `architecture/*.md` | Architecture docs (5 files) |
| `guides/*.md` | Walk-forward guide |
| `project/*.md` | Branding, timeline |
| `reports/*.md` | Validation reports |
| `research/*.md` | Strategy implementation, WFA docs |

## Dashboard (`frontend/maestro-ui/`)

- React application
- Live at https://virtuosocrypto.com/quant/
- UX redesign docs in `docs/ux-redesign/`

## Live State (`data/live/`)

| File | Description |
|------|-------------|
| `maestro_state.json` | Current engine state |

---

*Total project files: ~120 source files (excluding node_modules and venv)*
