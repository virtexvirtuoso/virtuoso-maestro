# Filos Revival Plan

## Goal
Revive the Filos quantitative trading platform and integrate VPS data sources.

## Phase 1: Get Filos Running Locally
- [ ] Fix config typo (`127.0.01` → `127.0.0.1`)
- [ ] Start RethinkDB container standalone first
- [ ] Test backend connectivity
- [ ] Verify frontend builds
- [ ] Full docker-compose up

## Phase 2: Update Dependencies
- [ ] Check Python version compatibility (requirements from 2020)
- [ ] Update pandas, scikit-learn to modern versions
- [ ] Fix pyfolio install (quantopian is dead, use pyfolio-reloaded)
- [ ] Test backtrader still works

## Phase 3: Add VPS Data Sources
Current Filos sources: Binance, BitMEX (batch downloaders)

New sources needed from VPS (5.223.63.4):
- [ ] Create `vps_btcwiz_downloader.py` - on-chain metrics from port 8004
- [ ] Create `vps_derivatives_downloader.py` - fusion signals from port 8888
- [ ] Create `vps_influxdb_downloader.py` - historical OHLCV from InfluxDB 8086

### Data to Port:
| Source | Endpoint | Data Type |
|--------|----------|-----------|
| BTC Wiz (8004) | /metrics, /composite | On-chain, ETF, macro |
| Derivatives (8888) | /signals/fusion/{symbol} | FR, OI, LSR, CVD, Basis, IV |
| InfluxDB (8086) | VirtuosoDB bucket | Historical OHLCV |

## Phase 4: Add New Strategies
Port the "early gainer detection" strategy from walk_forward_backtest.py:
- [ ] Create `early_gainer_strategy.py` in backend/strategy/
- [ ] Add to __STRATEGY_CATALOG__
- [ ] Define params: rsi_oversold, vol_spike, range_threshold, TP/SL

## Phase 5: Verify Walk-Forward Works
- [ ] Run test optimization on BTCUSDT 1d
- [ ] Verify split-by-split results in UI
- [ ] Compare to VPS backtest results

## Notes
- Filos location: ~/Desktop/_Personal/filos
- VPS SSH: `ssh vps`
- RethinkDB admin: http://localhost:8081
- Frontend: http://localhost:8080
- Backend API: http://localhost:5050
