# Maestro Trading Pipeline

## Overview

Three-stage pipeline for developing, validating, and deploying crypto derivatives strategies.

```
MAESTRO (Research) → JESSE (Simulation) → FREQTRADE (Execution)
```

**Total Cost: $0** (all free tiers)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1. MAESTRO (Research)                    │
│         "Is this strategy robust or overfit?"               │
├─────────────────────────────────────────────────────────────┤
│  • Walk-forward optimization (10 rolling splits)            │
│  • Tests 22 strategies across market regimes                │
│  • Backtrader engine + RethinkDB storage                    │
│  • OUTPUT: Ranked strategies + optimal param ranges         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Top 3-5 strategies
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. JESSE (Simulation)                       │
│        "How does it perform with real derivatives?"         │
├─────────────────────────────────────────────────────────────┤
│  • Native funding rates, liquidation simulation             │
│  • Accurate margin/leverage modeling                        │
│  • Multi-timeframe without look-ahead bias                  │
│  • OUTPUT: Realistic PnL curves, refined params             │
└─────────────────────┬───────────────────────────────────────┘
                      │ Validated strategies
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               3. FREQTRADE (Execution)                      │
│             "Run it live, monitor, iterate"                 │
├─────────────────────────────────────────────────────────────┤
│  • Live trading on Bybit/Binance (free)                     │
│  • Telegram bot for alerts & control                        │
│  • Hyperopt for continuous optimization                     │
│  • Dry-run mode for paper trading                           │
│  • OUTPUT: Real profits, trade logs                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Three Tools?

| Stage | Tool | Unique Capability |
|-------|------|-------------------|
| **Validation** | Maestro | Walk-forward optimization (prevents overfitting) |
| **Simulation** | Jesse | Native funding rates, liquidation modeling |
| **Execution** | Freqtrade | Free live trading, Telegram integration |

### What You'd Miss Without Each

| Tool | Risk If Skipped |
|------|-----------------|
| Maestro | Overfit strategies go live, lose money |
| Jesse | Funding rate drag surprises (-5-15% annual) |
| Freqtrade | Pay $199/mo for Jesse live, or build custom execution |

---

## Stage 1: Maestro (Research & Validation)

### Purpose
Identify which strategies are **robust** vs **overfit** using walk-forward analysis.

### Location
```
VPS: /home/linuxuser/maestro/
Local: ~/Desktop/_Personal/maestro/
```

### Available Strategies (22)

**Classic (9)**
| Strategy | Type | Best Timeframe |
|----------|------|----------------|
| RSIStrategy | Momentum | 1h, 4h |
| MACDStrategy | Momentum | 4h, 1d |
| MACDStrategyX | Momentum | 4h, 1d |
| EmaCrossStrategy | Trend | 1h, 4h |
| MaCrossStrategy | Trend | 4h, 1d |
| BollingerBandsStrategy | Range | 1h, 4h |
| ChannelStrategy | Breakout | 4h, 1d |
| IchimokuStrategy | Multi-signal | 4h, 1d |
| FernandoStrategy | Custom | 4h |

**Scalping (5)**
| Strategy | Type | Best Timeframe |
|----------|------|----------------|
| ScalpRSIStrategy | Quick reversal | 5m, 15m |
| StochRSIStrategy | Momentum | 5m, 15m |
| VWAPStrategy | Mean reversion | 5m, 15m |
| MomentumBreakoutStrategy | Breakout | 15m, 1h |
| EMARibbonStrategy | Trend | 15m, 1h |

**Derivatives (8)**
| Strategy | Type | Best Timeframe |
|----------|------|----------------|
| FundingRateArbitrage | Funding cycles | 8h |
| BasisTradingStrategy | Contango/backwardation | 4h |
| OpenInterestDivergence | OI divergence | 1h, 4h |
| LiquidationHuntStrategy | Post-liquidation | 15m, 1h |
| GridTradingStrategy | Range | 1h |
| VolatilityBreakoutStrategy | Squeeze | 1h, 4h |
| TrendFollowingATR | Trend + stops | 4h, 1d |
| MeanReversionBands | Fade extremes | 1h, 4h |

### Running Walk-Forward

```bash
# SSH to VPS
ssh linuxuser@5.223.63.4

# Activate environment
cd ~/maestro/backend
source venv/bin/activate

# Start Flask API
CONFIG_FILE=config/maestro-prd.yaml python main/rest_api.py &

# Trigger walk-forward via API
curl -X POST http://localhost:5000/optimization/new/ \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "TrendFollowingATR",
    "provider": "BYBIT",
    "symbol": "btcusdt",
    "timeframe": "4h",
    "start_date": "2024-01-01",
    "end_date": "2026-02-01",
    "optimization": {"kind": "walkforward", "num_splits": 10}
  }'
```

### Interpreting Results

**Good Strategy (Robust)**
- Sharpe ratio consistent across splits (std < 0.3)
- Positive returns in 7+ of 10 splits
- Train and test Sharpe within 20% of each other

**Bad Strategy (Overfit)**
- High train Sharpe, low test Sharpe
- Returns vary wildly between splits
- Only profitable in 3-4 splits

### Data Available

| Symbol | Timeframes | Date Range | Candles |
|--------|------------|------------|---------|
| BTC/USDT | 5m, 15m, 1h, 4h, 1d | 2023-05 → 2026-02 | 25k+ |
| ETH/USDT | 5m, 15m, 1h, 4h, 1d | 2023-05 → 2026-02 | 25k+ |
| SOL/USDT | 5m, 15m, 1h, 4h, 1d | 2023-05 → 2026-02 | 25k+ |

---

## Stage 2: Jesse (Derivatives Simulation)

### Purpose
Re-test winning strategies with **accurate funding rate and liquidation simulation**.

### Location
```
VPS: ~/jesse-env/ (Python environment)
VPS: ~/jesse-projects/ (Strategy projects)
```

### Installation
```bash
source ~/jesse-env/bin/activate
jesse --version  # 1.12.2
```

### Strategy Porting Template

Maestro (Backtrader):
```python
class TrendFollowingATR(BaseStrategy):
    params = (('trend_period', 50), ('atr_period', 14), ('atr_mult', 2.0))
    
    def _next(self):
        if self.data.close[0] > self.highest[-1]:
            self.buy()
```

Jesse:
```python
from jesse.strategies import Strategy

class TrendFollowingATR(Strategy):
    def hyperparameters(self):
        return [
            {'name': 'trend_period', 'type': int, 'min': 20, 'max': 100, 'default': 50},
            {'name': 'atr_period', 'type': int, 'min': 7, 'max': 21, 'default': 14},
            {'name': 'atr_mult', 'type': float, 'min': 1.0, 'max': 4.0, 'default': 2.0},
        ]
    
    def should_long(self):
        return self.close > self.highest(self.hp['trend_period'])
    
    def go_long(self):
        qty = utils.size_to_qty(self.balance * 0.1, self.price)
        self.buy = qty, self.price
        self.stop_loss = qty, self.price - self.atr * self.hp['atr_mult']
```

### Key Jesse Features
- Native `self.funding_rate` access
- `self.liquidation_price` calculation
- Multi-timeframe: `self.get_candles('Bybit Perpetual', 'BTC-USDT', '1h')`

### Running Backtest
```bash
source ~/jesse-env/bin/activate
cd ~/jesse-projects/my_project
jesse backtest 2024-01-01 2026-01-01
```

---

## Stage 3: Freqtrade (Live Execution)

### Purpose
**Free live trading** with Telegram monitoring and continuous optimization.

### Location
```
VPS: ~/freqtrade-env/ (Python environment)
VPS: ~/freqtrade-strategies/ (User data)
```

### Installation
```bash
source ~/freqtrade-env/bin/activate
freqtrade --version  # 2026.1
```

### Strategy Porting Template

Jesse → Freqtrade:
```python
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class TrendFollowingATR(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = True
    
    # ROI and stoploss
    minimal_roi = {"0": 0.10, "60": 0.05, "120": 0.02}
    stoploss = -0.05
    trailing_stop = True
    
    def populate_indicators(self, dataframe, metadata):
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['highest'] = dataframe['high'].rolling(50).max()
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['close'] > dataframe['highest'].shift(1)),
            'enter_long'
        ] = 1
        return dataframe
```

### Configuration (Bybit Futures)
```json
{
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "exchange": {
        "name": "bybit",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_SECRET"
    },
    "pairlists": [{"method": "StaticPairList"}],
    "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"]
}
```

### Commands
```bash
# Download data
freqtrade download-data --config config.json --timeframe 4h --days 365

# Backtest
freqtrade backtesting --config config.json --strategy TrendFollowingATR

# Hyperopt (optimize parameters)
freqtrade hyperopt --config config.json --strategy TrendFollowingATR \
  --hyperopt-loss SharpeHyperOptLoss --epochs 500

# Dry run (paper trading)
freqtrade trade --config config.json --strategy TrendFollowingATR --dry-run

# Live trading
freqtrade trade --config config.json --strategy TrendFollowingATR
```

### Telegram Integration
Add to config:
```json
{
    "telegram": {
        "enabled": true,
        "token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
}
```

Commands: `/status`, `/profit`, `/balance`, `/start`, `/stop`

---

## Complete Workflow Example

### 1. Research (Maestro)
```bash
# Run walk-forward on TrendFollowingATR
# Result: Sharpe 1.4 avg, consistent across 10 splits ✓
```

### 2. Simulate (Jesse)
```bash
# Port to Jesse, backtest with funding rates
# Result: Sharpe 1.2 (funding drag of -0.2) ✓
# Refined ATR multiplier: 2.0 → 2.3
```

### 3. Execute (Freqtrade)
```bash
# Port to Freqtrade, hyperopt for final tuning
freqtrade hyperopt --strategy TrendFollowingATR --epochs 500
# Result: Sharpe 1.35

# Paper trade 2 weeks
freqtrade trade --dry-run

# Go live
freqtrade trade
```

---

## Cost Summary

| Tool | Backtest | Live Trading | Total |
|------|----------|--------------|-------|
| Maestro | Free | N/A | $0 |
| Jesse | Free | $199/mo (skip) | $0 |
| Freqtrade | Free | **Free** | $0 |

**Pipeline Total: $0**

---

## Maintenance

### Daily
- Check Freqtrade Telegram for alerts
- Monitor open positions

### Weekly
- Review Freqtrade trade logs
- Check for strategy drift

### Monthly
- Re-run Maestro walk-forward with new data
- Freqtrade hyperopt to adapt to market

---

## Server Locations

| Component | Location | Access |
|-----------|----------|--------|
| Maestro Backend | VPS 5.223.63.4 | `ssh linuxuser@5.223.63.4` |
| Maestro Frontend | http://5.223.63.4:3001 | Browser |
| RethinkDB | VPS localhost:28015 | Admin: localhost:8080 |
| Jesse | VPS ~/jesse-env | SSH |
| Freqtrade | VPS ~/freqtrade-env | SSH |
| Local Maestro | ~/Desktop/_Personal/maestro | Local |

---

## Troubleshooting

### Maestro API not responding
```bash
cd ~/maestro/backend
source venv/bin/activate
CONFIG_FILE=config/maestro-prd.yaml python main/rest_api.py
```

### RethinkDB down
```bash
sudo systemctl start rethinkdb
# or
rethinkdb --bind all &
```

### Freqtrade data issues
```bash
freqtrade download-data --config config.json --timeframe 4h --days 365 --erase
```

---

## Next Steps

1. [ ] Run Maestro walk-forward on all 22 strategies
2. [ ] Identify top 5 performers (Sharpe > 1.0, consistent)
3. [ ] Port winners to Jesse for funding rate validation
4. [ ] Deploy to Freqtrade for live trading
5. [ ] Set up Telegram bot for monitoring

---

*Last updated: 2026-02-05*
