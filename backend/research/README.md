# Maestro Research Module

**Automated Strategy Research & Optimization System**

Run cross-asset, cross-timeframe strategy backtests with intelligent analysis of what works vs what doesn't.

## Overview

```
┌─────────────────────────────────────────────┐
│            ORCHESTRATOR                      │
│  - Receives research request                │
│  - Spawns parallel backtest agents          │
│  - Aggregates findings                      │
│  - Generates next iteration                 │
└─────────────────────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Agent 1 │ │ Agent 2 │ │ Agent 3 │
   │ BTC 6h  │ │ ETH 1d  │ │ SOL 1h  │
   │ Bolling │ │ OBV+Cap │ │ MACD    │
   └─────────┘ └─────────┘ └─────────┘
        │           │           │
        ▼           ▼           ▼
   ┌─────────────────────────────────────────┐
   │         GRID BACKTESTER                  │
   │  - Fast backtests (VectorBT optional)   │
   │  - Multiple strategies × assets × TFs   │
   │  - Results to structured JSON           │
   └─────────────────────────────────────────┘
```

## Quick Start

```bash
cd ~/Desktop/_Personal/maestro

# Activate virtual environment
source .venv/bin/activate

# Run full research
python -m backend.research.cli run \
  --strategies BollingerBreakout MACD RSI OBV \
  --assets BTC/USDT ETH/USDT SOL/USDT \
  --timeframes 1h 4h 1d

# Quick grid test
python -m backend.research.cli grid \
  --strategy BollingerBreakout \
  --all-assets --all-timeframes

# Generate hybrid strategies
python -m backend.research.cli combine --smart

# Analyze existing results
python -m backend.research.cli analyze --input results.json

# List available strategies
python -m backend.research.cli list
```

## Components

| File | Purpose |
|------|---------|
| `grid_backtest.py` | Run strategies across asset/timeframe grids |
| `strategy_combiner.py` | Generate hybrid strategy combinations |
| `results_aggregator.py` | Consolidate and rank results |
| `pattern_analyzer.py` | Find what works vs what doesn't |
| `orchestrator.py` | Coordinate multi-agent research |
| `cli.py` | Command-line interface |

## Built-in Strategies

### Trend Following
- `MACD` - MACD crossover
- `EMA_Cross` - EMA crossover (12/26)
- `SMA_Cross` - SMA crossover (10/30)
- `Momentum` - Price momentum

### Mean Reversion
- `RSI` - RSI oversold/overbought
- `BollingerBreakout` - Bollinger Bands breakout
- `MeanReversion` - Z-score mean reversion

### Volume Based
- `OBV` - On-Balance Volume
- `VolumeBreakout` - Volume spike breakout
- `CapitulationReversal` - Buy after liquidation cascades

## Filters

| Filter | Description |
|--------|-------------|
| `VolumeFilter` | Only trade on high volume (>2x average) |
| `TrendFilter` | Only trade in direction of trend (50 SMA) |
| `VolatilityFilter` | Only trade when volatility is moderate |
| `MomentumFilter` | Require momentum confirmation |
| `CapitulationFilter` | Only buy after volume spike + price drop |

## Hybrid Strategies

The `StrategyCombiner` generates intelligent combinations:

```python
from backend.research.strategy_combiner import StrategyCombiner

combiner = StrategyCombiner()
combiner.add_base(['BollingerBreakout', 'MACD', 'RSI'])
combiner.add_filters(['VolumeFilter', 'CapitulationFilter'])

# Smart combinations (complementary strategies only)
hybrids = combiner.generate_smart()

# Includes:
# - CapitulationReversal_Pure
# - Bollinger+Capitulation
# - OBV+CapitulationDivergence
# - MultiTF_Capitulation
# - MACD+RSI confluence
```

## Output Format

### Cross-Asset Performance Table

```
📊 BollingerBreakout
------------------------------------------------------------
Asset/TF                 Return %     Sharpe    MaxDD %   Trades
------------------------------------------------------------
BTC/USDT/1h               -3.97%     0.0000      0.00%        9
BTC/USDT/4h               -4.06%     0.0000      0.00%        7
BTC/USDT/1d              +81.52%     0.0000      0.00%       12
ETH/USDT/1d             +104.40%     0.0000      0.00%        9
SOL/USDT/1d             +348.35%     0.0000      0.00%       10
```

### Pattern Analysis

```
✅ WHAT WORKS:
   • BollingerBreakout is consistently profitable
   • Daily timeframe outperforms across all strategies
   • SOL shows highest returns

❌ WHAT DOESN'T:
   • Hourly timeframes lose to commission drag
   • RSI underperforms in trending markets
```

## Python API

```python
from backend.research.orchestrator import ResearchOrchestrator, quick_research

# Quick research
report = quick_research(
    strategies=['BollingerBreakout', 'MACD'],
    assets=['BTC/USDT', 'ETH/USDT'],
    timeframes=['1h', '4h', '1d']
)
print(report)

# Full control
orchestrator = ResearchOrchestrator()
orchestrator.configure(
    strategies=['BollingerBreakout', 'MACD', 'RSI'],
    assets=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    timeframes=['1h', '4h', '1d'],
    generate_hybrids=True,
    parallel=True,
    max_workers=4,
)
report = orchestrator.run()
```

## Grid Backtest API

```python
from backend.research.grid_backtest import GridBacktester, GridConfig

config = GridConfig(
    strategies=['BollingerBreakout', 'MACD'],
    assets=['BTC/USDT', 'ETH/USDT'],
    timeframes=['1h', '4h', '1d'],
    exchange='binance',  # or bybit, okx, gate, kucoin, mexc
    initial_capital=10000,
    commission=0.001,
)

grid = GridBacktester(config=config)

def progress(current, total, result):
    print(f"[{current}/{total}] {result.strategy}: {result.total_return:+.2f}%")

results = grid.run(progress_callback=progress)
print(grid.summary())
grid.to_json('results.json')
```

## Requirements

```bash
pip install pandas numpy ccxt

# Optional for faster backtests
pip install vectorbt
```

## Notes

- **Geo-blocking**: Binance and Bybit are blocked from US IPs. Run on VPS for full access.
- **VectorBT**: Optional but recommended for faster backtests (100-1000x speedup).
- **Data caching**: OHLCV data is cached per session to avoid redundant API calls.

## Sample Research Results

From initial testing (2026-02-05):

| Strategy | Best Asset/TF | Return |
|----------|---------------|--------|
| BollingerBreakout | SOL/USDT/1d | +348% |
| MACD | SOL/USDT/1d | +254% |
| BollingerBreakout | ETH/USDT/1d | +104% |

**Key Finding**: Daily timeframe consistently outperforms hourly/4h across all strategies.

## Next Steps

1. Run walk-forward validation on winning strategies
2. Optimize parameters with Optuna
3. Test hybrid combinations
4. Paper trade best strategies for 2 weeks
5. Deploy to production

---

*Built for Maestro - The Master Conductor of Trading Strategies*
