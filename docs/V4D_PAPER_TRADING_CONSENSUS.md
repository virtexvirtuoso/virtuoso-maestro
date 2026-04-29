# V4d Paper Trading System — Cross-Agent Consensus Matrix

**Date**: 2026-02-17 (Updated 2026-02-17 22:00 EST)
**Agents**: Tech Lead Advisor, Trading Validator, Infrastructure Maintainer, Sprint Prioritizer
**Subject**: Architecture plan evaluation for V4d paper trading system

---

## Key Update: Perpetual Futures (Not Spot)

After reviewing the existing Virtuoso shadow mode infrastructure (CAS/CTUS/CREW/FRAPS tracking 20 symbols on perps), the decision is to paper trade **perpetual futures**, not spot. This aligns with Virtuoso's actual trading infrastructure.

### Implications of Perps vs Spot

| Factor | Spot (Original Plan) | Perps (Updated) |
|--------|---------------------|-----------------|
| Funding drag | None | 8-15%/yr — **must track and include** |
| Leverage | 1x only | 1-2x (V4d uses vol-targeted sizing) |
| Price source | yfinance/Binance spot | Binance/Bybit perps mark price |
| Short capability | No | Available (but V4d doesn't short) |
| Exchange alignment | Different from production | **Matches Virtuoso production** |
| Realistic P&L | Optimistic | Honest (includes all costs) |
| Data source | CCXT spot OHLCV | CCXT perps OHLCV + funding rates |

### Additional Perps Requirements
- **Funding rate tracking**: Fetch 8h funding rates for BTC/ETH/SOL, apply to open positions
- **Mark price**: Use perpetual mark price (not spot) for SMA50 and trailing stops
- **Funding cost in P&L**: Deduct cumulative funding from equity curve
- **Liquidation distance**: Track (informational) even though V4d uses low leverage

---

## Existing Infrastructure Context

Virtuoso already runs a shadow mode system on VPS:
- `virtuoso-shadow.service` — Market Psychology shadow mode
- `virtuoso-alpha-shadow.service` — Alpha Trading Insights shadow mode
- Tracking 20 symbols, 42+ hours uptime, 11.6K orderbook updates
- Discord alerts via "Optimizations" webhook

**V4d should follow the same pattern**: standalone service, own DB, Discord alerts to #development, but architecturally similar to existing shadow modes for operational consistency.

---

## Architecture Plan Summary

| Component | Description |
|-----------|-------------|
| **Signal Engine** | Python daemon on VPS, daily SMA50 + trailing stops + vol ceiling + DD breaker |
| **Assets** | BTC, ETH, SOL perpetual futures (equal weight 33/33/33) |
| **Storage** | SQLite with WAL mode (trades, signals, equity snapshots, funding costs) |
| **Alerts** | Discord development webhook (entry/exit/daily summary/risk/funding) |
| **Dashboard** | Single-page HTML at virtuosocrypto.com/quant/paper/ |
| **API** | FastAPI on port 8006 (GET /status, /trades, /equity, /signals) |
| **Data** | Binance/Bybit perps OHLCV + funding rates via CCXT |

---

## Design Decisions — Consensus Matrix

| # | Decision | Tech Lead | Trading Validator | Infra | Sprint | **Verdict** |
|---|----------|-----------|-------------------|-------|--------|-------------|
| 1 | $100K notional | $100K | $100K | — | — | **DO IT** |
| 2 | Daily rebalance only | Daily | Daily (log intraday) | — | — | **DO IT** |
| 3 | Price source: yfinance | yfinance | **REJECT — Binance** | — | yfinance (fallback) | **DISCUSS** |
| 4 | Daily trailing stop | Daily | Daily (log violations) | — | Daily | **DO IT** |
| 5 | No regime overlay | Add after 30d | Accept | — | SKIP entirely | **DEFER** |
| 6 | Single file vs modular | **Modular** (4 files) | — | — | — | **MODULAR** |
| 7 | Backtest overlay | Live overlay | — | — | V1.1 | **DEFER to V1.1** |
| 8 | Monitoring | Full stack | Discord + healthcheck | Discord + healthcheck | MVP: healthcheck | **DO IT** |

### Infrastructure-Specific Decisions

| # | Decision | Recommendation | Priority |
|---|----------|---------------|----------|
| 1 | Scheduling | systemd timer | Should — matches existing VPS pattern |
| 2 | Database | SQLite w/ WAL mode | **Must** — WAL prevents corruption |
| 3 | Deployment | git pull (not scp) | Should — rollback + audit trail |
| 4 | Backup | Daily SQLite cp | **Must** — one cron line, zero cost |
| 5 | Venv | Separate venv | **Must** — 50 services already on box |
| 6 | Resources | VPS has 10 GB RAM available | **OK** — audited 2026-02-18, not a constraint |

---

## Key Disagreement: yfinance vs Binance — RESOLVED

| Agent | Position | Reasoning |
|-------|----------|-----------|
| Tech Lead | yfinance | No API keys, simpler, one less failure mode |
| Trading Validator | **Binance perps** | Must match production price source; yfinance divergence invalidates forward test |
| Sprint Prioritizer | yfinance (with fallback) | Ship faster, worry about precision later |

**Resolution**: Use **Binance perpetual futures OHLCV via CCXT** (primary) with Bybit as fallback. This is mandatory now that we're paper trading perps — spot prices diverge from perp mark prices, especially during high funding periods. We already have CCXT installed and Binance perps historical data in `data/perps/`.

### Funding Rate Data Source
- **Primary**: Binance funding rates via CCXT (`fetch_funding_rate_history`)
- **Fallback**: CoinGlass API (we have the key, daily FR data for 19 tokens already downloaded)
- **Frequency**: Every 8 hours (Binance standard), applied to open positions

---

## Critical Implementation Risks

All agents flagged these risks. Any one of them can invalidate the paper test.

| Risk | Flagged By | Severity | Mitigation |
|------|-----------|----------|------------|
| `position.shift(0)` fragility | Trading Validator | **CRITICAL** | Paper system must replicate exact 2-stage shift logic from `v4_honest_system.py` |
| SQLite WAL mode | Tech Lead + Infra | **HIGH** | Set `PRAGMA journal_mode=WAL` at connection — prevents corruption from concurrent read/write |
| sqrt(252) contamination | Trading Validator | **HIGH** | 97 instances of `sqrt(252)` in codebase; `v4_honest_system.py` is clean but shared utils may not be |
| Silent death (no alerts) | All 4 agents | **HIGH** | Staleness alert if no run in >26 hours + `/health` endpoint |
| VPS resource contention | Infra | **LOW** | Audited 2026-02-18: 10 GB RAM available; stagger timer to 00:15 UTC to avoid cron pile-up with V3.1-H2 at 00:05 |
| DD breaker state machine | Trading Validator | **HIGH** | Must match exact re-entry logic: recovery to -12.5% AND new SMA50 cross required |

### Look-Ahead Bias Risk

The `compute_returns` function in `v4_honest_system.py` uses `position.shift(0)` (a no-op). This is only correct because the upstream `apply_no_lookahead()` already shifted signals by 1. The paper trading system must replicate this exact timing:

1. Signal generated AFTER daily close (using yesterday's data)
2. Position taken at NEXT day's open
3. Return earned = today's close-to-close

If paper and backtest disagree on execution timing, all comparisons are invalid.

### sqrt(365) Verification

The vol ceiling check computes annualized volatility. Using `sqrt(252)` (equity markets) instead of `sqrt(365)` (crypto, 24/7) **underestimates vol by 17%**. A realized vol of 83% (should halve positions) would compute as 69% (below the 80% ceiling), failing to trigger the safety mechanism.

**Status**: `v4_honest_system.py` is correct. But 97 instances of `sqrt(252)` exist elsewhere in the codebase — any shared utility function could reintroduce the bug.

---

## Recommended MVP (48-Hour Ship)

The sprint prioritizer recommends cutting the dashboard and API from the initial build.

### Day 1: Core Engine

| Component | Details |
|-----------|---------|
| Signal function | Pure function: DataFrame in → signals dict out. Test against 5 known backtest dates. |
| SQLite schema | 3 tables: `signals`, `positions`, `equity`. WAL mode enabled. |
| Discord alerts | 4 message types: entry, exit, daily summary, error/warning |
| Manual verification | Compare output against known backtest results for specific dates |

### Day 2: Deploy & Harden

| Component | Details |
|-----------|---------|
| VPS deployment | git clone to `~/v4_paper/`, separate venv |
| systemd timer | Daily execution at market close, `Restart=always` |
| `/health` endpoint | Returns last signal timestamp; register with existing health-check infra |
| SQLite backup | Cron: `cp v4_paper.db v4_paper.db.bak` |
| 24-hour burn-in | One full daily cycle end-to-end |

### Day 3: Buffer

Absorbs Day 1-2 overruns. If clean, start V1.1 (FastAPI + dashboard).

### V1.1 (Week 2)

- FastAPI API (4 endpoints)
- Dashboard HTML (equity curve, positions, trade log)
- Backtest overlay on equity chart

### V2 (After 30 Days)

- Regime overlay (if validated)
- Intraday trailing stop monitoring
- Multi-asset expansion

### Build Order

```
1. Signal computation function (pure, testable, no I/O)
2. SQLite schema + write helpers
3. Wire signal + storage into daily runner
4. Discord webhook integration
5. Deploy to VPS + systemd timer
6. /health endpoint
7. 24-hour burn-in
```

---

## Metrics to Track: Paper vs Backtest

### Signal Fidelity

| Metric | Target | Purpose |
|--------|--------|---------|
| Signal agreement rate | >95% | Core validity check |
| SMA50 value divergence | <0.5% | Price source drift |
| Entry date difference | 0 ideal, <2/yr ok | Signal timing match |

### Execution Quality

| Metric | Target | Purpose |
|--------|--------|---------|
| Realized slippage | <20 bps | Fill quality |
| Realized TX cost | Track vs 10 bps assumption | Cost model validation |
| Fill rate | 100% | Daily spot should always fill |

### Risk Management

| Metric | Target | Purpose |
|--------|--------|---------|
| Trailing stop trigger agreement | >98% | Stop logic match |
| Vol ceiling activation agreement | >95% | Regime detection match |
| Intraday stop violation count | Track (info only) | Measure daily-vs-intraday gap |

### Performance Comparison

| Metric | Target | Purpose |
|--------|--------|---------|
| Rolling 30d Sharpe correlation | >0.85 | Strategy tracking |
| Monthly return tracking error | <2% | Performance match |
| Drawdown timing agreement | Same month | Risk profile match |
| CAGR difference after 6 months | Within 5% absolute | Long-term drift |

### Meta-Validation

| Metric | Target | Purpose |
|--------|--------|---------|
| Days until first trade divergence | >30 days | System stability |
| Price source divergence (daily) | <0.5% median | Data quality |
| Data availability gaps | 0 | Uptime tracking |

---

## Agent-Specific Insights

### Tech Lead Advisor

- **Architecture**: Modular (4 files: engine.py, api.py, alerts.py, models.py) costs 30 min more upfront, saves hours of debugging. A monolith will hit 800+ lines in weeks.
- **Idempotency**: Daily runs must be keyed by date. `UNIQUE(date)` constraint on positions table prevents double-counting from timer re-fires.
- **Estimated complexity**: Small (S). ~600 lines of Python total.

### Trading Validator

- **Price source**: yfinance uses CoinMarketCap aggregation, not exchange prices. On volatile days, 0.5-2% divergence from Binance. Even 1 different entry/exit can shift Sharpe by 0.1-0.2.
- **DD breaker re-entry**: Requires BOTH recovery to -12.5% AND a new SMA50 crossover. Simpler "re-enter when DD < 25%" logic would produce different trades.
- **Forward-fill risk**: yfinance may have gaps that `.ffill()` papers over. Binance has clean 365-day/year data. SMA50 computation will be subtly different if using yfinance for paper vs Binance for backtest.

### Infrastructure Maintainer

- **VPS state** (audited 2026-02-18): 50 running services, 15 GB RAM (10 GB available), 4 GB swap (2.5 GB used). Virtuoso trading uses 786 MB RSS / 96% CPU. Memory is NOT a blocker.
- **Pattern to follow**: Every project uses its own venv. Deployment via git pull (not scp). journalctl for logs (not file logging).
- **Maintenance burden**: ~0.5 hours/month under normal operation.

### Sprint Prioritizer

- **Meta-risk**: "If you can't write down V4d's exact rules in plain text, you're still doing research, not engineering." Strategy must be frozen before building infrastructure.
- **MVP scope**: Signal engine + SQLite + Discord + systemd. That's it. Dashboard is V1.1.
- **yfinance risk**: Have CCXT/Binance fallback identified before deploying. yfinance is an unofficial scraper that breaks when Yahoo changes their frontend.

---

## Research Foundation (Feb 6-17, 2026)

V4d's design is backed by the most rigorous testing in Maestro's history:

### What's Validated
- **4,000+ walk-forward tests**, 80+ signal variants, 48+ tokens
- **V4d BTC/ETH/SOL**: Sharpe 1.52, CAGR 41.9%, MaxDD -18.7% (spot)
- **Selection bias holdout**: Top 5 portfolio Sharpe 1.91 on unseen 2024-2025 data
- **Bootstrap CI**: [0.67, 2.31] excludes zero
- **2022 bear**: +3.1% vs B&H -79.6%

### What Failed (don't add these)
- On-chain signals (NUPL, MVRV, SOPR, NVT): 36 tests, only NUPL capitulation survived Bonferroni standalone, but failed paired permutation when added to V4 (p=0.077)
- Derivatives signals: 816 tests, 0 survive Bonferroni standalone
- ML regime classifier: Worked in-sample, noise OOS
- Macro timing (M2, HYG): Dead after full-cycle retest
- Dynamic signal weighting / rotation: No improvement over static

### Perps-Specific Expected Performance
With 8-15%/yr funding drag, realistic V4d perps expectations:
- **Sharpe**: ~1.1-1.3 (vs 1.52 spot)
- **CAGR**: ~28-35% (vs 41.9% spot)
- **MaxDD**: Similar or slightly worse (~-20%)
- **The funding drag is the COST of using leverage infrastructure** — this is the honest number

### Regime Overlay (Deferred to V2)
- Composite derivatives regime (LSR 35%, FR 35%, Liq 15%, Taker 15%): BTC p=0.012
- Regime position sizing: Sharpe 3.16, MaxDD -9.7% in backtest
- Wait for 30 days of clean V4d paper data before adding

---

## External Research Report Assessment (Feb 17)

An external optimization report was received recommending HMM regime detection, XGBoost meta-learners, pairs trading, and options overlays. Our assessment after 10 days of exhaustive testing:

- ✅ **Agree**: Execution & cost realism (priority #1), testing framework
- ⚠️ **Caution**: Adaptive vol targeting (our V3.2 showed mixed results)
- ❌ **Reject**: HMM/ML regime (noise OOS), F&G features (backwards in crypto), pairs trading (all negative), on-chain signals (marginal at best)

**The meta-lesson**: Complexity → noise. The edge is trend following + risk management + diversification. V4d embodies this.

---

## Final Verdict

All 4 agents approve the architecture. Four changes from the original plan:

1. **Use Binance perps prices** (not spot/yfinance) — matches production, includes funding drag
2. **Track funding rates** — 8h funding applied to open positions, deducted from equity
3. **Modular from day 1** (4 files) — tech lead emphatic, costs 30 min, saves hours
4. **SQLite WAL + backup + healthcheck from day 1** — all agents flagged silent failure as the #1 operational risk

### Build Order (Updated)

```
Day 1: Signal engine (SMA50 + trailing stops + vol ceiling + DD breaker)
       SQLite schema (positions, trades, equity, funding_costs)
       CCXT perps price + funding rate fetcher
       Discord alerts (entry/exit/daily/risk/funding)
       
Day 2: Deploy to VPS (~/v4_paper/, own venv, systemd timer)
       Health endpoint + staleness alert
       24-hour burn-in
       
Week 2: FastAPI (port 8006) + Dashboard (virtuosocrypto.com/quant/paper/)
        Backtest overlay on equity chart
        
Day 30: Evaluate regime overlay addition
        Compare paper vs backtest divergence
        Decision: scale up or iterate
```

The system is well-scoped, the strategy is rigorously validated, and the infrastructure pattern (shadow mode) is proven. Ship the MVP in 48 hours, add the dashboard in week 2, evaluate regime overlay after 30 days of clean data.
