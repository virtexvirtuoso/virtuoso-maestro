# V4d Paper Trading — Deployment Spec

**Date**: 2026-02-17 (Updated 2026-02-20 — reflects actual deployment)
**Strategy**: SMA50 + trailing stops + vol ceiling + DD breaker on BTC/ETH/SOL perps (equal weight)
**Status**: Live since 2026-02-19, running daily at 00:15 UTC

> "If you can't write down the rules in plain text, you're still doing research, not engineering."

---

## Actual Deployment Architecture

V4D was integrated into the **Virtuoso codebase** rather than deployed as a standalone system. This was the better choice — it reuses Virtuoso's existing web server, database layer, and nginx config with zero additional processes.

| Component | Location on VPS | Notes |
|-----------|----------------|-------|
| Signal engine | `~/trading/Virtuoso/scripts/v4d_paper_trading.py` (714 lines) | Single file, all signal functions self-contained |
| API routes | `~/trading/Virtuoso/src/api/routes/v4d_paper.py` | 4 read-only endpoints, served via existing web server |
| Database | `~/trading/Virtuoso/data/virtuoso.db` | Tables `v4d_daily_state` + `v4d_trade_log` inside existing SQLite |
| Storage layer | `~/trading/Virtuoso/src/database/shadow_storage.py` | V4D functions added to existing shadow storage module |
| Dashboard | `/var/www/virtuosocrypto.com/quant/paper/index.html` (68 KB) | Static HTML, fetches from `/api/v4d/` |
| Service | `/etc/systemd/system/virtuoso-v4d-paper.service` | oneshot, triggered by timer |
| Timer | `/etc/systemd/system/virtuoso-v4d-paper.timer` | Daily at 00:15 UTC, `Persistent=true` |
| Nginx proxy | `/api/v4d/` → `127.0.0.1:8002` | Piggybacks on Virtuoso's existing web server (port 8002) |
| Python env | `~/trading/Virtuoso/venv311` | Shared with Virtuoso — no extra venv needed |
| Discord alerts | `DEVELOPMENT_WEBHOOK_URL` from `.env` | Daily summary + trade notifications |

### What the spec originally proposed vs what was built

| Spec Said | Actually Built |
|-----------|---------------|
| Standalone `~/v4_paper/` with own venv | Integrated into Virtuoso, reuses `venv311` |
| 4 files → later simplified to 1 | Single file (714 lines) |
| Own SQLite database | Tables inside existing `virtuoso.db` |
| FastAPI on port 8006 (socket-activated) | API routes on Virtuoso's existing port 8002 via nginx |
| Binance perps via CCXT | **Binance** perps via CCXT (initially deployed with Bybit, switched to Binance 2026-02-20) |
| `/health` endpoint deferred | Health via Virtuoso's existing health check infra |

**Key win**: Zero additional memory footprint for the API — no extra process, no extra port.

---

## VPS Resource Snapshot (2026-02-18)

Audited via SSH — replaces previous estimates from consensus doc.

| Resource | Actual | Notes |
|----------|--------|-------|
| **RAM** | 15 GB total, **10 GB available** | NOT the "4G/4G exhausted" from consensus doc |
| **Swap** | 4 GB total, 2.5 GB used, 1.5 GB free | |
| **Disk** | 150 GB total, 83 GB used, **61 GB free** (58%) | Tick collector adds 5-10 GB/day — needs retention policy |
| **CPU** | Load avg 1.28 | Virtuoso trading burns 96% of one core |
| **Services** | **50 running** (not 15) | 20+ custom Python processes |

**V4D actual footprint**: ~2s CPU per daily run (oneshot service). No persistent process. API served by existing Virtuoso web server.

### Cron/Timer Schedule

| Time (UTC) | Job | Conflict? |
|------------|-----|-----------|
| `0 * * * *` | Tick collector monitor | No |
| `5 0 * * *` | V3.1-H2 signal engine | No — V4D at 00:15, staggered by 10 min |
| Every 5 min | Virtuoso health monitor | No |
| **`00:15`** | **V4D paper trading** | No |

---

## Coexisting Systems

V4D is the third paper/signal system. They test different strategies with different data sources.

| | Existing Paper Engine | V3.1-H2 Signals | V4D |
|---|---|---|---|
| **Strategy** | 4-signal + hedge (derivatives) | 5-signal macro confluence | SMA50 + trailing stops |
| **Runs on** | Mac Mini (local cron) | VPS cron | VPS systemd timer |
| **Data source** | DuckDB (local) | yfinance + FRED | CCXT Binance perps |
| **Storage** | JSONL | JSON | SQLite (inside `virtuoso.db`) |
| **Alerts** | None | Telegram + Discord | Discord |
| **API** | None | None | `/api/v4d/` via Virtuoso web server |

No shared code, no conflicts. V4D reuses Virtuoso's infrastructure without interfering with existing services.

---

## Decisions Made

| # | Decision | Verdict | Notes |
|---|----------|---------|-------|
| 1 | Instrument | Perps (not spot) | Matches Virtuoso production; includes funding drag |
| 2 | Price source | Binance perps via CCXT | Matches Virtuoso's exchange connection |
| 3 | Funding rates | Binance via CCXT | Fetched per-asset each run |
| 4 | Notional | $100K simulated | Starting equity after backfill: $98,681 |
| 5 | Rebalance | Daily only | Full history recompute each run (~2s) |
| 6 | Trailing stop | Daily | Per-asset: BTC 12%, ETH 15%, SOL 8% |
| 7 | Regime overlay | Deferred to Day 30+ | Need 30d clean paper data first |
| 8 | Architecture | Single file inside Virtuoso | `scripts/v4d_paper_trading.py` + `src/api/routes/v4d_paper.py` |
| 9 | Storage | `v4d_daily_state` + `v4d_trade_log` in `virtuoso.db` | Reuses shadow_storage layer |
| 10 | Deployment | Integrated into Virtuoso codebase | Reuses venv, web server, nginx, DB |
| 11 | Scheduling | systemd timer at **00:15 UTC** | Staggered from V3.1-H2 at 00:05 |
| 12 | Monitoring | Discord + Virtuoso health check | Staleness alert if no run in >26h |
| 13 | Dashboard | Live at `virtuosocrypto.com/quant/paper/` | Static HTML, auto-refresh 60s |
| 14 | Forward vs backfill metrics | Separated in API and dashboard | `forward_*` fields track live-only performance |

---

## API Endpoints

Served via Virtuoso's web server. Nginx proxies `/api/v4d/` → `127.0.0.1:8002`.

| Endpoint | Purpose | Key Fields |
|----------|---------|------------|
| `GET /api/v4d/status` | Current state + performance summary | `state`, `performance` (includes `forward_*` metrics) |
| `GET /api/v4d/equity?days=365` | Equity curve for charting | `equity_curve[]` with per-day equity, positions, prices |
| `GET /api/v4d/trades?days=90` | Trade log | `trades[]` with action, price, reason |
| `GET /api/v4d/signals?days=30` | Signal history | `signals[]` with per-asset signal state |

### Forward vs Backfill Metrics

The API separates backfilled simulation from live forward-test data:

| Field | Meaning |
|-------|---------|
| `live_since` | Date forward test started (2026-02-19) |
| `starting_equity` | Equity at forward test start ($98,681 — includes backfill history) |
| `forward_pnl` | P&L since `live_since` only |
| `forward_pnl_pct` | Return since `live_since` |
| `forward_max_dd` | Max drawdown since `live_since` |
| `forward_sharpe` | Sharpe since `live_since` |
| `forward_cagr` | CAGR since `live_since` |
| `sharpe`, `cagr`, `max_dd` | Full-history (includes backfill) — for reference only |

---

## Dashboard

Live at [virtuosocrypto.com/quant/paper/](https://virtuosocrypto.com/quant/paper/).

**Components:**
- Header: equity, forward P&L, days running, drawdown, live-since date
- Backtest reference bar (Sharpe 1.52, CAGR 41.9%, MaxDD -25%, Win Rate 58%)
- Risk flags (DD breaker, vol ceiling — shown when active)
- 6 metric cards: Sharpe, CAGR, MaxDD, Win Rate, Total Trades, Cumulative Funding
- Equity curve chart with drawdown overlay and **vertical dashed line** at forward test start
- Per-asset mini charts (BTC/ETH/SOL) with position shading
- Current positions table with trailing stop distances
- 7-day signal heatmap
- Collapsible frozen strategy parameters
- Trade log (most recent 50)
- Auto-refresh every 60 seconds

**Backfill distinction**: Chart dims backfilled data and brightens forward-test data. Collapsible details section explains the backfilled period.

---

## SQLite Schema

Two tables inside `virtuoso.db`, managed by `shadow_storage.py`.

```sql
CREATE TABLE v4d_daily_state (
    date TEXT PRIMARY KEY,
    btc_signal INTEGER,
    eth_signal INTEGER,
    sol_signal INTEGER,
    btc_position REAL,
    eth_position REAL,
    sol_position REAL,
    btc_price REAL,
    eth_price REAL,
    sol_price REAL,
    equity REAL,
    drawdown REAL,
    dd_breaker_active INTEGER,
    vol_ceiling_active INTEGER,
    btc_funding REAL,
    eth_funding REAL,
    sol_funding REAL,
    cumulative_funding REAL,
    btc_peak REAL,
    eth_peak REAL,
    sol_peak REAL,
    btc_stopped INTEGER,
    eth_stopped INTEGER,
    sol_stopped INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE v4d_trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    asset TEXT NOT NULL,
    action TEXT NOT NULL,  -- 'entry', 'exit', 'stop', 'dd_breaker', 'vol_ceiling'
    price REAL,
    position_before REAL,
    position_after REAL,
    reason TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_v4d_trade_date ON v4d_trade_log(date);
CREATE INDEX idx_v4d_trade_asset ON v4d_trade_log(asset);
```

---

## Signal Pipeline

Recomputes full 100-day history each run (~2s). Pure functions, no cross-day state persistence.

```
Binance OHLCV (100 daily candles per asset)
    ↓
Layer 1: SMA50 signal (close > SMA → 1, else → 0)
    ↓
Layer 2: No-lookahead shift (signal.shift(1) — execute next day)
    ↓
Layer 3: Trailing stop (per-asset: BTC 12%, ETH 15%, SOL 8%)
    ↓
Layer 4: Vol ceiling (halve position if 30d vol > 80% annualized, sqrt(365))
    ↓
Layer 5: DD breaker (go flat if portfolio DD > 25%, re-enter on recovery to -12.5% + new SMA50 cross)
    ↓
Equity = (1 + weighted_returns).cumprod() * $100,000
    ↓
Store daily_state + detect trades vs previous day → trade_log
    ↓
Discord alert (daily summary)
```

---

## Risks

| Risk | Severity | Status |
|------|----------|--------|
| `position.shift(0)` fragility | **CRITICAL** | Mitigated — exact 2-stage shift logic replicated from `v4_honest_system.py` |
| sqrt(252) contamination | **HIGH** | Mitigated — V4D uses `ANNUALIZATION_FACTOR = 365`, no shared vol utils imported |
| DD breaker state machine | **HIGH** | Implemented — re-entry requires recovery to -12.5% AND new SMA50 cross |
| SQLite corruption | **HIGH** | Mitigated — `INSERT OR REPLACE` on `date` PK prevents double-counting |
| Silent death | **HIGH** | Mitigated — staleness alert if no run in >26h; Virtuoso health check monitors timer |
| Price source divergence | **MEDIUM** | Accepted — backtest uses spot CSVs, paper uses Binance perps mark prices |
| Cron pile-up | **MEDIUM** | Mitigated — V4D at 00:15, V3.1-H2 at 00:05, staggered by 10 min |
| Backfill confusion | **MEDIUM** | Mitigated — forward vs backfill metrics separated in API and dashboard |

---

## Current State (as of 2026-02-20)

| Metric | Value |
|--------|-------|
| Days running | 2 (live since 2026-02-19) |
| Equity | $98,681 (starting equity after backfill) |
| Forward P&L | $0.00 (all positions flat) |
| BTC/ETH/SOL signals | All FLAT (below SMA50) |
| Vol ceiling | **ACTIVE** (ETH ~99%, SOL ~101% annualized vol) |
| DD breaker | Inactive (-9.1% drawdown, threshold -25%) |
| Cumulative funding | $0.00 (no open positions) |
| Total trades | 0 (forward period only) |

---

## Key Metrics (Paper vs Backtest)

Track 3 things from Day 1. Everything else derived from equity curve on demand.

| Metric | Target | Purpose |
|--------|--------|---------|
| Signal agreement rate | >95% | Core validity — do paper and backtest agree on entries/exits? |
| Equity curve (daily P&L) | Track | Compare to backtest equity curve at Day 30 |
| Cumulative funding drag | Track vs 8-15%/yr estimate | Cost model validation |

Day 30 derived metrics (from equity curve, not separately tracked):
- CAGR difference vs backtest (target: within 5% absolute)
- Rolling Sharpe correlation (target: >0.85)
- Drawdown timing agreement

---

## Expected Performance (Perps)

With 8-15%/yr funding drag vs spot backtest (Sharpe 1.52, CAGR 41.9%, MaxDD -18.7%):

- **Sharpe**: ~1.1-1.3
- **CAGR**: ~28-35%
- **MaxDD**: ~-20%

Funding drag is the cost of leverage infrastructure — these are the honest numbers.

---

## References

- Signal engine (VPS): `~/trading/Virtuoso/scripts/v4d_paper_trading.py`
- API routes (VPS): `~/trading/Virtuoso/src/api/routes/v4d_paper.py`
- Storage layer (VPS): `~/trading/Virtuoso/src/database/shadow_storage.py` (v4d functions)
- Dashboard (VPS): `/var/www/virtuosocrypto.com/quant/paper/index.html`
- Service files (VPS): `~/trading/Virtuoso/deploy/virtuoso-v4d-paper.{service,timer}`
- Strategy source (Maestro): `backend/strategies/composite/v4_honest_system.py`
- Full validation results: `docs/09-reports/audits/ULTRATHINK_REVIEW_V4.md`
- Original consensus matrix: `docs/V4D_PAPER_TRADING_CONSENSUS.md`
- Dashboard URL: [virtuosocrypto.com/quant/paper/](https://virtuosocrypto.com/quant/paper/)
