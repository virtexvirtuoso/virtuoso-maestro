# Maestro → Virtuoso MCP Integration Architecture

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ yFinance │  │   FRED   │  │Coinalyze │  │ Virtuoso MCP  │  │
│  │(BTC,ETH, │  │(M2,CPI,  │  │(OI,Fund, │  │(27 existing   │  │
│  │ SOL,LINK)│  │ FFR,T10Y)│  │ Liq,LSR) │  │ tools)        │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │              │              │                │          │
└───────┼──────────────┼──────────────┼────────────────┼──────────┘
        │              │              │                │
        ▼              ▼              ▼                │
┌─────────────────────────────────────────┐            │
│         MAESTRO ENGINE (Daily Cron)     │            │
│                                         │            │
│  1. Fetch prices + macro + cross-asset  │            │
│  2. Confluence engine (5 signals)       │            │
│  3. Regime detection                    │            │
│  4. Per-asset signals (entry/exit)      │            │
│  5. Adaptive leverage                   │            │
│  6. Portfolio allocation                │            │
│  7. ML enhancement (optional)           │            │
│  8. Write state JSON (atomic)           │            │
│                                         │            │
└──────────────┬──────────────────────────┘            │
               │                                       │
               ▼                                       │
┌──────────────────────────┐                           │
│  maestro_state.json      │                           │
│  ~/data/live/            │                           │
│  (single source of truth)│                           │
└──────┬───────────┬───────┘                           │
       │           │                                   │
       ▼           ▼                                   │
┌──────────┐ ┌─────────────────┐                       │
│ Maestro  │ │   MCP Bridge    │                       │
│ REST API │ │ (5 new tools)   │◄──────────────────────┘
│ :8090    │ │                 │
│          │ │ maestro.regime  │
│ /regime  │ │ maestro.signals │
│ /signals │ │ maestro.alloc   │
│ /alloc   │ │ maestro.risk    │
│ /risk    │ │ maestro.report  │
│ /state   │ │                 │
│ /report  │ └────────┬────────┘
│ /health  │          │
└──────┬───┘          │
       │              ▼
       │    ┌────────────────────┐
       │    │  Virtuoso MCP      │
       │    │  virtuosocrypto.com│
       │    │  (27 + 5 = 32     │
       │    │   total tools)    │
       │    └────────┬──────────┘
       │             │
       ▼             ▼
┌──────────────────────────┐
│      CONSUMERS           │
│                          │
│  ┌──────────┐ ┌────────┐│
│  │Freqtrade │ │Claude/ ││
│  │IStrategy │ │ChatGPT ││
│  │(reads    │ │(via MCP)││
│  │ state.json│ │        ││
│  └──────────┘ └────────┘│
│  ┌──────────┐ ┌────────┐│
│  │Dashboard │ │Telegram││
│  │(REST API)│ │(alerts)││
│  └──────────┘ └────────┘│
└──────────────────────────┘
```

## Cron Schedule

| Time (UTC) | Job | Description |
|---|---|---|
| 06:00 | `maestro_cron.py` | Full daily run (prices + macro + signals) |
| 18:00 | `maestro_cron.py` | Mid-day refresh (catch intraday moves) |

```bash
# Crontab
0 6 * * * cd ~/trading/maestro && ./.venv/bin/python maestro_cron.py >> ~/trading/maestro/logs/cron.log 2>&1
0 18 * * * cd ~/trading/maestro && ./.venv/bin/python maestro_cron.py >> ~/trading/maestro/logs/cron.log 2>&1
```

## Deployment Steps (VPS)

```bash
# 1. Clone/sync to VPS
rsync -avz ~/Desktop/maestro/ vps:~/trading/maestro/ --exclude .venv --exclude __pycache__

# 2. Setup venv on VPS
ssh vps "cd ~/trading/maestro && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install fastapi uvicorn requests"

# 3. Set environment variables
ssh vps "echo 'export MAESTRO_API_KEY=your-secure-key' >> ~/.zshrc"
ssh vps "echo 'export FRED_API_KEY=your-fred-key' >> ~/.zshrc"

# 4. Create systemd service for API
cat <<EOF | ssh vps "sudo tee /etc/systemd/system/maestro-api.service"
[Unit]
Description=Maestro Quant API
After=network.target

[Service]
Type=simple
User=linuxuser
WorkingDirectory=/home/linuxuser/trading/maestro
ExecStart=/home/linuxuser/trading/maestro/.venv/bin/uvicorn maestro_api:app --host 127.0.0.1 --port 8090
Restart=always
Environment=MAESTRO_API_KEY=your-secure-key

[Install]
WantedBy=multi-user.target
EOF

# 5. Enable and start
ssh vps "sudo systemctl daemon-reload && sudo systemctl enable maestro-api && sudo systemctl start maestro-api"

# 6. Setup cron
ssh vps 'crontab -l 2>/dev/null; echo "0 6 * * * cd ~/trading/maestro && ./.venv/bin/python maestro_cron.py >> ~/trading/maestro/logs/cron.log 2>&1"' | ssh vps 'crontab -'

# 7. Run initial engine
ssh vps "cd ~/trading/maestro && source .venv/bin/activate && python maestro_cron.py"

# 8. Test API
curl -H "X-API-Key: your-secure-key" http://127.0.0.1:8090/api/health
```

## Adding New Signals/Features

### Add a new confluence signal:
1. Edit `strategies/composite/mega_strategy_v3.py` → add `sig6` in `compute_confluence()`
2. Update score range (0-6)
3. Update `maestro_engine.py` confluence signal names
4. Run integration test

### Add a new asset:
1. Add to `CRYPTO_TICKERS` and `ASSET_CONFIGS` in mega_strategy_v3.py
2. Add to `ASSETS` list in maestro_engine.py
3. Add base weight in `BASE_WEIGHTS`
4. Run engine, verify signals

### Add a new MCP tool:
1. Add handler function in `maestro_mcp_bridge.py`
2. Add tool definition to `MCP_TOOLS` list
3. Add corresponding API endpoint in `maestro_api.py` (if needed)
4. Register with Virtuoso MCP server

### Add ML model:
1. Train model, save to `~/trading/maestro/ml/models/`
2. ML enhancement in engine auto-detects trained models
3. Falls back to rule-based if models not available

## Key Files

| File | Purpose |
|---|---|
| `maestro_engine.py` | Core daily signal engine |
| `maestro_api.py` | FastAPI REST server |
| `maestro_mcp_bridge.py` | MCP tool definitions for Virtuoso |
| `maestro_cron.py` | Cron runner with alerting |
| `maestro_integration_test.py` | Full system test |
| `data/live/maestro_state.json` | Single source of truth (atomic write) |
