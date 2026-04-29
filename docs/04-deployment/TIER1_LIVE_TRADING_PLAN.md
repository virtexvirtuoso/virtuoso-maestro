# Tier 1 — Live Trading Execution Plan
## Maestro V3 Mega Strategy → Freqtrade on Bybit

**Created:** 2026-02-13  
**Strategy:** Macro Momentum V3 (5-signal confluence → regime → adaptive leverage)  
**Backtest Performance:** IS Sharpe 1.71, OOS Sharpe 0.84 (p=0.036), MaxDD -5.5%  
**Assets:** BTC (40%), ETH (25%), SOL (20%), LINK (15%)  
**Exchange:** Bybit Futures, Isolated Margin  
**VPS:** 5.223.63.4 (`ssh vps`), user `linuxuser`

---

## Table of Contents

1. [Pre-Deployment Checklist](#1-pre-deployment-checklist)
2. [VPS Setup](#2-vps-setup)
3. [Maestro API Deployment](#3-maestro-api-deployment)
4. [Dry-Run Phase](#4-dry-run-phase-2-4-weeks)
5. [Live Deployment](#5-live-deployment)
6. [Monitoring & Alerting](#6-monitoring--alerting)
7. [Risk Management](#7-risk-management)
8. [Timeline](#8-timeline)
9. [Capital Requirements](#9-capital-requirements)
10. [Success Metrics](#10-success-metrics)

---

## 1. Pre-Deployment Checklist

### 1.1 API Keys & Secrets

- [ ] **Bybit API key** — Create at https://www.bybit.com/app/user/api-management
  - Permissions: **Futures trading** (read + write), **Wallet** (read only)
  - IP whitelist: `5.223.63.4` (VPS IP only)
  - Do NOT enable withdrawal permissions
- [ ] **FRED API key** — Get from https://fred.stlouisfed.org/docs/api/api_key.html
- [ ] **Telegram bot token** — Create via @BotFather, get chat_id via @userinfobot
- [ ] **Maestro API key** — Generate: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

### 1.2 Strategy Files Verified

- [ ] `macro_momentum_v3_strategy.py` — compiles cleanly
- [ ] `macro_data_provider.py` — `--update` runs without errors
- [ ] `freqtrade_config.json` — valid JSON, all placeholders filled
- [ ] `maestro_engine.py` — generates `maestro_state.json` successfully
- [ ] `maestro_api.py` — starts and serves `/health` endpoint

```bash
# Verify locally
cd ~/Desktop/maestro/backend
python3 -m py_compile freqtrade/macro_momentum_v3_strategy.py
python3 -m py_compile freqtrade/macro_data_provider.py
python3 -m py_compile maestro_engine.py
python3 -m py_compile maestro_api.py
```

### 1.3 Bybit Account Setup

- [ ] Futures account funded with test amount ($100 minimum)
- [ ] All 4 pairs set to **Isolated Margin** mode manually in Bybit UI
- [ ] Leverage set to **1x** as default in Isolated mode (strategy controls actual leverage)
- [ ] Confirm trading fees: Maker 0.02%, Taker 0.055% (or VIP tier)

### 1.4 VPS Resources

- [ ] Confirm VPS has at least 2GB RAM free (`ssh vps "free -h"`)
- [ ] Confirm disk space > 5GB free (`ssh vps "df -h"`)
- [ ] Confirm no port conflicts: 8080 (FreqUI), 8090 (Maestro API)
- [ ] Confirm Virtuoso is stable and won't conflict

```bash
ssh vps "free -h && df -h / && ss -tlnp | grep -E '808[0-9]|809[0-9]'"
```

---

## 2. VPS Setup

### 2.1 Install Freqtrade

```bash
ssh vps

# Create directory structure
mkdir -p ~/trading/freqtrade-maestro
cd ~/trading/freqtrade-maestro

# Install via pip in venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install freqtrade

# Verify
freqtrade --version

# Create Freqtrade user_data structure
freqtrade create-userdir --userdir user_data
```

### 2.2 Deploy Strategy Files

```bash
# From LOCAL machine — run all at once
rsync -avz ~/Desktop/maestro/backend/freqtrade/macro_momentum_v3_strategy.py \
  vps:~/trading/freqtrade-maestro/user_data/strategies/

rsync -avz ~/Desktop/maestro/backend/freqtrade/macro_data_provider.py \
  vps:~/trading/freqtrade-maestro/user_data/strategies/

rsync -avz ~/Desktop/maestro/backend/freqtrade/freqtrade_config.json \
  vps:~/trading/freqtrade-maestro/user_data/
```

### 2.3 Configure Secrets on VPS

```bash
ssh vps

# Store secrets in environment file (not in config!)
cat > ~/trading/freqtrade-maestro/.env << 'EOF'
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here
FRED_API_KEY=your_fred_api_key_here
MAESTRO_API_KEY=your_generated_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
FREQTRADE_JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
FREQTRADE_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
EOF

chmod 600 ~/trading/freqtrade-maestro/.env
```

### 2.4 Update Freqtrade Config with Secrets

Create a wrapper script that injects secrets into config at runtime:

```bash
cat > ~/trading/freqtrade-maestro/start.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail

cd ~/trading/freqtrade-maestro
source .env
source .venv/bin/activate

# Inject secrets into config using jq
jq --arg key "$BYBIT_API_KEY" \
   --arg secret "$BYBIT_API_SECRET" \
   --arg jwt "$FREQTRADE_JWT_SECRET" \
   --arg pwd "$FREQTRADE_PASSWORD" \
   --arg tg_token "$TELEGRAM_BOT_TOKEN" \
   --arg tg_chat "$TELEGRAM_CHAT_ID" \
   '.exchange.key = $key |
    .exchange.secret = $secret |
    .api_server.jwt_secret_key = $jwt |
    .api_server.password = $pwd |
    .telegram.enabled = true |
    .telegram.token = $tg_token |
    .telegram.chat_id = $tg_chat' \
   user_data/freqtrade_config.json > /tmp/ft_config_live.json

exec freqtrade trade \
  --config /tmp/ft_config_live.json \
  --strategy MacroMomentumV3Strategy \
  --strategy-path user_data/strategies
SCRIPT

chmod +x ~/trading/freqtrade-maestro/start.sh
```

### 2.5 Install jq (if needed)

```bash
ssh vps "which jq || sudo apt-get install -y jq"
```

### 2.6 Create systemd Service

```bash
ssh vps

sudo tee /etc/systemd/system/freqtrade-maestro.service << 'EOF'
[Unit]
Description=Freqtrade Maestro V3 Mega Strategy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=linuxuser
Group=linuxuser
WorkingDirectory=/home/linuxuser/trading/freqtrade-maestro
ExecStart=/home/linuxuser/trading/freqtrade-maestro/start.sh
Restart=on-failure
RestartSec=60
StartLimitIntervalSec=300
StartLimitBurst=3

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/home/linuxuser/trading/freqtrade-maestro
ReadWritePaths=/tmp

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=freqtrade-maestro

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable freqtrade-maestro
```

### 2.7 Test Dry Run Start

```bash
# Start and watch logs
sudo systemctl start freqtrade-maestro
sudo journalctl -u freqtrade-maestro -f --no-pager

# Verify it connects to Bybit and loads strategy
# Expected: "Using strategy MacroMomentumV3Strategy" in logs
# Expected: "Pair whitelist: ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'LINK/USDT:USDT']"
```

---

## 3. Maestro API Deployment

### 3.1 Deploy Maestro Engine + API to VPS

```bash
# From LOCAL machine
rsync -avz ~/Desktop/maestro/backend/maestro_engine.py vps:~/trading/maestro/
rsync -avz ~/Desktop/maestro/backend/maestro_api.py vps:~/trading/maestro/
rsync -avz ~/Desktop/maestro/backend/maestro_cron.py vps:~/trading/maestro/

# Deploy dependencies (datasource, strategies modules)
rsync -avz ~/Desktop/maestro/backend/datasource/ vps:~/trading/maestro/datasource/
rsync -avz ~/Desktop/maestro/backend/strategies/ vps:~/trading/maestro/strategies/
rsync -avz ~/Desktop/maestro/backend/requirements.txt vps:~/trading/maestro/

# Create data directories
ssh vps "mkdir -p ~/trading/maestro/data/live"
```

### 3.2 Install Maestro Dependencies

```bash
ssh vps
cd ~/trading/maestro
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 3.3 Create Maestro Engine Systemd Service

```bash
sudo tee /etc/systemd/system/maestro-api.service << 'EOF'
[Unit]
Description=Maestro Quant API
After=network-online.target

[Service]
Type=simple
User=linuxuser
WorkingDirectory=/home/linuxuser/trading/maestro
Environment=MAESTRO_API_KEY=your_generated_api_key_here
Environment=FRED_API_KEY=your_fred_api_key_here
ExecStart=/home/linuxuser/trading/maestro/.venv/bin/uvicorn maestro_api:app --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable maestro-api
sudo systemctl start maestro-api
```

### 3.4 Daily Signal Cron (Maestro Engine)

```bash
ssh vps

# Create the daily runner script
cat > ~/trading/maestro/run_daily_signals.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd ~/trading/maestro
source .venv/bin/activate
source ~/trading/freqtrade-maestro/.env

export FRED_API_KEY
export MAESTRO_API_KEY

# Run engine — generates maestro_state.json
python3 maestro_engine.py 2>&1 | tee -a /tmp/maestro_engine.log

# Also update macro cache for Freqtrade strategy
cd ~/trading/freqtrade-maestro/user_data/strategies
python3 macro_data_provider.py --update 2>&1 | tee -a /tmp/macro_update.log

echo "[$(date -u)] Daily signal update complete"
SCRIPT

chmod +x ~/trading/maestro/run_daily_signals.sh

# Add to cron — run at 06:00 UTC daily (after most daily candle closes)
(crontab -l 2>/dev/null; echo "0 6 * * * /home/linuxuser/trading/maestro/run_daily_signals.sh >> /tmp/maestro_cron.log 2>&1") | crontab -

# Verify
crontab -l
```

### 3.5 Verify Signal Pipeline

```bash
ssh vps

# Run manually to test
~/trading/maestro/run_daily_signals.sh

# Check output
cat ~/trading/maestro/data/live/maestro_state.json | python3 -m json.tool | head -50

# Check macro cache
cat ~/trading/freqtrade-maestro/user_data/strategies/macro_cache.json | python3 -m json.tool

# Test API
curl -s -H "X-API-Key: your_api_key" http://127.0.0.1:8090/health
curl -s -H "X-API-Key: your_api_key" http://127.0.0.1:8090/signals | python3 -m json.tool
```

---

## 4. Dry-Run Phase (2-4 Weeks)

### 4.1 Dry-Run Configuration

The default config already has `"dry_run": true` and `"dry_run_wallet": 10000`. No changes needed. Verify:

```bash
ssh vps "cat ~/trading/freqtrade-maestro/user_data/freqtrade_config.json | python3 -c \"import json,sys; c=json.load(sys.stdin); print(f'dry_run={c[\\\"dry_run\\\"]}, wallet={c[\\\"dry_run_wallet\\\"]}')\""
# Expected: dry_run=True, wallet=10000
```

### 4.2 Daily Monitoring Checklist

Run this every day during dry-run:

```bash
ssh vps

# 1. Service health
sudo systemctl status freqtrade-maestro --no-pager
sudo systemctl status maestro-api --no-pager

# 2. Recent trades
source ~/trading/freqtrade-maestro/.venv/bin/activate
cd ~/trading/freqtrade-maestro
freqtrade show-trades --config user_data/freqtrade_config.json --print-json | python3 -m json.tool

# 3. Current open positions
freqtrade status --config user_data/freqtrade_config.json

# 4. Profit summary
freqtrade profit --config user_data/freqtrade_config.json

# 5. Recent errors
sudo journalctl -u freqtrade-maestro --since "24 hours ago" --no-pager | grep -i "error\|warning\|exception" | tail -20

# 6. Macro cache freshness
ls -la ~/trading/freqtrade-maestro/user_data/strategies/macro_cache.json
cat ~/trading/freqtrade-maestro/user_data/strategies/macro_cache.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Updated: {d[\"timestamp\"]}, Boost: {d[\"confluence_boost\"]}')"
```

### 4.3 Metrics to Track

| Metric | Target | Red Flag |
|--------|--------|----------|
| Trade count (per week) | 2-8 trades | 0 or >20 |
| Win rate | >45% | <30% |
| Avg profit per trade | >0.5% | <-1% |
| Max single loss | <-8% | >-15% |
| Sharpe (rolling 2w) | >0.5 | <-0.5 |
| Max drawdown | <-8% | >-12% |
| Service uptime | >99% | <95% |
| Macro cache age | <48h | >72h |
| Regime distribution | Mix of BULL/MILD_BULL/NEUTRAL | Stuck on one |
| Leverage used | 0.3x-2.0x range | Always max |

### 4.4 Success Criteria to Go Live

**ALL must be met before switching off dry-run:**

1. ✅ **Minimum 14 days** of uninterrupted dry-run
2. ✅ **At least 10 closed trades** (statistical minimum)
3. ✅ **Positive P&L** on dry-run wallet
4. ✅ **No service crashes** lasting >1 hour
5. ✅ **Macro cache updating** reliably (no stale data >48h)
6. ✅ **Signal alignment** — entry/exit tags match expected regime behavior
7. ✅ **Leverage behavior** — observed leverage within 0.3x-2.0x range, not always max
8. ✅ **Pyramiding behavior** — position adjustments follow dip rules correctly
9. ✅ **Max drawdown <10%** during dry-run period

### 4.5 Weekly Dry-Run Reports

Create a tracking spreadsheet or log at `~/trading/freqtrade-maestro/dry_run_log.md`:

```markdown
# Dry Run Log

## Week 1 (YYYY-MM-DD to YYYY-MM-DD)
- Trades opened: X
- Trades closed: X
- Win rate: X%
- P&L: +/-$X
- Max drawdown: -X%
- Issues: None / [describe]
- Regime distribution: BULL X%, MILD_BULL X%, NEUTRAL X%, BEAR X%
```

---

## 5. Live Deployment

### 5.1 Go-Live Procedure

```bash
ssh vps

# 1. Stop the bot
sudo systemctl stop freqtrade-maestro

# 2. Verify Bybit balance
source ~/trading/freqtrade-maestro/.venv/bin/activate
cd ~/trading/freqtrade-maestro
python3 -c "
import ccxt
import os
ex = ccxt.bybit({'apiKey': os.environ.get('BYBIT_API_KEY',''), 'secret': os.environ.get('BYBIT_API_SECRET','')})
bal = ex.fetch_balance()
print(f'USDT Free: {bal[\"USDT\"][\"free\"]:.2f}')
print(f'USDT Total: {bal[\"USDT\"][\"total\"]:.2f}')
"

# 3. Update config: dry_run=false, conservative balance ratio
cd ~/trading/freqtrade-maestro/user_data
# Edit freqtrade_config.json:
python3 -c "
import json
with open('freqtrade_config.json') as f: c = json.load(f)
c['dry_run'] = False
c['tradable_balance_ratio'] = 0.30  # Start with 30% of capital
c['max_open_trades'] = 4  # Start with fewer positions
with open('freqtrade_config.json', 'w') as f: json.dump(c, f, indent=4)
print('Config updated: dry_run=False, balance_ratio=0.30, max_trades=4')
"

# 4. Restart
sudo systemctl start freqtrade-maestro

# 5. Watch first candle processing
sudo journalctl -u freqtrade-maestro -f --no-pager
```

### 5.2 Capital Allocation — Phased Scaling

| Phase | Duration | `tradable_balance_ratio` | `max_open_trades` | Effective Capital |
|-------|----------|--------------------------|--------------------|--------------------|
| **Phase 1** | Week 1-2 | 0.30 | 4 | 30% of account |
| **Phase 2** | Week 3-4 | 0.50 | 6 | 50% of account |
| **Phase 3** | Month 2+ | 0.75 | 8 | 75% of account |
| **Phase 4** | Month 3+ (if performing) | 0.95 | 8 | 95% of account |

**Scaling rules:**
- Scale up ONLY if cumulative P&L is positive at end of each phase
- Scale up ONLY if max drawdown stayed within -8%
- If drawdown exceeds -5%, scale DOWN one phase
- If drawdown exceeds -10%, STOP trading (see kill switch §7.1)

### 5.3 Scaling Config Updates

```bash
# Phase 2 example (after 2 weeks of profitable live trading):
ssh vps "cd ~/trading/freqtrade-maestro && source .venv/bin/activate && python3 -c \"
import json
with open('user_data/freqtrade_config.json') as f: c = json.load(f)
c['tradable_balance_ratio'] = 0.50
c['max_open_trades'] = 6
with open('user_data/freqtrade_config.json', 'w') as f: json.dump(c, f, indent=4)
print('Scaled to Phase 2')
\""
ssh vps "sudo systemctl restart freqtrade-maestro"
```

---

## 6. Monitoring & Alerting

### 6.1 Telegram Alerts (Built-in Freqtrade)

Already configured via `start.sh` which injects Telegram credentials. Alerts include:
- ✅ Entry signals (pair, direction, leverage, entry_tag)
- ✅ Exit signals (pair, profit%, exit_tag)
- ✅ Bot startup/shutdown
- ✅ Warnings and errors

### 6.2 Daily P&L Report Script

```bash
cat > ~/trading/freqtrade-maestro/daily_report.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail
cd ~/trading/freqtrade-maestro
source .venv/bin/activate
source .env

# Get profit summary
REPORT=$(freqtrade profit --config user_data/freqtrade_config.json 2>/dev/null || echo "Error fetching profit")

# Get open trades
OPEN=$(freqtrade status --config user_data/freqtrade_config.json 2>/dev/null || echo "No open trades")

# Service uptime
UPTIME=$(systemctl show freqtrade-maestro --property=ActiveEnterTimestamp --value 2>/dev/null || echo "Unknown")

# Macro cache status
MACRO_AGE=""
if [ -f user_data/strategies/macro_cache.json ]; then
    MACRO_TS=$(python3 -c "import json; print(json.load(open('user_data/strategies/macro_cache.json'))['timestamp'])" 2>/dev/null || echo "unknown")
    MACRO_BOOST=$(python3 -c "import json; print(json.load(open('user_data/strategies/macro_cache.json'))['confluence_boost'])" 2>/dev/null || echo "?")
    MACRO_AGE="Macro cache: ${MACRO_TS}, boost: ${MACRO_BOOST}"
fi

# Send via Telegram
MSG="📊 *Maestro V3 Daily Report*
$(date -u '+%Y-%m-%d %H:%M UTC')

*Profit Summary:*
\`\`\`
${REPORT}
\`\`\`

*Open Positions:*
\`\`\`
${OPEN}
\`\`\`

*Service Up Since:* ${UPTIME}
*${MACRO_AGE}*"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" \
  -d parse_mode="Markdown" \
  -d text="${MSG}" > /dev/null

echo "[$(date -u)] Daily report sent"
SCRIPT

chmod +x ~/trading/freqtrade-maestro/daily_report.sh

# Cron: daily at 00:05 UTC
(crontab -l 2>/dev/null; echo "5 0 * * * /home/linuxuser/trading/freqtrade-maestro/daily_report.sh >> /tmp/daily_report.log 2>&1") | crontab -
```

### 6.3 Health Check Script (Every 5 Minutes)

```bash
cat > ~/trading/freqtrade-maestro/healthcheck.sh << 'SCRIPT'
#!/bin/bash
source ~/trading/freqtrade-maestro/.env

# Check if freqtrade service is running
if ! systemctl is-active --quiet freqtrade-maestro; then
    MSG="🚨 *ALERT: Freqtrade Maestro V3 is DOWN!*
$(date -u '+%Y-%m-%d %H:%M UTC')
Attempting restart..."
    
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      -d chat_id="${TELEGRAM_CHAT_ID}" \
      -d parse_mode="Markdown" \
      -d text="${MSG}" > /dev/null
    
    sudo systemctl restart freqtrade-maestro
fi

# Check maestro-api
if ! systemctl is-active --quiet maestro-api; then
    sudo systemctl restart maestro-api
fi
SCRIPT

chmod +x ~/trading/freqtrade-maestro/healthcheck.sh

# Cron: every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/linuxuser/trading/freqtrade-maestro/healthcheck.sh >> /tmp/healthcheck.log 2>&1") | crontab -
```

### 6.4 Circuit Breakers

Built into the strategy:
- **`custom_exit` → `emergency_drawdown`**: Closes position if loss > 2× trail_stop (24% BTC, 40% ETH, 20% SOL, 17.2% LINK)
- **`custom_exit` → `stale_trade_exit`**: Closes if >30 days with <0.5% profit
- **`confirm_trade_entry`**: Blocks entries when confluence=0 (leverage_mult=0)

External circuit breaker (add to healthcheck):

```bash
# Add to healthcheck.sh — kill switch if equity drops >15% from peak
cat >> ~/trading/freqtrade-maestro/healthcheck.sh << 'BREAKER'

# Circuit breaker: check drawdown
cd ~/trading/freqtrade-maestro
source .venv/bin/activate
DD=$(python3 -c "
import json, ccxt, os
ex = ccxt.bybit({'apiKey': os.environ.get('BYBIT_API_KEY',''), 'secret': os.environ.get('BYBIT_API_SECRET','')})
bal = ex.fetch_balance()
equity = float(bal.get('USDT',{}).get('total',0))
# Read peak from file
peak_file = '/tmp/maestro_equity_peak.txt'
try:
    peak = float(open(peak_file).read().strip())
except:
    peak = equity
if equity > peak:
    peak = equity
    open(peak_file,'w').write(str(peak))
dd = (equity - peak) / peak if peak > 0 else 0
print(f'{dd:.4f}')
" 2>/dev/null)

if [ -n "$DD" ]; then
    DD_PCT=$(echo "$DD" | python3 -c "import sys; v=float(sys.stdin.read()); print(f'{v*100:.1f}')")
    if (( $(echo "$DD < -0.15" | bc -l 2>/dev/null || echo 0) )); then
        sudo systemctl stop freqtrade-maestro
        MSG="🛑 *KILL SWITCH ACTIVATED*
Equity drawdown: ${DD_PCT}% (threshold: -15%)
Bot STOPPED. Manual review required."
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
          -d chat_id="${TELEGRAM_CHAT_ID}" \
          -d parse_mode="Markdown" \
          -d text="${MSG}" > /dev/null
    fi
fi
BREAKER
```

---

## 7. Risk Management

### 7.1 Max Drawdown Kill Switch

| Level | Drawdown | Action |
|-------|----------|--------|
| ⚠️ Warning | -8% | Telegram alert, review positions |
| 🟠 Reduce | -10% | Scale down to Phase 1 (30% capital) |
| 🛑 Stop | -15% | Stop bot, close all positions, manual review |
| ☠️ Emergency | -20% | Stop bot, withdraw remaining capital |

### 7.2 Per-Position Limits

Already enforced by strategy parameters:

| Asset | Max Leverage | Trail Stop | Emergency Exit (2× trail) | Max Pyramid Entries |
|-------|-------------|------------|---------------------------|---------------------|
| BTC | 3.0x | 12% | -24% | 4 |
| ETH | 2.5x | 20% | -40% | 4 |
| SOL | 2.0x | 10% | -20% | 4 |
| LINK | 2.0x | 8.6% | -17.2% | 4 |

### 7.3 Correlation Risk

All 4 assets are crypto — correlation is HIGH (typically 0.7-0.9).

**Mitigations:**
- `max_open_trades = 4-8` limits total exposure
- `tradable_balance_ratio < 1.0` keeps cash buffer
- Regime detection tends to go bearish across all assets simultaneously → shorts or no-trade
- Adaptive leverage reduces exposure in weak regimes

**Manual rule:** If all 4 assets have open longs simultaneously at >1.5x leverage, send a warning alert.

### 7.4 Liquidity Risk

All 4 pairs are top-20 by volume on Bybit. No liquidity concerns at <$500K position sizes.

### 7.5 Exchange Risk

- [ ] Never keep more than 50% of total crypto wealth on Bybit
- [ ] Enable 2FA on Bybit account
- [ ] API key restricted to VPS IP only
- [ ] No withdrawal permissions on API key

---

## 8. Timeline

### Week-by-Week Gantt

```
Week  0 (Feb 13-14): ████ VPS Setup + Freqtrade Install + File Deploy
Week  1 (Feb 17-21): ████ Maestro API Deploy + Signal Pipeline Test
Week  1 (Feb 17-21): ████ Start Dry-Run
Week  2 (Feb 24-28): ░░░░ Dry-Run Monitoring
Week  3 (Mar 03-07): ░░░░ Dry-Run Monitoring + First Weekly Report
Week  4 (Mar 10-14): ░░░░ Dry-Run Final Review + Go/No-Go Decision
Week  5 (Mar 17-21): ▓▓▓▓ LIVE Phase 1 — 30% capital, 4 max trades
Week  6 (Mar 24-28): ▓▓▓▓ LIVE Phase 1 — monitoring
Week  7 (Mar 31-Apr 4): ▓▓▓▓ LIVE Phase 2 — 50% capital (if Phase 1 profitable)
Week  8 (Apr 07-11): ▓▓▓▓ LIVE Phase 2 — monitoring
Week  9+ (Apr 14+):  ████ LIVE Phase 3 — 75% capital (if cumulative profitable)

████ = Active setup/deployment
░░░░ = Dry-run (paper trading)
▓▓▓▓ = Live trading
```

### Key Milestones

| Date | Milestone | Decision |
|------|-----------|----------|
| Feb 14 | VPS + Freqtrade operational | Go/no-go for dry-run |
| Feb 21 | Signal pipeline validated | Confirm signals make sense |
| Mar 14 | Dry-run complete | Go/no-go for live (see §4.4 criteria) |
| Mar 28 | Phase 1 review | Scale up or hold |
| Apr 11 | Phase 2 review | Scale up or scale down |
| May 14 | 3-month live review | Full strategy assessment |

---

## 9. Capital Requirements

### 9.1 Minimum Viable Capital

```
Minimum per-asset position: $25 (Bybit minimum ~$10 but need room for pyramiding)
4 assets × $25 = $100 base
With pyramiding (4 adds at 50% decreasing): $100 × 1.9375 = ~$194
With leverage headroom: $194 / 0.95 (balance ratio) = ~$204

Absolute minimum: $250 USDT (impractical for fees)
```

### 9.2 Recommended Capital

```
Target position size per asset (meaningful but not reckless):
  BTC (40%): $2,000
  ETH (25%): $1,250
  SOL (20%): $1,000
  LINK (15%): $750
  Total: $5,000

With Phase 1 (30% ratio): Need $5,000 / 0.30 = ~$16,700 in account
With Phase 3 (75% ratio): Need $5,000 / 0.75 = ~$6,700 in account

Recommended starting balance: $5,000 - $10,000 USDT
Ideal balance for full strategy: $15,000 - $25,000 USDT
```

### 9.3 Fee Analysis

```
Bybit Futures Fees (non-VIP):
  Maker: 0.02%
  Taker: 0.055%

Strategy uses limit orders (maker) for entry/exit, market for stoploss.

Expected fee drag per round-trip:
  Normal exit: 0.02% + 0.02% = 0.04% (both limit)
  Stoploss exit: 0.02% + 0.055% = 0.075% (limit entry + market stop)

With average 2x leverage: fees are on notional, so effective:
  Normal: 0.08%
  Stoploss: 0.15%

Estimated trades per month: 8-16 round trips
Monthly fee drag: 8 × 0.08% = 0.64% to 16 × 0.15% = 2.4%
Expected range: 0.6% - 1.5% per month

Funding rate impact (holding overnight on futures):
  Average: ±0.01% per 8h = ±0.03% per day
  30-day hold: ±0.9% (can be positive or negative)
```

---

## 10. Success Metrics

### 10.1 One Week (Live)

| Metric | "Working" | "Concerning" |
|--------|-----------|--------------|
| Bot uptime | >99% | <95% |
| Trades executed | 1-4 | 0 or >10 |
| Signal quality | Entry tags match regime | Random entries |
| Leverage | Varies with confluence | Always 1x or always max |
| P&L | Any (too early) | >-5% |
| Drawdown | <-3% | >-5% |

### 10.2 One Month (Live)

| Metric | "Working" | "Concerning" |
|--------|-----------|--------------|
| Sharpe ratio | >0.3 (annualized) | <0 |
| Total return | >0% | <-5% |
| Max drawdown | <-8% | >-12% |
| Win rate | >40% | <25% |
| Avg win/loss ratio | >1.5 | <0.8 |
| Trade count | 8-20 | <4 or >40 |
| Regime accuracy | Shorts in downtrends, longs in uptrends | Reversed |

### 10.3 Three Months (Live)

| Metric | "Working" | "Beating baseline" |
|--------|-----------|-------------------|
| Sharpe ratio | >0.5 annualized | >0.8 (approaching OOS) |
| Total return | >3% | >8% |
| Max drawdown | <-8% | <-5.5% (matching backtest) |
| Return/MaxDD | >0.5 | >1.0 |
| vs Buy & Hold | Outperforming risk-adjusted | Outperforming absolute |
| Monthly consistency | >2 of 3 months green | All 3 green |

### 10.4 Decision Framework at 3 Months

| Outcome | Action |
|---------|--------|
| Sharpe > 0.8, DD < 8% | Scale to Phase 4 (95% capital), consider adding assets |
| Sharpe 0.3-0.8, DD < 10% | Hold at current phase, review parameters |
| Sharpe 0-0.3, DD < 12% | Scale down to Phase 1, run 1 more month |
| Sharpe < 0, DD > 12% | STOP. Review strategy assumptions. Paper trade any changes. |
| DD > 15% at any point | STOP immediately (kill switch should have triggered) |

---

## Appendix A: Complete Cron Schedule

```
# Maestro engine — daily signal generation (06:00 UTC)
0 6 * * * /home/linuxuser/trading/maestro/run_daily_signals.sh >> /tmp/maestro_cron.log 2>&1

# Daily P&L report (00:05 UTC)
5 0 * * * /home/linuxuser/trading/freqtrade-maestro/daily_report.sh >> /tmp/daily_report.log 2>&1

# Health check (every 5 minutes)
*/5 * * * * /home/linuxuser/trading/freqtrade-maestro/healthcheck.sh >> /tmp/healthcheck.log 2>&1
```

## Appendix B: Quick Reference Commands

```bash
# Start/stop/restart
sudo systemctl start freqtrade-maestro
sudo systemctl stop freqtrade-maestro
sudo systemctl restart freqtrade-maestro

# Logs
sudo journalctl -u freqtrade-maestro -f
sudo journalctl -u freqtrade-maestro --since "1 hour ago"
sudo journalctl -u maestro-api -f

# Freqtrade CLI
cd ~/trading/freqtrade-maestro && source .venv/bin/activate
freqtrade status --config user_data/freqtrade_config.json
freqtrade profit --config user_data/freqtrade_config.json
freqtrade show-trades --config user_data/freqtrade_config.json

# FreqUI (web dashboard) — SSH tunnel from local
ssh -L 8080:localhost:8080 vps
# Then open http://localhost:8080 in browser

# Emergency: close all positions and stop
sudo systemctl stop freqtrade-maestro
cd ~/trading/freqtrade-maestro && source .venv/bin/activate
# Then manually close positions via Bybit UI
```

## Appendix C: File Locations on VPS

```
~/trading/freqtrade-maestro/
├── .env                              # API keys (chmod 600)
├── .venv/                            # Python venv
├── start.sh                          # Startup script (injects secrets)
├── daily_report.sh                   # Daily Telegram report
├── healthcheck.sh                    # Service health + circuit breaker
├── dry_run_log.md                    # Manual tracking log
└── user_data/
    ├── freqtrade_config.json         # Main config
    ├── strategies/
    │   ├── macro_momentum_v3_strategy.py
    │   ├── macro_data_provider.py
    │   └── macro_cache.json          # Auto-updated by cron
    ├── logs/
    │   └── freqtrade.log
    └── tradesv3.sqlite               # Trade database

~/trading/maestro/
├── .venv/                            # Python venv
├── maestro_engine.py                 # Daily signal generator
├── maestro_api.py                    # REST API (port 8090)
├── run_daily_signals.sh              # Cron runner
├── datasource/                       # Data loaders
└── strategies/                       # Strategy modules

~/trading/maestro/data/live/
└── maestro_state.json                # Generated state file
```

---

*Plan authored 2026-02-13. Review and update after each phase transition.*
