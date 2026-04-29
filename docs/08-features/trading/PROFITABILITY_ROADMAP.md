# PROFITABILITY ROADMAP — Maestro/Virtuoso Trading Ecosystem

**Version:** 1.0
**Date:** 2026-02-13
**Author:** Maestro Quant System
**Status:** ACTIVE

---

## Executive Summary

Three-tier monetization plan built on validated infrastructure: a walk-forward proven strategy (V3 Macro Momentum, OOS Sharpe 0.84, p=0.036), 73 MCP tools, and production systems already running on VPS. Each tier funds and validates the next.

| Tier | Revenue Source | Timeline | Capital Required | Monthly Target (Steady State) |
|------|---------------|----------|-----------------|-------------------------------|
| 1 | Own Capital Trading | Weeks 1-8 | $10,000-$25,000 | $300-$1,250/mo (3-5% mo) |
| 2 | UNDERTOW Signal Subscription | Months 2-4 | $2,000 (tech/brand) | $5,000-$20,000/mo |
| 3 | Managed Capital / Fund | Months 4-12 | $5,000 (legal/infra) | $10,000-$100,000+/mo |

---

# TIER 0: INTELLECTUAL PROPERTY PROTECTION

Protect the edge before monetizing it. Quant funds treat IP protection as infrastructure, not overhead. Total budget for meaningful coverage: under $2,000.

## 0.1 Trademarks

Two marks to file via USPTO TEAS Plus (teas.uspto.gov):

| Mark | Classes | Goods/Services | Filing Fee |
|------|---------|---------------|------------|
| UNDERTOW | Class 36 | Financial analysis; providing financial information relating to cryptocurrency markets; financial signal services | $250/class |
| VIRTUOSO CRYPTO | Class 36, Class 42 | Class 36: same as above. Class 42: Software as a service (SaaS) featuring quantitative trading tools; computer software development in the field of cryptocurrency trading | $250/class x 2 = $500 |

**Total trademark cost: $750-$1,050** (depending on whether both marks filed simultaneously).

### Filing Steps

1. Search TESS (tmsearch.uspto.gov) for both "UNDERTOW" in Class 36 and "VIRTUOSO CRYPTO" in Classes 36/42. Look for live marks with identical or confusingly similar names in the same class. If clear, proceed.
2. Create a USPTO.gov account.
3. File via TEAS Plus ($250/class). Select pre-approved descriptions from the ID Manual where possible -- this avoids office actions and keeps the fee at $250 rather than $350.
4. Basis: Intent-to-Use (Section 1(b)) if not yet offering the service publicly. Switch to Use in Commerce (Section 1(a)) once Tier 2 launches.
5. Expect 8-12 months from filing to registration. Initial office action response (if any) is due within 3 months.
6. After registration: file Section 8 declaration between years 5-6, and Section 9 renewal every 10 years.

### Interim Protection

Use the TM symbol (not the registered R symbol) immediately upon filing. Common law rights attach at first commercial use even without registration.

## 0.2 Trade Secrets

This is the highest-value, lowest-cost IP protection available.

### What Quant Funds Protect

Renaissance Technologies, Two Sigma, and DE Shaw treat their alpha signals as trade secrets, not patents. Patents require public disclosure. Trade secrets last indefinitely -- as long as secrecy is maintained. The M2 acceleration thesis, confluence weights, regime detection parameters, walk-forward optimized thresholds, and ML model weights are all protectable as trade secrets.

### What Qualifies in This System

| Asset | Classification | Protection Method |
|-------|---------------|-------------------|
| M2 acceleration thesis and macro signal logic | Trade secret | Access controls, NDA |
| Optimized parameters (mega_v3_best_params.json) | Trade secret | Encrypted storage, access logs |
| Confluence weights and scoring methodology | Trade secret | NDA, need-to-know basis |
| Walk-forward validation pipeline | Trade secret | Private repos, access controls |
| Whale Hunter tracking methodology | Trade secret | NDA, compartmentalization |
| ML model weights and training data | Trade secret | Encrypted at rest, NDA |

### Required Measures

Trade secret protection requires demonstrating "reasonable measures" to maintain secrecy. Without these, courts will not enforce claims.

1. Mark all proprietary documents and code files with "CONFIDENTIAL -- TRADE SECRET" headers.
2. Restrict repository access. Private GitHub/GitLab repos with named collaborators only.
3. Use encrypted storage for parameter files and model weights (GPG or age encryption).
4. Maintain an access log documenting who has access to what.
5. Require NDAs from every contractor, collaborator, or beta tester who sees strategy logic.
6. Compartmentalize: signal subscribers see outputs, never inputs. UNDERTOW customers receive trade signals, not the model that generates them.

### Cost: $0 (internal discipline) + NDA drafting time.

## 0.3 Copyright

All original code is automatically copyrighted upon creation under the Copyright Act of 1976. No registration is required for protection to exist. However, registration at copyright.gov unlocks statutory damages ($750-$150,000 per infringement) and attorney fee recovery -- neither is available without registration.

### Registration Details

| Item | Detail |
|------|--------|
| Filing portal | copyright.gov/registration |
| Fee | $65 per work (single author, single work) |
| Turnaround | 3-6 months for standard processing |
| What to register | Source code (deposited as first 25 and last 25 pages, with trade secret portions redacted) |

### Priority Files for Registration

| File / Module | Reason |
|---------------|--------|
| maestro_engine.py | Core signal generation logic |
| macro_momentum_v3_strategy.py | Primary trading strategy |
| Whale Hunter system (aggregated as single work) | Proprietary tracking methodology |
| BTC Wiz on-chain analytics (aggregated as single work) | 27-signal composite system |

Register these four works as a group or individually. Total cost: $260 (4 x $65). File after Tier 1 deployment when the code is stable.

## 0.4 Patent Assessment

### Post-Alice Corp Reality

Since Alice Corp. v. CLS Bank (2014), the USPTO rejects approximately 90% of software patent applications directed to abstract ideas -- including financial algorithms and trading methods. The two-step Alice test asks: (1) Is the claim directed to an abstract idea? (2) Does it contain an "inventive concept" beyond the abstract idea?

Trading algorithms almost always fail Step 1. "Buy when M2 accelerates" is an abstract financial concept regardless of implementation.

### Where a Patent Might Survive

The ML pipeline component of the system -- specifically the walk-forward optimization with Optuna, the regime detection model, and the multi-signal confluence architecture -- might qualify as a "technical improvement to computer functionality" under Step 2. Courts have allowed patents for novel data processing pipelines that produce a concrete technical result.

| Factor | Assessment |
|--------|-----------|
| Odds of grant | 30-40% for the ML/pipeline claims, near 0% for the trading logic itself |
| Cost | $10,000-$20,000 (patent attorney filing + prosecution over 2-3 years) |
| Time to grant | 18-36 months |
| Provisional patent (12-month placeholder) | $1,600-$3,000 |

### Recommendation

Skip patent filing unless raising $1M+ in outside capital where a patent portfolio strengthens the pitch. The cost-benefit does not justify it at current scale. Trade secret protection covers the same assets at zero cost, without requiring public disclosure.

If circumstances change, file a provisional patent application first ($1,600-$3,000) to secure a priority date, then evaluate the full filing within the 12-month provisional window.

## 0.5 Timeline and Costs

| Action | Cost | Priority | Timeline |
|--------|------|----------|----------|
| Implement trade secret measures (headers, access controls, encryption) | $0 | P0 -- do immediately | Week 1 |
| Draft NDA for contractors and collaborators | $0 (template below) | P0 -- do immediately | Week 1 |
| Search TESS for UNDERTOW and VIRTUOSO CRYPTO | $0 | P0 | Week 1 |
| File UNDERTOW trademark (Class 36, TEAS Plus) | $250 | P1 | Week 2 |
| File VIRTUOSO CRYPTO trademark (Classes 36 + 42, TEAS Plus) | $500 | P1 | Week 2 |
| Register copyright for 4 key works at copyright.gov | $260 | P2 | Month 2 (after code stabilizes) |
| LLC formation (Wyoming, also protects IP holding) | $500-$800 | P1 | Month 2 |
| Patent provisional filing (only if raising $1M+) | $1,600-$3,000 | P3 -- skip unless fundraising | Month 6+ |
| Patent full filing (only if provisional filed) | $10,000-$20,000 | P3 -- skip unless fundraising | Month 12+ |

**Total for P0 + P1 + P2 actions: $1,010-$1,310.** Well under $2,000 for trademark, trade secret, and copyright protection across the entire system.

## 0.6 Contractor NDA Clause

Use the following clause in any agreement with contractors, collaborators, freelancers, or beta testers who access strategy logic, parameters, or source code:

```
CONFIDENTIALITY AND NON-DISCLOSURE

1. Definition. "Confidential Information" means all non-public information
   disclosed by the Company to the Recipient, including but not limited to:
   trading strategies, algorithms, source code, model parameters, optimization
   results, signal generation logic, confluence scoring methodology, data
   pipelines, and any derivative analysis -- whether disclosed orally, in
   writing, or through access to systems.

2. Obligations. The Recipient shall: (a) hold all Confidential Information
   in strict confidence; (b) not disclose Confidential Information to any
   third party without prior written consent; (c) not use Confidential
   Information for any purpose other than the agreed engagement; (d) limit
   access to Confidential Information to those with a need to know.

3. Duration. This obligation survives termination of the engagement and
   remains in effect for five (5) years from the date of disclosure, or
   indefinitely for trade secrets as defined under the Defend Trade Secrets
   Act (18 U.S.C. 1836).

4. Return of Materials. Upon termination or request, the Recipient shall
   return or destroy all copies of Confidential Information and certify
   destruction in writing.

5. Remedies. The Recipient acknowledges that breach may cause irreparable
   harm and that the Company is entitled to seek injunctive relief in
   addition to any other remedies available at law.
```

Embed this clause in contractor agreements, consulting agreements, and partnership MOUs. For UNDERTOW subscribers, the Terms of Service should include a prohibition on reverse-engineering signal logic.

---

# TIER 1: TRADE OWN CAPITAL

## 1.1 Pre-Deployment Checklist

Complete every item before deploying a single dollar.

| # | Item | Command / Action | Status |
|---|------|-----------------|--------|
| 1 | Bybit sub-account created (isolated from main) | Bybit web UI: Create Sub Account > API Trading | [ ] |
| 2 | API keys generated (IP-restricted to VPS) | Bybit > Sub Account > API Management > Whitelist 5.223.63.4 | [ ] |
| 3 | Freqtrade installed on VPS | See 1.2 | [ ] |
| 4 | Strategy files deployed | See 1.2 | [ ] |
| 5 | FRED API key set | `echo 'export FRED_API_KEY="xxx"' >> ~/.bashrc` | [ ] |
| 6 | Macro cache populated | `python macro_data_provider.py --update` | [ ] |
| 7 | Cron job active | `crontab -l` shows 6am UTC macro update | [ ] |
| 8 | Telegram bot configured | BotFather > token + chat_id in config | [ ] |
| 9 | Backtest on VPS matches local results | Compare Sharpe/DD within 5% | [ ] |
| 10 | Dry-run started | See 1.3 | [ ] |
| 11 | Kill switch tested | `systemctl stop freqtrade` closes cleanly | [ ] |
| 12 | Monitoring dashboard accessible | FreqUI via SSH tunnel or nginx | [ ] |

## 1.2 VPS Setup (Exact Commands)

```bash
# --- SSH into VPS ---
ssh vps

# --- Install Freqtrade ---
cd ~/trading
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
# Activate venv
source .venv/bin/activate

# --- Deploy strategy files from local ---
# (Run from local machine)
rsync -avz ~/Desktop/maestro/backend/freqtrade/ vps:~/trading/freqtrade-maestro/user_data/strategies/macro_v3/

# --- Deploy Maestro engine (signal generation) ---
rsync -avz ~/Desktop/maestro/backend/maestro_engine.py vps:~/trading/maestro/
rsync -avz ~/Desktop/maestro/backend/maestro_api.py vps:~/trading/maestro/
rsync -avz ~/Desktop/maestro/backend/maestro_cron.py vps:~/trading/maestro/
rsync -avz ~/Desktop/maestro/backend/datasource/ vps:~/trading/maestro/datasource/
rsync -avz ~/Desktop/maestro/backend/strategies/ vps:~/trading/maestro/strategies/
rsync -avz ~/Desktop/maestro/backend/ml/ vps:~/trading/maestro/ml/
rsync -avz ~/Desktop/maestro/data/optimization/mega_v3_best_params.json vps:~/trading/maestro/data/optimization/

# --- Back on VPS: configure ---
ssh vps
cd ~/trading/freqtrade-maestro

# Copy strategy to correct location
cp user_data/strategies/macro_v3/macro_momentum_v3_strategy.py user_data/strategies/
cp user_data/strategies/macro_v3/macro_data_provider.py user_data/strategies/
cp user_data/strategies/macro_v3/freqtrade_config.json user_data/

# Edit config
nano user_data/freqtrade_config.json
# Set: exchange.key, exchange.secret, telegram.token, telegram.chat_id
# Set: dry_run = true (initially)

# --- Environment variables ---
cat >> ~/.bashrc << 'EOF'
export FRED_API_KEY="your_fred_api_key"
export ALPHAVANTAGE_API_KEY="your_av_key"
export MAESTRO_ENV=production
export MAESTRO_LOG_LEVEL=INFO
EOF
source ~/.bashrc

# --- Macro data cron ---
(crontab -l 2>/dev/null; echo "0 6 * * * cd ~/trading/freqtrade-maestro/user_data/strategies && python macro_data_provider.py --update >> /tmp/macro_update.log 2>&1") | crontab -

# --- Initial macro cache build ---
cd ~/trading/freqtrade-maestro/user_data/strategies
python macro_data_provider.py --update

# --- Backtest validation on VPS ---
cd ~/trading/freqtrade-maestro
freqtrade backtesting \
    --config user_data/freqtrade_config.json \
    --strategy MacroMomentumV3Strategy \
    --timerange 20230101-20260101 \
    --timeframe 1d \
    --enable-position-stacking

# --- Systemd service ---
sudo tee /etc/systemd/system/freqtrade.service << 'EOF'
[Unit]
Description=Freqtrade MacroMomentumV3
After=network.target

[Service]
User=linuxuser
WorkingDirectory=/home/linuxuser/trading/freqtrade-maestro
ExecStart=/home/linuxuser/trading/freqtrade-maestro/.venv/bin/freqtrade trade --config user_data/freqtrade_config.json --strategy MacroMomentumV3Strategy
Restart=on-failure
RestartSec=30
Environment=FRED_API_KEY=your_fred_key

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable freqtrade

# --- Maestro API service ---
sudo tee /etc/systemd/system/maestro-api.service << 'EOF'
[Unit]
Description=Maestro API Server
After=network.target

[Service]
User=linuxuser
WorkingDirectory=/home/linuxuser/trading/maestro
ExecStart=/home/linuxuser/trading/maestro/.venv/bin/uvicorn maestro_api:app --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable maestro-api
sudo systemctl start maestro-api
```

## 1.3 Dry-Run Phase (Weeks 1-4)

**Duration:** Minimum 2 weeks, target 4 weeks.
**Purpose:** Validate signal generation, order execution, and system stability without capital at risk.

```bash
# Start dry run
sudo systemctl start freqtrade
# (config has dry_run: true)

# Monitor
journalctl -u freqtrade -f
# FreqUI access
ssh -L 8080:localhost:8080 vps
# Open http://localhost:8080
```

### Dry-Run Success Criteria

| Metric | Minimum | Target | Fail Action |
|--------|---------|--------|-------------|
| Uptime | 95% | 99% | Fix crash cause, restart timer |
| Signal alignment vs backtest | 80% match | 90% match | Debug signal pipeline |
| Regime detection accuracy | Matches Maestro API | Exact match | Fix data sync |
| Macro cache freshness | < 48h stale | < 24h | Fix cron |
| Telegram alerts firing | All entries/exits | + regime changes | Fix config |
| No spurious trades | 0 ghost trades | 0 | Debug immediately |

### Dry-Run Weekly Review

- Week 1: System stability, signal accuracy, alert reliability
- Week 2: Compare dry-run P&L to backtest expectations for same period
- Week 3: Stress test: manually trigger regime changes, verify behavior
- Week 4: Final sign-off. If all criteria met, proceed to live.

## 1.4 Live Deployment

### Capital Requirements

| Scenario | Starting Capital | Allocation | Expected Monthly Return | Monthly $ (conservative) |
|----------|-----------------|------------|------------------------|--------------------------|
| Minimum viable | $5,000 | 30% tradable | 3-5% | $45-$75 |
| Recommended | $10,000 | 50% tradable | 3-5% | $150-$250 |
| Full deployment | $25,000 | 80% tradable | 3-5% | $600-$1,000 |

### Position Sizing (per $10,000 capital, 50% tradable = $5,000 active)

| Asset | Weight | Notional | Max Leverage | Max Position |
|-------|--------|----------|-------------|-------------|
| BTC | 40% | $2,000 | 3.0x | $6,000 |
| ETH | 25% | $1,250 | 2.5x | $3,125 |
| SOL | 20% | $1,000 | 2.0x | $2,000 |
| LINK | 15% | $750 | 2.0x | $1,500 |

### Go-Live Commands

```bash
ssh vps
cd ~/trading/freqtrade-maestro

# Edit config: set dry_run to false, tradable_balance_ratio to 0.3
nano user_data/freqtrade_config.json
# Change: "dry_run": false
# Change: "tradable_balance_ratio": 0.3

# Restart
sudo systemctl restart freqtrade
journalctl -u freqtrade -f
```

### Scaling Schedule

| Week | tradable_balance_ratio | Effective Capital ($10k account) | Condition |
|------|----------------------|----------------------------------|-----------|
| 1-2 | 0.30 | $3,000 | No bugs, signals match |
| 3-4 | 0.50 | $5,000 | Positive P&L or < -3% DD |
| 5-8 | 0.80 | $8,000 | Track record validates |
| 9+ | 1.00 | $10,000 | Full confidence |

## 1.5 Risk Management and Kill Switches

### Automated Kill Switches (in freqtrade_config.json)

```json
{
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "stoploss": -0.08,
    "trailing_stop": true,
    "trailing_stop_positive": 0.03,
    "trailing_stop_positive_offset": 0.05,
    "max_open_trades": 4,
    "protections": [
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 30,
            "trade_limit": 5,
            "stop_duration_candles": 7,
            "max_allowed_drawdown": 0.15
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 14,
            "trade_limit": 3,
            "stop_duration_candles": 7,
            "only_per_pair": false
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ]
}
```

### Manual Kill Switch

```bash
# Emergency stop: closes all positions and halts
sudo systemctl stop freqtrade
# Verify positions closed on Bybit UI

# Graceful stop: finish current candle then halt
# Send /stop to Telegram bot
# Or: freqtrade show-trades, then freqtrade force-exit --all
```

### Drawdown Limits

| Level | Drawdown | Action |
|-------|----------|--------|
| Warning | -5% from peak | Telegram alert, reduce leverage to 0.5x max |
| Caution | -10% from peak | Auto-halt new entries for 7 candles (protection) |
| Critical | -15% from peak | Manual review required, systemctl stop |
| Absolute | -20% from peak | Full stop, no restart without code review |

## 1.6 Monitoring and Alerting

### Monitoring Stack

```bash
# Telegram alerts (built into Freqtrade)
# Config section:
{
    "telegram": {
        "enabled": true,
        "token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID",
        "notification_settings": {
            "status": "on",
            "warning": "on",
            "startup": "on",
            "entry": "on",
            "exit": "on",
            "entry_cancel": "on",
            "exit_cancel": "on",
            "protection_trigger": "on"
        }
    }
}

# Health check script (add to cron, every 5 min)
cat > ~/trading/scripts/health_check.sh << 'SCRIPT'
#!/bin/bash
FT_STATUS=$(systemctl is-active freqtrade)
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8090/health)
MACRO_AGE=$(find ~/trading/freqtrade-maestro/user_data/strategies/macro_cache.json -mmin +2880 2>/dev/null)

if [ "$FT_STATUS" != "active" ]; then
    curl -s -X POST "https://api.telegram.org/botYOUR_TOKEN/sendMessage" \
        -d chat_id=YOUR_CHAT_ID -d text="ALERT: Freqtrade service DOWN"
fi
if [ "$API_STATUS" != "200" ]; then
    curl -s -X POST "https://api.telegram.org/botYOUR_TOKEN/sendMessage" \
        -d chat_id=YOUR_CHAT_ID -d text="ALERT: Maestro API non-200: $API_STATUS"
fi
if [ -n "$MACRO_AGE" ]; then
    curl -s -X POST "https://api.telegram.org/botYOUR_TOKEN/sendMessage" \
        -d chat_id=YOUR_CHAT_ID -d text="ALERT: Macro cache stale (>48h)"
fi
SCRIPT
chmod +x ~/trading/scripts/health_check.sh

# Add to cron
(crontab -l 2>/dev/null; echo "*/5 * * * * ~/trading/scripts/health_check.sh") | crontab -
```

## 1.7 Week-by-Week Timeline

| Week | Phase | Key Actions | Success Gate |
|------|-------|-------------|-------------|
| 1 | Setup | VPS install, deploy files, configure, backtest on VPS | Backtest matches local within 5% |
| 2 | Dry Run | Start dry run, monitor signals, verify alerts | 95% uptime, signals firing |
| 3 | Dry Run | Compare dry-run performance to backtest expectations | No bugs, regime detection working |
| 4 | Dry Run | Final validation, stress test edge cases | All checklist items green |
| 5 | Live (0.3x) | Go live with 30% capital, monitor hourly first day | First trades execute correctly |
| 6 | Live (0.3x) | Daily monitoring, compare to expectations | DD < -5%, system stable |
| 7 | Live (0.5x) | Scale to 50% if Week 5-6 positive | Positive cumulative P&L |
| 8 | Live (0.5x) | Full monitoring, document track record | Track record sheet started |

## 1.8 Success Metrics

| Timeframe | Metric | Target | Red Flag |
|-----------|--------|--------|----------|
| 1 week | System uptime | > 99% | < 90% |
| 1 week | Signals match backtest regime | > 85% | < 70% |
| 1 week | Max intra-week DD | < -3% | > -5% |
| 1 month | Cumulative return | > +3% | < -8% |
| 1 month | Sharpe (annualized from daily) | > 0.5 | < 0 |
| 1 month | Number of trades | 5-15 | 0 or > 30 |
| 3 months | Cumulative return | > +12% | < -5% |
| 3 months | Sharpe (annualized) | > 0.8 | < 0.3 |
| 3 months | Max drawdown | < -8% | > -15% |
| 3 months | Win rate | > 45% | < 35% |

---

# TIER 2: SIGNAL SUBSCRIPTION -- UNDERTOW BRAND

## 2.1 Product Definition

**Brand:** UNDERTOW -- "The invisible force beneath the surface"
**Tagline:** Institutional-grade macro signals for crypto. The current you can't see.
**Positioning:** Not another TA group. Macro-liquidity-driven signals backed by M2 acceleration data, on-chain analytics (BTC Wiz, 27 signals), and whale flow intelligence (57% WR elite traders). Quant-validated. Walk-forward proven.

### Signal Products

| Signal Type | Source | Frequency | Format |
|-------------|--------|-----------|--------|
| Macro Regime | Maestro V3 (M2/DXY/Yield/Cross-Asset) | Daily | Regime label + confluence score (0-5) |
| Trade Signals | V3 strategy entries/exits | As generated (~2-5/week) | Asset, direction, leverage, entry zone, SL, TP |
| Whale Alerts | Whale Hunter (elite tracker) | Real-time | Trader, asset, direction, size, platform |
| On-Chain Pulse | BTC Wiz (RHODL/MVRV/NVT/etc) | Daily | Signal name, value, z-score, interpretation |
| Market Structure | MCP composite (73 tools) | Daily | Fear/Greed, liquidation risk, sector rotation |

### Signal Format (Telegram)

```
UNDERTOW SIGNAL | 2026-02-14

MACRO REGIME: BULL (Confluence 4/5)
M2 Accelerating | DXY Weakening | Yield Curve Steepening

TRADE: LONG BTC/USDT
Entry Zone: $94,200 - $95,000
Stop Loss: $91,500 (-3.2%)
Take Profit 1: $98,000 (+3.4%)
Take Profit 2: $102,000 (+7.6%)
Leverage: 1.5x (regime-adjusted)
Confidence: HIGH

Whale Confirmation: 3 elite traders opened BTC longs
  in last 4h (avg size $2.1M)
On-Chain: RHODL ratio rising, MVRV below 2.5 (not overheated)

---
Risk: This is analysis, not financial advice.
```

## 2.2 Pricing Tiers

Market context: Crypto signal services range from $30/month (basic TA groups) to $500+/month (institutional-grade). Popular services like CryptoSignals.org charge $42-$100/month. Palm Beach Confidential charges $199/month. Whale-tracking services like Copin.io charge $50-$200/month.

| Tier | Price | Content | Target Audience |
|------|-------|---------|-----------------|
| CURRENT | $49/month ($399/year) | Daily macro regime, weekly market brief, on-chain dashboard access | Retail traders wanting edge |
| SIGNAL | $99/month ($799/year) | Everything in CURRENT + real-time trade signals + whale alerts | Active traders, $10k-$100k accounts |
| DEPTH | $199/month ($1,599/year) | Everything in SIGNAL + parameter updates, regime model access, 1-on-1 monthly call | Serious traders, $100k+ accounts |

### Founding Member Pricing (first 50 subscribers)

| Tier | Launch Price | Savings |
|------|-------------|---------|
| CURRENT | $29/month | 41% off |
| SIGNAL | $69/month | 30% off |
| DEPTH | $149/month | 25% off |

## 2.3 Revenue Projections

| Month | CURRENT Subs | SIGNAL Subs | DEPTH Subs | MRR |
|-------|-------------|-------------|------------|-----|
| 1 (soft launch) | 10 | 5 | 1 | $984 |
| 3 | 30 | 15 | 5 | $3,860 |
| 6 | 80 | 40 | 12 | $9,688 |
| 12 | 200 | 100 | 30 | $25,600 |

Assumptions: 5% monthly churn, 10-15% month-over-month growth after launch.

## 2.4 Telegram Channel Setup

### Channel Structure

| Channel | Type | Purpose |
|---------|------|---------|
| @undertow_public | Public | Free content, market commentary, 1-2 delayed signals/week |
| @undertow_current | Private (paid) | Daily regime, weekly briefs, on-chain dashboard |
| @undertow_signal | Private (paid) | Real-time signals, whale alerts |
| @undertow_depth | Private (paid) | Full access + model updates |
| @undertow_bot | Bot | Subscription management, signal delivery |

### Payment Integration

Use one of:
- **InviteMember** (invitemember.com) -- Telegram-native subscription bot, Stripe integration, $0.29 + 3% per transaction
- **SendPulse** -- chatbot + payments
- Manual via Stripe Checkout links + webhook to grant channel access

```
Setup flow:
1. Create Telegram bot via @BotFather
2. Create private channels, add bot as admin
3. Connect InviteMember, configure tiers
4. Set up Stripe account for payments
5. Create invite links that route through InviteMember
```

## 2.5 Signal Pipeline (Automated)

```
Maestro Engine (cron, daily 6:15 UTC)
    |
    v
maestro_api.py /signals endpoint
    |
    v
Signal Formatter Script (Python)
    |
    +--> Telegram Bot API --> @undertow_signal channel
    +--> Telegram Bot API --> @undertow_current channel (regime only)
    |
Whale Hunter (real-time webhook)
    |
    v
Whale Alert Formatter
    |
    +--> Telegram Bot API --> @undertow_signal channel

BTC Wiz (cron, daily 7:00 UTC)
    |
    v
On-Chain Summary Formatter
    |
    +--> Telegram Bot API --> @undertow_current channel
```

### Signal Bot Code (skeleton)

```python
# ~/trading/undertow/signal_bot.py
import requests
import asyncio
from telegram import Bot

BOT_TOKEN = "YOUR_BOT_TOKEN"
SIGNAL_CHANNEL = "@undertow_signal"
CURRENT_CHANNEL = "@undertow_current"

MAESTRO_API = "http://localhost:8090"

async def publish_signals():
    bot = Bot(token=BOT_TOKEN)

    # Fetch from Maestro API
    resp = requests.get(f"{MAESTRO_API}/signals")
    data = resp.json()

    # Format regime message
    regime_msg = format_regime(data)
    await bot.send_message(chat_id=CURRENT_CHANNEL, text=regime_msg, parse_mode="Markdown")

    # Format trade signals
    if data.get("signals"):
        signal_msg = format_trade_signal(data)
        await bot.send_message(chat_id=SIGNAL_CHANNEL, text=signal_msg, parse_mode="Markdown")

if __name__ == "__main__":
    asyncio.run(publish_signals())
```

## 2.6 Content Strategy

| Day | Content | Channel |
|-----|---------|---------|
| Monday | Weekly Macro Outlook (M2, DXY, rates) | CURRENT + Public (delayed) |
| Tuesday | On-Chain Pulse (BTC Wiz top signals) | CURRENT |
| Wednesday | Whale Watch Weekly (top moves, success rates) | SIGNAL |
| Thursday | Mid-week Signal Update | SIGNAL |
| Friday | Week in Review + Weekend Positioning | CURRENT |
| Daily | Regime update + any triggered signals | CURRENT/SIGNAL |
| Real-time | Whale alerts, regime changes | SIGNAL |

### Public Channel Funnel Content (free, 2-3x/week)

- Delayed signals (24-48h after paid delivery)
- Educational: "Why M2 acceleration matters for crypto"
- Performance recaps (redacted entry/exit prices)
- On-chain chart of the week

## 2.7 Soft Launch Plan

| Week | Action |
|------|--------|
| 1-2 | Build signal bot, test formatting, set up channels |
| 3 | Create landing page on virtuosocrypto.com/undertow |
| 4 | Invite 10-20 beta testers (free access, collect feedback) |
| 5-6 | Iterate on signal format, timing, content based on feedback |
| 7 | Open founding member pricing (50 spots), announce on public channel |
| 8 | Begin paid content delivery, track retention |

## 2.8 Tech Build List

| Item | Effort | Priority |
|------|--------|----------|
| Telegram signal bot (Python) | 2-3 days | P0 |
| InviteMember integration | 1 day | P0 |
| Stripe account + webhook | 1 day | P0 |
| Landing page (virtuosocrypto.com/undertow) | 2 days | P0 |
| Signal formatter (Maestro API -> Telegram format) | 1 day | P0 |
| Whale alert auto-forward | 1 day | P1 |
| BTC Wiz daily summary bot | 1 day | P1 |
| Performance tracking dashboard | 3 days | P1 |
| Public channel content scheduler | 1 day | P2 |
| Email list + drip sequence | 2 days | P2 |

**Total build time: 2-3 weeks of focused work.**

## 2.9 Legal Disclaimers

Every signal message must include:

```
This is market analysis and commentary, not financial advice.
Past performance does not guarantee future results. Trading
crypto derivatives involves substantial risk of loss. UNDERTOW
does not manage funds or execute trades on your behalf. You are
solely responsible for your own trading decisions.
```

Additional requirements:
- Terms of Service on landing page
- No guaranteed returns in any marketing
- Disclose that the operator trades the same signals (Tier 1)
- Consider LLC formation for liability protection ($500-$800, Wyoming or Delaware)

---

# TIER 3: MANAGED CAPITAL / FUND

## 3.1 Milestone Gates (do not proceed without meeting these)

| Gate | Requirement | Measured By |
|------|-------------|-------------|
| G1: Track Record | 3 months live trading, Sharpe > 0.5 | Verified trade logs |
| G2: Drawdown Discipline | Max DD never exceeded -15% live | Equity curve |
| G3: Signal Revenue | UNDERTOW generating > $2,000/mo MRR | Stripe dashboard |
| G4: Legal Entity | LLC or equivalent formed | State filing |
| G5: Compliance Review | Lawyer consultation completed | Written opinion |

## 3.2 Track Record Platforms

### Crypto-Specific

| Platform | Type | Cost | Verification | Best For |
|----------|------|------|-------------|----------|
| Copin.io | Copy trading / track record | Free | On-chain (HyperLiquid) | Crypto-native audience |
| STFX | Vault-based trading | Free | On-chain | DeFi-native investors |
| dHEDGE | On-chain fund | Gas costs | Fully on-chain | Transparent track record |

### Traditional (crypto-compatible)

| Platform | Type | Cost | Verification | Best For |
|----------|------|------|-------------|----------|
| Collective2 | Signal tracking | $49-$99/mo | API-verified | Retail investors |
| Darwinex | Broker/fund platform | Free (funded) | Broker-verified | Prop capital (up to $500k) |
| IBKR Hedge Fund Marketplace | Listed fund | IBKR account | Broker-verified | Institutional LPs |

**Recommended path:** Start with Copin.io (free, crypto-native, verifiable), then add Collective2 for broader reach.

## 3.3 Fund Structures

### Option A: Friends & Family (Lowest Barrier)

| Aspect | Detail |
|--------|--------|
| Structure | Multi-member LLC (Wyoming) |
| AUM Target | $100,000 - $500,000 |
| Investors | < 10 accredited investors |
| Fee | 1% management + 15% performance (high-water mark) |
| Regulation | Exempt under SEC Rule 506(b) -- no general solicitation |
| Formation cost | $800 LLC + $3,000-$5,000 legal (operating agreement, PPM) |
| Timeline | 1-2 months after Gate G5 |

### Option B: Prop Firm Capital

| Firm | Capital Available | Profit Split | Requirements |
|------|-------------------|-------------|-------------|
| Darwinex Zero | Up to EUR 500,000 | 15% of profits | 6-month track record, D-Score > 60 |
| FTMO (crypto) | $10,000-$200,000 | 80-90% of profits | Pass challenge (10% profit, 5% max DD) |

| TopStep (futures) | $50,000-$150,000 | 80% of profits | Evaluation period |

**V3 stats vs prop firm requirements:**

| Metric | V3 (OOS) | FTMO Requirement | Darwinex Requirement |
|--------|----------|-----------------|---------------------|
| Max DD | -5.5% | < -10% (pass) | No hard limit (D-Score) |
| Monthly return | ~3-5% est. | > 10% in challenge | Consistent positive |
| Sharpe | 0.84 | N/A | > 0.5 preferred |
| Win rate | ~50% est. | N/A | N/A |

V3 is well within prop firm parameters. FTMO challenge is realistic.

### Option C: LP Fund (Highest Upside, Highest Barrier)

| Aspect | Detail |
|--------|--------|
| Structure | Cayman Islands LP or BVI fund |
| AUM Target | $1,000,000+ |
| Investors | Accredited / qualified purchasers |
| Fee | 2% management + 20% performance |
| Regulation | Cayman CIMA registration, US 3(c)(1) or 3(c)(7) exemption |
| Formation cost | $25,000-$50,000 (legal, admin, audit) |
| Timeline | 6-12 months |
| Admin | Fund administrator required ($1,000-$3,000/mo) |

**Recommendation: Start with Option A (F&F) and Option B (prop firms) simultaneously. Option C only after $500k+ AUM proven.**

## 3.4 Regulatory Landscape

### United States

- SEC: Investment Adviser registration exempt if < 15 clients and no public holding out (Section 203(b)(3))
- CFTC: Crypto futures may require CTA registration. Consult lawyer.
- State: Wyoming most favorable for crypto LLC
- Key risk: If managing > $25M or > 15 clients, SEC registration required

### Offshore Options

| Jurisdiction | Formation Cost | Annual Cost | Regulatory Body | Tax |
|-------------|---------------|-------------|-----------------|-----|
| Cayman Islands | $25,000-$40,000 | $10,000-$20,000 | CIMA | 0% |
| BVI | $15,000-$25,000 | $5,000-$10,000 | FSC | 0% |
| Singapore | $10,000-$20,000 | $5,000-$15,000 | MAS | 0-17% |
| Dubai (DIFC) | $15,000-$30,000 | $10,000-$20,000 | DFSA | 0% |

**Recommendation:** Wyoming LLC for F&F (Option A). Defer offshore until AUM justifies cost.

## 3.5 Pitch Deck Outline (10 slides)

| Slide | Content |
|-------|---------|
| 1. Cover | UNDERTOW Capital -- Macro-Liquidity Alpha in Crypto |
| 2. Problem | Crypto markets driven by macro (M2), but 95% of traders use only TA |
| 3. Edge | M2 acceleration is THE signal: BTC 102.8%/yr when M2 accelerating vs 2.1% when not |
| 4. Strategy | 5-signal confluence, regime detection, adaptive leverage, long/short |
| 5. Performance | Walk-forward OOS: Sharpe 0.84, MaxDD -5.5%, p=0.036, +23.1% in 2022 bear |
| 6. Technology | 73 MCP tools, Whale Hunter (57% WR), BTC Wiz (27 on-chain signals), automated execution |
| 7. Track Record | Live trading results (Tier 1), Copin.io verification, prop firm allocation |
| 8. Market | $3.2T crypto market, $50B+ crypto fund AUM, growing institutional demand |
| 9. Terms | 1/15 fee structure, high-water mark, monthly liquidity, $25k minimum |
| 10. Team | Founder background, technology stack, advisory |

## 3.6 Fee Structures

| AUM Tier | Management Fee | Performance Fee | High-Water Mark |
|----------|---------------|-----------------|-----------------|
| < $500k (F&F) | 1.0% | 15% | Yes |
| $500k - $2M | 1.5% | 20% | Yes |
| > $2M | 2.0% | 20% | Yes, with annual reset |

### Revenue at Scale

| AUM | Management Fee | Performance Fee (20% of 30% annual) | Total Annual |
|-----|---------------|--------------------------------------|-------------|
| $250,000 | $2,500 | $15,000 | $17,500 |
| $500,000 | $7,500 | $30,000 | $37,500 |
| $1,000,000 | $15,000 | $60,000 | $75,000 |
| $5,000,000 | $100,000 | $300,000 | $400,000 |

## 3.7 Capacity Analysis

| Factor | Limit | Notes |
|--------|-------|-------|
| BTC liquidity | $50M+ daily | No constraint below $10M AUM |
| ETH liquidity | $30M+ daily | No constraint below $5M AUM |
| SOL liquidity | $10M+ daily | May see slippage above $2M positions |
| LINK liquidity | $5M+ daily | Binding constraint: limit LINK to $1M positions |
| Estimated alpha decay | Begins ~$5M AUM | Monitor execution slippage quarterly |
| Hard capacity | ~$20M AUM | Beyond this, strategy diversification required |

## 3.8 Capital Raising Strategy

| Phase | Timeline | Target | Source | Method |
|-------|----------|--------|--------|--------|
| Seed | Month 4-6 | $100k-$250k | Personal network, F&F | Direct outreach, pitch deck |
| Early | Month 6-9 | $250k-$500k | UNDERTOW subscribers, crypto community | Track record + pitch |
| Growth | Month 9-12 | $500k-$2M | HNW individuals, crypto funds-of-funds | Introductions, conferences |
| Scale | Month 12+ | $2M-$10M | Family offices, allocators | Formal fund structure |

## 3.9 Twelve-Month Timeline

| Month | Milestone | Gate Required |
|-------|-----------|--------------|
| 1 | Tier 1 live trading begins | -- |
| 2 | UNDERTOW soft launch, prop firm challenge started | -- |
| 3 | 3-month live track record complete | G1 |
| 4 | Copin.io / Collective2 track record published | G1, G2 |
| 4 | Wyoming LLC formed | G4 |
| 5 | Lawyer consultation on fund structure | G5 |
| 5 | F&F fund opens to first investors | G1-G5 |
| 6 | FTMO / Darwinex challenge completed | -- |
| 6 | $250k AUM target | -- |
| 8 | UNDERTOW at $5k+ MRR | G3 |
| 9 | $500k AUM, evaluate offshore structure | -- |
| 12 | $1M+ AUM or decision to stay boutique | -- |

---

# COMBINED REVENUE PROJECTION

| Month | Tier 1 (Trading P&L) | Tier 2 (UNDERTOW MRR) | Tier 3 (Fund Fees) | Total Monthly |
|-------|---------------------|----------------------|-------------------|--------------|
| 1 | $0 (dry run) | $0 | $0 | $0 |
| 2 | $90-$150 | $0 | $0 | $90-$150 |
| 3 | $150-$250 | $984 | $0 | $1,134-$1,234 |
| 4 | $250-$400 | $2,000 | $0 | $2,250-$2,400 |
| 6 | $400-$700 | $5,000 | $1,500 | $6,900-$7,200 |
| 9 | $500-$1,000 | $10,000 | $3,000 | $13,500-$14,000 |
| 12 | $600-$1,250 | $20,000 | $6,000 | $26,600-$27,250 |

---

# APPENDIX A: CRITICAL COMMANDS REFERENCE

```bash
# --- Tier 1: Trading Operations ---
# Start/stop trading
sudo systemctl start freqtrade
sudo systemctl stop freqtrade
sudo systemctl restart freqtrade
journalctl -u freqtrade -f

# Emergency: force close all positions
cd ~/trading/freqtrade-maestro
source .venv/bin/activate
freqtrade force-exit --all --config user_data/freqtrade_config.json

# Check trades
freqtrade show-trades --config user_data/freqtrade_config.json

# FreqUI access
ssh -L 8080:localhost:8080 vps
# Then open http://localhost:8080

# Maestro API
sudo systemctl start maestro-api
curl http://localhost:8090/health
curl http://localhost:8090/signals

# --- Tier 2: Signal Operations ---
# Run signal bot manually
cd ~/trading/undertow
python signal_bot.py

# Check subscriber count (InviteMember API)
curl https://api.invitemember.com/v1/subscribers -H "Authorization: Bearer YOUR_KEY"
```

# APPENDIX B: RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Strategy underperforms OOS | Medium | High | 4-week dry run, gradual scaling, 15% DD kill switch |
| VPS downtime | Low | Medium | Health check cron, Telegram alerts, manual backup |
| Exchange API changes | Low | Medium | Pin Freqtrade version, test updates in dry run first |
| Macro data source failure | Low | Medium | Cache lasts 48h, multiple data sources (FRED + AV) |
| Regulatory action | Low | High | LLC formation, legal consultation, no US securities language |
| Subscriber churn (Tier 2) | Medium | Medium | Content quality, community, annual pricing discount |
| Key-person risk | Medium | High | Document everything, automate everything possible |

---

*Virtuoso's Take:*

The edge is real. M2 acceleration separating 102.8% annualized from 2.1% is not noise -- it survived walk-forward validation with p=0.036. The infrastructure exists: 73 MCP tools, whale tracking at 57% win rate, 27 on-chain signals, and a strategy that made money shorting through every major crash since 2020.

The temptation is to skip to Tier 3. Resist it. Tier 1 is the foundation -- without live trading proof, the signal service is just another Telegram group, and the fund pitch is fiction. Four weeks of dry run, then small live, then scale. Each tier finances and validates the next.

UNDERTOW is the right brand. The invisible force beneath the surface -- that is exactly what M2 liquidity is to crypto markets. Most traders watch price. We watch what moves price.

Execute Tier 1 this week. The rest follows.
