# Macro Momentum V3 — Freqtrade Deployment Guide

## Overview

Mega Strategy V3 ported to Freqtrade IStrategy format.
- **Architecture**: 5-signal confluence → regime detection → adaptive leverage → long/short
- **Assets**: BTC, ETH, SOL, LINK (per-asset optimized params)
- **Exchange**: Bybit Futures (isolated margin)
- **Walk-forward validated**: OOS Sharpe 0.84, p=0.036

## Files

| File | Purpose |
|------|---------|
| `macro_momentum_v3_strategy.py` | Main strategy (IStrategy) |
| `macro_data_provider.py` | FRED macro data fetcher + cache |
| `freqtrade_config.json` | Freqtrade configuration |
| `macro_cache.json` | Auto-generated macro data cache |

## 1. Install Freqtrade on VPS

```bash
ssh vps
cd ~/trading
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
# Or: pip install freqtrade
```

## 2. Deploy Strategy

```bash
# From local machine
scp freqtrade/macro_momentum_v3_strategy.py vps:~/trading/freqtrade/user_data/strategies/
scp freqtrade/macro_data_provider.py vps:~/trading/freqtrade/user_data/strategies/
scp freqtrade/freqtrade_config.json vps:~/trading/freqtrade/user_data/

# Or rsync the whole directory
rsync -avz freqtrade/ vps:~/trading/freqtrade/user_data/strategies/macro_v3/
```

## 3. Configure

Edit `freqtrade_config.json` on VPS:
```bash
ssh vps
cd ~/trading/freqtrade
nano user_data/freqtrade_config.json
```

**Required changes:**
- `exchange.key` / `exchange.secret` — Bybit API keys
- `api_server.jwt_secret_key` — random secret
- `api_server.password` — secure password
- `telegram.token` / `telegram.chat_id` — if using Telegram alerts
- `dry_run` — set `false` for live trading

## 4. Set Up Macro Data Cron

```bash
# Set FRED API key
echo 'export FRED_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

# Test update
cd ~/trading/freqtrade/user_data/strategies
python macro_data_provider.py --update

# Add to cron (daily at 6am UTC)
crontab -e
# Add: 0 6 * * * cd ~/trading/freqtrade/user_data/strategies && python macro_data_provider.py --update >> /tmp/macro_update.log 2>&1
```

## 5. Backtest

```bash
cd ~/trading/freqtrade
freqtrade backtesting \
    --config user_data/freqtrade_config.json \
    --strategy MacroMomentumV3Strategy \
    --timerange 20230101-20260101 \
    --timeframe 1d \
    --enable-position-stacking
```

## 6. Dry Run

```bash
freqtrade trade \
    --config user_data/freqtrade_config.json \
    --strategy MacroMomentumV3Strategy \
    --dry-run
```

## 7. Go Live

1. Verify dry-run results for at least 2 weeks
2. Set `"dry_run": false` in config
3. Start with reduced `tradable_balance_ratio` (e.g., 0.3)
4. Monitor closely for first week

```bash
freqtrade trade \
    --config user_data/freqtrade_config.json \
    --strategy MacroMomentumV3Strategy
```

## 8. Monitoring

```bash
# Status
freqtrade show-trades --config user_data/freqtrade_config.json

# Logs
tail -f user_data/logs/freqtrade.log

# FreqUI (web dashboard)
# Access at http://localhost:8080 (or via SSH tunnel)
ssh -L 8080:localhost:8080 vps
```

## Per-Asset Parameter Reference

| Asset | SMA Slow | Momentum Period | RSI Entry | RSI Exit | Trail Stop | Dip % | Pyramid Dip | Max Leverage |
|-------|----------|-----------------|-----------|----------|------------|-------|-------------|--------------|
| BTC   | 100      | 35              | 52        | 72       | 12%        | 3%    | 2%          | 3.0x         |
| ETH   | 140      | 15              | 32        | 75       | 20%        | 5%    | 3%          | 2.5x         |
| SOL   | 70       | 20              | 30        | 78       | 10%        | 6%    | 4%          | 2.0x         |
| LINK  | 190      | 25              | 53        | 70       | 8.6%       | 5%    | 3%          | 2.0x         |

## Confluence Score → Regime → Leverage

| Confluence | Regime    | Leverage Mult | Action |
|------------|-----------|---------------|--------|
| 5          | BULL      | 2.0x          | Aggressive long, pyramid |
| 4          | BULL      | 1.5x          | Long, pyramid on dips |
| 3          | MILD_BULL | 1.0x          | Cautious long |
| 2          | NEUTRAL   | 0.6x          | Minimal exposure |
| 1          | BEAR      | 0.3x          | Short opportunities |
| 0          | BEAR      | 0.0x          | No trade |

## Systemd Service (optional)

```ini
# /etc/systemd/system/freqtrade.service
[Unit]
Description=Freqtrade MacroMomentumV3
After=network.target

[Service]
User=linuxuser
WorkingDirectory=/home/linuxuser/trading/freqtrade
ExecStart=/home/linuxuser/trading/freqtrade/.venv/bin/freqtrade trade --config user_data/freqtrade_config.json --strategy MacroMomentumV3Strategy
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable freqtrade
sudo systemctl start freqtrade
sudo journalctl -u freqtrade -f
```
