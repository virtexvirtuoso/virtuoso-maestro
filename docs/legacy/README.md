# Filos Legacy Archive

Historical files from the original Filos project (October 2020).

## What's Here

### `original-codebase/`
Complete backup of the original Filos codebase from G-DRIVE (Oct 2020).

```
original-codebase/
├── backend/
│   ├── config/           # Config reader
│   ├── datafeed/         # RethinkDB data feed
│   ├── datasource/       # Binance/BitMEX batch downloaders
│   ├── engine/           # Backtesting, optimization, walk-forward
│   ├── main/             # REST API, data download scripts
│   ├── strategy/         # 9 original strategies
│   ├── storage/          # RethinkDB storage layer
│   ├── filos-dev.yaml    # Dev config
│   └── filos-prd.yaml    # Production config
├── frontend/filosui/     # React 16 + Material-UI frontend
├── docker-compose.yaml   # Original Docker setup
├── deploy.sh             # Bitbucket deployment script
└── bitbucket-pipelines.yml
```

### `filos-revival.md`
Task plan from Jan 2026 for reviving the platform (now completed as Maestro).

## Original Strategies

1. `base_strategy.py` - Base class for all strategies
2. `bollinger_bands_strategy.py` - BB breakout/mean reversion
3. `channel_strategy.py` - Channel breakout
4. `ema_cross_strategy.py` - EMA crossover
5. `fernando_strategy.py` - BBW + VLI (based on Glucksmann thesis)
6. `ichimoku_strategy.py` - Ichimoku cloud
7. `ma_cross_strategy.py` - MA crossover
8. `macd_strategy.py` - MACD signal
9. `rsi_strategy.py` - RSI overbought/oversold

## What's NOT Here (Dropbox Cloud-Only)

These files exist in Dropbox but are stored cloud-only and couldn't be copied:

**Location:** `~/Library/CloudStorage/Dropbox/VIRTEX/FILOS/`

| Folder | Contents |
|--------|----------|
| `Videos/` | 11 demo recordings with William & Pere (2020) - ~3GB |
| `Sketches/` | Filos Workflow diagrams, Dashboard mockups |
| `Outlines/` | Filos Prisma docs, trade specifications |
| `Strategies/` | Additional Freqtrade strategy collection |
| `Branding/` | Logo drafts (Sketch files) |
| `Notes/` | Development phases, Upwork contractor notes |

### Key Videos in Dropbox

| File | Date | Description |
|------|------|-------------|
| `1st Meeting with William Lucia.mp4` | May 2020 | 350 MB |
| `Demo with Pere on FILO beta.mp4` | May 2020 | 80 MB |
| `Filos demo 6_30_20.mkv` | Jun 30, 2020 | Platform walkthrough |
| `Filos meeting with William 7-7-2020 [deployment walk through].mkv` | Jul 7, 2020 | Deployment guide |
| `Filos F Strategy creation with William 07_16_2020.mkv` | Jul 16, 2020 | FernandoStrategy creation |
| `FILOS Walk-Forward Demo.mkv` | 2020 | WFA feature demo |

To access these videos, open Dropbox and download them manually, or run:
```bash
brctl download ~/Library/CloudStorage/Dropbox/VIRTEX/FILOS/Videos/
```

## Other Locations

- **G-DRIVE:** `/Volumes/G-DRIVE/filos/` - Original backup (source for `original-codebase/`)
- **G-DRIVE:** `/Volumes/G-DRIVE/VILLAR USB/Filos Trend.pbix` - PowerBI reports

---

*Archived: February 5, 2026*
*See: `docs/TIMELINE.md` for full project history*
