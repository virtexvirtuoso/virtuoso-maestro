#!/bin/bash
# Download all missing spot candle data
set -e
cd ~/Desktop/maestro
LOG_DIR=~/Desktop/maestro/data/spot/logs
mkdir -p $LOG_DIR

echo "$(date): Starting all downloads..."

# 15m - all 24 assets from 2023
echo "$(date): Starting 15m..."
python3 backend/research/download_spot_candles.py --timeframes 15m --start 2023-01-01 > $LOG_DIR/15m.log 2>&1
echo "$(date): 15m DONE"

# 5m - all 24 assets from 2024 (1 year is enough, keeps size manageable)
echo "$(date): Starting 5m..."
python3 backend/research/download_spot_candles.py --timeframes 5m --start 2024-01-01 > $LOG_DIR/5m.log 2>&1
echo "$(date): 5m DONE"

# 1d - redownload all to new format with FET/RENDER/TAO + fix SUI
echo "$(date): Starting 1d refresh..."
python3 backend/research/download_spot_candles.py --timeframes 1d --start 2019-01-01 > $LOG_DIR/1d.log 2>&1
echo "$(date): 1d DONE"

echo "$(date): ALL DOWNLOADS COMPLETE"

# Summary
echo ""
echo "=== FINAL INVENTORY ==="
for tf in 1d 4h 1h 15m 5m; do
  count=$(ls ~/Desktop/maestro/data/spot/$tf/*.csv 2>/dev/null | wc -l | tr -d ' ')
  if [ "$count" -gt "0" ]; then
    sample=$(ls ~/Desktop/maestro/data/spot/$tf/*.csv | head -1)
    rows=$(wc -l < "$sample" | tr -d ' ')
    echo "$tf: $count files, ~$rows rows each"
  else
    echo "$tf: $count files"
  fi
done
