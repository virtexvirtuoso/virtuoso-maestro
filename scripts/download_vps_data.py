#!/usr/bin/env python3
"""
VPS Data Downloader for Maestro Backtesting

Downloads historical price data, funding rates, and signals from VPS
for comprehensive strategy backtesting.

Usage:
    python scripts/download_vps_data.py --days 365 --symbols BTC,ETH,SOL
    
Requires: SSH access to VPS (configured as 'vps' in ~/.ssh/config)
"""

import subprocess
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time

# VPS Configuration
VPS_HOST = "vps"  # SSH alias
VPS_API_BASE = "http://127.0.0.1:8888"
VPS_BTCWIZ_BASE = "http://127.0.0.1:8004"

# Local data directory
DATA_DIR = Path(__file__).parent.parent / "backend" / "data"

# Symbols to download
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "ARBUSDT", "OPUSDT", "APTUSDT"
]


def run_ssh_command(command: str, timeout: int = 60) -> str:
    """Run command on VPS via SSH."""
    full_cmd = f'ssh {VPS_HOST} "{command}"'
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH command failed: {result.stderr}")
    return result.stdout


def download_existing_ohlcv():
    """Download existing OHLCV data from VPS backtest_data folder."""
    print("\n📊 Downloading existing OHLCV data from VPS...")
    
    # Create local directory
    ohlcv_dir = DATA_DIR / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    
    # List available files
    files = run_ssh_command("ls ~/backtest_data/*.csv").strip().split('\n')
    
    for remote_file in files:
        if not remote_file:
            continue
        filename = os.path.basename(remote_file)
        local_file = ohlcv_dir / filename
        
        print(f"  Downloading {filename}...")
        subprocess.run(
            f"scp {VPS_HOST}:{remote_file} {local_file}",
            shell=True,
            check=True
        )
    
    print(f"✅ Downloaded {len(files)} OHLCV files to {ohlcv_dir}")
    return ohlcv_dir


def download_funding_rates(symbols: list, days: int = 90):
    """Download historical funding rate data."""
    print(f"\n💰 Downloading funding rates for {len(symbols)} symbols...")
    
    funding_dir = DATA_DIR / "funding_rates"
    funding_dir.mkdir(parents=True, exist_ok=True)
    
    # Create VPS script to fetch historical funding
    vps_script = f'''
import requests
import json
from datetime import datetime

symbols = {symbols}
base_url = "{VPS_API_BASE}"

results = {{}}
for symbol in symbols:
    try:
        resp = requests.get(f"{{base_url}}/signals/funding-rate/{{symbol}}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                results[symbol] = data["signal"]
    except Exception as e:
        print(f"Error {{symbol}}: {{e}}")

print(json.dumps(results))
'''
    
    # Run on VPS
    output = run_ssh_command(f"python3 -c '{vps_script}'", timeout=120)
    
    try:
        funding_data = json.loads(output.strip().split('\n')[-1])
        
        # Save to file
        funding_file = funding_dir / f"funding_snapshot_{datetime.now().strftime('%Y%m%d')}.json"
        with open(funding_file, 'w') as f:
            json.dump(funding_data, f, indent=2)
        
        print(f"✅ Saved funding rates to {funding_file}")
        return funding_data
    except json.JSONDecodeError:
        print(f"⚠️ Could not parse funding data: {output[:200]}")
        return {}


def download_all_signals(symbols: list):
    """Download comprehensive signal data for all symbols."""
    print(f"\n📡 Downloading all signals for {len(symbols)} symbols...")
    
    signals_dir = DATA_DIR / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    all_signals = {}
    
    for symbol in symbols:
        cmd = f"curl -s '{VPS_API_BASE}/signals/all/{symbol}'"
        try:
            output = run_ssh_command(cmd, timeout=30)
            data = json.loads(output)
            if data.get("success"):
                all_signals[symbol] = data["signals"]
                print(f"  ✓ {symbol}: {len(data['signals'])} signals")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
    
    # Save combined signals
    signals_file = signals_dir / f"all_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(signals_file, 'w') as f:
        json.dump(all_signals, f, indent=2)
    
    print(f"✅ Saved all signals to {signals_file}")
    return all_signals


def download_sector_rotation():
    """Download sector rotation rankings."""
    print("\n🔄 Downloading sector rotation data...")
    
    sector_dir = DATA_DIR / "sector_rotation"
    sector_dir.mkdir(parents=True, exist_ok=True)
    
    endpoints = {
        "rankings": "/sector-rotation/rankings?limit=50",
        "summary": "/sector-rotation/summary",
        "signals": "/sector-rotation/signals",
    }
    
    sector_data = {}
    for name, endpoint in endpoints.items():
        try:
            cmd = f"curl -s '{VPS_API_BASE}{endpoint}'"
            output = run_ssh_command(cmd, timeout=30)
            sector_data[name] = json.loads(output)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # Save
    sector_file = sector_dir / f"sector_rotation_{datetime.now().strftime('%Y%m%d')}.json"
    with open(sector_file, 'w') as f:
        json.dump(sector_data, f, indent=2)
    
    print(f"✅ Saved sector data to {sector_file}")
    return sector_data


def download_btc_composite():
    """Download BTC composite signals from BTC Wiz."""
    print("\n🔮 Downloading BTC composite signals...")
    
    btc_dir = DATA_DIR / "btc_wiz"
    btc_dir.mkdir(parents=True, exist_ok=True)
    
    endpoints = {
        "composite": "/signals/composite",
        "metrics": "/metrics",
        "recommendation": "/analysis/recommendation",
    }
    
    btc_data = {}
    for name, endpoint in endpoints.items():
        try:
            cmd = f"curl -s '{VPS_BTCWIZ_BASE}{endpoint}'"
            output = run_ssh_command(cmd, timeout=30)
            btc_data[name] = json.loads(output)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # Save
    btc_file = btc_dir / f"btc_composite_{datetime.now().strftime('%Y%m%d')}.json"
    with open(btc_file, 'w') as f:
        json.dump(btc_data, f, indent=2)
    
    print(f"✅ Saved BTC data to {btc_file}")
    return btc_data


def create_combined_dataset():
    """Combine all downloaded data into a single backtesting dataset."""
    print("\n🔗 Creating combined backtesting dataset...")
    
    combined_dir = DATA_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OHLCV files
    ohlcv_dir = DATA_DIR / "ohlcv"
    ohlcv_data = {}
    
    if ohlcv_dir.exists():
        for csv_file in ohlcv_dir.glob("*.csv"):
            symbol = csv_file.stem.replace("_USDT_max", "USDT").replace("_", "")
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            ohlcv_data[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} rows")
    
    # Save as combined parquet for faster loading
    for symbol, df in ohlcv_data.items():
        parquet_file = combined_dir / f"{symbol}.parquet"
        df.to_parquet(parquet_file)
    
    print(f"✅ Created {len(ohlcv_data)} parquet files in {combined_dir}")
    return combined_dir


def main():
    parser = argparse.ArgumentParser(description="Download VPS data for Maestro backtesting")
    parser.add_argument("--days", type=int, default=90, help="Days of history to download")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--skip-ohlcv", action="store_true", help="Skip OHLCV download")
    parser.add_argument("--skip-signals", action="store_true", help="Skip signal download")
    args = parser.parse_args()
    
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    
    print("=" * 60)
    print("MAESTRO DATA DOWNLOADER")
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {args.days}")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download all data types
    if not args.skip_ohlcv:
        download_existing_ohlcv()
    
    if not args.skip_signals:
        download_funding_rates(symbols, args.days)
        download_all_signals(symbols)
        download_sector_rotation()
        download_btc_composite()
    
    # Create combined dataset
    create_combined_dataset()
    
    print("\n" + "=" * 60)
    print("✅ DATA DOWNLOAD COMPLETE")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
