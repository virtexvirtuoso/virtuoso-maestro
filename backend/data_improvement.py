#!/usr/bin/env python3
"""
Maestro Data Improvement Script
- Downloads extended OHLCV history
- Adds database indexes
- Sets up daily updates

Usage:
    python data_improvement.py --all              # Run everything
    python data_improvement.py --download --days 180
    python data_improvement.py --indexes
    python data_improvement.py --check
    python data_improvement.py --cron
"""

import asyncio
import ccxt.async_support as ccxt
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'data' / 'maestro.duckdb'

# Top symbols for extended history
TOP_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'DOGE/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
    'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'MATIC/USDT:USDT'
]


async def fetch_ohlcv_history(exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV history with pagination."""
    all_data = []
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    limit = 1000

    clean_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
    logger.info(f"Fetching {clean_symbol} {timeframe} - {days} days")

    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Next ms after last candle

            if len(ohlcv) < limit:
                break

            await asyncio.sleep(0.1)  # Rate limit

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = clean_symbol
    df['timeframe'] = timeframe

    logger.info(f"  Got {len(df):,} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


async def download_extended_history(symbols: list, timeframes: list, days: int):
    """Download extended OHLCV history for multiple symbols."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    try:
        await exchange.load_markets()

        conn = duckdb.connect(str(DB_PATH))

        for symbol in symbols:
            for tf in timeframes:
                df = await fetch_ohlcv_history(exchange, symbol, tf, days)

                if df.empty:
                    continue

                # Upsert into database (replace duplicates)
                clean_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')

                # Delete existing data for this symbol/timeframe in date range
                min_ts = df['timestamp'].min()
                conn.execute(f"""
                    DELETE FROM ohlcv_futures
                    WHERE symbol = '{clean_symbol}'
                    AND timeframe = '{tf}'
                    AND timestamp >= '{min_ts}'
                """)

                # Insert new data
                conn.execute("""
                    INSERT INTO ohlcv_futures
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                    FROM df
                """)

                logger.info(f"  Inserted {len(df):,} rows for {clean_symbol} {tf}")

        conn.close()

    finally:
        await exchange.close()


def add_indexes():
    """Add indexes to improve query performance."""
    logger.info("Adding database indexes...")

    conn = duckdb.connect(str(DB_PATH))

    indexes = [
        ("idx_ohlcv_symbol_tf_ts", "ohlcv_futures", "(symbol, timeframe, timestamp)"),
        ("idx_ohlcv_symbol", "ohlcv_futures", "(symbol)"),
        ("idx_oi_symbol_ts", "derivatives_oi", "(symbol, timestamp)"),
        ("idx_funding_symbol_ts", "derivatives_funding", "(symbol, timestamp)"),
        ("idx_liq_symbol_ts", "derivatives_liquidations", "(symbol, timestamp)"),
    ]

    for idx_name, table, columns in indexes:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} {columns}")
            logger.info(f"  Created index {idx_name}")
        except Exception as e:
            logger.warning(f"  Index {idx_name}: {e}")

    conn.close()
    logger.info("Indexes complete")


def check_data_quality():
    """Check for gaps and data quality issues."""
    logger.info("Checking data quality...")

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Check for gaps in hourly data
    gaps = conn.execute("""
        WITH hourly AS (
            SELECT symbol, timestamp,
                   LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev
            FROM ohlcv_futures
            WHERE timeframe = '1h'
        )
        SELECT symbol, COUNT(*) as gap_count
        FROM hourly
        WHERE EXTRACT(EPOCH FROM (timestamp - prev)) / 3600 > 1.5
        GROUP BY symbol
        HAVING gap_count > 0
        ORDER BY gap_count DESC
        LIMIT 10
    """).fetchall()

    if gaps:
        logger.warning("Found gaps in hourly data:")
        for symbol, count in gaps:
            logger.warning(f"  {symbol}: {count} gaps")
    else:
        logger.info("  No gaps found in hourly data")

    # Summary stats
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT symbol) as symbols,
            COUNT(*) as total_rows,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM ohlcv_futures
    """).fetchone()

    logger.info(f"  Symbols: {stats[0]}, Rows: {stats[1]:,}")
    logger.info(f"  Date range: {stats[2]} to {stats[3]}")

    conn.close()


def generate_cron_script():
    """Generate daily update script and cron entry."""
    script_dir = Path(__file__).parent
    update_script = script_dir / 'daily_update.py'

    script_content = '''#!/usr/bin/env python3
"""Daily OHLCV update script - run via cron."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_improvement import download_extended_history, TOP_SYMBOLS

async def main():
    # Update last 2 days to catch any gaps
    await download_extended_history(
        symbols=TOP_SYMBOLS,
        timeframes=['5m', '15m', '1h', '4h'],
        days=2
    )

if __name__ == '__main__':
    asyncio.run(main())
'''

    with open(update_script, 'w') as f:
        f.write(script_content)

    update_script.chmod(0o755)
    logger.info(f"Created {update_script}")

    # Create logs directory
    logs_dir = script_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Print cron entry
    venv_python = script_dir / 'venv' / 'bin' / 'python'
    cron_entry = f"0 1 * * * cd {script_dir} && {venv_python} daily_update.py >> logs/daily_update.log 2>&1"

    logger.info(f"\nAdd to crontab with: crontab -e")
    logger.info(f"\nCron entry:")
    print(cron_entry)

    return cron_entry


def show_stats():
    """Show current database statistics."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("\n" + "=" * 60)
    print("MAESTRO DATABASE STATISTICS")
    print("=" * 60)

    # Table sizes
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"\n{'Table':<30} {'Rows':>15}")
    print("-" * 45)
    for (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:<30} {count:>15,}")

    # OHLCV coverage by timeframe
    print(f"\n{'Timeframe':<10} {'Symbols':>10} {'Rows':>15}")
    print("-" * 35)
    result = conn.execute("""
        SELECT timeframe, COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows
        FROM ohlcv_futures
        GROUP BY timeframe
        ORDER BY rows DESC
    """).fetchall()
    for tf, symbols, rows in result:
        print(f"{tf:<10} {symbols:>10} {rows:>15,}")

    conn.close()


async def main():
    parser = argparse.ArgumentParser(description='Maestro Data Improvement')
    parser.add_argument('--download', action='store_true', help='Download extended history')
    parser.add_argument('--days', type=int, default=180, help='Days of history (default: 180)')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '15m'], help='Timeframes to download')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols (default: top 10)')
    parser.add_argument('--indexes', action='store_true', help='Add database indexes')
    parser.add_argument('--check', action='store_true', help='Check data quality')
    parser.add_argument('--cron', action='store_true', help='Generate cron update script')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--all', action='store_true', help='Run all improvements')

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.all or args.indexes:
        add_indexes()

    if args.all or args.check:
        check_data_quality()

    if args.all or args.download:
        symbols = args.symbols or TOP_SYMBOLS
        await download_extended_history(symbols, args.timeframes, args.days)

    if args.all or args.cron:
        generate_cron_script()

    if not any([args.download, args.indexes, args.check, args.cron, args.stats, args.all]):
        parser.print_help()
        print("\nExamples:")
        print("  python data_improvement.py --stats")
        print("  python data_improvement.py --indexes")
        print("  python data_improvement.py --download --days 180 --timeframes 5m 15m")
        print("  python data_improvement.py --all")


if __name__ == '__main__':
    asyncio.run(main())
