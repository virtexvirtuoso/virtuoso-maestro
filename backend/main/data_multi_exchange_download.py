#!/usr/bin/env python3
"""
Multi-Exchange Data Downloader using CCXT

Downloads OHLCV data from multiple exchanges:
- Binance, Bybit, Gate, KuCoin, MEXC

Usage:
    python3 data_multi_exchange_download.py
    
    # Or specify exchanges:
    python3 data_multi_exchange_download.py --exchanges binance bybit mexc
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent to path for imports
ROOT_DIR = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(ROOT_DIR))

from datasource.ccxt_batch_downloader import CCXTBatchDownloader
from datasource.providers import DataSourceProviders
from storage.rethinkdb_storage_layer import RethinkDbStorageLayer
from config.config_reader import ConfigReader
from logger.logger_builder import LoggerBuilder


# Map config keys to provider enums
PROVIDER_MAP = {
    'binance': DataSourceProviders.BINANCE,
    'bybit': DataSourceProviders.BYBIT,
    'gate': DataSourceProviders.GATE,
    'kucoin': DataSourceProviders.KUCOIN,
    'mexc': DataSourceProviders.MEXC,
    'okx': DataSourceProviders.OKX,
    'bitmex': DataSourceProviders.BITMEX,
}


def main():
    parser = argparse.ArgumentParser(description='Multi-exchange OHLCV downloader')
    parser.add_argument('--config', default='maestro-dev.yaml', help='Config file name')
    parser.add_argument('--exchanges', nargs='+', default=None, 
                        help='Specific exchanges to download (default: all)')
    args = parser.parse_args()
    
    logger = LoggerBuilder(name='MultiExchangeDownloader').build()
    
    # Load config
    config_path = ROOT_DIR.joinpath(args.config)
    if not config_path.exists():
        config_path = ROOT_DIR.joinpath('config', args.config)
    
    logger.info(f'Loading config from {config_path}')
    
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)['config']
    
    # Connect to RethinkDB
    db_config = config['rethinkdb']
    storage = RethinkDbStorageLayer(
        host=db_config['host'], 
        port=db_config['port'], 
        db=db_config['db']
    )
    
    # Get datasources to process
    datasources = config.get('datasource', {})
    exchanges_to_process = args.exchanges or list(datasources.keys())
    
    for exchange_name in exchanges_to_process:
        if exchange_name not in datasources:
            logger.warning(f'Exchange {exchange_name} not in config, skipping')
            continue
            
        provider = PROVIDER_MAP.get(exchange_name.lower())
        if not provider:
            logger.warning(f'Unknown provider: {exchange_name}, skipping')
            continue
        
        exchange_config = datasources[exchange_name]
        symbols = exchange_config.get('symbols', {})
        
        if not symbols:
            logger.info(f'No symbols for {exchange_name}, skipping')
            continue
        
        logger.info(f'=== Starting {exchange_name.upper()} download ({len(symbols)} symbols) ===')
        
        try:
            downloader = CCXTBatchDownloader(
                provider=provider,
                storage=storage,
                logger=LoggerBuilder(name=f'CCXT-{exchange_name}').build()
            )
            
            # Register all symbols
            for symbol, sconf in symbols.items():
                for bin_size in sconf.get('bin_size', ['1d']):
                    downloader.register(symbol=symbol, bin_size=bin_size)
            
            # Start download
            downloader.start()
            
        except Exception as e:
            logger.error(f'Error with {exchange_name}: {e}')
            continue
    
    logger.info('=== All downloads complete ===')


if __name__ == '__main__':
    main()
