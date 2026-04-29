#!/usr/bin/env python3
"""
Standalone Bybit perpetuals tick data collector.
Streams trades + L2 orderbook snapshots → daily Parquet files.

No dependency on Virtuoso — runs independently for research.
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd
from pybit.unified_trading import WebSocket

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('tick_collector')

# ── Config ─────────────────────────────────────────────────────────────
def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Storage ────────────────────────────────────────────────────────────
class ParquetWriter:
    """Buffers data in memory, flushes to daily Parquet files."""

    def __init__(self, data_dir: str, flush_interval: int = 300):
        self.data_dir = Path(data_dir)
        self.flush_interval = flush_interval
        self.buffers = defaultdict(list)  # key: (type, symbol, date_str) → rows
        self.lock = threading.Lock()
        self._last_flush = time.time()
        self.rows_written = 0

    def append_trade(self, symbol: str, timestamp_ms: int, price: float, size: float, side: str):
        date_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
        key = ('trades', symbol, date_str)
        with self.lock:
            self.buffers[key].append({
                'timestamp': timestamp_ms,
                'price': price,
                'size': size,
                'side': side,
            })

    def append_orderbook(self, symbol: str, timestamp_ms: int, snapshot: dict):
        date_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
        key = ('orderbook', symbol, date_str)
        with self.lock:
            self.buffers[key].append({
                'timestamp': timestamp_ms,
                **snapshot,
            })

    def maybe_flush(self):
        if time.time() - self._last_flush < self.flush_interval:
            return
        self.flush()

    def flush(self):
        with self.lock:
            items = list(self.buffers.items())
            self.buffers.clear()
        
        for (dtype, symbol, date_str), rows in items:
            if not rows:
                continue
            
            out_dir = self.data_dir / dtype / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{date_str}.parquet'
            
            df_new = pd.DataFrame(rows)
            
            # Append to existing file if present
            if out_path.exists():
                df_old = pd.read_parquet(out_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            
            df_new.to_parquet(out_path, index=False)
            self.rows_written += len(rows)
        
        if items:
            self._last_flush = time.time()
            log.info(f'Flushed {sum(len(r) for _, r in items)} rows | Total: {self.rows_written:,}')


# ── Collector ──────────────────────────────────────────────────────────
class TickCollector:
    def __init__(self, config: dict):
        self.config = config
        self.symbols = config['symbols']
        self.ob_depth = config.get('orderbook_depth', 25)
        self.ob_interval = config.get('orderbook_interval', 1)
        self.writer = ParquetWriter(
            config.get('data_dir', './data'),
            config.get('flush_interval', 300),
        )
        self.ws_trade = None
        self.ws_ob = None
        self.running = False
        self._trade_count = defaultdict(int)
        self._ob_count = defaultdict(int)

    def _handle_trade(self, message):
        """Process incoming trade messages."""
        try:
            topic = message.get('topic', '')
            data = message.get('data', [])
            symbol = topic.split('.')[-1] if '.' in topic else ''
            
            for trade in data:
                self.writer.append_trade(
                    symbol=symbol,
                    timestamp_ms=int(trade['T']),
                    price=float(trade['p']),
                    size=float(trade['v']),
                    side=trade['S'],  # Buy or Sell
                )
                self._trade_count[symbol] += 1
        except Exception as e:
            log.error(f'Trade handler error: {e}')

    def _handle_orderbook(self, message):
        """Process incoming orderbook messages."""
        try:
            topic = message.get('topic', '')
            data = message.get('data', {})
            symbol = topic.split('.')[-1] if '.' in topic else ''
            ts = int(data.get('ts', time.time() * 1000))
            
            bids = data.get('b', [])[:self.ob_depth]
            asks = data.get('a', [])[:self.ob_depth]
            
            if not bids or not asks:
                return
            
            bid_prices = [float(b[0]) for b in bids]
            bid_sizes = [float(b[1]) for b in bids]
            ask_prices = [float(a[0]) for a in asks]
            ask_sizes = [float(a[1]) for a in asks]
            
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Imbalance: bid volume / total volume (top 5 levels)
            top_n = min(5, len(bid_sizes), len(ask_sizes))
            bid_vol = sum(bid_sizes[:top_n])
            ask_vol = sum(ask_sizes[:top_n])
            imbalance = bid_vol / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.5
            
            snapshot = {
                'bid_prices': json.dumps(bid_prices),
                'bid_sizes': json.dumps(bid_sizes),
                'ask_prices': json.dumps(ask_prices),
                'ask_sizes': json.dumps(ask_sizes),
                'mid_price': mid,
                'spread': spread,
                'imbalance': imbalance,
            }
            
            self.writer.append_orderbook(symbol, ts, snapshot)
            self._ob_count[symbol] += 1
        except Exception as e:
            log.error(f'Orderbook handler error: {e}')

    def start(self):
        """Connect and subscribe."""
        self.running = True
        testnet = self.config.get('testnet', False)
        
        log.info(f'Starting collector | Symbols: {self.symbols} | Depth: {self.ob_depth}')
        log.info(f'Testnet: {testnet} | Flush every {self.config.get("flush_interval", 300)}s')
        
        # Separate WS connections for trades and orderbook (pybit limitation)
        self.ws_trade = WebSocket(testnet=testnet, channel_type='linear')
        self.ws_ob = WebSocket(testnet=testnet, channel_type='linear')
        
        # Subscribe to trades
        for sym in self.symbols:
            self.ws_trade.trade_stream(symbol=sym, callback=self._handle_trade)
            log.info(f'Subscribed: publicTrade.{sym}')
        
        time.sleep(1)  # Let first connection stabilize
        
        # Subscribe to orderbook on separate connection
        for sym in self.symbols:
            self.ws_ob.orderbook_stream(
                depth=self.ob_depth,
                symbol=sym,
                callback=self._handle_orderbook,
            )
            log.info(f'Subscribed: orderbook.{self.ob_depth}.{sym}')
        
        log.info('All subscriptions active. Collecting...')
        
        # Main loop: periodic flush + stats
        try:
            while self.running:
                time.sleep(10)
                self.writer.maybe_flush()
                
                # Stats every minute
                total_trades = sum(self._trade_count.values())
                total_obs = sum(self._ob_count.values())
                if total_trades > 0 or total_obs > 0:
                    stats = ' | '.join(f'{s}: {self._trade_count[s]}t/{self._ob_count[s]}ob' for s in self.symbols)
                    log.info(f'Stats: {stats} | Buffered: {sum(len(v) for v in self.writer.buffers.values())}')
        except KeyboardInterrupt:
            log.info('Interrupted')
        finally:
            self.stop()

    def stop(self):
        self.running = False
        log.info('Flushing remaining data...')
        self.writer.flush()
        for ws in [self.ws_trade, self.ws_ob]:
            if ws:
                try:
                    ws.exit()
                except:
                    pass
        log.info(f'Stopped. Total rows written: {self.writer.rows_written:,}')


# ── Main ───────────────────────────────────────────────────────────────
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    if not os.path.exists(config_path):
        log.error(f'Config not found: {config_path}')
        sys.exit(1)
    
    config = load_config(config_path)
    collector = TickCollector(config)
    
    # Graceful shutdown
    def shutdown(sig, frame):
        log.info(f'Signal {sig} received, shutting down...')
        collector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    collector.start()


if __name__ == '__main__':
    main()
