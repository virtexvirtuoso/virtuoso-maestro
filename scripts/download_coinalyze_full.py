#!/usr/bin/env python3
"""Download full derivatives history from Coinalyze API for all 13 tokens."""
import requests
import pandas as pd
import os
import time
from datetime import datetime

API_KEY = '***COINGLASS_KEY_REDACTED***'
BASE = 'https://api.coinalyze.net/v1'
OUTPUT = os.path.expanduser('~/Desktop/maestro/data/derivatives')
os.makedirs(OUTPUT, exist_ok=True)

TOKENS = ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'SUI', 'TIA', 'INJ', 'LINK', 'AVAX', 'FET', 'TAO', 'RENDER']
TO_TS = int(datetime(2026, 2, 14).timestamp())
FROM_TS = int(datetime(2024, 2, 1).timestamp())


def fetch(endpoint, symbol, interval=None, from_ts=FROM_TS, to_ts=TO_TS):
    params = {'symbols': symbol, 'from': from_ts, 'to': to_ts, 'api_key': API_KEY}
    if interval:
        params['interval'] = interval
    resp = requests.get(f'{BASE}/{endpoint}', params=params, timeout=30)
    if resp.status_code == 429:
        wait = int(resp.headers.get('Retry-After', 60))
        print(f'  Rate limited, waiting {wait}s...')
        time.sleep(wait)
        return fetch(endpoint, symbol, interval, from_ts, to_ts)
    if resp.status_code != 200:
        print(f'  Error {resp.status_code}: {resp.text[:200]}')
        return None
    data = resp.json()
    if not data or 'history' not in data[0]:
        return None
    return data[0]


def parse_ohlc(data, fields=None):
    """Parse OHLC-style history (OI, liquidations)."""
    records = []
    for h in data.get('history', []):
        rec = {'timestamp': pd.to_datetime(h['t'], unit='s')}
        if fields:
            for f in fields:
                rec[f] = h.get(f)
        else:
            rec.update({k: v for k, v in h.items() if k != 't'})
        records.append(rec)
    return pd.DataFrame(records)


def main():
    print(f"Downloading derivatives data for {len(TOKENS)} tokens")
    print(f"Range: {datetime.fromtimestamp(FROM_TS).date()} → {datetime.fromtimestamp(TO_TS).date()}")
    print()

    summary = {}
    for token in TOKENS:
        sym = f'{token}USDT_PERP.A'
        print(f'\n=== {token} ===')

        # OI daily
        print('  OI daily...', end=' ', flush=True)
        data = fetch('open-interest-history', sym, 'daily')
        if data:
            df = parse_ohlc(data)
            df.to_csv(f'{OUTPUT}/{token.lower()}_oi_daily_full.csv', index=False)
            print(f'{len(df)} rows ({df.timestamp.min().date()} → {df.timestamp.max().date()})')
            summary[f'{token}_oi'] = len(df)
        else:
            print('no data')
        time.sleep(1.6)

        # Funding
        print('  Funding...', end=' ', flush=True)
        data = fetch('funding-rate-history', sym, 'daily')
        if data:
            records = []
            for h in data.get('history', []):
                records.append({'timestamp': pd.to_datetime(h['t'], unit='s'), 'funding_rate': h.get('r')})
            df = pd.DataFrame(records)
            df.to_csv(f'{OUTPUT}/{token.lower()}_funding_full.csv', index=False)
            print(f'{len(df)} rows')
            summary[f'{token}_funding'] = len(df)
        else:
            print('no data')
        time.sleep(1.6)

        # LSR daily
        print('  LSR daily...', end=' ', flush=True)
        data = fetch('long-short-ratio-history', sym, 'daily')
        if data:
            records = []
            for h in data.get('history', []):
                records.append({
                    'timestamp': pd.to_datetime(h['t'], unit='s'),
                    'long_ratio': h.get('l'), 'short_ratio': h.get('s')
                })
            df = pd.DataFrame(records)
            df.to_csv(f'{OUTPUT}/{token.lower()}_lsr_daily_full.csv', index=False)
            print(f'{len(df)} rows')
            summary[f'{token}_lsr'] = len(df)
        else:
            print('no data')
        time.sleep(1.6)

        # Liquidations daily
        print('  Liquidations...', end=' ', flush=True)
        data = fetch('liquidation-history', sym, 'daily')
        if data:
            records = []
            for h in data.get('history', []):
                records.append({
                    'timestamp': pd.to_datetime(h['t'], unit='s'),
                    'long_liq': h.get('l'), 'short_liq': h.get('s')
                })
            df = pd.DataFrame(records)
            df.to_csv(f'{OUTPUT}/{token.lower()}_liquidations_daily.csv', index=False)
            print(f'{len(df)} rows')
            summary[f'{token}_liq'] = len(df)
        else:
            print('no data')
        time.sleep(1.6)

    print('\n\n=== SUMMARY ===')
    for k, v in sorted(summary.items()):
        print(f'  {k}: {v} rows')


if __name__ == '__main__':
    main()
