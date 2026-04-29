#!/usr/bin/env python3
"""
Tick Collector Monitor — sends progress alerts to Discord admin webhook.
Run via cron every hour (or as standalone check).
"""

import os
import sys
import json
import glob
import requests
from datetime import datetime, timezone
from pathlib import Path

WEBHOOK_URL = os.environ.get('SYSTEM_ALERTS_WEBHOOK_URL',
    'https://discord.com/api/webhooks/1379097202613420163/IJXNvNxw09zXGvQe2oZZ-8TwYc91hZH4PqD6XtVEQa5fH6TpBt9hBLuTZiejUPjW9m8i')

DATA_DIR = Path(os.environ.get('TICK_DATA_DIR', '/home/linuxuser/tick_collector/data'))
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT']
AVATAR_URL = 'https://virtuosocrypto.com/static/avatar.png'

# Milestones (days of data collected)
MILESTONES = [1, 3, 7, 14, 21, 30]
MILESTONE_FILE = Path('/home/linuxuser/tick_collector/.milestones_sent')


def get_stats():
    """Get collection statistics."""
    stats = {}
    total_size = 0
    total_trade_files = 0
    total_ob_files = 0
    dates_seen = set()

    for sym in SYMBOLS:
        trade_files = sorted(glob.glob(str(DATA_DIR / 'trades' / sym / '*.parquet')))
        ob_files = sorted(glob.glob(str(DATA_DIR / 'orderbook' / sym / '*.parquet')))
        
        sym_size = sum(os.path.getsize(f) for f in trade_files + ob_files)
        total_size += sym_size
        total_trade_files += len(trade_files)
        total_ob_files += len(ob_files)
        
        for f in trade_files:
            dates_seen.add(Path(f).stem)
        
        stats[sym] = {
            'trade_days': len(trade_files),
            'ob_days': len(ob_files),
            'size_mb': round(sym_size / 1024 / 1024, 1),
        }

    days_collected = len(dates_seen)
    return {
        'symbols': stats,
        'total_size_gb': round(total_size / 1024 / 1024 / 1024, 2),
        'total_trade_files': total_trade_files,
        'total_ob_files': total_ob_files,
        'days_collected': days_collected,
        'dates': sorted(dates_seen),
    }


def get_sent_milestones():
    if MILESTONE_FILE.exists():
        return set(json.loads(MILESTONE_FILE.read_text()))
    return set()


def save_milestone(day):
    sent = get_sent_milestones()
    sent.add(day)
    MILESTONE_FILE.write_text(json.dumps(sorted(sent)))


def send_discord(title, description, color=0xF59E0B, fields=None):
    embed = {
        'title': title,
        'description': description,
        'color': color,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'footer': {'text': 'Tick Collector Monitor'},
    }
    if fields:
        embed['fields'] = fields

    payload = {
        'username': 'Virtuoso Tick Collector',
        'avatar_url': AVATAR_URL,
        'embeds': [embed],
    }

    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f'Discord send failed: {e}')
        return False


def check_health():
    """Check if collector is running and data is fresh."""
    import subprocess
    result = subprocess.run(['systemctl', 'is-active', 'tick-collector'],
                          capture_output=True, text=True)
    is_running = result.stdout.strip() == 'active'
    
    # Check latest file timestamp
    latest_mtime = 0
    for sym in SYMBOLS:
        for dtype in ['trades', 'orderbook']:
            files = glob.glob(str(DATA_DIR / dtype / sym / '*.parquet'))
            for f in files:
                mt = os.path.getmtime(f)
                if mt > latest_mtime:
                    latest_mtime = mt
    
    now = datetime.now(timezone.utc).timestamp()
    minutes_stale = (now - latest_mtime) / 60 if latest_mtime > 0 else 999
    
    return is_running, minutes_stale


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    stats = get_stats()
    is_running, minutes_stale = check_health()

    if mode == 'startup':
        send_discord(
            '🟢 Tick Collector Started',
            f'Collecting L2 orderbook + trades for {len(SYMBOLS)} symbols.\n'
            f'Symbols: {", ".join(SYMBOLS)}\n'
            f'Depth: 50 levels per side\n'
            f'Storage: `{DATA_DIR}`',
            color=0x22C55E,
        )
        return

    if mode == 'check':
        # Alert if service is down
        if not is_running:
            send_discord(
                '🔴 Tick Collector DOWN',
                'Service `tick-collector` is not running!\n'
                'Run: `sudo systemctl start tick-collector`',
                color=0xEF4444,
            )
            return

        # Alert if data is stale (> 10 min)
        if minutes_stale > 10:
            send_discord(
                '⚠️ Tick Collector Stale Data',
                f'Last data written {minutes_stale:.0f} minutes ago.\n'
                f'Service is {"running" if is_running else "STOPPED"}.',
                color=0xEAB308,
            )
            return

        # Check milestones
        days = stats['days_collected']
        sent = get_sent_milestones()
        
        for milestone in MILESTONES:
            if days >= milestone and milestone not in sent:
                sym_lines = []
                for sym, s in stats['symbols'].items():
                    sym_lines.append(f'**{sym}**: {s["trade_days"]}d trades, {s["ob_days"]}d orderbook ({s["size_mb"]} MB)')
                
                target_msg = ""
                if milestone < 14:
                    target_msg = f"\n\n📊 Target: 14-21 days for VPIN + order flow research."
                elif milestone == 14:
                    target_msg = "\n\n🔬 **Ready for initial VPIN + order book imbalance research!**"
                elif milestone == 21:
                    target_msg = "\n\n🔬 **Full microstructure research dataset ready.**"
                
                send_discord(
                    f'📈 Tick Collector: {milestone} Day{"s" if milestone > 1 else ""} Milestone',
                    f'Collected **{days} days** of tick data.\n'
                    f'Total size: **{stats["total_size_gb"]} GB**\n\n'
                    + '\n'.join(sym_lines)
                    + target_msg,
                    color=0x8B5CF6,
                    fields=[{
                        'name': 'Date Range',
                        'value': f'{stats["dates"][0]} → {stats["dates"][-1]}' if stats['dates'] else 'N/A',
                        'inline': True,
                    }],
                )
                save_milestone(milestone)
                break  # Only one milestone alert per check

    if mode == 'status':
        # Force a status report
        sym_lines = []
        for sym, s in stats['symbols'].items():
            sym_lines.append(f'**{sym}**: {s["trade_days"]}d trades, {s["ob_days"]}d orderbook ({s["size_mb"]} MB)')
        
        send_discord(
            '📊 Tick Collector Status',
            f'Service: {"🟢 Running" if is_running else "🔴 Stopped"}\n'
            f'Days collected: **{stats["days_collected"]}**\n'
            f'Total size: **{stats["total_size_gb"]} GB**\n'
            f'Data freshness: {minutes_stale:.0f} min ago\n\n'
            + '\n'.join(sym_lines),
            color=0x3B82F6,
        )


if __name__ == '__main__':
    main()
