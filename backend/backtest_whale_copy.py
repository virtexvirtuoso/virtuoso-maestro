#!/usr/bin/env python3
"""
Whale Copy-Trading Signal Backtest
===================================
Strategy: Follow Elite Whale Entries from Whale Hunter DB
- When a whale opens/flips to a position, enter same direction on next daily close
- Exit after N days (test 1, 3, 7, 14 day holds)
- Commission: 20bps per trade (entry + exit = 40bps round trip)

Data: whale_trades.csv extracted from whale_hunter/data/whale_alerts.db
Price: Downloaded via ccxt (Bybit perpetuals, 1d)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try ccxt for price data
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data")
RESULTS_DIR = os.path.join(DATA_DIR, "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMISSION_BPS = 20  # per side
COMMISSION = COMMISSION_BPS / 10000  # 0.002

# ─── Price Data ────────────────────────────────────────────────

def fetch_daily_prices(symbols, start_date="2025-11-01", end_date="2026-02-15"):
    """Fetch daily OHLCV from Bybit via ccxt."""
    if not HAS_CCXT:
        raise RuntimeError("ccxt not installed")
    
    # Try multiple exchanges (Bybit blocked in US, fallback to Binance/OKX)
    for ex_name in ['binanceusdm', 'okx', 'bybit']:
        try:
            exchange = getattr(ccxt, ex_name)({'enableRateLimit': True})
            exchange.load_markets()
            print(f"  Using exchange: {ex_name}")
            break
        except Exception as e:
            print(f"  {ex_name} failed: {e}")
            continue
    else:
        raise RuntimeError("No exchange available")
    
    since = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    prices = {}
    for sym in symbols:
        try:
            # Try perp, then spot
            for pair in [f"{sym}/USDT:USDT", f"{sym}/USDT"]:
                if pair in exchange.markets:
                    break
            ohlcv = exchange.fetch_ohlcv(pair, '1d', since=since, limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[df.index < pd.Timestamp(end_date)]
            prices[sym] = df
            print(f"  ✅ {sym}: {len(df)} daily bars")
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
    return prices


def load_or_fetch_prices(symbols):
    """Load cached prices or fetch from exchange."""
    cache_path = os.path.join(DATA_DIR, "whale_price_cache.parquet")
    
    if os.path.exists(cache_path):
        print("Loading cached prices...")
        df = pd.read_parquet(cache_path)
        prices = {sym: grp.droplevel(0) for sym, grp in df.groupby(level=0)}
        # Check if all symbols present
        missing = [s for s in symbols if s not in prices]
        if not missing:
            return prices
        print(f"  Missing: {missing}, fetching...")
    
    print("Fetching daily prices from Bybit...")
    prices = fetch_daily_prices(symbols)
    
    # Cache
    if prices:
        dfs = []
        for sym, df in prices.items():
            df2 = df.copy()
            df2['symbol'] = sym
            df2 = df2.set_index('symbol', append=True).swaplevel()
            dfs.append(df2)
        pd.concat(dfs).to_parquet(cache_path)
    
    return prices


# ─── Whale Data Processing ─────────────────────────────────────

def load_whale_trades():
    """Load and process whale trade data."""
    path = os.path.join(DATA_DIR, "whale_trades.csv")
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['dt'])
    df['date'] = df['dt'].dt.date
    print(f"Loaded {len(df)} whale trade records")
    print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
    print(f"  Traders: {df['trader_name'].nunique()}, Symbols: {df['symbol'].nunique()}")
    print(f"  Actions: {df['action'].value_counts().to_dict()}")
    return df


def compute_trader_stats(df):
    """Compute per-trader performance stats from position data.
    
    We reconstruct trades: a flip/increase = entry, close = exit.
    For each trader+symbol, we match opens to closes chronologically.
    """
    # Focus on flip (new position) and close actions
    opens = df[df['action'].isin(['flip', 'increase'])].copy()
    closes = df[df['action'] == 'close'].copy()
    
    trades = []
    for (trader, symbol), group_opens in opens.groupby(['trader_name', 'symbol']):
        group_closes = closes[(closes['trader_name'] == trader) & (closes['symbol'] == symbol)].sort_values('timestamp')
        
        for _, open_row in group_opens.sort_values('timestamp').iterrows():
            # Find next close after this open
            future_closes = group_closes[group_closes['timestamp'] > open_row['timestamp']]
            if len(future_closes) > 0:
                close_row = future_closes.iloc[0]
                # PnL: use entry prices (open's entry_price vs close's entry_price which is the avg entry of the closed position)
                # Since close action's entry_price = the entry of what was closed, and we know position_side
                # The close's entry_price should be the same as open's entry_price (same position)
                # We'd need market price at close time — we'll compute from OHLCV data instead
                trades.append({
                    'trader': trader,
                    'tier': open_row['trader_tier'],
                    'symbol': symbol,
                    'side': open_row['position_side'],
                    'entry_time': open_row['dt'],
                    'exit_time': close_row['dt'],
                    'entry_price': open_row['entry_price'],
                    'leverage': open_row['leverage'],
                    'trade_score': open_row['trade_score'],
                })
    
    trades_df = pd.DataFrame(trades)
    print(f"\nReconstructed {len(trades_df)} complete trades (open→close pairs)")
    return trades_df


def classify_traders(trades_df, prices):
    """Classify traders as elite based on actual P&L using market prices."""
    results = []
    
    for _, trade in trades_df.iterrows():
        sym = trade['symbol']
        if sym not in prices:
            continue
        
        price_df = prices[sym]
        entry_date = pd.Timestamp(trade['entry_time']).normalize()
        exit_date = pd.Timestamp(trade['exit_time']).normalize()
        
        # Get close prices on entry and exit dates
        entry_prices = price_df[price_df.index >= entry_date]
        exit_prices = price_df[price_df.index >= exit_date]
        
        if len(entry_prices) == 0 or len(exit_prices) == 0:
            continue
        
        entry_px = entry_prices.iloc[0]['close']
        exit_px = exit_prices.iloc[0]['close']
        
        if trade['side'] == 'LONG':
            ret = (exit_px - entry_px) / entry_px
        else:
            ret = (entry_px - exit_px) / entry_px
        
        ret -= 2 * COMMISSION  # round trip
        
        results.append({
            'trader': trade['trader'],
            'tier': trade['tier'],
            'symbol': sym,
            'side': trade['side'],
            'return': ret,
            'entry_time': trade['entry_time'],
            'exit_time': trade['exit_time'],
        })
    
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute per-trader stats
    stats = results_df.groupby('trader').agg(
        tier=('tier', 'first'),
        total_trades=('return', 'count'),
        wins=('return', lambda x: (x > 0).sum()),
        total_return=('return', 'sum'),
        avg_return=('return', 'mean'),
        std_return=('return', 'std'),
    ).reset_index()
    stats['win_rate'] = stats['wins'] / stats['total_trades']
    stats['sharpe'] = stats['avg_return'] / stats['std_return'].replace(0, np.nan) * np.sqrt(252)
    stats = stats.sort_values('total_return', ascending=False)
    
    print(f"\n{'='*70}")
    print("TRADER PERFORMANCE RANKING (all reconstructed trades)")
    print(f"{'='*70}")
    for _, row in stats.head(15).iterrows():
        print(f"  {row['trader']:25s} | {row['tier']:15s} | trades={row['total_trades']:3.0f} | "
              f"WR={row['win_rate']:.1%} | total={row['total_return']:+.2%} | avg={row['avg_return']:+.3%}")
    
    return results_df, stats


# ─── Copy-Trading Backtest ──────────────────────────────────────

def generate_signals(whale_df, elite_traders, prices):
    """Generate daily signals: when an elite whale opens, enter same direction next day."""
    # Filter to elite traders and open/flip actions
    opens = whale_df[
        (whale_df['action'].isin(['flip', 'increase'])) &
        (whale_df['trader_name'].isin(elite_traders))
    ].copy()
    
    signals = []
    for _, row in opens.iterrows():
        sym = row['symbol']
        if sym not in prices:
            continue
        
        signal_date = pd.Timestamp(row['dt']).normalize() + timedelta(days=1)  # enter next day
        direction = 1 if row['position_side'] == 'LONG' else -1
        
        signals.append({
            'date': signal_date,
            'symbol': sym,
            'direction': direction,
            'trader': row['trader_name'],
            'leverage': row['leverage'],
        })
    
    return pd.DataFrame(signals)


def backtest_hold_period(signals_df, prices, hold_days=7):
    """Backtest: enter on signal date close, exit after hold_days."""
    if len(signals_df) == 0:
        return {'sharpe': 0, 'total_return': 0, 'trades': 0, 'win_rate': 0}
    
    trade_returns = []
    
    for _, sig in signals_df.iterrows():
        sym = sig['symbol']
        if sym not in prices:
            continue
        
        price_df = prices[sym]
        entry_bars = price_df[price_df.index >= sig['date']]
        if len(entry_bars) < hold_days + 1:
            continue
        
        entry_px = entry_bars.iloc[0]['close']
        exit_px = entry_bars.iloc[min(hold_days, len(entry_bars)-1)]['close']
        
        ret = sig['direction'] * (exit_px - entry_px) / entry_px
        ret -= 2 * COMMISSION  # round trip
        
        trade_returns.append({
            'date': sig['date'],
            'symbol': sym,
            'direction': sig['direction'],
            'return': ret,
            'trader': sig['trader'],
        })
    
    if not trade_returns:
        return {'sharpe': 0, 'total_return': 0, 'trades': 0, 'win_rate': 0}
    
    tr_df = pd.DataFrame(trade_returns)
    
    # Portfolio: equal weight all active signals per day
    daily_returns = tr_df.groupby('date')['return'].mean()
    
    total_ret = (1 + daily_returns).prod() - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    win_rate = (tr_df['return'] > 0).mean()
    avg_ret = tr_df['return'].mean()
    
    return {
        'sharpe': round(sharpe, 3),
        'total_return': round(total_ret, 4),
        'avg_trade_return': round(avg_ret, 5),
        'trades': len(tr_df),
        'win_rate': round(win_rate, 4),
        'max_trade': round(tr_df['return'].max(), 4),
        'min_trade': round(tr_df['return'].min(), 4),
        'symbols': tr_df['symbol'].nunique(),
    }


def walk_forward_test(whale_df, prices, elite_traders, hold_days=7, n_folds=10):
    """Expanding window walk-forward test."""
    signals_df = generate_signals(whale_df, elite_traders, prices)
    if len(signals_df) == 0:
        return []
    
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    min_date = signals_df['date'].min()
    max_date = signals_df['date'].max()
    total_days = (max_date - min_date).days
    fold_size = total_days // n_folds
    
    oos_results = []
    for fold in range(2, n_folds + 1):  # need at least 1 fold for IS
        cutoff = min_date + timedelta(days=fold * fold_size)
        oos_start = cutoff
        oos_end = min_date + timedelta(days=(fold + 1) * fold_size) if fold < n_folds else max_date + timedelta(days=30)
        
        oos_signals = signals_df[(signals_df['date'] >= oos_start) & (signals_df['date'] < oos_end)]
        if len(oos_signals) < 3:
            continue
        
        result = backtest_hold_period(oos_signals, prices, hold_days)
        result['fold'] = fold
        result['oos_start'] = str(oos_start.date())
        result['oos_end'] = str(oos_end.date()) if fold < n_folds else str(max_date.date())
        oos_results.append(result)
    
    return oos_results


def permutation_test(whale_df, prices, elite_traders, hold_days=7, n_perms=200):
    """Shuffle entry dates to test if signal is better than random timing."""
    signals_df = generate_signals(whale_df, elite_traders, prices)
    if len(signals_df) == 0:
        return {'p_value': 1.0, 'actual_sharpe': 0}
    
    actual = backtest_hold_period(signals_df, prices, hold_days)
    actual_sharpe = actual['sharpe']
    
    perm_sharpes = []
    for i in range(n_perms):
        perm_signals = signals_df.copy()
        # Shuffle dates within each symbol
        for sym in perm_signals['symbol'].unique():
            mask = perm_signals['symbol'] == sym
            dates = perm_signals.loc[mask, 'date'].values
            np.random.shuffle(dates)
            perm_signals.loc[mask, 'date'] = dates
        
        perm_result = backtest_hold_period(perm_signals, prices, hold_days)
        perm_sharpes.append(perm_result['sharpe'])
    
    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    
    return {
        'actual_sharpe': actual_sharpe,
        'perm_mean_sharpe': round(np.mean(perm_sharpes), 3),
        'perm_std_sharpe': round(np.std(perm_sharpes), 3),
        'p_value': round(p_value, 4),
        'n_perms': n_perms,
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    print("🐋 WHALE COPY-TRADING SIGNAL BACKTEST")
    print("=" * 70)
    
    # Step 1: Load whale data
    whale_df = load_whale_trades()
    
    # Step 2: Get price data for traded symbols
    top_symbols = whale_df['symbol'].value_counts().head(6).index.tolist()
    # Filter to symbols likely on Bybit
    tradeable = ['BTC', 'ETH', 'SOL', 'XRP', 'ZEC', 'HYPE', 'DOGE', 'AAVE']
    symbols = [s for s in top_symbols if s in tradeable]
    if not symbols:
        symbols = ['BTC', 'ETH', 'SOL']
    
    print(f"\nFocusing on {len(symbols)} symbols: {symbols}")
    prices = load_or_fetch_prices(symbols)
    
    if not prices:
        print("❌ No price data available!")
        return
    
    # Step 3: Reconstruct trades and rank traders
    trades_df = compute_trader_stats(whale_df)
    trade_results, trader_stats = classify_traders(trades_df, prices)
    
    if len(trader_stats) == 0:
        print("❌ No complete trades found!")
        return
    
    # Step 4: Define "elite" filter criteria
    # Since trader_stats win_rate from DB is 0, use our computed stats
    # Try multiple elite definitions
    elite_configs = {
        'top_tier': trader_stats[trader_stats['tier'] == 'elite']['trader'].tolist(),
        'profitable': trader_stats[trader_stats['total_return'] > 0]['trader'].tolist(),
        'high_wr': trader_stats[(trader_stats['win_rate'] > 0.50) & (trader_stats['total_trades'] >= 5)]['trader'].tolist(),
        'all_traders': trader_stats['trader'].tolist(),
    }
    
    print(f"\n{'='*70}")
    print("ELITE FILTER CONFIGS")
    print(f"{'='*70}")
    for name, traders in elite_configs.items():
        print(f"  {name}: {len(traders)} traders")
    
    # Step 5: Backtest each config x hold period
    all_results = {}
    hold_periods = [1, 3, 7, 14]
    
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS (Full Sample)")
    print(f"{'='*70}")
    
    best_sharpe = -999
    best_config = None
    
    for config_name, elite_traders in elite_configs.items():
        if not elite_traders:
            continue
        
        config_results = {}
        for hold in hold_periods:
            signals = generate_signals(whale_df, elite_traders, prices)
            result = backtest_hold_period(signals, prices, hold)
            config_results[f"hold_{hold}d"] = result
            
            tag = f"{config_name} / {hold}d hold"
            print(f"  {tag:35s} | Sharpe={result['sharpe']:+.2f} | "
                  f"Return={result['total_return']:+.2%} | WR={result['win_rate']:.1%} | "
                  f"Trades={result['trades']}")
            
            if result['sharpe'] > best_sharpe and result['trades'] >= 10:
                best_sharpe = result['sharpe']
                best_config = (config_name, hold, elite_traders)
        
        all_results[config_name] = config_results
    
    # Step 6: Walk-forward validation on best config
    wf_results = {}
    perm_results = {}
    
    if best_config:
        config_name, hold, elite_traders = best_config
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD VALIDATION: {config_name} / {hold}d hold")
        print(f"{'='*70}")
        
        wf = walk_forward_test(whale_df, prices, elite_traders, hold, n_folds=10)
        wf_results = wf
        
        if wf:
            oos_sharpes = [r['sharpe'] for r in wf]
            oos_returns = [r['total_return'] for r in wf]
            print(f"\n  OOS Folds: {len(wf)}")
            print(f"  OOS Sharpe: mean={np.mean(oos_sharpes):.3f}, std={np.std(oos_sharpes):.3f}")
            print(f"  OOS Return: mean={np.mean(oos_returns):.4f}")
            for r in wf:
                print(f"    Fold {r['fold']}: {r['oos_start']} to {r['oos_end']} | "
                      f"Sharpe={r['sharpe']:+.2f} | Ret={r['total_return']:+.2%} | Trades={r['trades']}")
        
        # Step 7: Permutation test
        print(f"\n{'='*70}")
        print(f"PERMUTATION TEST (200 shuffles)")
        print(f"{'='*70}")
        
        perm = permutation_test(whale_df, prices, elite_traders, hold, n_perms=200)
        perm_results = perm
        
        print(f"  Actual Sharpe:  {perm['actual_sharpe']:+.3f}")
        print(f"  Random Sharpe:  {perm['perm_mean_sharpe']:+.3f} ± {perm['perm_std_sharpe']:.3f}")
        print(f"  p-value:        {perm['p_value']:.4f}")
        print(f"  Significant:    {'✅ YES' if perm['p_value'] < 0.05 else '❌ NO'} (α=0.05)")
    
    # Save results
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'whale_trades': len(whale_df),
            'symbols': list(prices.keys()),
            'commission_bps': COMMISSION_BPS,
            'date_range': f"{whale_df['dt'].min()} to {whale_df['dt'].max()}",
        },
        'trader_stats': trader_stats.to_dict(orient='records') if len(trader_stats) > 0 else [],
        'backtest_results': all_results,
        'best_config': {
            'filter': best_config[0] if best_config else None,
            'hold_days': best_config[1] if best_config else None,
            'n_traders': len(best_config[2]) if best_config else 0,
        },
        'walk_forward': wf_results,
        'permutation_test': perm_results,
    }
    
    output_path = os.path.join(RESULTS_DIR, "whale_copy_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if best_config:
        print(f"  Best config: {best_config[0]} filter, {best_config[1]}d hold")
        print(f"  Best Sharpe: {best_sharpe:+.3f}")
        if perm_results:
            print(f"  Statistically significant: {'YES' if perm_results.get('p_value', 1) < 0.05 else 'NO'} (p={perm_results.get('p_value', 'N/A')})")
    else:
        print("  No viable config found with >= 10 trades")


if __name__ == "__main__":
    main()
