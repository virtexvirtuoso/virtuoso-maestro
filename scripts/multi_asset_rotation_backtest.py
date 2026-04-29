#!/usr/bin/env python3
"""
Multi-Asset Momentum + Funding Rotation Strategy

Uses funding rate data across 13 assets to:
1. Rank assets by momentum (price appreciation)
2. Filter by funding rate signal (contrarian)
3. Rotate to top N assets weekly

Assets: BTC, ETH, SOL, ARB, OP, SUI, TIA, INJ, LINK, AVAX, FET, TAO, RENDER
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data'

# Assets with both funding and OHLCV data
ASSETS = ['btc', 'eth', 'sol', 'arb', 'op', 'sui', 'tia', 'inj', 'link', 'avax', 'fet', 'tao', 'render']


def load_asset_data(asset: str, timeframe: str = '1d') -> pd.DataFrame:
    """Load OHLCV and funding for an asset."""
    ohlcv_path = f'{DATA_DIR}/merged/binance_{asset}_usdt_{timeframe}.csv'
    funding_path = f'{DATA_DIR}/derivatives/{asset}_funding.csv'

    try:
        ohlcv = pd.read_csv(ohlcv_path)
        ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
        ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
        ohlcv['asset'] = asset.upper()

        # Drop existing funding column if present (may be mostly NaN)
        if 'funding_rate' in ohlcv.columns:
            ohlcv = ohlcv.drop(columns=['funding_rate'])

        # Load real funding
        funding = pd.read_csv(funding_path)
        funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
        funding = funding.rename(columns={'fundingRate': 'funding_rate_real'})

        # Resample to daily
        funding_daily = funding.set_index('timestamp').resample('D').last().reset_index()

        # Merge
        ohlcv = pd.merge_asof(
            ohlcv,
            funding_daily[['timestamp', 'funding_rate_real']],
            on='timestamp',
            direction='backward'
        )

        ohlcv['funding_rate'] = ohlcv['funding_rate_real'].fillna(0)
        ohlcv = ohlcv.drop(columns=['funding_rate_real'], errors='ignore')

        return ohlcv
    except Exception as e:
        print(f"Error loading {asset}: {e}")
        return None


def load_all_assets(start_date: str = '2023-06-01') -> dict:
    """Load all assets and align to common date range."""
    data = {}

    for asset in ASSETS:
        df = load_asset_data(asset)
        if df is not None and len(df) > 100:
            df = df[df['timestamp'] >= start_date].reset_index(drop=True)
            data[asset.upper()] = df

    print(f"Loaded {len(data)} assets from {start_date}")
    return data


def calculate_rankings(data: dict, lookback: int = 7) -> pd.DataFrame:
    """
    Calculate asset rankings based on momentum and funding.

    Returns DataFrame with daily rankings for each asset.
    """
    # Get common dates
    dates_lists = [set(df['timestamp'].dt.date) for df in data.values()]
    common_dates = sorted(set.intersection(*dates_lists))

    rankings = []

    for date in common_dates:
        date_dt = pd.Timestamp(date)

        scores = {}
        for asset, df in data.items():
            mask = df['timestamp'].dt.date == date
            if not mask.any():
                continue

            idx = mask.idxmax()

            # Need at least lookback days of history
            if idx < lookback:
                continue

            # Momentum: N-day return
            close_now = df.loc[idx, 'close']
            close_then = df.loc[idx - lookback, 'close']
            momentum = (close_now / close_then - 1) * 100

            # Funding: Average funding over lookback
            funding = df.loc[idx - lookback + 1:idx, 'funding_rate'].mean() * 100

            # Funding z-score (for contrarian signal)
            funding_series = df['funding_rate'].iloc[:idx+1]
            if len(funding_series) > 20:
                funding_ma = funding_series.rolling(20).mean().iloc[-1]
                funding_std = funding_series.rolling(20).std().iloc[-1]
                if funding_std > 0:
                    funding_zscore = (df.loc[idx, 'funding_rate'] - funding_ma) / funding_std
                else:
                    funding_zscore = 0
            else:
                funding_zscore = 0

            # Combined score: momentum + contrarian funding
            # Positive momentum is good
            # Extreme negative funding is good (contrarian long)
            # Extreme positive funding is bad (crowded long)
            score = momentum - funding_zscore * 2  # Penalize high funding z-score

            scores[asset] = {
                'date': date_dt,
                'asset': asset,
                'momentum': momentum,
                'funding': funding,
                'funding_zscore': funding_zscore,
                'score': score,
                'close': close_now
            }

        if scores:
            df_scores = pd.DataFrame(scores.values())
            df_scores['rank'] = df_scores['score'].rank(ascending=False)
            rankings.append(df_scores)

    return pd.concat(rankings, ignore_index=True)


def rotation_backtest(
    data: dict,
    rankings: pd.DataFrame,
    top_n: int = 3,
    rebalance_freq: int = 7,  # days
    position_size: float = 0.1,
    commission: float = 0.001
) -> dict:
    """
    Backtest rotation strategy.

    Equal weight across top N assets, rebalance weekly.
    """
    dates = sorted(rankings['date'].unique())
    n = len(dates)

    pnl_history = []
    holdings = {}  # asset -> weight
    last_rebalance = dates[0]

    for i in range(1, n):
        date = dates[i]
        prev_date = dates[i-1]

        # Calculate returns from holdings
        daily_pnl = 0
        for asset, weight in holdings.items():
            df = data[asset]
            curr_mask = df['timestamp'].dt.date == date.date()
            prev_mask = df['timestamp'].dt.date == prev_date.date()

            if curr_mask.any() and prev_mask.any():
                curr_price = df.loc[curr_mask.idxmax(), 'close']
                prev_price = df.loc[prev_mask.idxmax(), 'close']
                ret = (curr_price / prev_price - 1) * weight * position_size
                daily_pnl += ret

        # Rebalance check
        days_since_rebalance = (date - last_rebalance).days
        if days_since_rebalance >= rebalance_freq:
            # Get today's rankings
            today_ranks = rankings[rankings['date'] == date].sort_values('rank')
            new_top = today_ranks.head(top_n)['asset'].tolist()

            # Calculate turnover cost
            old_assets = set(holdings.keys())
            new_assets = set(new_top)
            turnover = len(old_assets.symmetric_difference(new_assets))
            daily_pnl -= turnover * commission * position_size / top_n

            # Update holdings
            holdings = {asset: 1/top_n for asset in new_top}
            last_rebalance = date

        pnl_history.append({
            'date': date,
            'pnl': daily_pnl,
            'n_holdings': len(holdings)
        })

    # Calculate metrics
    pnl_df = pd.DataFrame(pnl_history)
    pnl_df['cumulative'] = (1 + pnl_df['pnl']).cumprod() - 1

    returns = pnl_df['pnl']
    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0

    cummax = (1 + pnl_df['pnl']).cumprod().cummax()
    dd = (1 + pnl_df['pnl']).cumprod() / cummax - 1
    max_dd = dd.min()

    return {
        'total_return': pnl_df['cumulative'].iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_days': n,
        'history': pnl_df
    }


def walk_forward_rotation(data: dict, n_splits: int = 5, top_n: int = 3):
    """Walk-forward test rotation strategy."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: Multi-Asset Rotation (Top {top_n})")
    print(f"{'='*70}")

    # Calculate full rankings
    rankings = calculate_rankings(data, lookback=7)
    dates = sorted(rankings['date'].unique())
    n = len(dates)
    fold_size = n // n_splits

    results = []

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)
        train_end = start + int((end - start) * 0.7)

        train_dates = dates[start:train_end]
        test_dates = dates[train_end:end]

        train_rankings = rankings[rankings['date'].isin(train_dates)]
        test_rankings = rankings[rankings['date'].isin(test_dates)]

        if len(test_rankings) < 20:
            continue

        # Train (just to get performance)
        train_result = rotation_backtest(data, train_rankings, top_n=top_n)

        # Test
        test_result = rotation_backtest(data, test_rankings, top_n=top_n)

        print(f"Fold {i}: Train Sharpe={train_result['sharpe_ratio']:.3f}, "
              f"Test Sharpe={test_result['sharpe_ratio']:.3f}, "
              f"Test Return={test_result['total_return']*100:.1f}%")

        results.append({
            'fold': i,
            'train_sharpe': train_result['sharpe_ratio'],
            'test_sharpe': test_result['sharpe_ratio'],
            'test_return': test_result['total_return'],
            'test_max_dd': test_result['max_drawdown']
        })

    if not results:
        return {'strategy': f'Rotation_Top{top_n}', 'avg_test_sharpe': 0}

    avg_sharpe = np.mean([r['test_sharpe'] for r in results])
    compounded = np.prod([1 + r['test_return'] for r in results]) - 1
    consistency = sum(1 for r in results if r['test_sharpe'] > 0) / len(results)

    print(f"\nSUMMARY: Sharpe={avg_sharpe:.3f}, Return={compounded*100:.1f}%, "
          f"Consistency={consistency*100:.0f}%")

    return {
        'strategy': f'Rotation_Top{top_n}',
        'avg_test_sharpe': avg_sharpe,
        'compounded_return': compounded,
        'consistency': consistency,
        'folds': results
    }


def main():
    print("=" * 80)
    print("MULTI-ASSET MOMENTUM + FUNDING ROTATION BACKTEST")
    print(f"Assets: {', '.join([a.upper() for a in ASSETS])}")
    print("=" * 80)

    data = load_all_assets(start_date='2023-06-01')

    # Test different portfolio sizes
    all_results = []
    for top_n in [3, 5, 7]:
        result = walk_forward_rotation(data, top_n=top_n)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    all_results.sort(key=lambda x: x['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<20} {'Sharpe':>8} {'Return':>10} {'Consist':>8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['strategy']:<20} {r['avg_test_sharpe']:>8.3f} "
              f"{r.get('compounded_return', 0)*100:>9.1f}% "
              f"{r.get('consistency', 0)*100:>7.0f}%")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/multi_asset_rotation_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
