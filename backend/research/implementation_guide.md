# Strategy Implementation Guide
*Quick-start code templates for Tier 1 strategies*

## 1. Funding Rate Carry Strategy

### Data Requirements
- Funding rate data: `~/Desktop/maestro/data/derivatives/funding_rates/`
- OHLCV data: `DataLoader().get_ohlcv_full(symbol, timeframe)`

### Implementation Skeleton

```python
import pandas as pd
import numpy as np
from data_loader import DataLoader

class FundingRateCarry:
    """
    Harvest funding rate carry by taking positions opposite to crowded trades.
    
    Entry: Long when funding < -0.05% (shorts pay longs)
           Short when funding > 0.05% (longs pay shorts)
    Exit: When funding crosses zero or reverses >50%
    """
    
    def __init__(self, min_funding_threshold=0.0005, max_holding_days=14):
        self.min_funding_threshold = min_funding_threshold
        self.max_holding_days = max_holding_days
        
    def calculate_signals(self, ohlcv_df, funding_df):
        """
        ohlcv_df: DataFrame with columns [timestamp, open, high, low, close, volume]
        funding_df: DataFrame with columns [timestamp, funding_rate]
        """
        # Merge OHLCV with funding rate
        df = ohlcv_df.merge(funding_df, on='timestamp', how='left')
        
        # Calculate 7-day average funding rate
        df['funding_7d_avg'] = df['funding_rate'].rolling(7).mean()
        
        # Entry signals
        df['entry_long'] = df['funding_7d_avg'] < -self.min_funding_threshold
        df['entry_short'] = df['funding_7d_avg'] > self.min_funding_threshold
        
        # Exit signals
        df['exit_long'] = (df['funding_7d_avg'] > 0) | (df['funding_7d_avg'] > df['funding_7d_avg'].shift(1) * 0.5)
        df['exit_short'] = (df['funding_7d_avg'] < 0) | (df['funding_7d_avg'] < df['funding_7d_avg'].shift(1) * 0.5)
        
        # Generate position signals
        df['signal'] = 0
        df.loc[df['entry_long'], 'signal'] = 1
        df.loc[df['entry_short'], 'signal'] = -1
        df.loc[df['exit_long'] | df['exit_short'], 'signal'] = 0
        
        # Forward fill signals (hold until exit)
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Max holding period filter
        df['days_held'] = df.groupby((df['position'] != df['position'].shift()).cumsum()).cumcount()
        df.loc[df['days_held'] > self.max_holding_days, 'position'] = 0
        
        return df['position']
    
    def backtest(self, symbols, start_date, end_date):
        """
        Backtest across multiple symbols.
        """
        loader = DataLoader()
        results = {}
        
        for symbol in symbols:
            ohlcv = loader.get_ohlcv_full(symbol, '1d').to_pandas()
            funding = loader.get_funding_rates(symbol)  # Implement this
            
            signals = self.calculate_signals(ohlcv, funding)
            
            # Calculate returns
            ohlcv['returns'] = ohlcv['close'].pct_change()
            ohlcv['strategy_returns'] = signals.shift(1) * ohlcv['returns']
            
            # Add funding P&L
            ohlcv['funding_pnl'] = signals.shift(1) * funding['funding_rate']
            ohlcv['total_returns'] = ohlcv['strategy_returns'] + ohlcv['funding_pnl']
            
            results[symbol] = ohlcv
        
        return results

# Usage
strategy = FundingRateCarry(min_funding_threshold=0.0005)
symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
results = strategy.backtest(symbols, '2023-01-01', '2026-03-09')
```

### Key Metrics to Track
- Cumulative funding P&L (separate from price returns)
- Sharpe ratio (total returns = price + funding)
- Max drawdown
- Win rate
- Average holding period

---

## 2. NR4/NR7 Trend Continuation

### Implementation Skeleton

```python
import pandas as pd
import numpy as np

class NR4TrendContinuation:
    """
    Trade NR4/NR7 breakouts in direction of larger trend.
    
    Entry: NR4/7 in uptrend + close in top 1/3 of range → buy break of NR high
    Exit: Stop at NR low, target 3x NR range
    """
    
    def __init__(self, nr_lookback=4, trend_ema=50, profit_mult=3.0, stop_atr=1.5):
        self.nr_lookback = nr_lookback
        self.trend_ema = trend_ema
        self.profit_mult = profit_mult
        self.stop_atr = stop_atr
        
    def calculate_signals(self, df):
        """
        df: DataFrame with [timestamp, open, high, low, close, volume]
        """
        # Calculate range
        df['range'] = df['high'] - df['low']
        
        # Identify NR4/NR7
        df['min_range'] = df['range'].rolling(self.nr_lookback).min()
        df['is_nr'] = df['range'] == df['min_range']
        
        # Trend filter
        df['ema'] = df['close'].ewm(span=self.trend_ema).mean()
        df['uptrend'] = df['close'] > df['ema']
        df['downtrend'] = df['close'] < df['ema']
        
        # Close position in range
        df['close_pct'] = (df['close'] - df['low']) / df['range']
        
        # Entry conditions
        df['long_setup'] = (
            df['is_nr'] & 
            df['uptrend'] & 
            (df['close_pct'] > 0.67)  # Top 1/3
        )
        df['short_setup'] = (
            df['is_nr'] & 
            df['downtrend'] & 
            (df['close_pct'] < 0.33)  # Bottom 1/3
        )
        
        # Breakout triggers
        df['long_entry'] = (df['long_setup'].shift(1)) & (df['close'] > df['high'].shift(1))
        df['short_entry'] = (df['short_setup'].shift(1)) & (df['close'] < df['low'].shift(1))
        
        # ATR for stops
        df['atr'] = df['range'].rolling(14).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['long_entry'], 'signal'] = 1
        df.loc[df['short_entry'], 'signal'] = -1
        
        # TODO: Implement proper stop loss and take profit logic
        # (requires tracking entry price and NR range at entry)
        
        return df['signal']

# Usage
strategy = NR4TrendContinuation(nr_lookback=4, trend_ema=50)
df = loader.get_ohlcv_full('BTC', '4h').to_pandas()
signals = strategy.calculate_signals(df)
```

### Optimization Targets
- `nr_lookback`: 4 vs 7 (or both)
- `trend_ema`: 20, 50, 100
- `profit_mult`: 2.0-4.0
- Close position threshold: top/bottom 20%, 33%, 50%

---

## 3. Kalman Filter Pairs Trading

### Implementation Skeleton

```python
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

class KalmanPairsTrading:
    """
    Dynamic hedge ratio pairs trading using Kalman filter.
    
    Entry: Spread z-score < -1.5 (long spread) or > 1.5 (short spread)
    Exit: Spread crosses zero or |z| > 3.0
    """
    
    def __init__(self, z_threshold=1.5, z_stop=3.0, ma_period=20):
        self.z_threshold = z_threshold
        self.z_stop = z_stop
        self.ma_period = ma_period
        
    def estimate_hedge_ratio(self, price_a, price_b):
        """
        Estimate dynamic hedge ratio using Kalman filter.
        """
        # Prepare observation matrix
        obs_mat = np.vstack([price_b, np.ones(len(price_b))]).T
        
        # Kalman filter (estimate beta and alpha)
        kf = KalmanFilter(
            n_dim_obs=1, 
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=1.0,
            transition_covariance=0.01 * np.eye(2)
        )
        
        state_means, _ = kf.filter(price_a)
        beta = state_means[:, 0]  # Hedge ratio
        alpha = state_means[:, 1]  # Intercept
        
        return beta, alpha
    
    def calculate_signals(self, df_a, df_b):
        """
        df_a, df_b: DataFrames with close prices for asset A and B
        """
        # Merge on timestamp
        df = df_a[['timestamp', 'close']].merge(
            df_b[['timestamp', 'close']], 
            on='timestamp', 
            suffixes=('_a', '_b')
        )
        
        # Estimate dynamic hedge ratio
        beta, alpha = self.estimate_hedge_ratio(df['close_a'].values, df['close_b'].values)
        df['beta'] = beta
        
        # Calculate spread
        df['spread'] = df['close_a'] - df['beta'] * df['close_b']
        
        # Calculate z-score
        df['spread_ma'] = df['spread'].rolling(self.ma_period).mean()
        df['spread_sd'] = df['spread'].rolling(self.ma_period).std()
        df['z_score'] = (df['spread'] - df['spread_ma']) / df['spread_sd']
        
        # Entry signals
        df['entry_long_spread'] = df['z_score'] < -self.z_threshold
        df['entry_short_spread'] = df['z_score'] > self.z_threshold
        
        # Exit signals
        df['exit'] = (df['z_score'].abs() > self.z_stop) | ((df['z_score'] * df['z_score'].shift(1)) < 0)
        
        # Generate position signals
        df['signal'] = 0
        df.loc[df['entry_long_spread'], 'signal'] = 1  # Long A, Short B
        df.loc[df['entry_short_spread'], 'signal'] = -1  # Short A, Long B
        df.loc[df['exit'], 'signal'] = 0
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df

# Usage
strategy = KalmanPairsTrading(z_threshold=1.5)
df_btc = loader.get_ohlcv_full('BTC', '1h').to_pandas()
df_eth = loader.get_ohlcv_full('ETH', '1h').to_pandas()
result = strategy.calculate_signals(df_btc, df_eth)
```

### Data Requirements
- Install: `pip install pykalman`
- Test pairs: BTC-ETH, SOL-AVAX, USDT-USDC, BTC_spot-BTC_perp

### Key Considerations
- Transaction costs CRITICAL (10bps each leg = 20bps round-trip for pair)
- Test cointegration first (ADF test p < 0.05)
- Recalculate cointegration every 30 days

---

## 4. EWMAC Trend Following

### Implementation Skeleton

```python
import pandas as pd
import numpy as np

class EWMAC:
    """
    Exponentially Weighted Moving Average Crossover (Carver variant).
    
    Uses multiple EWMA pairs to capture trends at different speeds.
    Volatility-scaled position sizing.
    """
    
    def __init__(self, fast_periods=[8, 16, 32, 64], slow_periods=[32, 64, 128, 256], target_vol=0.10):
        self.fast_periods = fast_periods
        self.slow_periods = slow_periods
        self.target_vol = target_vol
        
    def calculate_forecast(self, df, fast, slow):
        """
        Calculate single EWMA forecast.
        """
        df[f'ewma_{fast}'] = df['close'].ewm(span=fast).mean()
        df[f'ewma_{slow}'] = df['close'].ewm(span=slow).mean()
        
        # Raw forecast: (fast - slow) / price_volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
        
        raw_forecast = (df[f'ewma_{fast}'] - df[f'ewma_{slow}']) / (df['close'] * df['volatility'])
        
        # Scale and cap forecast to [-20, +20]
        forecast = raw_forecast * 10  # Scaling factor
        forecast = forecast.clip(-20, 20)
        
        return forecast
    
    def calculate_signals(self, df):
        """
        Calculate combined forecast from multiple EWMA pairs.
        """
        forecasts = []
        
        for fast, slow in zip(self.fast_periods, self.slow_periods):
            forecast = self.calculate_forecast(df.copy(), fast, slow)
            forecasts.append(forecast)
        
        # Average forecasts
        df['combined_forecast'] = pd.concat(forecasts, axis=1).mean(axis=1)
        
        # Position sizing: scale by target volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['position'] = (df['combined_forecast'] / 10) * (self.target_vol / df['volatility'])
        
        # Clip position to [-1, 1]
        df['position'] = df['position'].clip(-1, 1)
        
        # Only trade if |forecast| > 2 (minimum signal strength)
        df.loc[df['combined_forecast'].abs() < 2, 'position'] = 0
        
        return df['position']

# Usage
strategy = EWMAC(
    fast_periods=[8, 16, 32, 64],
    slow_periods=[32, 64, 128, 256],
    target_vol=0.10
)
df = loader.get_ohlcv_full('BTC', '1d').to_pandas()
signals = strategy.calculate_signals(df)
```

### Optimization Targets
- Test 4-6 EWMA pairs (fast/slow combinations)
- `target_vol`: 10%, 15%, 20%
- Forecast combination: average vs weighted by recent Sharpe
- Minimum forecast threshold: 2, 5, 10

---

## 5. OU Process Mean Reversion (Half-Life)

### Implementation Skeleton

```python
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

class OUMeanReversion:
    """
    Ornstein-Uhlenbeck mean reversion using half-life estimation.
    
    Entry: |z-score| > 1.5 AND half-life < 20 days
    Exit: z-score crosses zero OR half-life exceeds threshold
    """
    
    def __init__(self, z_threshold=1.5, z_stop=3.0, max_half_life=20, min_half_life=2):
        self.z_threshold = z_threshold
        self.z_stop = z_stop
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        
    def calculate_half_life(self, prices, lookback=60):
        """
        Estimate half-life using AR(1) model.
        Half-life = -log(2) / log(lambda)
        """
        if len(prices) < lookback:
            return np.nan
        
        # Fit AR(1): log(price_t) = lambda * log(price_{t-1}) + epsilon
        lag_prices = prices.shift(1).dropna()
        prices_aligned = prices.loc[lag_prices.index]
        
        model = OLS(prices_aligned, lag_prices)
        result = model.fit()
        lambda_param = result.params[0]
        
        if lambda_param >= 1 or lambda_param <= 0:
            return np.nan  # Not mean-reverting
        
        half_life = -np.log(2) / np.log(lambda_param)
        return half_life
    
    def calculate_signals(self, df, lookback=60):
        """
        df: DataFrame with [timestamp, close]
        """
        # Calculate rolling z-score
        df['ma'] = df['close'].rolling(lookback).mean()
        df['sd'] = df['close'].rolling(lookback).std()
        df['z_score'] = (df['close'] - df['ma']) / df['sd']
        
        # Calculate rolling half-life
        df['half_life'] = df['close'].rolling(lookback).apply(
            lambda x: self.calculate_half_life(pd.Series(x)), raw=False
        )
        
        # Entry conditions
        df['entry_long'] = (
            (df['z_score'] < -self.z_threshold) & 
            (df['half_life'] < self.max_half_life) &
            (df['half_life'] > self.min_half_life)
        )
        df['entry_short'] = (
            (df['z_score'] > self.z_threshold) & 
            (df['half_life'] < self.max_half_life) &
            (df['half_life'] > self.min_half_life)
        )
        
        # Exit conditions
        df['exit'] = (
            (df['z_score'].abs() > self.z_stop) |  # Stop loss
            ((df['z_score'] * df['z_score'].shift(1)) < 0) |  # Zero cross
            (df['half_life'] > self.max_half_life)  # Half-life too slow
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['entry_long'], 'signal'] = 1
        df.loc[df['entry_short'], 'signal'] = -1
        df.loc[df['exit'], 'signal'] = 0
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df

# Usage
strategy = OUMeanReversion(z_threshold=1.5, max_half_life=20)
df = loader.get_ohlcv_full('BTC', '1d').to_pandas()
result = strategy.calculate_signals(df, lookback=60)
```

### Key Metrics
- Half-life distribution (should be 2-20 days)
- z-score at entry (should be >1.5)
- Mean reversion time (should match half-life * 3)
- ADF test p-value (should be <0.05 for stationarity)

---

## Testing Framework

### Walk-Forward Validation Template

```python
from datetime import datetime, timedelta

def walk_forward_backtest(strategy, df, train_days=252, test_days=63, step_days=63):
    """
    Walk-forward optimization.
    
    train_days: In-sample optimization window (e.g., 252 days = 1 year)
    test_days: Out-of-sample test window (e.g., 63 days = 3 months)
    step_days: Step size for rolling window (e.g., 63 days)
    """
    results = []
    
    start_idx = train_days
    while start_idx + test_days < len(df):
        # In-sample data
        train_df = df.iloc[start_idx - train_days:start_idx]
        
        # Optimize parameters on in-sample (use Optuna here)
        best_params = optimize_strategy(strategy, train_df)
        
        # Out-of-sample data
        test_df = df.iloc[start_idx:start_idx + test_days]
        
        # Test with best params
        strategy.set_params(**best_params)
        test_result = strategy.backtest(test_df)
        
        results.append({
            'train_start': train_df.index[0],
            'train_end': train_df.index[-1],
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1],
            'params': best_params,
            'sharpe': test_result['sharpe'],
            'returns': test_result['total_return'],
            'max_dd': test_result['max_drawdown']
        })
        
        start_idx += step_days
    
    return pd.DataFrame(results)

def optimize_strategy(strategy, df):
    """
    Use Optuna to find best parameters.
    """
    import optuna
    
    def objective(trial):
        # Suggest parameters (example for NR4)
        nr_lookback = trial.suggest_int('nr_lookback', 4, 10)
        trend_ema = trial.suggest_int('trend_ema', 20, 100)
        profit_mult = trial.suggest_float('profit_mult', 2.0, 5.0)
        
        strategy.set_params(nr_lookback=nr_lookback, trend_ema=trend_ema, profit_mult=profit_mult)
        result = strategy.backtest(df)
        
        return result['sharpe']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    return study.best_params
```

---

## Final Checklist Before Production

- [ ] Walk-forward validation passed (OOS Sharpe > 0.5)
- [ ] Backtest includes transaction costs (10bps)
- [ ] Backtest includes slippage (0.05-0.1%)
- [ ] Max drawdown acceptable (<30%)
- [ ] Sharpe ratio > 1.0 (OOS)
- [ ] Win rate > 45% (for mean reversion) or trend capture > 60% (for trend following)
- [ ] Strategy is NOT curve-fit (parameter sensitivity tested)
- [ ] Bonferroni correction applied (if testing multiple variants)
- [ ] Paper trade results confirm backtest (2-4 weeks minimum)
- [ ] Risk management in place (max position size, portfolio heat)
- [ ] Monitoring dashboard created (track live performance vs backtest)

---

**Next**: Start with Funding Rate Carry (easiest to implement, data ready). Then move to NR4/NR7 and OU Process.
