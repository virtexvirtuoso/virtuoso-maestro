# Maestro V2 Strategy Implementation Plan

**Academic Paper Analysis & Implementation Roadmap**

*Generated: 2025-02-06*

---

## Executive Summary

This document synthesizes 6 academic papers into actionable implementation plans for the Maestro trading platform. Each strategy is evaluated for:
- **Priority** (based on Sharpe ratio, implementation complexity, and synergy with existing infrastructure)
- **Integration path** with Maestro V2's VectorBT + Optuna engine
- **Timeline estimates** and dependencies

### Priority Ranking Overview

| Rank | Strategy | Sharpe (Reported) | Complexity | Est. Timeline |
|------|----------|-------------------|------------|---------------|
| 1 | Time Series Momentum (TSMOM) | >1.0 | ⭐⭐ (2/5) | 1-2 weeks |
| 2 | Funding Rate Arbitrage | 6.5-58.4 (6-mo) | ⭐⭐ (2/5) | 1-2 weeks |
| 3 | Wasserstein Regime Clustering | N/A (filter) | ⭐⭐⭐ (3/5) | 2-3 weeks |
| 4 | Copula Pairs Trading | Variable | ⭐⭐⭐⭐ (4/5) | 3-4 weeks |
| 5 | Multi-Level DQN | 2.74 | ⭐⭐⭐⭐⭐ (5/5) | 6-8 weeks |
| 6 | Attention Factors StatArb | 4.0 (2.3 net) | ⭐⭐⭐⭐⭐ (5/5) | 8-12 weeks |

---

## Paper 1: Time Series Momentum (AQR/Moskowitz et al.)

### Core Strategy
Trend-following across multiple asset classes using past 12-month returns as signals.

### Key Findings
- **Sharpe Ratio**: >1.0 annualized (diversified portfolio)
- **Test Period**: 1985-2009, 58 futures contracts
- **Key Insight**: 100% of instruments showed positive TSMOM returns
- **Reversal**: Partial reversal after 12 months (mean reversion kicks in)

### Key Parameters
```python
TSMOM_CONFIG = {
    'lookback_months': 12,           # Primary signal period
    'holding_period': 1,              # Rebalance monthly
    'volatility_target': 0.40,        # 40% annualized vol per position
    'volatility_lookback_days': 60,   # EWMA center of mass
    'volatility_decay': 0.9836,       # δ = 60/(60+1) ≈ 0.9836
}
```

### Volatility Estimation Formula
```python
def calculate_ewma_volatility(returns, delta=0.9836):
    """
    σ²_t = 261 * Σ(1-δ)δⁱ(r_{t-1-i} - r̄)²
    """
    weights = (1 - delta) * (delta ** np.arange(len(returns)))
    weighted_var = np.sum(weights * (returns - returns.mean())**2)
    return np.sqrt(261 * weighted_var)
```

### Signal Generation
```python
def tsmom_signal(prices, lookback=252):
    """
    signal = sign(r_{t-12,t}) * (40% / σ_t)
    Go long if past 12-month return positive, short otherwise
    """
    returns_12m = prices.pct_change(lookback)
    signal = np.sign(returns_12m)
    vol = calculate_ewma_volatility(prices.pct_change())
    position_size = 0.40 / vol  # Target 40% vol
    return signal * position_size
```

### Data Requirements
- **Frequency**: Daily OHLCV
- **History**: 12+ months for signal, 60 days for volatility
- **Assets**: BTC, ETH perpetual futures (easily extendable)
- **Source**: Binance/BitMEX API (already in Maestro)

### Maestro V2 Integration

```python
# strategies/tsmom.py
from engine_v2.strategy_adapter import VectorBTStrategy, SignalOutput

class TSMOMStrategy(VectorBTStrategy):
    """Time Series Momentum Strategy"""
    
    @staticmethod
    def get_params():
        return {
            'lookback': {'type': 'int', 'low': 60, 'high': 365, 'default': 252},
            'vol_target': {'type': 'float', 'low': 0.1, 'high': 0.6, 'default': 0.4},
            'vol_lookback': {'type': 'int', 'low': 20, 'high': 120, 'default': 60},
        }
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        close = data['close']
        lookback = params['lookback']
        vol_target = params['vol_target']
        
        # 12-month (or lookback) return
        momentum = close.pct_change(lookback)
        
        # EWMA volatility
        daily_ret = close.pct_change()
        vol = daily_ret.ewm(span=params['vol_lookback']).std() * np.sqrt(365)
        
        # Position sizing
        raw_signal = np.sign(momentum)
        position_size = vol_target / vol.clip(lower=0.05)
        
        entries_long = (raw_signal > 0).astype(int)
        entries_short = (raw_signal < 0).astype(int)
        exits = raw_signal.diff().abs() > 0
        
        return SignalOutput(
            entries_long=entries_long,
            entries_short=entries_short,
            exits_long=exits,
            exits_short=exits,
            size=position_size.abs()
        )
```

### Implementation Timeline: 1-2 weeks
- Week 1: Core strategy implementation + unit tests
- Week 2: Walk-forward optimization tuning

---

## Paper 2: Funding Rate Arbitrage (Werapun et al., 2025)

### Core Strategy
Capture funding rate payments while hedging directional exposure through spot/futures positions.

### Key Findings
- **Returns**: Up to 115.9% over 6 months (Drift XRP 7x)
- **Max Drawdown**: -1.92% (vs -25.92% for HODL)
- **Sharpe Ratios**: 6.50-58.40 (vs 2.89 HODL)
- **Correlation with HODL**: Near zero (diversification benefit)

### Key Parameters
```python
FUNDING_ARB_CONFIG = {
    'leverage': {'range': [1, 3, 5, 7], 'optimal': 5},
    'stop_loss': 0.05,                # 5% stop loss before liquidation
    'rebalance_threshold': 0.03,      # 3% imbalance triggers rebalance
    'funding_interval_hours': 8,      # CEX standard (1h for DEX)
    'trading_fee': 0.0005,            # 5 bps per trade
}
```

### Backtesting Algorithm
```python
def funding_arbitrage_backtest(prices_long, prices_short, 
                               funding_long, funding_short,
                               fee=0.0005, stop_loss=-0.05, leverage=1):
    """
    Core funding rate arbitrage backtest
    - Long spot/futures on one exchange
    - Short perpetual on another
    - Collect funding differential
    """
    init_balance = 1.0
    bal_long = (init_balance - leverage * fee) / 2
    bal_short = (init_balance - leverage * fee) / 2
    
    entry_long = prices_long[0]
    entry_short = prices_short[0]
    
    for i in range(1, len(prices_long)):
        # PnL from price movement
        pnl_long = leverage * (prices_long[i] - entry_long) / entry_long
        pnl_short = -leverage * (prices_short[i] - entry_short) / entry_short
        
        if pnl_long >= stop_loss and pnl_short >= stop_loss:
            # Accumulate funding
            bal_long += leverage * funding_long[i]
            bal_short += leverage * funding_short[i]
        else:
            # Rebalance
            temp_long = bal_long + pnl_long - 2 * fee * leverage
            temp_short = bal_short + pnl_short - 2 * fee * leverage
            bal_long = bal_short = (temp_long + temp_short) / 2
            entry_long = prices_long[i]
            entry_short = prices_short[i]
    
    return (bal_long + bal_short) - init_balance
```

### Data Requirements
- **Primary**: Funding rate history (8h intervals)
- **Secondary**: Mark prices for both legs
- **Sources**: 
  - CEX: Binance, BitMEX APIs
  - DEX: Drift, ApolloX subgraphs
- **History**: 6+ months for meaningful backtest

### Maestro V2 Integration

```python
# strategies/funding_arbitrage.py
class FundingRateArbitrageStrategy(VectorBTStrategy):
    """
    Funding rate arbitrage: delta-neutral position collecting funding
    """
    
    @staticmethod
    def get_params():
        return {
            'leverage': {'type': 'int', 'low': 1, 'high': 7, 'default': 3},
            'stop_loss': {'type': 'float', 'low': 0.02, 'high': 0.10, 'default': 0.05},
            'min_funding_rate': {'type': 'float', 'low': 0.0001, 'high': 0.001, 'default': 0.0003},
        }
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        """
        Entry when funding rate exceeds threshold
        Exit when funding rate flips or hits stop loss
        """
        funding_rate = data['funding_rate']  # Needs custom datafeed
        min_rate = params['min_funding_rate']
        
        # Long spot + short perp when funding positive
        entries_long = funding_rate > min_rate
        entries_short = funding_rate < -min_rate
        
        # Position size based on leverage
        size = params['leverage'] / 2  # Split between legs
        
        return SignalOutput(
            entries_long=entries_long.astype(int),
            entries_short=entries_short.astype(int),
            exits_long=(funding_rate <= 0).astype(int),
            exits_short=(funding_rate >= 0).astype(int),
            size=size
        )
```

### Custom Data Feed Required
```python
# datasource/funding_rate_downloader.py
class FundingRateDownloader:
    """Download funding rate history from exchanges"""
    
    ENDPOINTS = {
        'binance': 'https://fapi.binance.com/fapi/v1/fundingRate',
        'bitmex': 'https://www.bitmex.com/api/v1/funding',
        'drift': 'https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com/...'
    }
    
    async def fetch_funding_history(self, symbol: str, exchange: str,
                                    start_date: datetime, end_date: datetime):
        # Implementation...
        pass
```

### Implementation Timeline: 1-2 weeks
- Week 1: Data infrastructure (funding rate feeds) + core logic
- Week 2: Multi-exchange support + optimization

---

## Paper 3: Wasserstein Market Regime Clustering (Horvath et al., 2021)

### Core Strategy
Unsupervised clustering of market regimes using Wasserstein distance on return distributions. **This is a META-STRATEGY** that enhances other strategies by conditioning on regime.

### Key Findings
- **Method**: WK-means algorithm on distribution space
- **Validation**: MMD scores show superior clustering vs HMM and moment-based methods
- **Application**: Regime-conditional strategy selection, risk management

### Algorithm Overview
```python
def wasserstein_kmeans(return_segments, k=4, p=2, max_iter=100):
    """
    k-means on the space of probability distributions
    using p-Wasserstein distance
    
    Args:
        return_segments: List of return series (each is a "distribution")
        k: Number of regimes
        p: Wasserstein order (typically 2)
    
    Returns:
        labels: Regime assignment for each segment
        centroids: Wasserstein barycenters of each cluster
    """
    from scipy.stats import wasserstein_distance
    
    # Initialize centroids randomly
    centroids = [return_segments[i] for i in np.random.choice(len(return_segments), k, replace=False)]
    
    for iteration in range(max_iter):
        # Assign labels based on Wasserstein distance
        labels = []
        for segment in return_segments:
            distances = [wasserstein_distance(segment, c) for c in centroids]
            labels.append(np.argmin(distances))
        
        # Update centroids (Wasserstein barycenter)
        new_centroids = []
        for cluster_id in range(k):
            cluster_segments = [s for s, l in zip(return_segments, labels) if l == cluster_id]
            if cluster_segments:
                # Simple average for 1D barycenter (exact for p=2)
                barycenter = np.mean([np.sort(s) for s in cluster_segments], axis=0)
                new_centroids.append(barycenter)
            else:
                new_centroids.append(centroids[cluster_id])
        
        # Check convergence
        if centroids == new_centroids:
            break
        centroids = new_centroids
    
    return labels, centroids
```

### Key Parameters
```python
REGIME_CONFIG = {
    'segment_length': 20,        # Days per segment (h1 in paper)
    'overlap': 5,                # Overlap between segments (h2)
    'n_regimes': 4,              # Typical: Low-vol, High-vol, Bull, Bear
    'wasserstein_p': 2,          # L2 Wasserstein
    'min_samples': 50,           # Min segments for reliable clustering
}
```

### Data Requirements
- **Frequency**: Daily returns (minimum)
- **History**: 2+ years for regime identification
- **Computation**: O(N log N) per distance calculation

### Maestro V2 Integration

```python
# analytics/regime_detector.py
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import numpy as np

class WassersteinRegimeDetector:
    """
    Market regime detection using Wasserstein k-means
    Use as a filter for other strategies
    """
    
    def __init__(self, n_regimes=4, segment_length=20, overlap=5):
        self.n_regimes = n_regimes
        self.segment_length = segment_length
        self.overlap = overlap
        self.centroids = None
        self.regime_labels = None
    
    def fit(self, returns: pd.Series):
        """Fit regime model on historical returns"""
        segments = self._create_segments(returns)
        self.regime_labels, self.centroids = self._wasserstein_kmeans(segments)
        self._label_regimes()  # Assign semantic labels (bull, bear, etc.)
        return self
    
    def predict(self, recent_returns: np.ndarray) -> int:
        """Classify current regime based on recent returns"""
        if self.centroids is None:
            raise ValueError("Model not fitted")
        
        distances = [wasserstein_distance(recent_returns, c) for c in self.centroids]
        return np.argmin(distances)
    
    def get_regime_stats(self) -> dict:
        """Return statistics for each regime"""
        return {
            regime: {
                'mean_return': np.mean(self.centroids[regime]),
                'volatility': np.std(self.centroids[regime]),
                'skewness': scipy.stats.skew(self.centroids[regime])
            }
            for regime in range(self.n_regimes)
        }
    
    def _create_segments(self, returns):
        """Lift return series to segments"""
        segments = []
        step = self.segment_length - self.overlap
        for i in range(0, len(returns) - self.segment_length, step):
            segments.append(returns[i:i + self.segment_length].values)
        return segments
    
    def _wasserstein_kmeans(self, segments, max_iter=100):
        # Implementation from above
        pass
```

### Integration with Existing Strategies
```python
# engine_v2/regime_conditional_strategy.py
class RegimeConditionalStrategy(VectorBTStrategy):
    """
    Meta-strategy that switches between sub-strategies based on regime
    """
    
    def __init__(self, regime_detector: WassersteinRegimeDetector,
                 regime_strategies: dict):
        self.regime_detector = regime_detector
        self.regime_strategies = regime_strategies  # {regime_id: strategy}
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        # Detect current regime
        recent_returns = data['close'].pct_change().iloc[-20:].values
        current_regime = self.regime_detector.predict(recent_returns)
        
        # Delegate to regime-specific strategy
        strategy = self.regime_strategies.get(current_regime)
        if strategy:
            return strategy.generate_signals(data, params)
        else:
            return self._neutral_signal(data)
```

### Implementation Timeline: 2-3 weeks
- Week 1: Core WK-means algorithm + validation
- Week 2: Integration with VectorBT engine
- Week 3: Regime-conditional strategy wrapper + testing

---

## Paper 4: Copula-Based Cointegrated Pairs Trading (Tadi & Witzany, 2025)

### Core Strategy
Statistical arbitrage using copula-modeled dependencies for cointegrated cryptocurrency pairs.

### Key Findings
- **Improvement**: Outperforms distance and cointegration-only approaches
- **Method**: Reference asset-based copula with stationary spread
- **Innovation**: Uses BTC as reference asset for spread calculation

### Mathematical Framework

#### Cointegration Testing
```python
def test_cointegration(price1, price2, significance=0.05):
    """
    Test for cointegration using Engle-Granger and nonlinear (KSS) tests
    """
    from statsmodels.tsa.stattools import adfuller, coint
    
    # Linear cointegration (Engle-Granger)
    score, pvalue, _ = coint(price1, price2)
    linear_coint = pvalue < significance
    
    # Calculate spread
    beta = np.cov(price1, price2)[0, 1] / np.var(price2)
    spread = price1 - beta * price2
    
    # Nonlinear unit root (KSS test approximation)
    # ΔS_t = δ(S_{t-1})³ + ε_t
    spread_cubed = spread.shift(1) ** 3
    delta_spread = spread.diff()
    # Regress and check significance of δ
    
    return {
        'linear_cointegrated': linear_coint,
        'beta': beta,
        'spread': spread,
        'half_life': calculate_half_life(spread)
    }

def calculate_half_life(spread):
    """Mean reversion half-life via AR(1)"""
    spread_lag = spread.shift(1)
    delta_spread = spread.diff()
    # y = a + b*y_{t-1} + e
    # half_life = -ln(2) / ln(b)
    b = np.cov(delta_spread[1:], spread_lag[1:])[0, 1] / np.var(spread_lag[1:])
    if b >= 1 or b <= 0:
        return np.inf
    return -np.log(2) / np.log(abs(b))
```

#### Copula Trading Signals
```python
def copula_trading_signal(spread, btc_price, copula_family='gaussian'):
    """
    Generate signals using conditional copula probabilities
    
    h_{1|2} = P(U1 <= u1 | U2 = u2) = ∂C(u1, u2) / ∂u2
    
    Trading rules:
    - If h_{1|2} < α and h_{2|1} > 1-α: Long asset 1, Short asset 2
    - If h_{1|2} > 1-α and h_{2|1} < α: Short asset 1, Long asset 2
    - If |h_{1|2} - 0.5| < α2 and |h_{2|1} - 0.5| < α2: Close positions
    """
    from scipy.stats import norm
    from copulas.bivariate import GaussianCopula, ClaytonCopula, FrankCopula
    
    # Fit copula
    u1 = norm.cdf(spread, spread.mean(), spread.std())
    u2 = norm.cdf(btc_price.pct_change(), 
                  btc_price.pct_change().mean(), 
                  btc_price.pct_change().std())
    
    if copula_family == 'gaussian':
        copula = GaussianCopula()
    elif copula_family == 'clayton':
        copula = ClaytonCopula()
    elif copula_family == 'frank':
        copula = FrankCopula()
    
    copula.fit(pd.DataFrame({'u1': u1, 'u2': u2}))
    
    # Conditional probabilities
    h1_given_2 = copula.partial_derivative(u1, u2, which=1)
    h2_given_1 = copula.partial_derivative(u1, u2, which=2)
    
    return h1_given_2, h2_given_1
```

### Key Parameters
```python
COPULA_PAIRS_CONFIG = {
    'formation_period': 180,          # Days for pair selection
    'trading_period': 30,             # Days before re-evaluation
    'entry_threshold': 0.05,          # α1: Entry when h < 0.05 or h > 0.95
    'exit_threshold': 0.4,            # α2: Exit when |h - 0.5| < 0.1
    'min_correlation': 0.7,           # Minimum Pearson correlation
    'copula_families': ['gaussian', 'student_t', 'clayton', 'gumbel'],
    'half_life_max': 30,              # Max mean reversion half-life (days)
}
```

### Data Requirements
- **Pairs**: 20 cryptocurrency futures (USDT-margined)
- **History**: 2+ years for cointegration testing
- **Frequency**: Hourly or 5-minute for intraday trading
- **Reference Asset**: BTC (for spread calculation)

### Maestro V2 Integration

```python
# strategies/copula_pairs.py
class CopulaPairsTradingStrategy(VectorBTStrategy):
    """
    Copula-based pairs trading for cointegrated crypto pairs
    """
    
    @staticmethod
    def get_params():
        return {
            'entry_threshold': {'type': 'float', 'low': 0.01, 'high': 0.15, 'default': 0.05},
            'exit_threshold': {'type': 'float', 'low': 0.3, 'high': 0.49, 'default': 0.4},
            'copula_family': {'type': 'categorical', 'choices': ['gaussian', 'student_t', 'clayton']},
            'lookback': {'type': 'int', 'low': 60, 'high': 365, 'default': 180},
        }
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        """
        Multi-asset strategy: needs paired data
        data should have columns: ['close_asset1', 'close_asset2', 'close_btc']
        """
        asset1 = data['close_asset1']
        asset2 = data['close_asset2']
        btc = data['close_btc']
        
        # Calculate spread with BTC reference
        beta = self._calculate_beta(asset1, asset2, btc)
        spread = asset1 - beta * asset2
        
        # Copula signals
        h1, h2 = copula_trading_signal(spread, btc, params['copula_family'])
        
        alpha1 = params['entry_threshold']
        alpha2 = 0.5 - params['exit_threshold']
        
        entries_long = (h1 < alpha1) & (h2 > 1 - alpha1)
        entries_short = (h1 > 1 - alpha1) & (h2 < alpha1)
        exits = (np.abs(h1 - 0.5) < alpha2) & (np.abs(h2 - 0.5) < alpha2)
        
        return SignalOutput(
            entries_long=entries_long.astype(int),
            entries_short=entries_short.astype(int),
            exits_long=exits.astype(int),
            exits_short=exits.astype(int),
        )
```

### Implementation Timeline: 3-4 weeks
- Week 1: Cointegration testing framework
- Week 2: Copula fitting and signal generation
- Week 3: Multi-pair portfolio construction
- Week 4: Integration and optimization

---

## Paper 5: Multi-Level Deep Q-Networks (Otabek & Choi, 2024)

### Core Strategy
Reinforcement learning trading using multi-level DQN that integrates:
1. **Trade-DQN**: Historical price signals
2. **Predictive-DQN**: Twitter sentiment → price prediction
3. **Main-DQN**: Combines both for final decisions

### Key Findings
- **ROI**: +29.93% annualized
- **Sharpe Ratio**: 2.74
- **Test Period**: Oct 2014 - Mar 2019 (hourly data)
- **Key Innovation**: Novel reward function balancing profit, risk, and trading activity

### Architecture
```
┌─────────────────┐     ┌─────────────────┐
│   Trade-DQN     │     │ Predictive-DQN  │
│ (Price → Action)│     │(Price+Sentiment │
│   Buy/Sell/Hold │     │  → % Change)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │   x1 ∈ {-1,0,1}      │  x2 ∈ [-100,100]
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Main-DQN   │
              │  (x1, x2)   │
              │  → Trade    │
              └─────────────┘
```

### Reward Function
```python
def mdqn_reward(action, pnl, balance, threshold_risk=0.3, 
                threshold_trades=20, consecutive_holds=0):
    """
    Novel reward function balancing:
    1. Profit (PnL from trades)
    2. Risk (penalize when balance drops below threshold)
    3. Active trading (penalize excessive holding)
    
    r_t = PnL_k      if action=sell and m <= ω
        = -1         if m > ω or (action=buy and balance < threshold)
        = 0          otherwise
    """
    if consecutive_holds > threshold_trades or \
       (action == 'buy' and balance < threshold_risk):
        return -1.0
    
    if action == 'sell':
        return pnl
    
    return 0.0
```

### Network Architecture
```python
# Main-DQN architecture
MAIN_DQN_CONFIG = {
    'input_dim': 2,           # [trade_recommendation, price_prediction]
    'hidden_layers': [64, 64, 64],
    'output_dim': 3,          # Buy, Sell, Hold
    'activation': 'relu',
    'output_activation': 'linear',
    'learning_rate': 0.001,
    'discount_factor': 0.95,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 64,
    'memory_size': 10000,
    'target_update_freq': 400,
}
```

### Data Requirements
- **Price Data**: Hourly BTC OHLCV (5+ years)
- **Sentiment Data**: Twitter/X API (7M+ tweets in paper)
- **Processing**: VADER sentiment analysis
- **Features**: [price, sentiment_score]

### Maestro V2 Integration

```python
# strategies/mdqn_strategy.py
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MDQNStrategy(VectorBTStrategy):
    """
    Multi-level DQN trading strategy
    Requires pre-trained DQN models or training during optimization
    """
    
    def __init__(self, trade_dqn_path=None, pred_dqn_path=None):
        self.trade_dqn = self._load_model(trade_dqn_path)
        self.pred_dqn = self._load_model(pred_dqn_path)
        self.main_dqn = DQN(input_dim=2, hidden_dims=[64, 64, 64], output_dim=3)
        self.memory = deque(maxlen=10000)
    
    @staticmethod
    def get_params():
        return {
            'risk_threshold': {'type': 'float', 'low': 0.1, 'high': 0.5, 'default': 0.3},
            'max_trades_per_day': {'type': 'int', 'low': 4, 'high': 24, 'default': 16},
            'transaction_fee': {'type': 'float', 'low': 0.001, 'high': 0.02, 'default': 0.015},
        }
    
    def train_episode(self, data: pd.DataFrame, params: dict):
        """Train Main-DQN on historical data"""
        # Get preprocessing outputs
        trade_signals = self._get_trade_signals(data)  # From Trade-DQN
        price_predictions = self._get_price_predictions(data)  # From Pred-DQN
        
        # Combine as state
        states = np.column_stack([trade_signals, price_predictions])
        
        # Run through environment
        for t in range(len(states) - 1):
            state = torch.FloatTensor(states[t])
            action = self._select_action(state)
            
            # Execute action, get reward
            reward = self._calculate_reward(action, data.iloc[t:t+2], params)
            next_state = torch.FloatTensor(states[t + 1])
            
            # Store experience
            self.memory.append((state, action, reward, next_state))
            
            # Train
            if len(self.memory) >= 64:
                self._replay(64)
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        """Generate signals using trained M-DQN"""
        trade_signal = self._get_trade_signals(data)
        price_pred = self._get_price_predictions(data)
        
        state = torch.FloatTensor([trade_signal[-1], price_pred[-1]])
        with torch.no_grad():
            q_values = self.main_dqn(state)
            action = q_values.argmax().item()
        
        entries_long = pd.Series(0, index=data.index)
        entries_short = pd.Series(0, index=data.index)
        exits = pd.Series(0, index=data.index)
        
        if action == 0:  # Buy
            entries_long.iloc[-1] = 1
        elif action == 1:  # Sell
            entries_short.iloc[-1] = 1
        # action == 2 is Hold
        
        return SignalOutput(
            entries_long=entries_long,
            entries_short=entries_short,
            exits_long=exits,
            exits_short=exits,
        )
```

### Implementation Timeline: 6-8 weeks
- Weeks 1-2: Data pipeline (Twitter sentiment, preprocessing)
- Weeks 3-4: Trade-DQN and Predictive-DQN implementation
- Weeks 5-6: Main-DQN integration and training
- Weeks 7-8: VectorBT wrapper and optimization

### Dependencies
- PyTorch or TensorFlow
- VADER sentiment analyzer
- Twitter/X API access (or historical sentiment data)

---

## Paper 6: Attention Factors for Statistical Arbitrage (Epstein et al., 2025)

### Core Strategy
End-to-end deep learning for statistical arbitrage using attention-based factor models with transaction cost optimization.

### Key Findings
- **Gross Sharpe**: 4.0+ (out-of-sample, 24 years)
- **Net Sharpe**: 2.3 (after transaction costs) - 84% improvement over prior SOTA
- **Annual Return**: 16% uncorrelated to market
- **Key Innovation**: Joint optimization of factors and trading policy

### Architecture Overview
```
Firm Characteristics X_t ∈ R^(N×M)
         │
         ▼
    ┌────────────┐
    │ Embed W^K  │  X̃_t = X_t · W^K
    └────────────┘
         │
         ▼
    ┌────────────┐
    │  Attention │  ω^F = softmax(Q · X̃^T / √d)
    │  (K query  │
    │  vectors)  │
    └────────────┘
         │
         ▼
    Factors F_t = ω^F · R_t
    Residuals ε_t = R_t - β^T · F_t
         │
         ▼
    ┌────────────┐
    │  LongConv  │  ω^port = LongConv(ε_{t-s:t-1})
    │  (32 conv) │
    └────────────┘
         │
         ▼
    Portfolio R^port = ε^T · ω^port
```

### Key Parameters
```python
ATTENTION_FACTORS_CONFIG = {
    # Model architecture
    'n_factors': 30,              # K: number of attention factors
    'embed_dim': 32,              # d: embedding dimension
    'n_characteristics': 39,       # M: firm characteristics
    'sequence_length': 30,         # s: lookback for LongConv
    
    # LongConv model
    'longconv_hidden_dim': 32,
    'longconv_layers': 1,
    'n_convolutions': 32,
    
    # Optimization
    'lambda_var': 100,            # Weight on explained variance objective
    'ridge_penalty': 0.01,        # λ_ridge for stability
    
    # Transaction costs
    'transaction_cost': 0.0005,   # 5 bps per trade
    'shorting_cost': 0.0001,      # 1 bp for shorts
    
    # Training
    'train_window_years': 8,
    'retrain_frequency': 'yearly',
}
```

### Firm Characteristics (39 features)
```python
CHARACTERISTICS = {
    'past_returns': ['r2_1', 'r12_2', 'r12_7', 'r36_13', 'ST_Rev', 'Ret_D1', 'Ret_W1', 'STD_W1'],
    'value': ['A2ME', 'BEME', 'C', 'CF', 'CF2P', 'Q', 'Lev', 'E2P'],
    'profitability': ['PROF', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'D2A'],
    'investment': ['Investment', 'NOA', 'DPI2A'],
    'intangibles': ['OA', 'OL', 'PCM'],
    'trading_frictions': ['AT', 'LME', 'LTurnover', 'Rel2High', 'Resid_Var', 'Spread', 'SUV', 'Variance', 'Vol', 'Beta'],
}
```

### Core Implementation

```python
# strategies/attention_factors.py
import torch
import torch.nn as nn

class AttentionFactorModel(nn.Module):
    """
    Attention-based factor model for statistical arbitrage
    """
    
    def __init__(self, n_assets, n_characteristics, n_factors, embed_dim, seq_length):
        super().__init__()
        self.n_factors = n_factors
        self.embed_dim = embed_dim
        
        # Characteristic embedding
        self.W_K = nn.Linear(n_characteristics, embed_dim, bias=False)
        
        # Query vectors (one per factor)
        self.Q = nn.Parameter(torch.randn(n_factors, embed_dim))
        
        # LongConv for time-series patterns
        self.longconv = nn.Conv1d(n_assets, n_assets, kernel_size=seq_length, 
                                   groups=n_assets)  # Depthwise conv
        
        # Output projection
        self.ridge_penalty = 0.01
    
    def forward(self, characteristics, returns, past_residuals):
        """
        Args:
            characteristics: (batch, n_assets, n_characteristics)
            returns: (batch, n_assets)
            past_residuals: (batch, n_assets, seq_length)
        
        Returns:
            portfolio_return: (batch,)
            explained_var: scalar
        """
        batch_size, n_assets, _ = characteristics.shape
        
        # Embed characteristics
        X_embed = self.W_K(characteristics)  # (batch, n_assets, embed_dim)
        
        # Attention weights (factor loadings)
        scores = torch.matmul(self.Q, X_embed.transpose(-1, -2))  # (batch, n_factors, n_assets)
        omega_F = torch.softmax(scores / np.sqrt(self.embed_dim), dim=-1)
        
        # Factors
        factors = torch.matmul(omega_F, returns.unsqueeze(-1)).squeeze(-1)  # (batch, n_factors)
        
        # Factor loadings via ridge regression
        omega_FT = omega_F.transpose(-1, -2)  # (batch, n_assets, n_factors)
        cov = torch.matmul(omega_F, omega_FT)
        cov_reg = cov + self.ridge_penalty * torch.eye(self.n_factors).to(cov.device)
        beta = torch.linalg.solve(cov_reg, omega_F)  # (batch, n_factors, n_assets)
        
        # Residuals
        residuals = returns - torch.matmul(beta.transpose(-1, -2), factors.unsqueeze(-1)).squeeze(-1)
        
        # Portfolio weights from LongConv
        omega_port = self.longconv(past_residuals).squeeze(-1)  # (batch, n_assets)
        omega_port = omega_port / omega_port.abs().sum(dim=-1, keepdim=True)  # Normalize
        
        # Portfolio return
        portfolio_return = (residuals * omega_port).sum(dim=-1)
        
        # Explained variance
        total_var = returns.var(dim=-1)
        resid_var = residuals.var(dim=-1)
        explained_var = 1 - resid_var / total_var
        
        return portfolio_return, explained_var
    
    def calculate_transaction_costs(self, omega_t, omega_tm1, tc=0.0005, sc=0.0001):
        """Transaction and shorting costs"""
        turnover = torch.abs(omega_t - omega_tm1).sum(dim=-1)
        short_penalty = torch.clamp(-omega_t, min=0).sum(dim=-1)
        return tc * turnover + sc * short_penalty


class AttentionFactorsStrategy(VectorBTStrategy):
    """
    VectorBT wrapper for Attention Factors model
    Note: This requires significant adaptation for crypto markets
    """
    
    def __init__(self, model_path=None, retrain_frequency='monthly'):
        self.model = None
        self.retrain_frequency = retrain_frequency
        if model_path:
            self.model = torch.load(model_path)
    
    @staticmethod
    def get_params():
        return {
            'n_factors': {'type': 'int', 'low': 5, 'high': 50, 'default': 30},
            'embed_dim': {'type': 'int', 'low': 16, 'high': 64, 'default': 32},
            'seq_length': {'type': 'int', 'low': 10, 'high': 60, 'default': 30},
            'lambda_var': {'type': 'float', 'low': 10, 'high': 500, 'default': 100},
        }
    
    def generate_signals(self, data: pd.DataFrame, params: dict) -> SignalOutput:
        """
        For crypto: Adapt to use available characteristics
        - Momentum features
        - Volume features
        - Volatility features
        - On-chain metrics (if available)
        """
        # Build characteristic matrix from available data
        chars = self._build_characteristics(data)
        returns = data['close'].pct_change()
        
        # Get model predictions
        with torch.no_grad():
            weights = self.model.predict(chars, returns)
        
        # Convert weights to signals
        entries_long = weights > 0.1
        entries_short = weights < -0.1
        
        return SignalOutput(
            entries_long=entries_long.astype(int),
            entries_short=entries_short.astype(int),
            exits_long=(weights <= 0).astype(int),
            exits_short=(weights >= 0).astype(int),
            size=weights.abs()
        )
```

### Implementation Timeline: 8-12 weeks
- Weeks 1-2: Characteristic engineering for crypto (adapt from equities)
- Weeks 3-4: Attention factor model implementation
- Weeks 5-6: LongConv time-series model
- Weeks 7-8: Joint optimization with transaction costs
- Weeks 9-10: Integration with VectorBT
- Weeks 11-12: Training pipeline and validation

### Key Challenges for Crypto Adaptation
1. **Fewer assets**: Paper uses 500 stocks; crypto has ~50-100 liquid pairs
2. **Characteristic availability**: Need to adapt 39 equity characteristics
3. **Shorter history**: Less data for training deep models
4. **Higher volatility**: May need different hyperparameters

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
**Priority: High-value, low-complexity strategies**

| Week | Deliverable | Strategy |
|------|-------------|----------|
| 1 | TSMOM core implementation | Time Series Momentum |
| 2 | TSMOM optimization + tests | Time Series Momentum |
| 3 | Funding rate data pipeline | Funding Rate Arbitrage |
| 4 | Funding arbitrage strategy | Funding Rate Arbitrage |

**Milestone**: Two production-ready strategies

### Phase 2: Market Intelligence (Weeks 5-8)
**Priority: Regime detection as strategy filter**

| Week | Deliverable | Strategy |
|------|-------------|----------|
| 5 | WK-means algorithm | Regime Clustering |
| 6 | Regime integration | Regime Clustering |
| 7 | Copula framework | Pairs Trading |
| 8 | Pairs selection + signals | Pairs Trading |

**Milestone**: Regime-conditional strategy switching

### Phase 3: Advanced ML (Weeks 9-20)
**Priority: Research-grade implementations**

| Week | Deliverable | Strategy |
|------|-------------|----------|
| 9-10 | Sentiment pipeline | M-DQN |
| 11-14 | M-DQN training | M-DQN |
| 15-18 | Attention factors | Attention StatArb |
| 19-20 | Integration + testing | All |

**Milestone**: Full ML strategy suite

---

## Dependencies & Integration Points

### Strategy Dependencies
```
                    ┌─────────────────┐
                    │ Regime Detector │
                    │  (Wasserstein)  │
                    └────────┬────────┘
                             │ filters
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    TSMOM      │   │ Copula Pairs  │   │ Funding Arb   │
│ (Trend regime)│   │ (Mean-revert) │   │ (Low vol reg) │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Multi-Strategy │
                    │   Allocator     │
                    └─────────────────┘
```

### Data Dependencies
```python
DATA_REQUIREMENTS = {
    'tsmom': {
        'tables': ['ohlcv'],
        'frequency': 'daily',
        'history': '2 years',
    },
    'funding_arbitrage': {
        'tables': ['ohlcv', 'funding_rates'],
        'frequency': '8h',
        'history': '1 year',
        'custom_feeds': ['binance_funding', 'bitmex_funding'],
    },
    'regime_clustering': {
        'tables': ['ohlcv'],
        'frequency': 'daily',
        'history': '3 years',
    },
    'copula_pairs': {
        'tables': ['ohlcv'],
        'frequency': 'hourly',
        'history': '2 years',
        'multi_asset': True,
    },
    'mdqn': {
        'tables': ['ohlcv', 'sentiment'],
        'frequency': 'hourly',
        'history': '5 years',
        'external': ['twitter_sentiment'],
    },
    'attention_factors': {
        'tables': ['ohlcv', 'characteristics'],
        'frequency': 'daily',
        'history': '8 years',
        'multi_asset': True,
    },
}
```

---

## Risk Considerations

### Strategy-Specific Risks

| Strategy | Primary Risk | Mitigation |
|----------|--------------|------------|
| TSMOM | Trend reversal | Stop-loss, regime filter |
| Funding Arb | Liquidation | Conservative leverage, stop-loss |
| Pairs Trading | Pair breakdown | Half-life filter, cointegration monitoring |
| M-DQN | Overfitting | Walk-forward validation, regularization |
| Attention Factors | Model decay | Regular retraining, drift detection |

### Portfolio-Level Risks
- **Correlation risk**: Strategies may become correlated in crisis
- **Capacity constraints**: Funding arb and pairs have limited capacity
- **Execution risk**: High-frequency strategies need low latency

---

## Success Metrics

### Per-Strategy KPIs
- **Sharpe Ratio** > 1.0 (out-of-sample)
- **Max Drawdown** < 15%
- **Win Rate** > 50%
- **Profit Factor** > 1.5

### Platform-Level KPIs
- **Strategy diversity**: Correlation < 0.3 between strategies
- **Regime adaptability**: Positive returns in 3+ regime types
- **Implementation velocity**: New strategy in < 2 weeks

---

## Appendix: Academic Paper References

1. **Time Series Momentum** - Moskowitz, Ooi, Pedersen (2012) - AQR/Chicago Booth
2. **Funding Rate Arbitrage** - Werapun et al. (2025) - Blockchain: Research and Applications
3. **Copula Pairs Trading** - Tadi & Witzany (2025) - Financial Innovation
4. **Multi-Level DQN** - Otabek & Choi (2024) - Scientific Reports
5. **Attention Factors** - Epstein et al. (2025) - ICAIF '25
6. **Wasserstein Regime Clustering** - Horvath, Issa, Muguruza (2021) - SSRN

---

*Document maintained by Maestro Research Team*
*Last updated: 2025-02-06*
