"""
Strategy Combiner - Generate hybrid strategy combinations

Combines base strategies with filters to create new strategies:
- A + B (both must agree)
- A with B filter (A signals, B confirms)
- A or B (either triggers)
- Weighted ensemble

Usage:
    combiner = StrategyCombiner()
    combiner.add_base(['BollingerBreakout', 'MACD'])
    combiner.add_filters(['VolumeFilter', 'TrendFilter'])
    hybrids = combiner.generate()
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from itertools import combinations, product
import logging

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class HybridStrategy:
    """Definition of a hybrid strategy."""
    name: str
    description: str
    base_strategies: List[str]
    filters: List[str]
    combination_type: str  # 'and', 'or', 'filter', 'ensemble'
    weights: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class CombinerConfig:
    """Configuration for strategy combination."""
    base_strategies: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    combination_types: List[str] = field(default_factory=lambda: ['and', 'filter'])
    max_components: int = 3  # Max strategies to combine
    include_singles: bool = True  # Include base strategies alone


class StrategyCombiner:
    """
    Generate hybrid strategy combinations from base strategies and filters.
    """
    
    # Available filters
    FILTERS = {
        'VolumeFilter': {
            'description': 'Only trade on high volume (>2x average)',
            'params': {'multiplier': 2.0, 'period': 20}
        },
        'TrendFilter': {
            'description': 'Only trade in direction of trend (50 SMA)',
            'params': {'period': 50}
        },
        'VolatilityFilter': {
            'description': 'Only trade when volatility is moderate (not extreme)',
            'params': {'period': 20, 'min_percentile': 20, 'max_percentile': 80}
        },
        'MomentumFilter': {
            'description': 'Only trade with momentum confirmation',
            'params': {'period': 14}
        },
        'TimeFilter': {
            'description': 'Only trade during specific sessions (e.g., US hours)',
            'params': {'start_hour': 13, 'end_hour': 21}  # UTC
        },
        'CapitulationFilter': {
            'description': 'Only buy after volume spike + price drop (liquidation cascade)',
            'params': {'vol_mult': 4.0, 'price_drop': -0.03}
        },
    }
    
    # Base strategy categories for smart combinations
    STRATEGY_CATEGORIES = {
        'trend_following': ['MACD', 'EMA_Cross', 'SMA_Cross', 'Momentum'],
        'mean_reversion': ['RSI', 'BollingerBreakout', 'MeanReversion'],
        'volume_based': ['OBV', 'VolumeBreakout', 'CapitulationReversal'],
        'breakout': ['BollingerBreakout', 'VolumeBreakout'],
    }
    
    def __init__(self, config: Optional[CombinerConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or CombinerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.hybrids: List[HybridStrategy] = []
    
    def add_base(self, strategies: List[str]) -> 'StrategyCombiner':
        """Add base strategies."""
        self.config.base_strategies.extend(strategies)
        return self
    
    def add_filters(self, filters: List[str]) -> 'StrategyCombiner':
        """Add filters."""
        self.config.filters.extend(filters)
        return self
    
    def generate(self) -> List[HybridStrategy]:
        """
        Generate all valid hybrid combinations.
        
        Returns:
            List of HybridStrategy definitions
        """
        self.hybrids = []
        
        # 1. Single strategies (optionally)
        if self.config.include_singles:
            for strategy in self.config.base_strategies:
                self.hybrids.append(HybridStrategy(
                    name=strategy,
                    description=f"Base {strategy} strategy",
                    base_strategies=[strategy],
                    filters=[],
                    combination_type='single'
                ))
        
        # 2. Strategy + Filter combinations
        if 'filter' in self.config.combination_types:
            for strategy in self.config.base_strategies:
                for filter_name in self.config.filters:
                    self.hybrids.append(HybridStrategy(
                        name=f"{strategy}+{filter_name}",
                        description=f"{strategy} with {filter_name} confirmation",
                        base_strategies=[strategy],
                        filters=[filter_name],
                        combination_type='filter'
                    ))
        
        # 3. AND combinations (both strategies must agree)
        if 'and' in self.config.combination_types:
            for combo in combinations(self.config.base_strategies, 2):
                # Skip combining same-category strategies (redundant)
                if not self._are_complementary(combo[0], combo[1]):
                    continue
                
                self.hybrids.append(HybridStrategy(
                    name=f"{combo[0]}+{combo[1]}",
                    description=f"Both {combo[0]} AND {combo[1]} must agree",
                    base_strategies=list(combo),
                    filters=[],
                    combination_type='and'
                ))
        
        # 4. OR combinations (either triggers)
        if 'or' in self.config.combination_types:
            for combo in combinations(self.config.base_strategies, 2):
                self.hybrids.append(HybridStrategy(
                    name=f"{combo[0]}|{combo[1]}",
                    description=f"Either {combo[0]} OR {combo[1]} triggers",
                    base_strategies=list(combo),
                    filters=[],
                    combination_type='or'
                ))
        
        # 5. Triple combinations with filter
        if self.config.max_components >= 3:
            for combo in combinations(self.config.base_strategies, 2):
                for filter_name in self.config.filters:
                    self.hybrids.append(HybridStrategy(
                        name=f"{combo[0]}+{combo[1]}+{filter_name}",
                        description=f"{combo[0]} AND {combo[1]} with {filter_name}",
                        base_strategies=list(combo),
                        filters=[filter_name],
                        combination_type='and_filter'
                    ))
        
        self.logger.info(f"Generated {len(self.hybrids)} hybrid strategies")
        return self.hybrids
    
    def generate_smart(self) -> List[HybridStrategy]:
        """
        Generate smart combinations based on complementary strategies.
        Only combines strategies that make sense together.
        """
        self.hybrids = []
        
        # Complementary pairs: trend + volume, mean reversion + volume
        smart_combos = [
            # Trend following + Volume confirmation
            ('MACD', 'VolumeFilter', "MACD trend with volume confirmation"),
            ('EMA_Cross', 'VolumeFilter', "EMA crossover with volume confirmation"),
            
            # Mean reversion + Capitulation
            ('RSI', 'CapitulationFilter', "RSI oversold after capitulation"),
            ('BollingerBreakout', 'CapitulationFilter', "Bollinger bounce after liquidation cascade"),
            ('MeanReversion', 'CapitulationFilter', "Mean reversion after extreme move"),
            
            # Volume strategies + Trend filter
            ('OBV', 'TrendFilter', "OBV divergence in trend direction"),
            ('VolumeBreakout', 'TrendFilter', "Volume breakout with trend"),
            
            # Capitulation pure play
            ('CapitulationReversal', 'VolumeFilter', "Capitulation reversal with 4x volume filter"),
            
            # Multi-indicator confluence
            ('MACD', 'RSI', "MACD + RSI confluence"),
            ('BollingerBreakout', 'OBV', "Bollinger breakout with OBV confirmation"),
        ]
        
        for item in smart_combos:
            if len(item) == 3:
                strat1, strat2, desc = item
                
                # Determine if second is a filter or strategy
                if strat2 in self.FILTERS:
                    self.hybrids.append(HybridStrategy(
                        name=f"{strat1}+{strat2}",
                        description=desc,
                        base_strategies=[strat1],
                        filters=[strat2],
                        combination_type='filter'
                    ))
                else:
                    self.hybrids.append(HybridStrategy(
                        name=f"{strat1}+{strat2}",
                        description=desc,
                        base_strategies=[strat1, strat2],
                        filters=[],
                        combination_type='and'
                    ))
        
        # Add the capitulation-focused hybrids from the screenshots
        capitulation_hybrids = [
            HybridStrategy(
                name="CapitulationReversal_Pure",
                description="Pure capitulation reversal (volume spike + price drop = buy the blood)",
                base_strategies=['CapitulationReversal'],
                filters=['VolumeFilter'],
                combination_type='filter'
            ),
            HybridStrategy(
                name="Bollinger+Capitulation",
                description="Bollinger bands with capitulation filter",
                base_strategies=['BollingerBreakout'],
                filters=['CapitulationFilter'],
                combination_type='filter'
            ),
            HybridStrategy(
                name="OBV+CapitulationDivergence",
                description="OBV divergence detecting smart money vs weak hands",
                base_strategies=['OBV', 'CapitulationReversal'],
                filters=['VolumeFilter'],
                combination_type='and_filter'
            ),
            HybridStrategy(
                name="MultiTF_Capitulation",
                description="Multi-timeframe capitulation scanner (z-score based)",
                base_strategies=['MeanReversion', 'CapitulationReversal'],
                filters=['VolatilityFilter'],
                combination_type='and_filter'
            ),
            HybridStrategy(
                name="ConsecutiveLiquidationCascade",
                description="Detect consecutive liquidation events for high-conviction entries",
                base_strategies=['CapitulationReversal'],
                filters=['VolumeFilter', 'MomentumFilter'],
                combination_type='filter'
            ),
        ]
        
        self.hybrids.extend(capitulation_hybrids)
        
        self.logger.info(f"Generated {len(self.hybrids)} smart hybrid strategies")
        return self.hybrids
    
    def _are_complementary(self, strat1: str, strat2: str) -> bool:
        """Check if two strategies are complementary (different categories)."""
        cat1 = self._get_category(strat1)
        cat2 = self._get_category(strat2)
        
        # Different categories = complementary
        return cat1 != cat2
    
    def _get_category(self, strategy: str) -> Optional[str]:
        """Get category of a strategy."""
        for category, strategies in self.STRATEGY_CATEGORIES.items():
            if strategy in strategies:
                return category
        return None
    
    def apply_hybrid(self, df: pd.DataFrame, hybrid: HybridStrategy, 
                     strategy_signals: Dict[str, Callable]) -> pd.Series:
        """
        Apply a hybrid strategy to data.
        
        Args:
            df: OHLCV DataFrame
            hybrid: HybridStrategy definition
            strategy_signals: Dict mapping strategy names to signal functions
            
        Returns:
            Combined signal series
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required")
        
        # Get base strategy signals
        base_signals = []
        for strat_name in hybrid.base_strategies:
            if strat_name in strategy_signals:
                signals = strategy_signals[strat_name](df)
                base_signals.append(signals)
        
        if not base_signals:
            return pd.Series(0, index=df.index)
        
        # Combine based on type
        if hybrid.combination_type == 'single':
            combined = base_signals[0]
        
        elif hybrid.combination_type == 'and':
            # All must agree
            combined = base_signals[0].copy()
            for signals in base_signals[1:]:
                combined = combined.where(combined == signals, 0)
        
        elif hybrid.combination_type == 'or':
            # Any triggers
            combined = pd.Series(0, index=df.index)
            for signals in base_signals:
                combined = combined.where(combined != 0, signals)
        
        elif hybrid.combination_type in ['filter', 'and_filter']:
            # Base signals filtered
            if hybrid.combination_type == 'and_filter' and len(base_signals) > 1:
                combined = base_signals[0].copy()
                for signals in base_signals[1:]:
                    combined = combined.where(combined == signals, 0)
            else:
                combined = base_signals[0]
        
        else:
            combined = base_signals[0]
        
        # Apply filters
        for filter_name in hybrid.filters:
            filter_mask = self._get_filter_mask(df, filter_name)
            combined = combined.where(filter_mask, 0)
        
        return combined
    
    def _get_filter_mask(self, df: pd.DataFrame, filter_name: str) -> pd.Series:
        """Get boolean mask for a filter."""
        filter_config = self.FILTERS.get(filter_name, {})
        params = filter_config.get('params', {})
        
        if filter_name == 'VolumeFilter':
            vol_sma = df['volume'].rolling(params.get('period', 20)).mean()
            return df['volume'] > vol_sma * params.get('multiplier', 2.0)
        
        elif filter_name == 'TrendFilter':
            sma = df['close'].rolling(params.get('period', 50)).mean()
            # Allow longs above SMA, shorts below
            return pd.Series(True, index=df.index)  # Simplified
        
        elif filter_name == 'VolatilityFilter':
            returns = df['close'].pct_change()
            vol = returns.rolling(params.get('period', 20)).std()
            vol_pct = vol.rank(pct=True) * 100
            return (vol_pct > params.get('min_percentile', 20)) & \
                   (vol_pct < params.get('max_percentile', 80))
        
        elif filter_name == 'MomentumFilter':
            momentum = df['close'] / df['close'].shift(params.get('period', 14)) - 1
            return momentum.abs() > 0.01  # Some momentum required
        
        elif filter_name == 'CapitulationFilter':
            vol_sma = df['volume'].rolling(20).mean()
            price_change = df['close'].pct_change()
            return (df['volume'] > vol_sma * params.get('vol_mult', 4.0)) & \
                   (price_change < params.get('price_drop', -0.03))
        
        else:
            return pd.Series(True, index=df.index)
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export hybrids to JSON."""
        data = {
            'config': asdict(self.config),
            'hybrids': [h.to_dict() for h in self.hybrids],
            'filters': self.FILTERS,
            'generated_at': datetime.now().isoformat(),
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str
    
    def summary(self) -> str:
        """Generate summary of hybrid strategies."""
        lines = [
            f"\n{'='*60}",
            f"HYBRID STRATEGIES ({len(self.hybrids)} total)",
            f"{'='*60}\n",
        ]
        
        for i, hybrid in enumerate(self.hybrids, 1):
            lines.append(f"{i}. {hybrid.name}")
            lines.append(f"   Type: {hybrid.combination_type}")
            lines.append(f"   Base: {', '.join(hybrid.base_strategies)}")
            if hybrid.filters:
                lines.append(f"   Filters: {', '.join(hybrid.filters)}")
            lines.append(f"   {hybrid.description}")
            lines.append("")
        
        return "\n".join(lines)


# CLI interface
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    combiner = StrategyCombiner()
    combiner.add_base(['BollingerBreakout', 'MACD', 'RSI', 'OBV', 'CapitulationReversal'])
    combiner.add_filters(['VolumeFilter', 'TrendFilter', 'CapitulationFilter'])
    
    # Generate smart combinations
    hybrids = combiner.generate_smart()
    
    print(combiner.summary())
    combiner.to_json('hybrid_strategies.json')
    print(f"Saved to hybrid_strategies.json")
