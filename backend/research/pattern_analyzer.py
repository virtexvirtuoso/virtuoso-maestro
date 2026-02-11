"""
Pattern Analyzer - Find what works vs what doesn't

Analyzes backtest results to identify:
- Winning patterns (strategy × asset × timeframe combinations)
- Failure patterns
- Edge characteristics
- Recommendations for next iteration

Usage:
    analyzer = PatternAnalyzer(results)
    analysis = analyzer.analyze()
    print(analysis.what_works)
    print(analysis.what_doesnt)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import logging

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class Pattern:
    """A discovered pattern."""
    name: str
    description: str
    evidence: List[str]
    confidence: float  # 0-100
    profitability: float
    recommendation: str


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    what_works: List[Pattern]
    what_doesnt: List[Pattern]
    edge_characteristics: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'what_works': [asdict(p) for p in self.what_works],
            'what_doesnt': [asdict(p) for p in self.what_doesnt],
            'edge_characteristics': self.edge_characteristics,
            'recommendations': self.recommendations,
            'next_steps': self.next_steps,
        }


class PatternAnalyzer:
    """
    Analyze backtest results to find patterns and insights.
    """
    
    # Known pattern templates
    PATTERN_TEMPLATES = {
        'timeframe_dependency': {
            'check': '_check_timeframe_dependency',
            'description': 'Strategy performance varies significantly by timeframe',
        },
        'asset_dependency': {
            'check': '_check_asset_dependency',
            'description': 'Strategy performance varies significantly by asset',
        },
        'volume_correlation': {
            'check': '_check_volume_correlation',
            'description': 'Strategy works better with volume confirmation',
        },
        'trend_alignment': {
            'check': '_check_trend_alignment',
            'description': 'Strategy works better when aligned with trend',
        },
        'drawdown_pattern': {
            'check': '_check_drawdown_pattern',
            'description': 'Strategy has characteristic drawdown behavior',
        },
    }
    
    def __init__(self, results: Optional[List[Dict]] = None, logger: Optional[logging.Logger] = None):
        self.results = results or []
        self.logger = logger or logging.getLogger(__name__)
        self.analysis: Optional[AnalysisResult] = None
    
    def add_results(self, results: List[Dict]) -> 'PatternAnalyzer':
        """Add results to analyze."""
        self.results.extend(results)
        return self
    
    def analyze(self) -> AnalysisResult:
        """
        Run complete pattern analysis.
        
        Returns:
            AnalysisResult with findings
        """
        what_works = []
        what_doesnt = []
        
        # Group results
        by_strategy = self._group_by('strategy')
        by_asset = self._group_by('asset')
        by_timeframe = self._group_by('timeframe')
        
        # Analyze each strategy
        for strategy, results in by_strategy.items():
            patterns = self._analyze_strategy(strategy, results)
            
            for pattern in patterns:
                if pattern.profitability > 0:
                    what_works.append(pattern)
                else:
                    what_doesnt.append(pattern)
        
        # Find cross-cutting patterns
        cross_patterns = self._find_cross_patterns(by_asset, by_timeframe)
        for pattern in cross_patterns:
            if pattern.profitability > 0:
                what_works.append(pattern)
            else:
                what_doesnt.append(pattern)
        
        # Edge characteristics
        edge_chars = self._compute_edge_characteristics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(what_works, what_doesnt)
        
        # Next steps
        next_steps = self._generate_next_steps(what_works, what_doesnt)
        
        # Sort by confidence
        what_works.sort(key=lambda p: p.confidence, reverse=True)
        what_doesnt.sort(key=lambda p: p.confidence, reverse=True)
        
        self.analysis = AnalysisResult(
            what_works=what_works,
            what_doesnt=what_doesnt,
            edge_characteristics=edge_chars,
            recommendations=recommendations,
            next_steps=next_steps,
        )
        
        return self.analysis
    
    def _group_by(self, key: str) -> Dict[str, List[Dict]]:
        """Group results by a key."""
        groups = defaultdict(list)
        for r in self.results:
            groups[r.get(key, 'unknown')].append(r)
        return dict(groups)
    
    def _analyze_strategy(self, strategy: str, results: List[Dict]) -> List[Pattern]:
        """Analyze patterns for a single strategy."""
        patterns = []
        
        successful = [r for r in results if r.get('success', True)]
        if not successful:
            return patterns
        
        returns = [r['total_return'] for r in successful]
        sharpes = [r['sharpe_ratio'] for r in successful]
        drawdowns = [r['max_drawdown'] for r in successful]
        
        avg_return = np.mean(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # Pattern: Overall profitability
        if avg_return > 50 and win_rate > 0.6:
            patterns.append(Pattern(
                name=f"{strategy}_overall_winner",
                description=f"{strategy} is consistently profitable",
                evidence=[
                    f"Average return: {avg_return:+.2f}%",
                    f"Win rate: {win_rate*100:.0f}%",
                    f"Tests: {len(successful)}",
                ],
                confidence=min(95, win_rate * 100 + 20),
                profitability=avg_return,
                recommendation=f"Deploy {strategy} as core strategy"
            ))
        elif avg_return < -20:
            patterns.append(Pattern(
                name=f"{strategy}_overall_loser",
                description=f"{strategy} is consistently unprofitable",
                evidence=[
                    f"Average return: {avg_return:+.2f}%",
                    f"Win rate: {win_rate*100:.0f}%",
                ],
                confidence=min(95, (1-win_rate) * 100 + 20),
                profitability=avg_return,
                recommendation=f"Avoid {strategy} or major parameter adjustment needed"
            ))
        
        # Pattern: Timeframe dependency
        by_tf = defaultdict(list)
        for r in successful:
            by_tf[r['timeframe']].append(r['total_return'])
        
        if len(by_tf) > 1:
            tf_avgs = {tf: np.mean(rets) for tf, rets in by_tf.items()}
            best_tf = max(tf_avgs, key=tf_avgs.get)
            worst_tf = min(tf_avgs, key=tf_avgs.get)
            
            spread = tf_avgs[best_tf] - tf_avgs[worst_tf]
            
            if spread > 100:  # Significant spread
                patterns.append(Pattern(
                    name=f"{strategy}_timeframe_sensitive",
                    description=f"{strategy} performance heavily depends on timeframe",
                    evidence=[
                        f"Best: {best_tf} ({tf_avgs[best_tf]:+.2f}%)",
                        f"Worst: {worst_tf} ({tf_avgs[worst_tf]:+.2f}%)",
                        f"Spread: {spread:.2f}%",
                    ],
                    confidence=min(90, spread / 2),
                    profitability=tf_avgs[best_tf],
                    recommendation=f"Focus {strategy} on {best_tf} timeframe only"
                ))
        
        # Pattern: Asset dependency
        by_asset = defaultdict(list)
        for r in successful:
            by_asset[r['asset']].append(r['total_return'])
        
        if len(by_asset) > 1:
            asset_avgs = {a: np.mean(rets) for a, rets in by_asset.items()}
            best_asset = max(asset_avgs, key=asset_avgs.get)
            worst_asset = min(asset_avgs, key=asset_avgs.get)
            
            spread = asset_avgs[best_asset] - asset_avgs[worst_asset]
            
            if spread > 100:
                patterns.append(Pattern(
                    name=f"{strategy}_asset_sensitive",
                    description=f"{strategy} performance varies significantly by asset",
                    evidence=[
                        f"Best: {best_asset} ({asset_avgs[best_asset]:+.2f}%)",
                        f"Worst: {worst_asset} ({asset_avgs[worst_asset]:+.2f}%)",
                    ],
                    confidence=min(90, spread / 2),
                    profitability=asset_avgs[best_asset],
                    recommendation=f"Focus {strategy} on {best_asset}"
                ))
        
        # Pattern: Drawdown behavior
        avg_dd = np.mean(drawdowns)
        if avg_dd > 50:
            patterns.append(Pattern(
                name=f"{strategy}_high_drawdown",
                description=f"{strategy} has excessive drawdowns",
                evidence=[
                    f"Average max drawdown: {avg_dd:.2f}%",
                    f"This kills real trading even if profitable",
                ],
                confidence=85,
                profitability=-abs(avg_dd),  # Negative for what_doesnt
                recommendation=f"Add stop-loss or position sizing to {strategy}"
            ))
        
        return patterns
    
    def _find_cross_patterns(self, by_asset: Dict, by_timeframe: Dict) -> List[Pattern]:
        """Find patterns that apply across strategies."""
        patterns = []
        
        # Timeframe analysis across all strategies
        tf_returns = {}
        for tf, results in by_timeframe.items():
            successful = [r for r in results if r.get('success', True)]
            if successful:
                tf_returns[tf] = np.mean([r['total_return'] for r in successful])
        
        if tf_returns:
            best_tf = max(tf_returns, key=tf_returns.get)
            worst_tf = min(tf_returns, key=tf_returns.get)
            
            if tf_returns[best_tf] > 0:
                patterns.append(Pattern(
                    name="best_timeframe_overall",
                    description=f"{best_tf} is the sweet spot timeframe",
                    evidence=[
                        f"Average return on {best_tf}: {tf_returns[best_tf]:+.2f}%",
                        f"Works across multiple strategies",
                    ],
                    confidence=80,
                    profitability=tf_returns[best_tf],
                    recommendation=f"Focus all strategies on {best_tf} timeframe"
                ))
            
            if tf_returns[worst_tf] < -20:
                patterns.append(Pattern(
                    name="worst_timeframe_overall",
                    description=f"{worst_tf} timeframe destroys edge",
                    evidence=[
                        f"Average return on {worst_tf}: {tf_returns[worst_tf]:+.2f}%",
                        f"Commission drag likely culprit for short TFs",
                    ],
                    confidence=85,
                    profitability=tf_returns[worst_tf],
                    recommendation=f"Avoid {worst_tf} timeframe entirely"
                ))
        
        # Check for 5m/1m death pattern
        short_tfs = ['1m', '5m']
        short_returns = []
        for tf in short_tfs:
            if tf in tf_returns:
                short_returns.append(tf_returns[tf])
        
        if short_returns and np.mean(short_returns) < 0:
            patterns.append(Pattern(
                name="short_timeframe_death",
                description="5-minute data = death",
                evidence=[
                    "Commission drag kills everything at high frequency",
                    "Edge fades faster than trading costs",
                ],
                confidence=90,
                profitability=np.mean(short_returns),
                recommendation="Never use 5m or lower timeframes"
            ))
        
        return patterns
    
    def _compute_edge_characteristics(self) -> Dict[str, Any]:
        """Compute characteristics of the trading edge."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.get('success', True)]
        if not successful:
            return {}
        
        returns = [r['total_return'] for r in successful]
        sharpes = [r['sharpe_ratio'] for r in successful]
        win_rates = [r.get('win_rate', 50) for r in successful]
        
        return {
            'total_tests': len(self.results),
            'successful_tests': len(successful),
            'overall_profitable_pct': len([r for r in returns if r > 0]) / len(returns) * 100,
            'average_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'average_sharpe': np.mean(sharpes),
            'average_win_rate': np.mean(win_rates),
        }
    
    def _generate_recommendations(self, what_works: List[Pattern], what_doesnt: List[Pattern]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Top positive patterns
        for pattern in what_works[:3]:
            recommendations.append(pattern.recommendation)
        
        # Top negative patterns (things to avoid)
        for pattern in what_doesnt[:3]:
            recommendations.append(pattern.recommendation)
        
        return recommendations
    
    def _generate_next_steps(self, what_works: List[Pattern], what_doesnt: List[Pattern]) -> List[str]:
        """Generate next iteration steps."""
        steps = []
        
        if what_works:
            top_winner = what_works[0]
            steps.append(f"1. Optimize parameters for {top_winner.name.split('_')[0]}")
            steps.append(f"2. Run walk-forward validation on winning strategies")
        
        if what_doesnt:
            top_loser = what_doesnt[0]
            if 'timeframe' in top_loser.name:
                steps.append(f"3. Eliminate short timeframes from all tests")
            if 'drawdown' in top_loser.name:
                steps.append(f"4. Add position sizing / stop-loss overlay")
        
        steps.append("5. Test hybrid combinations of top 2-3 strategies")
        steps.append("6. Paper trade best combination for 2 weeks")
        
        return steps
    
    def generate_report(self) -> str:
        """Generate formatted analysis report."""
        if not self.analysis:
            self.analyze()
        
        lines = [
            "",
            "=" * 80,
            "PATTERN ANALYSIS REPORT",
            "=" * 80,
            "",
        ]
        
        # Edge characteristics
        ec = self.analysis.edge_characteristics
        if ec:
            lines.append("📊 EDGE CHARACTERISTICS:")
            lines.append("-" * 40)
            lines.append(f"   Total tests: {ec.get('total_tests', 0)}")
            lines.append(f"   Profitable: {ec.get('overall_profitable_pct', 0):.1f}%")
            lines.append(f"   Avg return: {ec.get('average_return', 0):+.2f}%")
            lines.append(f"   Avg Sharpe: {ec.get('average_sharpe', 0):.4f}")
            lines.append("")
        
        # What works
        lines.append("✅ WHAT WORKS:")
        lines.append("-" * 80)
        for i, pattern in enumerate(self.analysis.what_works[:5], 1):
            lines.append(f"\n{i}. {pattern.description}")
            for evidence in pattern.evidence:
                lines.append(f"   • {evidence}")
            lines.append(f"   Confidence: {pattern.confidence:.0f}%")
            lines.append(f"   → {pattern.recommendation}")
        
        # What doesn't
        lines.append("")
        lines.append("❌ WHAT DOESN'T WORK:")
        lines.append("-" * 80)
        for i, pattern in enumerate(self.analysis.what_doesnt[:5], 1):
            lines.append(f"\n{i}. {pattern.description}")
            for evidence in pattern.evidence:
                lines.append(f"   • {evidence}")
            lines.append(f"   → {pattern.recommendation}")
        
        # Recommendations
        lines.append("")
        lines.append("🎯 RECOMMENDATIONS:")
        lines.append("-" * 80)
        for rec in self.analysis.recommendations:
            lines.append(f"   • {rec}")
        
        # Next steps
        lines.append("")
        lines.append("📋 NEXT STEPS:")
        lines.append("-" * 80)
        for step in self.analysis.next_steps:
            lines.append(f"   {step}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export analysis to JSON."""
        if not self.analysis:
            self.analyze()
        
        data = {
            'analysis': self.analysis.to_dict(),
            'generated_at': datetime.now().isoformat(),
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pattern Analyzer")
    parser.add_argument('--input', required=True, help="JSON results file")
    parser.add_argument('--output', default='analysis.json')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    data = json.loads(Path(args.input).read_text())
    results = data.get('results', data)
    
    analyzer = PatternAnalyzer(results)
    analyzer.analyze()
    
    print(analyzer.generate_report())
    analyzer.to_json(args.output)
    print(f"\nSaved to {args.output}")
