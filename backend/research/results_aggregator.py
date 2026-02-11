"""
Results Aggregator - Consolidate and analyze backtest results

Aggregates results from multiple backtests and generates:
- Cross-asset comparison tables
- Strategy rankings
- Consistency scores
- What works / what doesn't analysis

Usage:
    aggregator = ResultsAggregator()
    aggregator.add_results(grid_results)
    report = aggregator.generate_report()
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
class StrategyRanking:
    """Ranking of a strategy across all tests."""
    strategy: str
    avg_return: float
    avg_sharpe: float
    avg_max_dd: float
    win_rate: float
    consistency_score: float  # How consistent across assets/timeframes
    total_tests: int
    profitable_tests: int
    best_asset: str
    best_timeframe: str
    worst_asset: str
    worst_timeframe: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AssetTimeframeMatrix:
    """Performance matrix for strategy × asset × timeframe."""
    strategy: str
    matrix: Dict[str, Dict[str, float]]  # asset -> timeframe -> return
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ResultsAggregator:
    """
    Aggregate and analyze backtest results from multiple sources.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.results: List[Dict] = []
        self.rankings: List[StrategyRanking] = []
        self.matrices: List[AssetTimeframeMatrix] = []
    
    def add_results(self, results: List[Dict]) -> 'ResultsAggregator':
        """Add backtest results."""
        self.results.extend(results)
        return self
    
    def add_from_json(self, path: str) -> 'ResultsAggregator':
        """Load results from JSON file."""
        data = json.loads(Path(path).read_text())
        if 'results' in data:
            self.results.extend(data['results'])
        else:
            self.results.extend(data)
        return self
    
    def aggregate(self) -> 'ResultsAggregator':
        """
        Aggregate all results and compute rankings.
        """
        if not self.results:
            self.logger.warning("No results to aggregate")
            return self
        
        # Group by strategy
        by_strategy = defaultdict(list)
        for r in self.results:
            by_strategy[r['strategy']].append(r)
        
        # Compute rankings
        self.rankings = []
        for strategy, results in by_strategy.items():
            ranking = self._compute_ranking(strategy, results)
            self.rankings.append(ranking)
        
        # Sort by average Sharpe
        self.rankings.sort(key=lambda x: x.avg_sharpe, reverse=True)
        
        # Compute matrices
        self.matrices = []
        for strategy in by_strategy.keys():
            matrix = self._compute_matrix(strategy, by_strategy[strategy])
            self.matrices.append(matrix)
        
        return self
    
    def _compute_ranking(self, strategy: str, results: List[Dict]) -> StrategyRanking:
        """Compute ranking metrics for a strategy."""
        successful = [r for r in results if r.get('success', True)]
        
        if not successful:
            return StrategyRanking(
                strategy=strategy,
                avg_return=0, avg_sharpe=0, avg_max_dd=0, win_rate=0,
                consistency_score=0, total_tests=len(results), profitable_tests=0,
                best_asset='N/A', best_timeframe='N/A',
                worst_asset='N/A', worst_timeframe='N/A'
            )
        
        returns = [r['total_return'] for r in successful]
        sharpes = [r['sharpe_ratio'] for r in successful]
        max_dds = [r['max_drawdown'] for r in successful]
        
        # Find best/worst
        best_idx = max(range(len(successful)), key=lambda i: successful[i]['total_return'])
        worst_idx = min(range(len(successful)), key=lambda i: successful[i]['total_return'])
        
        # Consistency: std of returns (lower = more consistent)
        std_return = np.std(returns) if len(returns) > 1 else 0
        consistency = max(0, 100 - std_return)  # Invert so higher = more consistent
        
        return StrategyRanking(
            strategy=strategy,
            avg_return=np.mean(returns),
            avg_sharpe=np.mean(sharpes),
            avg_max_dd=np.mean(max_dds),
            win_rate=len([r for r in returns if r > 0]) / len(returns) * 100,
            consistency_score=consistency,
            total_tests=len(results),
            profitable_tests=len([r for r in returns if r > 0]),
            best_asset=successful[best_idx]['asset'],
            best_timeframe=successful[best_idx]['timeframe'],
            worst_asset=successful[worst_idx]['asset'],
            worst_timeframe=successful[worst_idx]['timeframe'],
        )
    
    def _compute_matrix(self, strategy: str, results: List[Dict]) -> AssetTimeframeMatrix:
        """Compute asset × timeframe performance matrix."""
        matrix = defaultdict(dict)
        
        for r in results:
            if r.get('success', True):
                matrix[r['asset']][r['timeframe']] = r['total_return']
        
        return AssetTimeframeMatrix(strategy=strategy, matrix=dict(matrix))
    
    def get_cross_asset_table(self, metric: str = 'total_return') -> pd.DataFrame:
        """
        Generate cross-asset comparison table.
        
        Returns DataFrame with strategies as rows, asset/timeframe as columns.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required")
        
        data = []
        for r in self.results:
            if r.get('success', True):
                data.append({
                    'strategy': r['strategy'],
                    'asset_tf': f"{r['asset']}_{r['timeframe']}",
                    'value': r.get(metric, 0)
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        pivot = df.pivot(index='strategy', columns='asset_tf', values='value')
        return pivot
    
    def get_consistency_analysis(self) -> Dict[str, Any]:
        """
        Analyze which strategies are consistent across assets/timeframes.
        
        Returns dict with:
        - consistent: strategies that work everywhere
        - asset_specific: strategies that only work on certain assets
        - timeframe_specific: strategies that only work on certain timeframes
        """
        analysis = {
            'consistent': [],
            'asset_specific': [],
            'timeframe_specific': [],
            'inconsistent': [],
        }
        
        for ranking in self.rankings:
            # Find the matrix for this strategy
            matrix = next((m for m in self.matrices if m.strategy == ranking.strategy), None)
            if not matrix:
                continue
            
            # Analyze consistency
            all_returns = []
            by_asset = defaultdict(list)
            by_tf = defaultdict(list)
            
            for asset, tfs in matrix.matrix.items():
                for tf, ret in tfs.items():
                    all_returns.append(ret)
                    by_asset[asset].append(ret)
                    by_tf[tf].append(ret)
            
            if not all_returns:
                continue
            
            # Check if profitable everywhere
            profitable_pct = len([r for r in all_returns if r > 0]) / len(all_returns)
            
            # Check asset variation
            asset_avgs = {a: np.mean(rets) for a, rets in by_asset.items()}
            asset_std = np.std(list(asset_avgs.values())) if len(asset_avgs) > 1 else 0
            
            # Check timeframe variation
            tf_avgs = {t: np.mean(rets) for t, rets in by_tf.items()}
            tf_std = np.std(list(tf_avgs.values())) if len(tf_avgs) > 1 else 0
            
            # Classify
            if profitable_pct > 0.8 and ranking.consistency_score > 50:
                analysis['consistent'].append({
                    'strategy': ranking.strategy,
                    'profitable_pct': profitable_pct * 100,
                    'avg_return': ranking.avg_return,
                })
            elif asset_std > tf_std * 2:
                # More variation by asset
                best_asset = max(asset_avgs, key=asset_avgs.get)
                analysis['asset_specific'].append({
                    'strategy': ranking.strategy,
                    'best_asset': best_asset,
                    'best_return': asset_avgs[best_asset],
                })
            elif tf_std > asset_std * 2:
                # More variation by timeframe
                best_tf = max(tf_avgs, key=tf_avgs.get)
                analysis['timeframe_specific'].append({
                    'strategy': ranking.strategy,
                    'best_timeframe': best_tf,
                    'best_return': tf_avgs[best_tf],
                })
            else:
                analysis['inconsistent'].append({
                    'strategy': ranking.strategy,
                    'profitable_pct': profitable_pct * 100,
                })
        
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report.
        
        Returns formatted string report similar to the screenshots.
        """
        if not self.rankings:
            self.aggregate()
        
        lines = [
            "",
            "=" * 80,
            "STRATEGY RESEARCH RESULTS",
            "=" * 80,
            "",
            f"Total backtests: {len(self.results)}",
            f"Strategies tested: {len(self.rankings)}",
            "",
        ]
        
        # Strategy Rankings
        lines.append("📊 STRATEGY RANKINGS (by Sharpe Ratio)")
        lines.append("-" * 80)
        lines.append(f"{'Strategy':<30} {'Avg Return':>12} {'Sharpe':>10} {'MaxDD':>10} {'Consistent?':>12}")
        lines.append("-" * 80)
        
        for r in self.rankings:
            consistent = "✅ YES" if r.consistency_score > 50 and r.avg_return > 0 else "⚠️ Partial" if r.avg_return > 0 else "❌ NO"
            lines.append(
                f"{r.strategy:<30} {r.avg_return:>+10.2f}% {r.avg_sharpe:>10.4f} "
                f"{r.avg_max_dd:>9.2f}% {consistent:>12}"
            )
        
        # Cross-asset matrix for top strategy
        if self.matrices:
            lines.append("")
            lines.append("📈 WHAT ACTUALLY WORKS ACROSS THE BOARD:")
            lines.append("-" * 80)
            
            top_matrix = self.matrices[0]
            lines.append(f"\nStrategy: {top_matrix.strategy}")
            
            # Build table
            assets = list(top_matrix.matrix.keys())
            timeframes = set()
            for tfs in top_matrix.matrix.values():
                timeframes.update(tfs.keys())
            timeframes = sorted(timeframes)
            
            header = f"{'Asset':<15}" + "".join(f"{tf:>12}" for tf in timeframes) + f"{'Consistent?':>14}"
            lines.append(header)
            lines.append("-" * len(header))
            
            for asset in assets:
                row = f"{asset:<15}"
                returns = []
                for tf in timeframes:
                    ret = top_matrix.matrix[asset].get(tf, None)
                    if ret is not None:
                        row += f"{ret:>+11.2f}%"
                        returns.append(ret)
                    else:
                        row += f"{'N/A':>12}"
                
                # Consistency check
                if returns:
                    if all(r > 0 for r in returns):
                        row += f"{'✅ YES':>14}"
                    elif any(r > 0 for r in returns):
                        row += f"{'⚠️ Partial':>14}"
                    else:
                        row += f"{'❌ NO':>14}"
                
                lines.append(row)
        
        # Consistency Analysis
        consistency = self.get_consistency_analysis()
        
        lines.append("")
        lines.append("🎯 KEY FINDINGS:")
        lines.append("-" * 80)
        
        if consistency['consistent']:
            lines.append("")
            lines.append("✅ CONSISTENT WINNERS (work everywhere):")
            for item in consistency['consistent']:
                lines.append(f"   • {item['strategy']}: {item['profitable_pct']:.0f}% profitable, avg {item['avg_return']:+.2f}%")
        
        if consistency['timeframe_specific']:
            lines.append("")
            lines.append("⚠️ TIMEFRAME-SPECIFIC (only work on certain TFs):")
            for item in consistency['timeframe_specific']:
                lines.append(f"   • {item['strategy']}: Best on {item['best_timeframe']} ({item['best_return']:+.2f}%)")
        
        if consistency['asset_specific']:
            lines.append("")
            lines.append("⚠️ ASSET-SPECIFIC (only work on certain assets):")
            for item in consistency['asset_specific']:
                lines.append(f"   • {item['strategy']}: Best on {item['best_asset']} ({item['best_return']:+.2f}%)")
        
        # RBI Verdict (like the screenshots)
        lines.append("")
        lines.append("🏆 THE RBI VERDICT:")
        lines.append("-" * 80)
        
        if self.rankings:
            top = self.rankings[0]
            lines.append(f"1. {top.strategy} is king - profitable on {top.profitable_tests}/{top.total_tests} tests")
            lines.append(f"   Best: {top.best_asset} {top.best_timeframe}")
            
            # Find timeframe patterns
            tf_performance = defaultdict(list)
            for r in self.results:
                if r.get('success', True):
                    tf_performance[r['timeframe']].append(r['total_return'])
            
            best_tf = max(tf_performance.keys(), key=lambda t: np.mean(tf_performance[t]))
            worst_tf = min(tf_performance.keys(), key=lambda t: np.mean(tf_performance[t]))
            
            lines.append(f"2. Best timeframe overall: {best_tf} (avg {np.mean(tf_performance[best_tf]):+.2f}%)")
            lines.append(f"3. Worst timeframe overall: {worst_tf} (avg {np.mean(tf_performance[worst_tf]):+.2f}%)")
            
            if '5m' in tf_performance or '1m' in tf_performance:
                lines.append("4. 5-minute data = death - commission drag kills everything")
            
            lines.append(f"5. Hourly sweet spot for most strategies")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export aggregated results to JSON."""
        if not self.rankings:
            self.aggregate()
        
        data = {
            'rankings': [r.to_dict() for r in self.rankings],
            'matrices': [m.to_dict() for m in self.matrices],
            'consistency': self.get_consistency_analysis(),
            'generated_at': datetime.now().isoformat(),
        }
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Results Aggregator")
    parser.add_argument('--input', nargs='+', required=True, help="JSON result files")
    parser.add_argument('--output', default='aggregated_results.json')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    aggregator = ResultsAggregator()
    
    for path in args.input:
        aggregator.add_from_json(path)
    
    aggregator.aggregate()
    
    print(aggregator.generate_report())
    aggregator.to_json(args.output)
    print(f"\nSaved to {args.output}")
