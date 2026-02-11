"""
Research Orchestrator - Coordinate multi-agent strategy research

Main coordinator that:
1. Receives research requests
2. Spawns parallel backtest agents
3. Aggregates findings
4. Generates next iteration recommendations

Usage:
    orchestrator = ResearchOrchestrator()
    orchestrator.configure(
        strategies=['BollingerBreakout', 'MACD', 'RSI'],
        assets=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        timeframes=['1h', '4h', '1d']
    )
    report = orchestrator.run()
"""

import json
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
import time

from .grid_backtest import GridBacktester, GridConfig, BacktestResult
from .strategy_combiner import StrategyCombiner, HybridStrategy
from .results_aggregator import ResultsAggregator
from .pattern_analyzer import PatternAnalyzer


@dataclass
class ResearchConfig:
    """Configuration for research run."""
    # What to test
    strategies: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    
    # How to test
    generate_hybrids: bool = True
    max_hybrid_components: int = 3
    
    # Data
    data_source: str = "ccxt"
    exchange: str = "binance"
    
    # Execution
    parallel: bool = True
    max_workers: int = 4
    use_subagents: bool = False  # Spawn external agents
    
    # Output
    output_dir: str = "research_output"
    
    def total_combinations(self) -> int:
        base = len(self.strategies) * len(self.assets) * len(self.timeframes)
        return base


@dataclass
class ResearchRun:
    """A complete research run."""
    id: str
    config: ResearchConfig
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    
    grid_results: List[Dict] = field(default_factory=list)
    hybrid_strategies: List[Dict] = field(default_factory=list)
    aggregated_report: str = ""
    pattern_analysis: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'config': asdict(self.config),
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'status': self.status,
            'results_count': len(self.grid_results),
            'hybrids_count': len(self.hybrid_strategies),
        }


class ResearchOrchestrator:
    """
    Orchestrate multi-agent strategy research.
    
    Workflow:
    1. Configure strategies, assets, timeframes
    2. Generate hybrid strategy combinations
    3. Run grid backtests (parallel)
    4. Aggregate results
    5. Analyze patterns
    6. Generate report with recommendations
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or ResearchConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.current_run: Optional[ResearchRun] = None
        self.progress_callback: Optional[Callable] = None
    
    def configure(
        self,
        strategies: List[str],
        assets: List[str],
        timeframes: List[str],
        **kwargs
    ) -> 'ResearchOrchestrator':
        """Configure the research run."""
        self.config.strategies = strategies
        self.config.assets = assets
        self.config.timeframes = timeframes
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        return self
    
    def on_progress(self, callback: Callable) -> 'ResearchOrchestrator':
        """Set progress callback."""
        self.progress_callback = callback
        return self
    
    def run(self) -> str:
        """
        Execute the research run.
        
        Returns:
            Formatted report string
        """
        # Initialize run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = ResearchRun(
            id=run_id,
            config=self.config,
            started_at=datetime.now().isoformat(),
        )
        
        self._log(f"Starting research run {run_id}")
        self._log(f"Testing {len(self.config.strategies)} strategies × "
                  f"{len(self.config.assets)} assets × "
                  f"{len(self.config.timeframes)} timeframes")
        
        try:
            # Step 1: Generate hybrid strategies (if enabled)
            if self.config.generate_hybrids:
                self._generate_hybrids()
            
            # Step 2: Run grid backtests
            self._run_backtests()
            
            # Step 3: Aggregate results
            aggregated = self._aggregate_results()
            
            # Step 4: Analyze patterns
            analysis = self._analyze_patterns()
            
            # Step 5: Generate report
            report = self._generate_report(aggregated, analysis)
            
            # Step 6: Save outputs
            self._save_outputs()
            
            self.current_run.status = "completed"
            self.current_run.completed_at = datetime.now().isoformat()
            
            return report
            
        except Exception as e:
            self.current_run.status = "failed"
            self._log(f"Research run failed: {e}", level="error")
            raise
    
    def _generate_hybrids(self):
        """Generate hybrid strategy combinations."""
        self._log("Generating hybrid strategies...")
        
        combiner = StrategyCombiner()
        combiner.add_base(self.config.strategies)
        combiner.add_filters(['VolumeFilter', 'TrendFilter', 'CapitulationFilter'])
        
        hybrids = combiner.generate_smart()
        self.current_run.hybrid_strategies = [h.to_dict() for h in hybrids]
        
        # Add hybrid names to strategies list
        for hybrid in hybrids:
            if hybrid.name not in self.config.strategies:
                self.config.strategies.append(hybrid.name)
        
        self._log(f"Generated {len(hybrids)} hybrid strategies")
    
    def _run_backtests(self):
        """Run all backtests."""
        self._log("Running backtests...")
        
        grid_config = GridConfig(
            strategies=self.config.strategies,
            assets=self.config.assets,
            timeframes=self.config.timeframes,
            data_source=self.config.data_source,
            exchange=self.config.exchange,
            parallel=self.config.parallel,
            max_workers=self.config.max_workers,
        )
        
        grid = GridBacktester(config=grid_config, logger=self.logger)
        
        # Progress callback
        total = grid_config.total_combinations()
        
        def progress(current, total_count, result):
            status = "✅" if result.success else "❌"
            self._log(f"[{current}/{total_count}] {status} {result.strategy}/{result.asset}/{result.timeframe}")
            if self.progress_callback:
                self.progress_callback(current, total_count, result)
        
        results = grid.run(progress_callback=progress)
        self.current_run.grid_results = [r.to_dict() for r in results]
        
        self._log(f"Completed {len(results)} backtests")
    
    def _aggregate_results(self) -> str:
        """Aggregate all results."""
        self._log("Aggregating results...")
        
        aggregator = ResultsAggregator(logger=self.logger)
        aggregator.add_results(self.current_run.grid_results)
        aggregator.aggregate()
        
        report = aggregator.generate_report()
        self.current_run.aggregated_report = report
        
        return report
    
    def _analyze_patterns(self) -> str:
        """Analyze patterns in results."""
        self._log("Analyzing patterns...")
        
        analyzer = PatternAnalyzer(self.current_run.grid_results, logger=self.logger)
        analyzer.analyze()
        
        self.current_run.pattern_analysis = analyzer.analysis.to_dict()
        
        return analyzer.generate_report()
    
    def _generate_report(self, aggregated_report: str, pattern_report: str) -> str:
        """Generate final combined report."""
        lines = [
            "",
            "🔬" * 30,
            "",
            "   MAESTRO STRATEGY RESEARCH REPORT",
            f"   Run ID: {self.current_run.id}",
            f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "🔬" * 30,
            "",
            aggregated_report,
            "",
            pattern_report,
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ]
        
        return "\n".join(lines)
    
    def _save_outputs(self):
        """Save all outputs to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_dir = output_dir / self.current_run.id
        run_dir.mkdir(exist_ok=True)
        
        # Save run metadata
        (run_dir / "run.json").write_text(
            json.dumps(self.current_run.to_dict(), indent=2)
        )
        
        # Save raw results
        (run_dir / "results.json").write_text(
            json.dumps(self.current_run.grid_results, indent=2)
        )
        
        # Save hybrid strategies
        if self.current_run.hybrid_strategies:
            (run_dir / "hybrids.json").write_text(
                json.dumps(self.current_run.hybrid_strategies, indent=2)
            )
        
        # Save aggregated report
        (run_dir / "report.txt").write_text(self.current_run.aggregated_report)
        
        # Save pattern analysis
        (run_dir / "analysis.json").write_text(
            json.dumps(self.current_run.pattern_analysis, indent=2)
        )
        
        self._log(f"Outputs saved to {run_dir}")
    
    def _log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level)(message)
    
    # ===== Sub-agent spawning (for external agent integration) =====
    
    def spawn_backtest_agent(self, strategy: str, asset: str, timeframe: str) -> Dict:
        """
        Spawn an external backtest agent.
        
        This can be used to parallelize across multiple machines
        or integrate with external agent systems.
        """
        if not self.config.use_subagents:
            # Run locally
            grid = GridBacktester()
            grid.add_strategies([strategy])
            grid.add_assets([asset])
            grid.add_timeframes([timeframe])
            results = grid.run()
            return results[0].to_dict() if results else {}
        
        # Spawn external agent (example with subprocess)
        cmd = [
            "python3", "-m", "backend.research.grid_backtest",
            "--strategies", strategy,
            "--assets", asset,
            "--timeframes", timeframe,
            "--output", f"/tmp/backtest_{strategy}_{asset}_{timeframe}.json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output_file = f"/tmp/backtest_{strategy}_{asset}_{timeframe}.json"
                if Path(output_file).exists():
                    data = json.loads(Path(output_file).read_text())
                    return data.get('results', [{}])[0]
        except Exception as e:
            self._log(f"Agent spawn failed: {e}", level="error")
        
        return {}
    
    def run_parallel_agents(self, combinations: List[tuple]) -> List[Dict]:
        """
        Run multiple backtest agents in parallel.
        
        Args:
            combinations: List of (strategy, asset, timeframe) tuples
            
        Returns:
            List of result dicts
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.spawn_backtest_agent, s, a, t): (s, a, t)
                for s, a, t in combinations
            }
            
            for future in as_completed(futures):
                combo = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._log(f"Completed: {combo}")
                except Exception as e:
                    self._log(f"Failed: {combo} - {e}", level="error")
        
        return results


# Convenience function for quick research
def quick_research(
    strategies: List[str],
    assets: List[str],
    timeframes: List[str],
    output_dir: str = "research_output"
) -> str:
    """
    Run quick strategy research.
    
    Example:
        report = quick_research(
            strategies=['BollingerBreakout', 'MACD', 'RSI'],
            assets=['BTC/USDT', 'ETH/USDT'],
            timeframes=['1h', '4h', '1d']
        )
        print(report)
    """
    orchestrator = ResearchOrchestrator()
    orchestrator.configure(
        strategies=strategies,
        assets=assets,
        timeframes=timeframes,
        output_dir=output_dir
    )
    return orchestrator.run()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Orchestrator")
    parser.add_argument('--strategies', nargs='+', 
                        default=['BollingerBreakout', 'MACD', 'RSI', 'OBV'])
    parser.add_argument('--assets', nargs='+',
                        default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
    parser.add_argument('--timeframes', nargs='+',
                        default=['1h', '4h', '1d'])
    parser.add_argument('--output', default='research_output')
    parser.add_argument('--no-hybrids', action='store_true')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    report = quick_research(
        strategies=args.strategies,
        assets=args.assets,
        timeframes=args.timeframes,
        output_dir=args.output
    )
    
    print(report)
