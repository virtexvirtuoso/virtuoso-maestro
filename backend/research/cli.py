#!/usr/bin/env python3
"""
Maestro Research CLI - Run strategy research from command line

Usage:
    python -m backend.research.cli run --strategies BollingerBreakout MACD --assets BTC/USDT ETH/USDT
    python -m backend.research.cli grid --strategy BollingerBreakout --all-assets --all-timeframes
    python -m backend.research.cli combine --base BollingerBreakout MACD --smart
    python -m backend.research.cli analyze --input results.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from backend.research.orchestrator import ResearchOrchestrator, quick_research
from backend.research.grid_backtest import GridBacktester
from backend.research.strategy_combiner import StrategyCombiner
from backend.research.results_aggregator import ResultsAggregator
from backend.research.pattern_analyzer import PatternAnalyzer


# Default configurations
DEFAULT_STRATEGIES = [
    'BollingerBreakout',
    'MACD', 
    'RSI',
    'OBV',
    'VolumeBreakout',
    'EMA_Cross',
    'MeanReversion',
    'CapitulationReversal',
]

DEFAULT_ASSETS = [
    'BTC/USDT',
    'ETH/USDT', 
    'SOL/USDT',
]

DEFAULT_TIMEFRAMES = [
    '1h',
    '4h',
    '1d',
]


def cmd_run(args):
    """Run full research orchestration."""
    strategies = args.strategies or DEFAULT_STRATEGIES
    assets = args.assets or DEFAULT_ASSETS
    timeframes = args.timeframes or DEFAULT_TIMEFRAMES
    
    print(f"🔬 Starting Maestro Strategy Research")
    print(f"   Strategies: {', '.join(strategies)}")
    print(f"   Assets: {', '.join(assets)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Total combinations: {len(strategies) * len(assets) * len(timeframes)}")
    print()
    
    report = quick_research(
        strategies=strategies,
        assets=assets,
        timeframes=timeframes,
        output_dir=args.output
    )
    
    print(report)
    print(f"\n📁 Results saved to: {args.output}/")


def cmd_grid(args):
    """Run grid backtest."""
    strategies = args.strategies or [args.strategy] if args.strategy else DEFAULT_STRATEGIES[:3]
    assets = DEFAULT_ASSETS if args.all_assets else (args.assets or DEFAULT_ASSETS[:2])
    timeframes = DEFAULT_TIMEFRAMES if args.all_timeframes else (args.timeframes or ['1h', '4h'])
    
    print(f"📊 Running Grid Backtest")
    print(f"   {len(strategies)} strategies × {len(assets)} assets × {len(timeframes)} timeframes")
    print()
    
    grid = GridBacktester()
    grid.add_strategies(strategies)
    grid.add_assets(assets)
    grid.add_timeframes(timeframes)
    
    def progress(current, total, result):
        status = "✅" if result.success else "❌"
        ret = f"{result.total_return:+.2f}%" if result.success else "ERROR"
        print(f"[{current}/{total}] {status} {result.strategy}/{result.asset}/{result.timeframe}: {ret}")
    
    results = grid.run(progress_callback=progress)
    
    print()
    print(grid.summary())
    
    if args.output:
        grid.to_json(args.output)
        print(f"\n📁 Saved to: {args.output}")


def cmd_combine(args):
    """Generate hybrid strategy combinations."""
    base = args.base or DEFAULT_STRATEGIES[:4]
    
    print(f"🧬 Generating Hybrid Strategies")
    print(f"   Base strategies: {', '.join(base)}")
    print()
    
    combiner = StrategyCombiner()
    combiner.add_base(base)
    combiner.add_filters(['VolumeFilter', 'TrendFilter', 'CapitulationFilter'])
    
    if args.smart:
        hybrids = combiner.generate_smart()
    else:
        hybrids = combiner.generate()
    
    print(combiner.summary())
    
    if args.output:
        combiner.to_json(args.output)
        print(f"\n📁 Saved to: {args.output}")


def cmd_analyze(args):
    """Analyze backtest results."""
    import json
    
    print(f"🔍 Analyzing Results")
    print(f"   Input: {args.input}")
    print()
    
    data = json.loads(Path(args.input).read_text())
    results = data.get('results', data)
    
    # Run aggregation
    aggregator = ResultsAggregator()
    aggregator.add_results(results)
    aggregator.aggregate()
    print(aggregator.generate_report())
    
    # Run pattern analysis
    analyzer = PatternAnalyzer(results)
    analyzer.analyze()
    print(analyzer.generate_report())
    
    if args.output:
        analyzer.to_json(args.output)
        print(f"\n📁 Analysis saved to: {args.output}")


def cmd_list(args):
    """List available strategies and filters."""
    print("📋 Available Strategies:")
    print("-" * 40)
    for s in GridBacktester.STRATEGIES.keys():
        print(f"   • {s}")
    
    print()
    print("🔧 Available Filters:")
    print("-" * 40)
    for f, info in StrategyCombiner.FILTERS.items():
        print(f"   • {f}: {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Maestro Strategy Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full research run
  python -m backend.research.cli run --strategies BollingerBreakout MACD RSI
  
  # Quick grid test
  python -m backend.research.cli grid --strategy BollingerBreakout --all-assets
  
  # Generate hybrid combinations
  python -m backend.research.cli combine --base BollingerBreakout MACD --smart
  
  # Analyze existing results
  python -m backend.research.cli analyze --input results.json
  
  # List available strategies
  python -m backend.research.cli list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run full research orchestration')
    run_parser.add_argument('--strategies', nargs='+', help='Strategies to test')
    run_parser.add_argument('--assets', nargs='+', help='Assets to test')
    run_parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')
    run_parser.add_argument('--output', default='research_output', help='Output directory')
    run_parser.add_argument('--no-hybrids', action='store_true', help='Skip hybrid generation')
    
    # Grid command
    grid_parser = subparsers.add_parser('grid', help='Run grid backtest')
    grid_parser.add_argument('--strategy', help='Single strategy to test')
    grid_parser.add_argument('--strategies', nargs='+', help='Multiple strategies')
    grid_parser.add_argument('--assets', nargs='+', help='Assets to test')
    grid_parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')
    grid_parser.add_argument('--all-assets', action='store_true', help='Test all default assets')
    grid_parser.add_argument('--all-timeframes', action='store_true', help='Test all default timeframes')
    grid_parser.add_argument('--output', help='Output JSON file')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Generate hybrid strategies')
    combine_parser.add_argument('--base', nargs='+', help='Base strategies')
    combine_parser.add_argument('--smart', action='store_true', help='Use smart combinations')
    combine_parser.add_argument('--output', help='Output JSON file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--input', required=True, help='Input JSON file')
    analyze_parser.add_argument('--output', help='Output analysis JSON')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available strategies/filters')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Dispatch command
    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'grid':
        cmd_grid(args)
    elif args.command == 'combine':
        cmd_combine(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
