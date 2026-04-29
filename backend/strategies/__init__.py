"""
Maestro Strategies - 66 Trading Strategies

Categories:
- technical: 19 strategies (Ichimoku, ADX, MACD, BB, RSI, etc.)
- scalping: 8 strategies (VWAP, StochRSI, EMA Ribbon, etc.)
- momentum: 6 strategies (TSMOM, ATR trend, volatility, etc.)
- composite: 6 strategies (Fernando, Combined, etc.)
- derivatives: 11 strategies (OI, CVD, Liquidation, Funding, etc.)
- hybrids: 16 strategies (combined strategies with filters)

Usage:
    from strategies import list_strategies, run_strategy, summary

    print(list_strategies())  # All 65
    print(list_strategies('derivatives'))  # Category only

    signals = run_strategy('Fernando', df)
    signals = run_strategy('MACD+RSI', df)
"""

from pathlib import Path
import importlib.util

STRATEGY_DIR = Path(__file__).parent
CATEGORIES = ['technical', 'scalping', 'momentum', 'composite', 'derivatives', 'hybrids']

def load_strategy(name: str):
    """Load a strategy module by name."""
    for subdir in CATEGORIES:
        subdir_path = STRATEGY_DIR / subdir
        if not subdir_path.exists():
            continue
        for file in subdir_path.glob('*.py'):
            if file.stem.startswith('_'):
                continue
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if getattr(module, 'NAME', None) == name:
                    return module
            except Exception:
                pass
    return None

def list_strategies(category: str = None) -> list:
    """List all strategy names, optionally filtered by category."""
    strategies = []
    subdirs = [category] if category else CATEGORIES
    
    for subdir in subdirs:
        subdir_path = STRATEGY_DIR / subdir
        if not subdir_path.exists():
            continue
        for file in subdir_path.glob('*.py'):
            if file.stem.startswith('_'):
                continue
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'NAME'):
                    strategies.append(module.NAME)
            except Exception:
                pass
    return sorted(set(strategies))

def list_categories() -> list:
    """List available categories."""
    return CATEGORIES.copy()

def get_description(name: str) -> str:
    """Get strategy description."""
    module = load_strategy(name)
    if module:
        return getattr(module, 'DESCRIPTION', 'No description')
    return 'Unknown strategy'

def run_strategy(name: str, df, **kwargs):
    """Run a strategy by name, returns signal Series."""
    module = load_strategy(name)
    if module is None:
        raise ValueError(f"Unknown strategy: {name}. Use list_strategies() to see available.")
    return module.generate_signals(df, **kwargs)

def strategy_info(name: str) -> dict:
    """Get full info about a strategy."""
    module = load_strategy(name)
    if module is None:
        return {'error': f"Unknown strategy: {name}"}

    return {
        'name': getattr(module, 'NAME', name),
        'category': getattr(module, 'CATEGORY', 'unknown'),
        'description': getattr(module, 'DESCRIPTION', 'No description'),
        'requires_derivatives': getattr(module, 'REQUIRES_DERIVATIVES', False),
    }


def get_param_metadata(name: str) -> dict:
    """
    Get detailed parameter metadata for a strategy.

    Returns dict of {param_name: {value, min, max, description, step, type}}.
    Uses function signature for defaults and types, plus optional PARAM_METADATA
    dict from the strategy module for descriptions and ranges.
    """
    import inspect

    module = load_strategy(name)
    if module is None:
        return {}

    if not hasattr(module, 'generate_signals'):
        return {}

    # Get function signature
    sig = inspect.signature(module.generate_signals)

    # Get optional PARAM_METADATA from module
    param_meta = getattr(module, 'PARAM_METADATA', {})

    # Default ranges for common parameter patterns
    DEFAULT_RANGES = {
        # Period-based params (lookback windows)
        'period': {'min': 2, 'max': 200, 'step': 1, 'description': 'Lookback period'},
        'fast': {'min': 2, 'max': 100, 'step': 1, 'description': 'Fast period'},
        'slow': {'min': 5, 'max': 200, 'step': 1, 'description': 'Slow period'},
        'signal': {'min': 2, 'max': 50, 'step': 1, 'description': 'Signal line period'},
        'tenkan': {'min': 5, 'max': 30, 'step': 1, 'description': 'Tenkan-sen (Conversion Line) period'},
        'kijun': {'min': 10, 'max': 60, 'step': 1, 'description': 'Kijun-sen (Base Line) period'},
        'senkou_b': {'min': 20, 'max': 120, 'step': 1, 'description': 'Senkou Span B (Leading Span B) period'},
        # RSI thresholds
        'oversold': {'min': 10, 'max': 40, 'step': 1, 'description': 'Oversold threshold'},
        'overbought': {'min': 60, 'max': 90, 'step': 1, 'description': 'Overbought threshold'},
        # Standard deviation multiplier
        'std': {'min': 0.5, 'max': 4.0, 'step': 0.1, 'description': 'Standard deviation multiplier'},
        # ATR multiplier
        'atr_mult': {'min': 0.5, 'max': 5.0, 'step': 0.1, 'description': 'ATR multiplier'},
        'multiplier': {'min': 0.5, 'max': 5.0, 'step': 0.1, 'description': 'Multiplier'},
        # Length/window
        'length': {'min': 2, 'max': 100, 'step': 1, 'description': 'Window length'},
        'window': {'min': 2, 'max': 100, 'step': 1, 'description': 'Rolling window size'},
        # Thresholds
        'threshold': {'min': 0.0, 'max': 100.0, 'step': 0.1, 'description': 'Threshold value'},
    }

    result = {}

    for param_name, param in sig.parameters.items():
        # Skip 'df' parameter (the DataFrame)
        if param_name == 'df':
            continue

        # Get default value
        default = param.default if param.default != inspect.Parameter.empty else None

        # Determine type from annotation or default value
        param_type = 'int'
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == float:
                param_type = 'float'
            elif param.annotation == bool:
                param_type = 'bool'
        elif default is not None:
            if isinstance(default, float) and not isinstance(default, bool):
                param_type = 'float'
            elif isinstance(default, bool):
                param_type = 'bool'

        # Start with defaults from DEFAULT_RANGES if param name matches
        meta = {}
        if param_name in DEFAULT_RANGES:
            meta = DEFAULT_RANGES[param_name].copy()

        # Override with module-specific PARAM_METADATA if available
        if param_name in param_meta:
            meta.update(param_meta[param_name])

        # Build the result entry
        result[param_name] = {
            'value': default,
            'type': param_type,
            'min': meta.get('min'),
            'max': meta.get('max'),
            'step': meta.get('step', 1 if param_type == 'int' else 0.1),
            'description': meta.get('description', f'{param_name.replace("_", " ").title()} parameter'),
        }

    return result

def summary() -> dict:
    """Get strategy count by category."""
    counts = {}
    for cat in CATEGORIES:
        counts[cat] = len(list_strategies(cat))
    counts['total'] = len(list_strategies())
    return counts

# Quick reference
TOTAL_STRATEGIES = len(list_strategies())
