"""
Engine Factory - Factory for creating V1 and V2 backtesting/optimization engines.

Provides a clean interface for selecting between V1 (Backtrader-based) and V2
(VectorBT + Optuna) engines.
"""


class EngineFactory:
    """
    Factory class for creating backtesting and walk-forward optimization engines.

    Supports both V1 (Backtrader-based) and V2 (VectorBT + Optuna) engine versions.

    Example usage:
        # Create V1 backtest engine (returns class)
        engine_cls = EngineFactory.create_backtest_engine('v1')

        # Create V2 backtest engine (returns instance)
        engine = EngineFactory.create_backtest_engine('v2')

        # Create walk-forward engines
        wf_v1 = EngineFactory.create_walkforward_engine('v1')
        wf_v2 = EngineFactory.create_walkforward_engine('v2')
    """

    @staticmethod
    def create_backtest_engine(version: str = 'v1'):
        """
        Create a backtesting engine of the specified version.

        Args:
            version: Engine version ('v1' for Backtrader, 'v2' for VectorBT)

        Returns:
            V1: BacktestingEngine class (for use with Optimizer)
            V2: VectorBTEngine instance (ready to use)

        Raises:
            ValueError: If version is not 'v1' or 'v2'
        """
        if version == 'v1':
            from engine.backtesting_engine import BacktestingEngine
            return BacktestingEngine
        elif version == 'v2':
            from engine_v2.vectorbt_engine import BacktestConfig, VectorBTEngine
            return VectorBTEngine(config=BacktestConfig())
        else:
            raise ValueError(f"Unknown engine version: {version}. Use 'v1' or 'v2'.")

    @staticmethod
    def create_walkforward_engine(version: str = 'v1'):
        """
        Create a walk-forward optimization engine of the specified version.

        Args:
            version: Engine version ('v1' for Backtrader, 'v2' for Optuna)

        Returns:
            V1: WalkForwardEngine class (for use with Optimizer)
            V2: WalkForwardOptuna class (to be instantiated with data/strategy)

        Raises:
            ValueError: If version is not 'v1' or 'v2'
        """
        if version == 'v1':
            from engine.walk_forward_engine import WalkForwardEngine
            return WalkForwardEngine
        elif version == 'v2':
            from engine_v2.walk_forward_optuna import WalkForwardOptuna
            return WalkForwardOptuna
        else:
            raise ValueError(f"Unknown engine version: {version}. Use 'v1' or 'v2'.")

    @staticmethod
    def get_available_versions() -> list:
        """Return list of available engine versions."""
        return ['v1', 'v2']

    @staticmethod
    def get_engine_info(version: str) -> dict:
        """
        Get information about a specific engine version.

        Args:
            version: Engine version ('v1' or 'v2')

        Returns:
            Dictionary with engine metadata
        """
        if version == 'v1':
            return {
                'version': 'v1',
                'name': 'Backtrader Engine',
                'backtest_engine': 'BacktestingEngine',
                'walkforward_engine': 'WalkForwardEngine',
                'optimization': 'Grid Search',
                'speed': 'Baseline (1x)',
                'description': 'Event-driven backtesting using Backtrader framework',
            }
        elif version == 'v2':
            return {
                'version': 'v2',
                'name': 'VectorBT + Optuna Engine',
                'backtest_engine': 'VectorBTEngine',
                'walkforward_engine': 'WalkForwardOptuna',
                'optimization': 'TPE (Bayesian)',
                'speed': '100-1000x faster',
                'description': 'Vectorized backtesting with smart hyperparameter optimization',
            }
        else:
            raise ValueError(f"Unknown engine version: {version}")
