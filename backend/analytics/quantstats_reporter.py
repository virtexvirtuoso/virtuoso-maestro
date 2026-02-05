"""
QuantStats Reporter - Portfolio analytics replacing abandoned PyFolio.

Uses 365-day annualization factor for cryptocurrency markets (24/7 trading)
instead of the traditional 252 trading days used for stocks.

Key Features:
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- HTML tearsheet reports
- Base64-encoded visualization images for API responses
- Proper crypto annualization (365 days)
"""

import base64
import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

try:
    import quantstats as qs
    QS_AVAILABLE = True
except ImportError:
    QS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Crypto markets trade 24/7, so use 365 days for annualization
ANNUALIZATION_FACTOR = 365


@dataclass
class QuantStatsMetrics:
    """Container for QuantStats metrics"""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    avg_daily_return: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)

    # Additional metrics
    extra_metrics: dict[str, Any] = field(default_factory=dict)


class QuantStatsReporter:
    """
    Portfolio analytics reporter using QuantStats.

    Replaces abandoned PyFolio with a modern, maintained library.
    Uses 365-day annualization for cryptocurrency 24/7 markets.

    Example:
        reporter = QuantStatsReporter(
            returns=backtest_result.returns,
            benchmark=None  # or SPY/BTC benchmark returns
        )
        metrics = reporter.get_metrics()
        html = reporter.generate_html_report()
        images = reporter.generate_tearsheet_images()
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        rf: float = 0.0,  # Risk-free rate
        periods_per_year: int = ANNUALIZATION_FACTOR,
    ):
        """
        Initialize QuantStats reporter.

        Args:
            returns: Series of portfolio returns (daily preferred)
            benchmark: Optional benchmark returns for comparison
            rf: Risk-free rate (annualized)
            periods_per_year: Annualization factor (365 for crypto, 252 for stocks)
        """
        if not QS_AVAILABLE:
            raise ImportError(
                "QuantStats is not installed. Install with: pip install quantstats>=0.0.62"
            )

        # Ensure returns is a proper Series with datetime index
        if returns is None or len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        self.returns = self._prepare_returns(returns)
        self.benchmark = self._prepare_returns(benchmark) if benchmark is not None else None
        self.rf = rf
        self.periods_per_year = periods_per_year

        # Configure QuantStats for crypto annualization
        qs.extend_pandas()

    def _prepare_returns(self, returns: pd.Series) -> pd.Series:
        """Prepare and clean returns series"""
        if returns is None:
            return None

        # Convert to Series if needed
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0] if returns.shape[1] > 0 else pd.Series()

        # Ensure datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        # Clean data
        returns = returns.dropna()

        # Replace inf values with NaN then drop
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        return returns

    def get_metrics(self) -> QuantStatsMetrics:
        """
        Calculate comprehensive portfolio metrics.

        Uses 365-day annualization for crypto markets.

        Returns:
            QuantStatsMetrics dataclass with all performance metrics
        """
        try:
            # Calculate metrics with crypto annualization
            metrics = QuantStatsMetrics()

            # Sharpe Ratio (annualized with 365 days)
            metrics.sharpe_ratio = self._safe_metric(
                lambda: qs.stats.sharpe(
                    self.returns,
                    rf=self.rf,
                    periods=self.periods_per_year
                )
            )

            # Sortino Ratio (downside risk adjusted)
            metrics.sortino_ratio = self._safe_metric(
                lambda: qs.stats.sortino(
                    self.returns,
                    rf=self.rf,
                    periods=self.periods_per_year
                )
            )

            # Calmar Ratio (return / max drawdown)
            metrics.calmar_ratio = self._safe_metric(
                lambda: qs.stats.calmar(self.returns)
            )

            # Maximum Drawdown
            metrics.max_drawdown = self._safe_metric(
                lambda: qs.stats.max_drawdown(self.returns),
                default=0.0
            )

            # Total Return
            metrics.total_return = self._safe_metric(
                lambda: qs.stats.comp(self.returns),
                default=0.0
            )

            # CAGR (Compound Annual Growth Rate)
            metrics.cagr = self._safe_metric(
                lambda: qs.stats.cagr(self.returns, periods=self.periods_per_year)
            )

            # Volatility (annualized)
            metrics.volatility = self._safe_metric(
                lambda: qs.stats.volatility(
                    self.returns,
                    periods=self.periods_per_year
                )
            )

            # Win Rate
            metrics.win_rate = self._safe_metric(
                lambda: qs.stats.win_rate(self.returns)
            )

            # Win/Loss Ratio
            metrics.win_loss_ratio = self._safe_metric(
                lambda: qs.stats.win_loss_ratio(self.returns),
                default=0.0
            )

            # Profit Factor
            metrics.profit_factor = self._safe_metric(
                lambda: qs.stats.profit_factor(self.returns),
                default=0.0
            )

            # Average Win
            metrics.avg_win = self._safe_metric(
                lambda: qs.stats.avg_win(self.returns)
            )

            # Average Loss
            metrics.avg_loss = self._safe_metric(
                lambda: qs.stats.avg_loss(self.returns)
            )

            # Best/Worst Day
            metrics.best_day = self._safe_metric(
                lambda: qs.stats.best(self.returns)
            )
            metrics.worst_day = self._safe_metric(
                lambda: qs.stats.worst(self.returns)
            )

            # Best/Worst Month
            monthly_returns = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            metrics.best_month = self._safe_metric(
                lambda: monthly_returns.max() if len(monthly_returns) > 0 else 0.0
            )
            metrics.worst_month = self._safe_metric(
                lambda: monthly_returns.min() if len(monthly_returns) > 0 else 0.0
            )

            # Average Daily Return
            metrics.avg_daily_return = self._safe_metric(
                lambda: self.returns.mean()
            )

            # Higher Moments
            metrics.skewness = self._safe_metric(
                lambda: qs.stats.skew(self.returns)
            )
            metrics.kurtosis = self._safe_metric(
                lambda: qs.stats.kurtosis(self.returns)
            )

            # Value at Risk
            metrics.var_95 = self._safe_metric(
                lambda: qs.stats.var(self.returns)
            )
            metrics.cvar_95 = self._safe_metric(
                lambda: qs.stats.cvar(self.returns)
            )

            # Drawdown duration (approximate days)
            try:
                dd_series = qs.stats.to_drawdown_series(self.returns)
                if dd_series is not None and len(dd_series) > 0:
                    # Find longest drawdown period
                    in_drawdown = dd_series < 0
                    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
                    drawdown_groups = drawdown_starts.cumsum()
                    drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
                    metrics.max_drawdown_duration = int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0
            except Exception:
                metrics.max_drawdown_duration = 0

            # Additional QuantStats metrics
            try:
                full_stats = qs.stats.metrics(
                    self.returns,
                    benchmark=self.benchmark,
                    rf=self.rf,
                    display=False,
                    mode='full',
                    periods_per_year=self.periods_per_year
                )
                if full_stats is not None:
                    metrics.extra_metrics = full_stats.to_dict() if hasattr(full_stats, 'to_dict') else {}
            except Exception as e:
                logger.debug(f"Could not calculate full stats: {e}")
                metrics.extra_metrics = {}

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return QuantStatsMetrics()

    def _safe_metric(self, fn, default: float = 0.0) -> float:
        """Safely calculate a metric with fallback"""
        try:
            result = fn()
            if result is None or pd.isna(result) or np.isinf(result):
                return default
            return float(result)
        except Exception:
            return default

    def generate_html_report(
        self,
        title: str = "Portfolio Performance Report",
        output_path: str | Path | None = None,
        benchmark_title: str = "Benchmark",
    ) -> str:
        """
        Generate an HTML tearsheet report.

        Args:
            title: Report title
            output_path: Optional path to save HTML file
            benchmark_title: Title for benchmark in charts

        Returns:
            HTML string of the report
        """
        try:
            # Create a temporary file if no output path provided
            if output_path is None:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.html',
                    delete=False
                ) as f:
                    output_path = f.name

            output_path = Path(output_path)

            # Generate report using QuantStats
            qs.reports.html(
                self.returns,
                benchmark=self.benchmark,
                rf=self.rf,
                title=title,
                output=str(output_path),
                periods_per_year=self.periods_per_year,
            )

            # Read and return HTML
            with open(output_path, encoding='utf-8') as f:
                html_content = f.read()

            return html_content

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{e}</p></body></html>"

    def generate_tearsheet_images(self) -> dict[str, str]:
        """
        Generate tearsheet images as base64-encoded strings.

        Returns:
            Dict mapping image names to base64-encoded PNG data
        """
        images = {}

        # QuantStats plots return matplotlib Figure objects when show=False
        # We capture them and convert to base64

        # 1. Cumulative Returns
        try:
            fig = qs.plots.returns(
                self.returns,
                benchmark=self.benchmark,
                compound=True,
                show=False
            )
            if fig is not None:
                images['cumulative_returns'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate cumulative returns plot: {e}")

        # 2. Drawdown Plot
        try:
            fig = qs.plots.drawdown(
                self.returns,
                show=False
            )
            if fig is not None:
                images['drawdown'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate drawdown plot: {e}")

        # 3. Monthly Returns Heatmap
        try:
            fig = qs.plots.monthly_heatmap(
                self.returns,
                show=False
            )
            if fig is not None:
                images['monthly_heatmap'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate monthly heatmap: {e}")

        # 4. Distribution of Returns
        try:
            fig = qs.plots.histogram(
                self.returns,
                show=False
            )
            if fig is not None:
                images['returns_distribution'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate returns distribution: {e}")

        # 5. Rolling Sharpe Ratio
        try:
            fig = qs.plots.rolling_sharpe(
                self.returns,
                show=False
            )
            if fig is not None:
                images['rolling_sharpe'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate rolling Sharpe plot: {e}")

        # 6. Rolling Volatility
        try:
            fig = qs.plots.rolling_volatility(
                self.returns,
                show=False
            )
            if fig is not None:
                images['rolling_volatility'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate rolling volatility plot: {e}")

        # 7. Underwater Plot (alternative drawdown view)
        try:
            fig = qs.plots.drawdowns_periods(
                self.returns,
                show=False
            )
            if fig is not None:
                images['drawdown_periods'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate drawdown periods plot: {e}")

        # 8. Daily Returns
        try:
            fig = qs.plots.daily_returns(
                self.returns,
                show=False
            )
            if fig is not None:
                images['daily_returns'] = self._fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate daily returns plot: {e}")

        return images

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64-encoded PNG string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str

    def get_report_data(self, title: str = "Portfolio Report") -> dict[str, Any]:
        """
        Get complete report data for API response.

        Returns a dict with:
        - metrics: Performance metrics
        - images: Base64-encoded tearsheet images
        - html: Full HTML report (optional, can be large)

        Returns:
            Dict with all report data
        """
        metrics = self.get_metrics()
        images = self.generate_tearsheet_images()

        return {
            'title': title,
            'annualization_factor': self.periods_per_year,
            'risk_free_rate': self.rf,
            'num_periods': len(self.returns),
            'start_date': self.returns.index[0].isoformat() if len(self.returns) > 0 else None,
            'end_date': self.returns.index[-1].isoformat() if len(self.returns) > 0 else None,
            'metrics': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'max_drawdown': metrics.max_drawdown,
                'max_drawdown_duration_days': metrics.max_drawdown_duration,
                'total_return': metrics.total_return,
                'cagr': metrics.cagr,
                'volatility': metrics.volatility,
                'win_rate': metrics.win_rate,
                'win_loss_ratio': metrics.win_loss_ratio,
                'profit_factor': metrics.profit_factor,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'best_day': metrics.best_day,
                'worst_day': metrics.worst_day,
                'best_month': metrics.best_month,
                'worst_month': metrics.worst_month,
                'avg_daily_return': metrics.avg_daily_return,
                'skewness': metrics.skewness,
                'kurtosis': metrics.kurtosis,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
            },
            'images': images,
        }
