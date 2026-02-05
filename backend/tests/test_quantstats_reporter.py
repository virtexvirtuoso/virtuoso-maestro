"""
Tests for QuantStats Reporter - Portfolio Analytics Module.

Validates:
- Metrics calculation correctness
- 365-day annualization for crypto
- HTML report generation
- Base64 image encoding
- Edge cases and error handling
"""

import base64
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.quantstats_reporter import (
    ANNUALIZATION_FACTOR,
    QuantStatsMetrics,
    QuantStatsReporter,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample daily returns series"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    # Realistic crypto-like returns: mean ~0.05% daily, 3% daily std
    returns = pd.Series(
        np.random.normal(0.0005, 0.03, len(dates)),
        index=dates
    )
    return returns


@pytest.fixture
def positive_returns():
    """Consistently positive returns for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns = pd.Series(
        [0.01, 0.015, 0.005, 0.02, 0.008] * 20,
        index=dates
    )
    return returns


@pytest.fixture
def negative_returns():
    """Consistently negative returns for testing"""
    np.random.seed(123)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns = pd.Series(
        np.random.normal(-0.01, 0.015, 100),
        index=dates
    )
    return returns


@pytest.fixture
def volatile_returns():
    """High volatility returns"""
    np.random.seed(456)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    returns = pd.Series(
        np.random.normal(0.001, 0.05, 200),
        index=dates
    )
    return returns


@pytest.fixture
def short_returns():
    """Very short returns series (edge case)"""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    returns = pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015, -0.005, 0.01, 0.005, -0.002, 0.008],
        index=dates
    )
    return returns


# =============================================================================
# Module Constants Tests
# =============================================================================

class TestAnnualizationFactor:
    """Test the annualization factor constant"""

    def test_annualization_factor_is_365(self):
        """Crypto uses 365-day annualization, not 252"""
        assert ANNUALIZATION_FACTOR == 365

    def test_annualization_factor_used_by_default(self, sample_returns):
        """Reporter should use 365-day factor by default"""
        reporter = QuantStatsReporter(returns=sample_returns)
        assert reporter.periods_per_year == 365


# =============================================================================
# Initialization Tests
# =============================================================================

class TestQuantStatsReporterInit:
    """Test QuantStatsReporter initialization"""

    def test_init_with_valid_returns(self, sample_returns):
        """Should initialize with valid returns"""
        reporter = QuantStatsReporter(returns=sample_returns)
        assert reporter.returns is not None
        assert len(reporter.returns) > 0

    def test_init_with_empty_returns_raises(self):
        """Should raise error for empty returns"""
        with pytest.raises(ValueError, match="cannot be empty"):
            QuantStatsReporter(returns=pd.Series([]))

    def test_init_with_none_returns_raises(self):
        """Should raise error for None returns"""
        with pytest.raises(ValueError, match="cannot be empty"):
            QuantStatsReporter(returns=None)

    def test_init_with_benchmark(self, sample_returns):
        """Should accept benchmark returns"""
        np.random.seed(999)
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.02, len(sample_returns)),
            index=sample_returns.index
        )
        reporter = QuantStatsReporter(returns=sample_returns, benchmark=benchmark)
        assert reporter.benchmark is not None

    def test_init_with_custom_rf(self, sample_returns):
        """Should accept custom risk-free rate"""
        reporter = QuantStatsReporter(returns=sample_returns, rf=0.04)
        assert reporter.rf == 0.04

    def test_init_with_custom_periods(self, sample_returns):
        """Should accept custom periods per year"""
        reporter = QuantStatsReporter(returns=sample_returns, periods_per_year=252)
        assert reporter.periods_per_year == 252

    def test_init_cleans_nan_values(self):
        """Should clean NaN values from returns"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        returns_with_nan = pd.Series(
            [0.01, np.nan, 0.02, np.nan, 0.015, 0.01, np.nan, 0.005, 0.02, 0.01],
            index=dates
        )
        reporter = QuantStatsReporter(returns=returns_with_nan)
        assert not reporter.returns.isna().any()

    def test_init_cleans_inf_values(self):
        """Should clean infinite values from returns"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        returns_with_inf = pd.Series(
            [0.01, np.inf, 0.02, -np.inf, 0.015, 0.01, 0.005, 0.005, 0.02, 0.01],
            index=dates
        )
        reporter = QuantStatsReporter(returns=returns_with_inf)
        assert not np.isinf(reporter.returns).any()


# =============================================================================
# Metrics Calculation Tests
# =============================================================================

class TestGetMetrics:
    """Test metrics calculation"""

    def test_get_metrics_returns_dataclass(self, sample_returns):
        """Should return QuantStatsMetrics dataclass"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics, QuantStatsMetrics)

    def test_sharpe_ratio_calculated(self, sample_returns):
        """Should calculate Sharpe ratio"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.sharpe_ratio, float)
        assert not np.isnan(metrics.sharpe_ratio)

    def test_sortino_ratio_calculated(self, sample_returns):
        """Should calculate Sortino ratio"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.sortino_ratio, float)
        assert not np.isnan(metrics.sortino_ratio)

    def test_calmar_ratio_calculated(self, sample_returns):
        """Should calculate Calmar ratio"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.calmar_ratio, float)

    def test_max_drawdown_calculated(self, sample_returns):
        """Should calculate max drawdown"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.max_drawdown, float)
        # Max drawdown should be negative or zero
        assert metrics.max_drawdown <= 0

    def test_total_return_calculated(self, sample_returns):
        """Should calculate total return"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.total_return, float)

    def test_win_rate_calculated(self, sample_returns):
        """Should calculate win rate"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.win_rate, float)
        # Win rate should be between 0 and 1
        assert 0 <= metrics.win_rate <= 1

    def test_var_cvar_calculated(self, sample_returns):
        """Should calculate VaR and CVaR"""
        reporter = QuantStatsReporter(returns=sample_returns)
        metrics = reporter.get_metrics()
        assert isinstance(metrics.var_95, float)
        assert isinstance(metrics.cvar_95, float)

    def test_positive_returns_positive_sharpe(self, positive_returns):
        """Positive returns should have positive Sharpe"""
        reporter = QuantStatsReporter(returns=positive_returns)
        metrics = reporter.get_metrics()
        assert metrics.sharpe_ratio > 0

    def test_negative_returns_negative_sharpe(self, negative_returns):
        """Negative returns should have negative Sharpe"""
        reporter = QuantStatsReporter(returns=negative_returns)
        metrics = reporter.get_metrics()
        assert metrics.sharpe_ratio < 0

    def test_metrics_with_short_series(self, short_returns):
        """Should handle short returns series"""
        reporter = QuantStatsReporter(returns=short_returns)
        metrics = reporter.get_metrics()
        # Should not raise and should return valid metrics
        assert isinstance(metrics.sharpe_ratio, float)


# =============================================================================
# Annualization Validation Tests
# =============================================================================

class TestAnnualizationValidation:
    """Test that 365-day annualization is properly applied"""

    def test_sharpe_uses_365_factor(self, sample_returns):
        """Sharpe should use 365-day annualization"""
        # Create two reporters with different annualization
        reporter_365 = QuantStatsReporter(returns=sample_returns, periods_per_year=365)
        reporter_252 = QuantStatsReporter(returns=sample_returns, periods_per_year=252)

        metrics_365 = reporter_365.get_metrics()
        metrics_252 = reporter_252.get_metrics()

        # Different annualization should give different Sharpe ratios
        # (unless both are zero)
        if metrics_365.sharpe_ratio != 0 and metrics_252.sharpe_ratio != 0:
            assert metrics_365.sharpe_ratio != metrics_252.sharpe_ratio

    def test_volatility_uses_365_factor(self, sample_returns):
        """Volatility should use 365-day annualization"""
        reporter_365 = QuantStatsReporter(returns=sample_returns, periods_per_year=365)
        reporter_252 = QuantStatsReporter(returns=sample_returns, periods_per_year=252)

        metrics_365 = reporter_365.get_metrics()
        metrics_252 = reporter_252.get_metrics()

        # 365-day annualized volatility should be higher than 252-day
        # because sqrt(365) > sqrt(252)
        if metrics_365.volatility != 0 and metrics_252.volatility != 0:
            ratio = metrics_365.volatility / metrics_252.volatility
            expected_ratio = np.sqrt(365) / np.sqrt(252)
            assert abs(ratio - expected_ratio) < 0.1  # Allow 10% tolerance

    def test_cagr_uses_365_factor(self, sample_returns):
        """CAGR should use 365-day annualization"""
        reporter_365 = QuantStatsReporter(returns=sample_returns, periods_per_year=365)
        reporter_252 = QuantStatsReporter(returns=sample_returns, periods_per_year=252)

        metrics_365 = reporter_365.get_metrics()
        metrics_252 = reporter_252.get_metrics()

        # CAGR should differ based on annualization assumption
        if metrics_365.cagr != 0 and metrics_252.cagr != 0:
            assert metrics_365.cagr != metrics_252.cagr


# =============================================================================
# Manual Calculation Comparison Tests
# =============================================================================

class TestManualCalculations:
    """Compare QuantStats metrics to manual calculations"""

    def test_sharpe_manual_calculation(self):
        """Verify Sharpe ratio against manual calculation"""
        # Create returns with known statistics
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        daily_return = 0.001  # 0.1% daily
        daily_std = 0.02  # 2% daily std

        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(daily_return, daily_std, 365),
            index=dates
        )

        reporter = QuantStatsReporter(returns=returns, rf=0.0, periods_per_year=365)
        metrics = reporter.get_metrics()

        # Manual calculation
        actual_mean = returns.mean()
        actual_std = returns.std()
        manual_sharpe = (actual_mean * np.sqrt(365)) / actual_std

        # Should be within 20% (QuantStats may use slightly different formula)
        if metrics.sharpe_ratio != 0:
            ratio = metrics.sharpe_ratio / manual_sharpe
            assert 0.8 < ratio < 1.2

    def test_max_drawdown_manual_calculation(self):
        """Verify max drawdown against manual calculation"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        # Returns that create a known drawdown
        returns = pd.Series(
            [0.10, 0.05, -0.15, -0.10, 0.05, 0.03, -0.05, 0.02, 0.04, 0.01],
            index=dates
        )

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        # Manual calculation of max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        manual_max_dd = drawdown.min()

        # Should match within reasonable tolerance
        assert abs(metrics.max_drawdown - manual_max_dd) < 0.05

    def test_win_rate_manual_calculation(self):
        """Verify win rate against manual calculation"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        returns = pd.Series(
            [0.01, -0.01, 0.02, -0.005, 0.015, -0.01, 0.01, 0.005, -0.002, 0.008],
            index=dates
        )

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        # Manual win rate: 7 positive returns out of 10
        manual_win_rate = (returns > 0).sum() / len(returns)

        assert abs(metrics.win_rate - manual_win_rate) < 0.01


# =============================================================================
# HTML Report Tests
# =============================================================================

class TestHTMLReport:
    """Test HTML report generation"""

    def test_generate_html_report_returns_string(self, sample_returns):
        """Should return HTML string"""
        reporter = QuantStatsReporter(returns=sample_returns)
        html = reporter.generate_html_report()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_required_elements(self, sample_returns):
        """HTML should contain key elements"""
        reporter = QuantStatsReporter(returns=sample_returns)
        html = reporter.generate_html_report(title="Test Report")

        # Should contain basic HTML structure
        assert '<html' in html.lower() or '<!doctype' in html.lower()

    def test_html_with_custom_title(self, sample_returns):
        """Should include custom title in report"""
        reporter = QuantStatsReporter(returns=sample_returns)
        html = reporter.generate_html_report(title="My Custom Report")

        # Title should appear somewhere in the HTML
        assert "My Custom Report" in html or "custom" in html.lower()


# =============================================================================
# Tearsheet Images Tests
# =============================================================================

class TestTearsheetImages:
    """Test tearsheet image generation"""

    def test_generate_images_returns_dict(self, sample_returns):
        """Should return dictionary of images"""
        reporter = QuantStatsReporter(returns=sample_returns)
        images = reporter.generate_tearsheet_images()
        assert isinstance(images, dict)

    def test_images_are_base64_encoded(self, sample_returns):
        """Images should be valid base64 strings"""
        reporter = QuantStatsReporter(returns=sample_returns)
        images = reporter.generate_tearsheet_images()

        for name, img_data in images.items():
            # Should be a string
            assert isinstance(img_data, str)
            # Should be valid base64
            try:
                decoded = base64.b64decode(img_data)
                # Should be PNG (starts with PNG magic bytes)
                assert decoded[:4] == b'\x89PNG'
            except Exception as e:
                pytest.fail(f"Image {name} is not valid base64 PNG: {e}")

    def test_key_images_generated(self, sample_returns):
        """Should generate key visualization images"""
        reporter = QuantStatsReporter(returns=sample_returns)
        images = reporter.generate_tearsheet_images()

        expected_keys = ['cumulative_returns', 'drawdown', 'monthly_heatmap']
        for key in expected_keys:
            assert key in images, f"Missing expected image: {key}"


# =============================================================================
# Get Report Data Tests
# =============================================================================

class TestGetReportData:
    """Test complete report data retrieval"""

    def test_get_report_data_structure(self, sample_returns):
        """Should return properly structured data"""
        reporter = QuantStatsReporter(returns=sample_returns)
        data = reporter.get_report_data()

        assert 'title' in data
        assert 'annualization_factor' in data
        assert 'metrics' in data
        assert 'images' in data

    def test_report_data_contains_all_metrics(self, sample_returns):
        """Should contain all key metrics"""
        reporter = QuantStatsReporter(returns=sample_returns)
        data = reporter.get_report_data()

        expected_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'total_return', 'volatility', 'win_rate'
        ]
        for metric in expected_metrics:
            assert metric in data['metrics'], f"Missing metric: {metric}"

    def test_report_data_annualization_factor(self, sample_returns):
        """Report data should show 365-day factor"""
        reporter = QuantStatsReporter(returns=sample_returns)
        data = reporter.get_report_data()

        assert data['annualization_factor'] == 365


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_short_series(self):
        """Should handle very short series gracefully"""
        dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
        returns = pd.Series([0.01, -0.005, 0.02], index=dates)

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        # Should not raise, metrics may be zero or limited
        assert isinstance(metrics.sharpe_ratio, float)

    def test_all_positive_returns(self):
        """Should handle all positive returns"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 100, index=dates)

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        assert metrics.win_rate == 1.0
        assert metrics.max_drawdown == 0.0

    def test_all_negative_returns(self):
        """Should handle all negative returns"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.Series([-0.01] * 100, index=dates)

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        assert metrics.win_rate == 0.0
        assert metrics.max_drawdown < 0

    def test_zero_returns(self):
        """Should handle zero returns"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.Series([0.0] * 100, index=dates)

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        # Should not raise
        assert isinstance(metrics.sharpe_ratio, float)

    def test_extreme_returns(self):
        """Should handle extreme but valid returns"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        # Extreme but realistic crypto returns
        returns = pd.Series(
            [0.5, -0.3, 0.4, -0.25, 0.3] * 10,
            index=dates
        )

        reporter = QuantStatsReporter(returns=returns)
        metrics = reporter.get_metrics()

        # Should calculate valid metrics
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.volatility)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with realistic scenarios"""

    def test_full_workflow(self, sample_returns):
        """Test complete workflow: init -> metrics -> report"""
        # Initialize
        reporter = QuantStatsReporter(
            returns=sample_returns,
            rf=0.02,
            periods_per_year=365
        )

        # Get metrics
        metrics = reporter.get_metrics()
        assert isinstance(metrics, QuantStatsMetrics)

        # Generate report data
        report_data = reporter.get_report_data()
        assert 'metrics' in report_data
        assert 'images' in report_data

        # Generate HTML (optional, slower)
        html = reporter.generate_html_report()
        assert len(html) > 0

    def test_comparison_with_benchmark(self, sample_returns):
        """Test metrics with benchmark comparison"""
        np.random.seed(999)
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.02, len(sample_returns)),
            index=sample_returns.index
        )

        reporter = QuantStatsReporter(
            returns=sample_returns,
            benchmark=benchmark
        )

        metrics = reporter.get_metrics()
        # Should calculate without errors
        assert isinstance(metrics.sharpe_ratio, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
