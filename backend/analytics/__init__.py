"""
Analytics module for portfolio performance analysis.

Replaces abandoned PyFolio with QuantStats for modern portfolio analytics.
"""

from .quantstats_reporter import QuantStatsReporter

__all__ = ['QuantStatsReporter']
