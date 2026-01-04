"""
Analysis and statistics module.

This module provides functionality for computing performance metrics,
statistical tests, information coefficients, and robustness analysis.

Classes
-------
PerformanceAnalyzer : Compute performance metrics
StatisticalTests : Statistical hypothesis tests
InformationCoefficientAnalyzer : IC analysis
RobustnessAnalyzer : Robustness tests
"""

from .performance import PerformanceAnalyzer, PerformanceMetrics
from .statistics import StatisticalTests, TestResult
from .information_coefficient import InformationCoefficientAnalyzer
from .robustness import RobustnessAnalyzer, RobustnessResult

__all__ = [
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "StatisticalTests",
    "TestResult",
    "InformationCoefficientAnalyzer",
    "RobustnessAnalyzer",
    "RobustnessResult",
]
