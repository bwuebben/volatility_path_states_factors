"""
Portfolio construction module.

This module provides functionality for building factor portfolios,
implementing state-conditioned exposure rules, and conducting
backtests.

Classes
-------
BaselinePortfolio : Standard unconditional factor portfolios
StateConditionedPortfolio : Path-state conditioned portfolios
VolatilityScaledPortfolio : Volatility-managed portfolios
ExposureOptimizer : Optimize state-conditional exposures
"""

from .baseline import BaselinePortfolio
from .state_conditioned import StateConditionedPortfolio
from .volatility_scaling import VolatilityScaledPortfolio
from .optimizer import ExposureOptimizer

__all__ = [
    "BaselinePortfolio",
    "StateConditionedPortfolio",
    "VolatilityScaledPortfolio",
    "ExposureOptimizer",
]
