"""
Factor construction module.

This module provides functionality for constructing equity factors
from CRSP and Compustat data.

Classes
-------
FactorBuilder : Build multiple factors
MomentumFactor : Momentum factor construction
ValueFactor : Value factor construction
QualityFactor : Quality factor construction
LowRiskFactor : Low-risk/beta factor construction
"""

from .factor_builder import FactorBuilder
from .momentum import MomentumFactor
from .value import ValueFactor
from .quality import QualityFactor
from .low_risk import LowRiskFactor

__all__ = [
    "FactorBuilder",
    "MomentumFactor",
    "ValueFactor",
    "QualityFactor",
    "LowRiskFactor",
]
