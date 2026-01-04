"""
Regime classification module.

This module provides functionality for computing volatility measures
and classifying multi-scale path states.

Classes
-------
VolatilityCalculator : Compute realized volatility at multiple horizons
PathStateClassifier : Construct multi-scale path states
RegimeClassifier : Classify path states into discrete regimes
"""

from .volatility import VolatilityCalculator
from .path_states import PathStateClassifier
from .regime_classifier import RegimeClassifier

__all__ = [
    "VolatilityCalculator",
    "PathStateClassifier", 
    "RegimeClassifier",
]
