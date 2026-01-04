"""
Volatility Path States
======================

A Python package for analyzing multi-scale path states and their impact
on equity factor performance.

This package implements the methodology from:
"How Volatility Gets Here: Multi-Scale Path States and the 
Conditional Performance of Equity Factors"

Modules
-------
data : Data loading and preprocessing
factors : Factor construction
regimes : Path state classification
portfolio : Portfolio construction and management
analysis : Performance analysis and statistics
visualization : Plotting and figure generation
utils : Utility functions

Example
-------
>>> from src import PathStateClassifier, StateConditionedPortfolio
>>> from src.data import SyntheticDataGenerator
>>> 
>>> # Generate data
>>> data = SyntheticDataGenerator().generate()
>>> 
>>> # Classify regimes
>>> classifier = PathStateClassifier()
>>> regimes = classifier.classify(data['market'])
>>> 
>>> # Build portfolio
>>> portfolio = StateConditionedPortfolio(data['factors'], regimes)
>>> results = portfolio.backtest()
"""

__version__ = "1.0.0"
__author__ = "Author Name"
__email__ = "author@institution.edu"

# Import main classes for convenient access
from .config import Config
from .regimes.path_states import PathStateClassifier
from .regimes.regime_classifier import RegimeClassifier
from .portfolio.state_conditioned import StateConditionedPortfolio
from .portfolio.baseline import BaselinePortfolio
from .analysis.performance import PerformanceAnalyzer

__all__ = [
    "Config",
    "PathStateClassifier",
    "RegimeClassifier",
    "StateConditionedPortfolio",
    "BaselinePortfolio",
    "PerformanceAnalyzer",
]
