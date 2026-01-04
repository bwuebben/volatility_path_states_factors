"""
Data loading and preprocessing module.

This module provides functionality for loading market and factor data
from various sources including WRDS/CRSP, Yahoo Finance, and synthetic
data generation for testing.

Classes
-------
DataLoader : Base class for data loading
WRDSDataLoader : Load data from WRDS/CRSP
YahooDataLoader : Load data from Yahoo Finance
SyntheticDataGenerator : Generate synthetic data for testing
"""

from .data_loader import DataLoader
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    "DataLoader",
    "SyntheticDataGenerator",
]

# Optional imports that require additional dependencies
try:
    from .wrds_loader import WRDSDataLoader
    __all__.append("WRDSDataLoader")
except ImportError:
    pass
