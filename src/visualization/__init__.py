"""
Visualization module.

This module provides functionality for creating publication-quality
figures and tables for the paper.

Classes
-------
FigureGenerator : Generate paper figures
TableGenerator : Generate LaTeX tables
PlotStyles : Consistent plot styling
"""

from .figures import FigureGenerator
from .tables import TableGenerator
from .styles import PlotStyles, set_publication_style

__all__ = [
    "FigureGenerator",
    "TableGenerator",
    "PlotStyles",
    "set_publication_style",
]
