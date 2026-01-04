"""
Plot styling utilities for publication-quality figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Optional, Tuple
import numpy as np


class PlotStyles:
    """
    Consistent styling for all figures.
    
    Attributes
    ----------
    REGIME_COLORS : dict
        Color mapping for regimes.
    FACTOR_COLORS : dict
        Color mapping for factors.
    REGIME_ORDER : list
        Canonical ordering of regimes.
    """
    
    # Regime colors
    REGIME_COLORS = {
        'Calm Trend': '#2E86AB',        # Blue
        'Choppy Transition': '#A7C957', # Green
        'Slow-Burn Stress': '#F9C74F',  # Yellow/Orange
        'Crash-Spike': '#E63946',       # Red
        'Recovery': '#9B5DE5',          # Purple
    }
    
    # Factor colors
    FACTOR_COLORS = {
        'Momentum': '#E63946',   # Red
        'Value': '#2E86AB',      # Blue
        'Quality': '#2A9D8F',    # Teal
        'Low-Risk': '#9B5DE5',   # Purple
        'MOM': '#E63946',
        'HML': '#2E86AB',
        'RMW': '#2A9D8F',
        'BAB': '#9B5DE5',
    }
    
    # Canonical regime ordering
    REGIME_ORDER = [
        'Calm Trend',
        'Choppy Transition',
        'Slow-Burn Stress',
        'Crash-Spike',
        'Recovery',
    ]
    
    # Default figure sizes
    FIGURE_SIZES = {
        'single': (8, 5),
        'double': (10, 6),
        'wide': (12, 5),
        'tall': (8, 8),
        'panel_2x1': (10, 8),
        'panel_2x2': (12, 10),
        'panel_3x1': (12, 10),
    }
    
    @classmethod
    def get_regime_color(cls, regime: str) -> str:
        """Get color for a regime."""
        return cls.REGIME_COLORS.get(regime, '#808080')
    
    @classmethod
    def get_factor_color(cls, factor: str) -> str:
        """Get color for a factor."""
        return cls.FACTOR_COLORS.get(factor, '#808080')
    
    @classmethod
    def get_figure_size(cls, size_name: str) -> Tuple[float, float]:
        """Get figure size by name."""
        return cls.FIGURE_SIZES.get(size_name, (8, 5))


def set_publication_style():
    """
    Set matplotlib style for publication-quality figures.
    
    This function configures matplotlib with appropriate settings
    for academic publication, including font sizes, line widths,
    and grid styling.
    """
    # Use a clean base style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Override with publication settings
    mpl.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        
        # Use LaTeX if available
        'text.usetex': False,  # Set to True if LaTeX is available
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        
        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,
        
        # Legend settings
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Figure settings
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Color cycle
        'axes.prop_cycle': mpl.cycler(
            color=['#2E86AB', '#E63946', '#2A9D8F', '#F9C74F', '#9B5DE5', '#264653']
        ),
    })


def create_regime_colormap():
    """
    Create a colormap for regime visualization.
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Colormap transitioning through regime colors.
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [PlotStyles.REGIME_COLORS[r] for r in PlotStyles.REGIME_ORDER]
    return LinearSegmentedColormap.from_list('regime', colors, N=256)


def add_regime_shading(
    ax: plt.Axes,
    regimes: 'pd.Series',
    alpha: float = 0.2,
    label_first: bool = True,
):
    """
    Add regime background shading to a plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add shading to.
    regimes : pd.Series
        Regime classification series.
    alpha : float, default 0.2
        Shading transparency.
    label_first : bool, default True
        Only label the first occurrence of each regime.
    """
    import pandas as pd
    
    labeled = set()
    dates = regimes.index
    
    for i in range(len(dates) - 1):
        regime = regimes.iloc[i]
        color = PlotStyles.get_regime_color(regime)
        
        if label_first and regime not in labeled:
            ax.axvspan(dates[i], dates[i+1], alpha=alpha, color=color, 
                      label=regime, linewidth=0)
            labeled.add(regime)
        else:
            ax.axvspan(dates[i], dates[i+1], alpha=alpha, color=color, linewidth=0)


def add_recession_shading(
    ax: plt.Axes,
    recessions: 'pd.Series',
    color: str = 'gray',
    alpha: float = 0.2,
):
    """
    Add NBER recession shading to a plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add shading to.
    recessions : pd.Series
        Boolean series indicating recession periods.
    color : str, default 'gray'
        Shading color.
    alpha : float, default 0.2
        Shading transparency.
    """
    in_recession = False
    start = None
    
    for date, is_recession in recessions.items():
        if is_recession and not in_recession:
            start = date
            in_recession = True
        elif not is_recession and in_recession:
            ax.axvspan(start, date, alpha=alpha, color=color, linewidth=0)
            in_recession = False
    
    # Handle recession extending to end
    if in_recession:
        ax.axvspan(start, recessions.index[-1], alpha=alpha, color=color, linewidth=0)


def format_axis_pct(ax: plt.Axes, axis: str = 'y'):
    """
    Format axis labels as percentages.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    axis : str, default 'y'
        Which axis to format: 'x', 'y', or 'both'.
    """
    from matplotlib.ticker import FuncFormatter
    
    pct_formatter = FuncFormatter(lambda x, p: f'{x:.0f}%')
    
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(pct_formatter)
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(pct_formatter)


def create_legend_patches(
    items: Dict[str, str],
    alpha: float = 0.7,
):
    """
    Create legend patches for regime/factor legends.
    
    Parameters
    ----------
    items : dict
        Mapping of names to colors.
    alpha : float, default 0.7
        Patch transparency.
        
    Returns
    -------
    list
        List of matplotlib patches.
    """
    import matplotlib.patches as mpatches
    
    return [
        mpatches.Patch(color=color, label=name, alpha=alpha)
        for name, color in items.items()
    ]


def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: list = ['pdf', 'png'],
    dpi: int = 300,
):
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Base filename (without extension).
    formats : list, default ['pdf', 'png']
        Output formats.
    dpi : int, default 300
        Resolution for raster formats.
    """
    for fmt in formats:
        fig.savefig(
            f"{filename}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches='tight',
        )


# Color utilities
def lighten_color(color: str, amount: float = 0.5) -> Tuple[float, float, float]:
    """
    Lighten a color by mixing with white.
    
    Parameters
    ----------
    color : str
        Color in any matplotlib format.
    amount : float, default 0.5
        Amount to lighten (0=original, 1=white).
        
    Returns
    -------
    tuple
        RGB color tuple.
    """
    import matplotlib.colors as mc
    
    c = mc.to_rgb(color)
    return tuple(
        min(1, ch + (1 - ch) * amount) for ch in c
    )


def darken_color(color: str, amount: float = 0.3) -> Tuple[float, float, float]:
    """
    Darken a color by mixing with black.
    
    Parameters
    ----------
    color : str
        Color in any matplotlib format.
    amount : float, default 0.3
        Amount to darken (0=original, 1=black).
        
    Returns
    -------
    tuple
        RGB color tuple.
    """
    import matplotlib.colors as mc
    
    c = mc.to_rgb(color)
    return tuple(
        max(0, ch * (1 - amount)) for ch in c
    )
