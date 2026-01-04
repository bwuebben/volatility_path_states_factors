"""
Figure generation for the paper.

This module generates all figures for the paper including
state space visualizations, cumulative performance, and
factor analysis plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from .styles import PlotStyles, set_publication_style, add_regime_shading

logger = logging.getLogger(__name__)


class FigureGenerator:
    """
    Generate all figures for the paper.
    
    Parameters
    ----------
    output_dir : str, default 'output/figures'
        Directory to save figures.
    formats : list, default ['pdf']
        Output formats.
    dpi : int, default 300
        Resolution for raster formats.
        
    Examples
    --------
    >>> generator = FigureGenerator()
    >>> generator.figure1_state_space(states, regimes)
    >>> generator.figure2_cumulative_performance(factor_returns, regimes)
    """
    
    def __init__(
        self,
        output_dir: str = 'output/figures',
        formats: list = ['pdf'],
        dpi: int = 300,
    ):
        """Initialize figure generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats
        self.dpi = dpi
        
        # Set publication style
        set_publication_style()
        
    def save_figure(self, fig: plt.Figure, name: str):
        """Save figure in all configured formats."""
        for fmt in self.formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved: {path}")
        plt.close(fig)
    
    def figure1_state_space(
        self,
        states: pd.DataFrame,
        regimes: pd.Series,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 1: State Space Visualization.
        
        Panel A: Scatter plot of volatility level vs ratio
        Panel B: Time series of regime membership
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state variables with sigma_1m and rho_sigma columns.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save the figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1.2, 0.8])
        
        # Panel A: State Space
        ax1 = axes[0]
        
        for regime in PlotStyles.REGIME_ORDER:
            mask = regimes == regime
            if not mask.any():
                continue
            
            ax1.scatter(
                states.loc[mask, 'sigma_1m'] * 100,
                states.loc[mask, 'rho_sigma'],
                c=PlotStyles.get_regime_color(regime),
                label=regime,
                alpha=0.6,
                s=20,
                edgecolors='none',
            )
        
        # Add threshold lines
        vol_33 = np.percentile(states['sigma_1m'].dropna(), 33) * 100
        vol_67 = np.percentile(states['sigma_1m'].dropna(), 67) * 100
        
        ax1.axvline(x=vol_33, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axvline(x=vol_67, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axhline(y=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        ax1.set_xlabel('One-Month Realized Volatility (%)')
        ax1.set_ylabel('Volatility Ratio ($\\rho^{\\sigma}$)')
        ax1.set_title('Panel A: State Space with Regime Boundaries', fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        
        # Panel B: Time series
        ax2 = axes[1]
        
        dates = regimes.index
        for i in range(len(dates) - 1):
            regime = regimes.iloc[i]
            ax2.axvspan(dates[i], dates[i+1], alpha=0.7, 
                       color=PlotStyles.get_regime_color(regime), linewidth=0)
        
        ax2.set_xlim([dates[0], dates[-1]])
        ax2.set_ylim([0, 1])
        ax2.set_yticks([])
        ax2.set_xlabel('Date')
        ax2.set_title('Panel B: Regime Time Series', fontweight='bold')
        
        # Legend
        patches = [mpatches.Patch(color=PlotStyles.get_regime_color(r), 
                                  label=r, alpha=0.7) for r in PlotStyles.REGIME_ORDER]
        ax2.legend(handles=patches, loc='upper center', ncol=5, 
                  bbox_to_anchor=(0.5, -0.15), framealpha=0.9)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure1_state_space')
        
        return fig
    
    def figure2_cumulative_performance(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        factor: str = 'Momentum',
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 2: Cumulative Performance Comparison.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
        regimes : pd.Series
            Regime classification.
        factor : str, default 'Momentum'
            Factor to plot.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Align data
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx, factor]
        reg = regimes.loc[common_idx]
        
        # Unconditional cumulative returns
        cum_uncond = np.cumsum(returns) * 100
        
        # State-conditioned (reduce exposure in crash states)
        exposures = {
            'Calm Trend': 1.0,
            'Choppy Transition': 0.7,
            'Slow-Burn Stress': 0.5,
            'Crash-Spike': 0.0,
            'Recovery': 0.7,
        }
        exposure = reg.map(exposures)
        cond_returns = returns * exposure
        cum_cond = np.cumsum(cond_returns) * 100
        
        # Add crash-spike shading
        in_crash = False
        crash_start = None
        first_labeled = False
        
        for i, (date, regime) in enumerate(reg.items()):
            if regime == 'Crash-Spike' and not in_crash:
                crash_start = date
                in_crash = True
            elif regime != 'Crash-Spike' and in_crash:
                if not first_labeled:
                    ax.axvspan(crash_start, date, alpha=0.3, 
                              color=PlotStyles.REGIME_COLORS['Crash-Spike'],
                              label='Crash-Spike State')
                    first_labeled = True
                else:
                    ax.axvspan(crash_start, date, alpha=0.3,
                              color=PlotStyles.REGIME_COLORS['Crash-Spike'])
                in_crash = False
        
        # Plot cumulative returns
        ax.plot(cum_uncond.index, cum_uncond.values, 'b--', 
                linewidth=1.5, label='Unconditional', alpha=0.8)
        ax.plot(cum_cond.index, cum_cond.values, 'b-', 
                linewidth=2, label='State-Conditioned')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Log Return (%)')
        ax.set_title(f'Cumulative Performance: Unconditional vs. State-Conditioned {factor}',
                    fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure2_cumulative_performance')
        
        return fig
    
    def figure3_dispersion(
        self,
        return_dispersion: pd.Series,
        fundamental_dispersion: pd.Series,
        regimes: pd.Series,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 3: Cross-Sectional Dispersion.
        
        Parameters
        ----------
        return_dispersion : pd.Series
            Cross-sectional return dispersion.
        fundamental_dispersion : pd.Series
            Cross-sectional fundamental dispersion.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Align data
        common_idx = return_dispersion.index.intersection(
            fundamental_dispersion.index
        ).intersection(regimes.index)
        
        ret_disp = return_dispersion.loc[common_idx] * 100
        fund_disp = fundamental_dispersion.loc[common_idx] * 100
        reg = regimes.loc[common_idx]
        
        # Plot by regime
        for regime in PlotStyles.REGIME_ORDER:
            mask = reg == regime
            if not mask.any():
                continue
            
            ax.scatter(
                fund_disp[mask],
                ret_disp[mask],
                c=PlotStyles.get_regime_color(regime),
                label=regime,
                alpha=0.6,
                s=25,
                edgecolors='none',
            )
        
        # Fit regression to Calm Trend
        calm_mask = reg == 'Calm Trend'
        if calm_mask.any():
            x = fund_disp[calm_mask]
            y = ret_disp[calm_mask]
            slope = np.cov(x, y)[0, 1] / np.var(x)
            intercept = y.mean() - slope * x.mean()
            
            x_line = np.linspace(fund_disp.min(), fund_disp.max(), 100)
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, 'k--', linewidth=1.5, 
                   label='Fit (Calm Trend)', alpha=0.7)
        
        ax.set_xlabel('Cross-Sectional Profitability Dispersion (%)')
        ax.set_ylabel('Cross-Sectional Return Dispersion (%)')
        ax.set_title('Cross-Sectional Dispersion: Returns vs. Fundamentals',
                    fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure3_dispersion')
        
        return fig
    
    def figure4_factor_returns_by_state(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 4: Factor Returns by State (Bar Chart).
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Align data
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        factors = list(returns.columns)
        
        # Compute means and SEs by regime
        means = {}
        stes = {}
        
        for factor in factors:
            means[factor] = {}
            stes[factor] = {}
            
            for regime in PlotStyles.REGIME_ORDER:
                mask = reg == regime
                r = returns.loc[mask, factor]
                means[factor][regime] = r.mean() * 100
                stes[factor][regime] = r.std() / np.sqrt(len(r)) * 100
        
        # Bar positions
        x = np.arange(len(PlotStyles.REGIME_ORDER))
        width = 0.2
        
        for i, factor in enumerate(factors):
            offset = width * i
            vals = [means[factor][r] for r in PlotStyles.REGIME_ORDER]
            errs = [stes[factor][r] for r in PlotStyles.REGIME_ORDER]
            
            ax.bar(x + offset, vals, width, label=factor,
                  color=PlotStyles.get_factor_color(factor),
                  alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.errorbar(x + offset, vals, yerr=errs, fmt='none',
                       color='black', capsize=2, linewidth=0.8)
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Path State Regime')
        ax.set_ylabel('Mean Monthly Return (%)')
        ax.set_title('Factor Returns by Path State', fontweight='bold')
        ax.set_xticks(x + width * (len(factors) - 1) / 2)
        ax.set_xticklabels(PlotStyles.REGIME_ORDER, rotation=15, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure4_factor_returns')
        
        return fig
    
    def figure5_ic_timeseries(
        self,
        ic_data: pd.DataFrame,
        regimes: pd.Series,
        window: int = 12,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 5: Information Coefficient Time Series.
        
        Parameters
        ----------
        ic_data : pd.DataFrame
            IC data by factor.
        regimes : pd.Series
            Regime classification.
        window : int, default 12
            Rolling window for smoothing.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        factors = list(ic_data.columns)
        n_factors = len(factors)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        common_idx = ic_data.index.intersection(regimes.index)
        ic = ic_data.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        for idx, factor in enumerate(factors[:4]):
            ax = axes[idx]
            
            ic_smooth = ic[factor].rolling(window, min_periods=1).mean()
            
            # Regime shading
            add_regime_shading(ax, reg, alpha=0.15)
            
            # Plot IC
            ax.plot(ic_smooth.index, ic_smooth.values,
                   color=PlotStyles.get_factor_color(factor),
                   linewidth=1.5, alpha=0.9)
            
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_title(factor, fontweight='bold')
            ax.set_ylabel('Information Coefficient')
            ax.set_ylim([-0.15, 0.15])
            
            if idx >= 2:
                ax.set_xlabel('Date')
        
        # Legend
        patches = [mpatches.Patch(color=PlotStyles.get_regime_color(r),
                                  label=r, alpha=0.3) for r in PlotStyles.REGIME_ORDER]
        fig.legend(handles=patches, loc='upper center', ncol=5,
                  bbox_to_anchor=(0.5, 0.02), framealpha=0.9)
        
        plt.suptitle('Information Coefficients by Factor (12-Month Rolling Average)',
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure5_ic_timeseries')
        
        return fig
    
    def figure6_drawdown(
        self,
        factor_returns: pd.Series,
        regimes: pd.Series,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 6: Drawdown Comparison.
        
        Parameters
        ----------
        factor_returns : pd.Series
            Factor returns.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        # Unconditional
        cum_uncond = np.cumsum(returns)
        running_max_uncond = np.maximum.accumulate(cum_uncond)
        dd_uncond = (running_max_uncond - cum_uncond) * 100
        
        # State-conditioned
        exposures = {
            'Calm Trend': 1.0, 'Choppy Transition': 0.7,
            'Slow-Burn Stress': 0.5, 'Crash-Spike': 0.0, 'Recovery': 0.7,
        }
        exposure = reg.map(exposures)
        cond_returns = returns * exposure
        cum_cond = np.cumsum(cond_returns)
        running_max_cond = np.maximum.accumulate(cum_cond)
        dd_cond = (running_max_cond - cum_cond) * 100
        
        # Panel A: Unconditional
        ax1 = axes[0]
        ax1.fill_between(returns.index, 0, -dd_uncond.values,
                        color=PlotStyles.FACTOR_COLORS['Momentum'], alpha=0.7)
        ax1.set_ylabel('Drawdown (%)')
        ax1.set_title('Panel A: Unconditional Momentum', fontweight='bold')
        ax1.set_ylim([-80, 5])
        ax1.axhline(y=0, color='black', linewidth=0.5)
        
        # Annotate max drawdown
        max_dd_idx = dd_uncond.idxmax()
        ax1.annotate(f'Max DD: {dd_uncond.max():.1f}%',
                    xy=(max_dd_idx, -dd_uncond.max()),
                    xytext=(max_dd_idx, -dd_uncond.max() - 15),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
        
        # Panel B: State-Conditioned
        ax2 = axes[1]
        ax2.fill_between(returns.index, 0, -dd_cond.values,
                        color=PlotStyles.FACTOR_COLORS['Momentum'], alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Panel B: State-Conditioned Momentum', fontweight='bold')
        ax2.set_ylim([-80, 5])
        ax2.axhline(y=0, color='black', linewidth=0.5)
        
        max_dd_idx_cond = dd_cond.idxmax()
        ax2.annotate(f'Max DD: {dd_cond.max():.1f}%',
                    xy=(max_dd_idx_cond, -dd_cond.max()),
                    xytext=(max_dd_idx_cond, -dd_cond.max() - 15),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure6_drawdown')
        
        return fig
    
    def figure7_regime_factor_panel(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        factors: List[str] = ['Momentum', 'Quality'],
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 7: Regime Classification and Factor Performance.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
        regimes : pd.Series
            Regime classification.
        factors : list
            Factors to plot.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        n_panels = 1 + len(factors)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2 + 3*len(factors)),
                                height_ratios=[0.3] + [1]*len(factors))
        
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        # Panel A: Regime classification
        ax1 = axes[0]
        for i in range(len(reg.index) - 1):
            regime = reg.iloc[i]
            ax1.axvspan(reg.index[i], reg.index[i+1], alpha=0.8,
                       color=PlotStyles.get_regime_color(regime), linewidth=0)
        
        ax1.set_xlim([reg.index[0], reg.index[-1]])
        ax1.set_ylim([0, 1])
        ax1.set_yticks([])
        ax1.set_title('Panel A: Regime Classification', fontweight='bold')
        ax1.set_xticklabels([])
        
        # Factor panels
        for idx, factor in enumerate(factors):
            ax = axes[idx + 1]
            
            cum_ret = np.cumsum(returns[factor]) * 100
            
            # Regime shading
            add_regime_shading(ax, reg, alpha=0.15)
            
            ax.plot(cum_ret.index, cum_ret.values,
                   color=PlotStyles.get_factor_color(factor), linewidth=1.5)
            ax.set_ylabel('Cumulative Return (%)')
            ax.set_title(f'Panel {chr(66+idx)}: {factor} Factor', fontweight='bold')
            ax.set_xlim([reg.index[0], reg.index[-1]])
            
            if idx == len(factors) - 1:
                ax.set_xlabel('Date')
            else:
                ax.set_xticklabels([])
        
        # Legend
        patches = [mpatches.Patch(color=PlotStyles.get_regime_color(r),
                                  label=r, alpha=0.7) for r in PlotStyles.REGIME_ORDER]
        fig.legend(handles=patches, loc='upper center', ncol=5,
                  bbox_to_anchor=(0.5, 0.02), framealpha=0.9)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure7_regime_factor_panel')
        
        return fig
    
    def figure8_volatility_ratio_distribution(
        self,
        rho_sigma: pd.Series,
        regimes: pd.Series,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate Figure 8: Volatility Ratio Distribution by Regime.
        
        Parameters
        ----------
        rho_sigma : pd.Series
            Volatility ratio series.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        common_idx = rho_sigma.index.intersection(regimes.index)
        rho = rho_sigma.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        bins = np.linspace(0.3, 2.5, 45)
        
        for regime in PlotStyles.REGIME_ORDER:
            mask = reg == regime
            if not mask.any():
                continue
            
            ax.hist(rho[mask], bins=bins, alpha=0.5, label=regime,
                   color=PlotStyles.get_regime_color(regime),
                   edgecolor='white', linewidth=0.5)
        
        # Threshold lines
        ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        
        y_max = ax.get_ylim()[1]
        ax.annotate('Decaying\n($\\rho^{\\sigma} < 0.8$)', xy=(0.55, y_max*0.85),
                   fontsize=9, ha='center', color='gray')
        ax.annotate('Sustained\n($0.8 \\leq \\rho^{\\sigma} \\leq 1.5$)', 
                   xy=(1.15, y_max*0.85), fontsize=9, ha='center', color='gray')
        ax.annotate('Spike\n($\\rho^{\\sigma} > 1.5$)', xy=(1.9, y_max*0.85),
                   fontsize=9, ha='center', color='gray')
        
        ax.set_xlabel('Volatility Ratio ($\\rho^{\\sigma}$)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Volatility Ratio by Regime', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim([0.3, 2.5])
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'figure8_vol_ratio_dist')
        
        return fig
    
    def generate_all_figures(
        self,
        data: Dict,
    ):
        """
        Generate all paper figures.
        
        Parameters
        ----------
        data : dict
            Dictionary containing all required data:
            - 'states': Path state variables
            - 'regimes': Regime classification
            - 'factors': Factor returns
            - 'ic': Information coefficients
        """
        logger.info("Generating all figures...")
        
        states = data.get('states', data.get('volatility'))
        regimes = data.get('regimes')
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        factors = data.get('factors')
        ic = data.get('ic')
        
        # Figure 1: State space
        self.figure1_state_space(states, regimes)
        
        # Figure 2: Cumulative performance
        self.figure2_cumulative_performance(factors, regimes)
        
        # Figure 3: Dispersion (if available)
        if 'return_dispersion' in data:
            self.figure3_dispersion(
                data['return_dispersion'],
                data['fundamental_dispersion'],
                regimes,
            )
        
        # Figure 4: Factor returns by state
        self.figure4_factor_returns_by_state(factors, regimes)
        
        # Figure 5: IC time series
        if ic is not None:
            self.figure5_ic_timeseries(ic, regimes)
        
        # Figure 6: Drawdown
        self.figure6_drawdown(factors['Momentum'], regimes)
        
        # Figure 7: Regime factor panel
        self.figure7_regime_factor_panel(factors, regimes)
        
        # Figure 8: Volatility ratio distribution
        if 'rho_sigma' in states.columns:
            self.figure8_volatility_ratio_distribution(states['rho_sigma'], regimes)
        
        logger.info("All figures generated successfully!")
