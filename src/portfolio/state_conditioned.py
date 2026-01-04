"""
State-conditioned portfolio implementation.

This module implements portfolios that adjust factor exposure
based on the current path state regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
import logging

from .baseline import BaselinePortfolio, PortfolioResult
from ..regimes.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class StateConditionedPortfolio:
    """
    Factor portfolio with state-conditioned exposures.
    
    This class adjusts factor exposure based on the current path state
    regime, implementing the methodology from the paper.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor return time series.
    regimes : pd.Series
        Regime classification for each period.
    exposures : dict, optional
        Exposure values by factor and regime. If None, uses defaults.
    target_volatility : float, optional
        Target volatility for scaling.
    transaction_cost : float, default 0.002
        One-way transaction cost.
        
    Attributes
    ----------
    exposures : dict
        State-conditional exposure values.
    effective_returns : pd.DataFrame
        Factor returns adjusted for exposures.
        
    Examples
    --------
    >>> portfolio = StateConditionedPortfolio(
    ...     factor_returns=factors,
    ...     regimes=regimes,
    ... )
    >>> portfolio.fit(training_end='1999-12-31')
    >>> result = portfolio.backtest(start='2000-01-01')
    >>> print(result.returns['net'].mean() * 12)
    """
    
    # Default exposure values from the paper
    DEFAULT_EXPOSURES = {
        'Momentum': {
            'Calm Trend': 1.0,
            'Choppy Transition': 0.7,
            'Slow-Burn Stress': 0.5,
            'Crash-Spike': 0.0,
            'Recovery': 0.7,
        },
        'Value': {
            'Calm Trend': 1.0,
            'Choppy Transition': 1.0,
            'Slow-Burn Stress': 0.8,
            'Crash-Spike': 0.4,
            'Recovery': 1.0,
        },
        'Quality': {
            'Calm Trend': 1.0,
            'Choppy Transition': 1.0,
            'Slow-Burn Stress': 1.0,
            'Crash-Spike': 1.0,
            'Recovery': 0.85,
        },
        'Low-Risk': {
            'Calm Trend': 1.0,
            'Choppy Transition': 1.0,
            'Slow-Burn Stress': 1.0,
            'Crash-Spike': 1.0,
            'Recovery': 0.75,
        },
    }
    
    REGIME_ORDER = [
        'Calm Trend',
        'Choppy Transition',
        'Slow-Burn Stress',
        'Crash-Spike',
        'Recovery',
    ]
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        exposures: Optional[Dict[str, Dict[str, float]]] = None,
        target_volatility: Optional[float] = None,
        transaction_cost: float = 0.002,
    ):
        """Initialize state-conditioned portfolio."""
        self.factor_returns = factor_returns.copy()
        self.regimes = regimes.copy()
        self.target_volatility = target_volatility
        self.transaction_cost = transaction_cost
        
        self.factor_names = list(factor_returns.columns)
        
        # Initialize exposures
        self.exposures = exposures or self._default_exposures()
        
        # Align data
        self._align_data()
        
        self._fitted = False
        
    def _default_exposures(self) -> Dict[str, Dict[str, float]]:
        """Get default exposures, filling in for unknown factors."""
        exposures = {}
        for factor in self.factor_names:
            if factor in self.DEFAULT_EXPOSURES:
                exposures[factor] = self.DEFAULT_EXPOSURES[factor].copy()
            else:
                # Default to full exposure for unknown factors
                exposures[factor] = {r: 1.0 for r in self.REGIME_ORDER}
        return exposures
    
    def _align_data(self):
        """Align factor returns and regimes."""
        common_index = self.factor_returns.index.intersection(self.regimes.index)
        self.factor_returns = self.factor_returns.loc[common_index]
        self.regimes = self.regimes.loc[common_index]
        
    def fit(
        self,
        training_end: str,
        regularization: float = 0.5,
        optimize: bool = True,
    ) -> 'StateConditionedPortfolio':
        """
        Fit exposure values from training data.
        
        Parameters
        ----------
        training_end : str
            End of training period.
        regularization : float, default 0.5
            Regularization strength (pulls exposures toward 1.0).
        optimize : bool, default True
            Whether to optimize exposures or use defaults.
            
        Returns
        -------
        self
        """
        training_end = pd.Timestamp(training_end)
        
        # Filter to training data
        train_mask = self.factor_returns.index <= training_end
        train_returns = self.factor_returns.loc[train_mask]
        train_regimes = self.regimes.loc[train_mask]
        
        if optimize:
            self.exposures = self._optimize_exposures(
                train_returns, train_regimes, regularization
            )
        
        self._fitted = True
        self._training_end = training_end
        
        logger.info(f"Fitted exposures using data through {training_end}")
        
        return self
    
    def _optimize_exposures(
        self,
        returns: pd.DataFrame,
        regimes: pd.Series,
        regularization: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize exposure values to maximize risk-adjusted returns.
        
        Uses a simple grid search with regularization.
        """
        from scipy.optimize import minimize
        
        optimized = {}
        
        for factor in self.factor_names:
            factor_returns = returns[factor]
            
            optimized[factor] = {}
            
            for regime in self.REGIME_ORDER:
                regime_mask = regimes == regime
                regime_returns = factor_returns.loc[regime_mask]
                
                if len(regime_returns) < 12:
                    # Not enough data; use default
                    optimized[factor][regime] = self.DEFAULT_EXPOSURES.get(
                        factor, {}
                    ).get(regime, 1.0)
                    continue
                
                # Optimize exposure for this regime
                def objective(g):
                    adj_returns = g * regime_returns
                    sharpe = adj_returns.mean() / adj_returns.std() if adj_returns.std() > 0 else 0
                    penalty = regularization * (g - 1) ** 2
                    return -sharpe + penalty
                
                result = minimize(
                    objective,
                    x0=1.0,
                    bounds=[(0.0, 1.0)],
                    method='L-BFGS-B',
                )
                
                optimized[factor][regime] = float(result.x[0])
        
        return optimized
    
    def get_exposure(self, factor: str, regime: str) -> float:
        """
        Get exposure value for a factor-regime combination.
        
        Parameters
        ----------
        factor : str
            Factor name.
        regime : str
            Regime name.
            
        Returns
        -------
        float
            Exposure value (0 to 1).
        """
        if factor in self.exposures and regime in self.exposures[factor]:
            return self.exposures[factor][regime]
        return 1.0  # Default to full exposure
    
    def compute_effective_returns(self) -> pd.DataFrame:
        """
        Compute factor returns adjusted for state-conditional exposures.
        
        Returns
        -------
        pd.DataFrame
            Exposure-adjusted factor returns.
        """
        effective = pd.DataFrame(index=self.factor_returns.index)
        
        for factor in self.factor_names:
            exposure_series = self.regimes.map(
                lambda r: self.get_exposure(factor, r)
            )
            effective[factor] = self.factor_returns[factor] * exposure_series
        
        return effective
    
    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """
        Run portfolio backtest with state conditioning.
        
        Parameters
        ----------
        start : str, optional
            Start date.
        end : str, optional
            End date.
        weights : dict, optional
            Factor weights. Default is equal weight.
            
        Returns
        -------
        PortfolioResult
            Backtest results.
        """
        # Compute effective returns
        effective_returns = self.compute_effective_returns()
        
        # Filter dates
        if start:
            effective_returns = effective_returns.loc[start:]
        if end:
            effective_returns = effective_returns.loc[:end]
        
        # Default weights
        if weights is None:
            weights = {f: 1.0 / len(self.factor_names) for f in self.factor_names}
        
        # Compute portfolio return
        portfolio_returns = pd.Series(0.0, index=effective_returns.index)
        for factor, weight in weights.items():
            if factor in effective_returns.columns:
                portfolio_returns += weight * effective_returns[factor]
        
        # Compute turnover from exposure changes
        turnover = self._compute_turnover(effective_returns)
        
        # Apply transaction costs
        net_returns = portfolio_returns - self.transaction_cost * turnover
        
        # Get exposures over time
        exposures = self._get_exposure_series()
        if start:
            exposures = exposures.loc[start:]
        if end:
            exposures = exposures.loc[:end]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'gross': portfolio_returns,
            'net': net_returns,
        })
        
        for factor in self.factor_names:
            result_df[f'{factor}_effective'] = effective_returns[factor]
        
        return PortfolioResult(
            returns=result_df,
            turnover=turnover,
            exposures=exposures,
        )
    
    def _compute_turnover(self, returns: pd.DataFrame) -> pd.Series:
        """Compute turnover from exposure changes."""
        exposures = self._get_exposure_series()
        
        # Turnover is sum of absolute exposure changes
        turnover = pd.Series(0.0, index=returns.index)
        
        for factor in self.factor_names:
            if factor in exposures.columns:
                exposure_changes = exposures[factor].diff().abs()
                turnover += exposure_changes
        
        # Add base portfolio turnover
        base_turnover = 0.10  # Base monthly turnover
        turnover += base_turnover
        
        return turnover.fillna(base_turnover)
    
    def _get_exposure_series(self) -> pd.DataFrame:
        """Get exposure values over time."""
        exposures = pd.DataFrame(index=self.regimes.index)
        
        for factor in self.factor_names:
            exposures[factor] = self.regimes.map(
                lambda r: self.get_exposure(factor, r)
            )
        
        return exposures
    
    def compare_with_baseline(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare state-conditioned performance with baseline.
        
        Parameters
        ----------
        start : str, optional
            Start date.
        end : str, optional
            End date.
            
        Returns
        -------
        pd.DataFrame
            Comparison statistics.
        """
        # State-conditioned backtest
        cond_result = self.backtest(start, end)
        
        # Baseline backtest
        baseline = BaselinePortfolio(
            self.factor_returns,
            self.target_volatility,
            self.transaction_cost,
        )
        base_result = baseline.backtest(start, end)
        
        # Compute statistics
        def compute_stats(returns):
            return {
                'mean_return': returns.mean() * 12,
                'volatility': returns.std() * np.sqrt(12),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(12),
                'skewness': returns.skew(),
                'max_drawdown': self._max_drawdown(returns),
            }
        
        cond_stats = compute_stats(cond_result.returns['net'])
        base_stats = compute_stats(base_result.returns['net'])
        
        comparison = pd.DataFrame({
            'Baseline': base_stats,
            'State-Conditioned': cond_stats,
        })
        
        comparison['Improvement'] = (
            comparison['State-Conditioned'] - comparison['Baseline']
        )
        comparison['Improvement %'] = (
            comparison['Improvement'] / comparison['Baseline'].abs() * 100
        )
        
        return comparison.T
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()
    
    def analyze_by_regime(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyze performance by regime.
        
        Parameters
        ----------
        start : str, optional
            Start date.
        end : str, optional
            End date.
            
        Returns
        -------
        pd.DataFrame
            Performance statistics by regime.
        """
        effective_returns = self.compute_effective_returns()
        regimes = self.regimes.copy()
        
        if start:
            effective_returns = effective_returns.loc[start:]
            regimes = regimes.loc[start:]
        if end:
            effective_returns = effective_returns.loc[:end]
            regimes = regimes.loc[:end]
        
        stats = []
        
        for regime in self.REGIME_ORDER:
            regime_mask = regimes == regime
            regime_returns = effective_returns.loc[regime_mask]
            
            if len(regime_returns) < 2:
                continue
            
            regime_stats = {
                'regime': regime,
                'observations': len(regime_returns),
                'frequency': len(regime_returns) / len(effective_returns) * 100,
            }
            
            for factor in self.factor_names:
                ret = regime_returns[factor]
                regime_stats[f'{factor}_mean'] = ret.mean() * 100
                regime_stats[f'{factor}_std'] = ret.std() * np.sqrt(12) * 100
                regime_stats[f'{factor}_exposure'] = self.get_exposure(factor, regime)
            
            stats.append(regime_stats)
        
        return pd.DataFrame(stats).set_index('regime')
    
    def get_current_exposure(self, current_regime: str) -> Dict[str, float]:
        """
        Get current exposure values for all factors.
        
        Parameters
        ----------
        current_regime : str
            Current regime classification.
            
        Returns
        -------
        dict
            Exposure values by factor.
        """
        return {
            factor: self.get_exposure(factor, current_regime)
            for factor in self.factor_names
        }
    
    def summary(self) -> str:
        """Generate summary of exposure rules."""
        lines = ["State-Conditioned Portfolio Exposures", "=" * 40]
        
        for factor in self.factor_names:
            lines.append(f"\n{factor}:")
            for regime in self.REGIME_ORDER:
                exp = self.get_exposure(factor, regime)
                lines.append(f"  {regime}: {exp:.2f}")
        
        return "\n".join(lines)
