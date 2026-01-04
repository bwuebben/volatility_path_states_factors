"""
Volatility-scaled portfolio implementation.

This module implements volatility-managed portfolios following
Moreira and Muir (2017).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from .baseline import BaselinePortfolio, PortfolioResult

logger = logging.getLogger(__name__)


class VolatilityScaledPortfolio:
    """
    Volatility-managed factor portfolio.
    
    Implements the volatility scaling approach of Moreira and Muir (2017),
    which adjusts factor exposure inversely with trailing realized volatility.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor return time series.
    lookback : int, default 21
        Lookback period for volatility estimation (trading days).
    target_volatility : float, optional
        Target portfolio volatility. If None, uses long-run average.
    max_leverage : float, default 2.0
        Maximum leverage multiple.
    transaction_cost : float, default 0.002
        One-way transaction cost.
        
    Attributes
    ----------
    scaling_factors : pd.DataFrame
        Volatility scaling factors over time.
        
    Examples
    --------
    >>> portfolio = VolatilityScaledPortfolio(factor_returns)
    >>> result = portfolio.backtest(start='2000-01-01')
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        lookback: int = 21,
        target_volatility: Optional[float] = None,
        max_leverage: float = 2.0,
        transaction_cost: float = 0.002,
    ):
        """Initialize volatility-scaled portfolio."""
        self.factor_returns = factor_returns.copy()
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        
        self.factor_names = list(factor_returns.columns)
        
        # Compute target volatility (long-run average if not specified)
        if target_volatility is not None:
            self.target_volatility = target_volatility
        else:
            self.target_volatility = None  # Will be computed per-factor
        
        # Compute scaling factors
        self.scaling_factors = self._compute_scaling_factors()
        
    def _compute_scaling_factors(self) -> pd.DataFrame:
        """Compute volatility scaling factors for each factor."""
        scaling = pd.DataFrame(index=self.factor_returns.index)
        
        for factor in self.factor_names:
            returns = self.factor_returns[factor]
            
            # Trailing realized volatility
            trailing_vol = returns.rolling(
                self.lookback, min_periods=max(5, self.lookback // 4)
            ).std()
            
            # Annualize
            trailing_vol_ann = trailing_vol * np.sqrt(252 / self.lookback * 12)
            
            # Target volatility
            if self.target_volatility is not None:
                target = self.target_volatility
            else:
                # Use long-run average
                target = trailing_vol_ann.expanding().mean()
            
            # Scaling factor
            scale = target / trailing_vol_ann
            
            # Clip to prevent extreme leverage
            scale = scale.clip(0.5, self.max_leverage)
            
            # Lag by one period (use prior month's volatility)
            scaling[factor] = scale.shift(1)
        
        return scaling
    
    def compute_scaled_returns(self) -> pd.DataFrame:
        """
        Compute volatility-scaled factor returns.
        
        Returns
        -------
        pd.DataFrame
            Scaled factor returns.
        """
        return self.factor_returns * self.scaling_factors
    
    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """
        Run portfolio backtest with volatility scaling.
        
        Parameters
        ----------
        start : str, optional
            Start date.
        end : str, optional
            End date.
        weights : dict, optional
            Factor weights.
            
        Returns
        -------
        PortfolioResult
            Backtest results.
        """
        scaled_returns = self.compute_scaled_returns()
        
        # Filter dates
        if start:
            scaled_returns = scaled_returns.loc[start:]
        if end:
            scaled_returns = scaled_returns.loc[:end]
        
        # Default weights
        if weights is None:
            weights = {f: 1.0 / len(self.factor_names) for f in self.factor_names}
        
        # Compute portfolio return
        portfolio_returns = pd.Series(0.0, index=scaled_returns.index)
        for factor, weight in weights.items():
            if factor in scaled_returns.columns:
                portfolio_returns += weight * scaled_returns[factor]
        
        # Compute turnover
        turnover = self._compute_turnover()
        if start:
            turnover = turnover.loc[start:]
        if end:
            turnover = turnover.loc[:end]
        
        # Apply transaction costs
        net_returns = portfolio_returns - self.transaction_cost * turnover
        
        # Create result
        result_df = pd.DataFrame({
            'gross': portfolio_returns,
            'net': net_returns,
        })
        
        return PortfolioResult(
            returns=result_df,
            turnover=turnover,
            exposures=self.scaling_factors,
        )
    
    def _compute_turnover(self) -> pd.Series:
        """Compute turnover from scaling factor changes."""
        # Turnover from scaling changes
        scale_changes = self.scaling_factors.diff().abs()
        turnover = scale_changes.mean(axis=1)
        
        # Add base turnover
        base_turnover = 0.10
        turnover += base_turnover
        
        return turnover.fillna(base_turnover)
    
    def compare_with_baseline(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare volatility-scaled with baseline performance.
        
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
        # Volatility-scaled backtest
        scaled_result = self.backtest(start, end)
        
        # Baseline backtest
        baseline = BaselinePortfolio(
            self.factor_returns,
            transaction_cost=self.transaction_cost,
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
                'avg_turnover': 0.15,  # Approximate
            }
        
        scaled_stats = compute_stats(scaled_result.returns['net'])
        base_stats = compute_stats(base_result.returns['net'])
        
        comparison = pd.DataFrame({
            'Baseline': base_stats,
            'Vol-Scaled': scaled_stats,
        })
        
        return comparison.T
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()


class VolatilityLevelConditionedPortfolio:
    """
    Portfolio conditioned on volatility level only (not path).
    
    This is used as a comparison benchmark to isolate the value
    of path information beyond volatility levels.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor returns.
    volatility : pd.Series
        Volatility series.
    quantiles : tuple, default (0.33, 0.67)
        Quantile thresholds for low/medium/high volatility.
    transaction_cost : float, default 0.002
        Transaction cost.
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        volatility: pd.Series,
        quantiles: tuple = (0.33, 0.67),
        transaction_cost: float = 0.002,
    ):
        """Initialize volatility-level conditioned portfolio."""
        self.factor_returns = factor_returns.copy()
        self.volatility = volatility.copy()
        self.quantiles = quantiles
        self.transaction_cost = transaction_cost
        
        self.factor_names = list(factor_returns.columns)
        
        # Default exposures by volatility level
        self.exposures = {
            'Momentum': {'low': 1.0, 'medium': 0.7, 'high': 0.3},
            'Value': {'low': 1.0, 'medium': 1.0, 'high': 0.6},
            'Quality': {'low': 1.0, 'medium': 1.0, 'high': 1.0},
            'Low-Risk': {'low': 1.0, 'medium': 1.0, 'high': 1.0},
        }
        
    def classify_vol_level(
        self,
        expanding: bool = True,
    ) -> pd.Series:
        """
        Classify volatility into low/medium/high.
        
        Parameters
        ----------
        expanding : bool, default True
            Use expanding window for thresholds.
            
        Returns
        -------
        pd.Series
            Volatility level classification.
        """
        vol = self.volatility
        
        if expanding:
            levels = pd.Series(index=vol.index, dtype=object)
            
            for i, (date, v) in enumerate(vol.items()):
                if i < 24:  # Minimum observations
                    levels.loc[date] = 'medium'
                    continue
                
                prior_vol = vol.iloc[:i]
                q_low = prior_vol.quantile(self.quantiles[0])
                q_high = prior_vol.quantile(self.quantiles[1])
                
                if v <= q_low:
                    levels.loc[date] = 'low'
                elif v <= q_high:
                    levels.loc[date] = 'medium'
                else:
                    levels.loc[date] = 'high'
        else:
            q_low = vol.quantile(self.quantiles[0])
            q_high = vol.quantile(self.quantiles[1])
            
            levels = pd.Series('medium', index=vol.index)
            levels[vol <= q_low] = 'low'
            levels[vol > q_high] = 'high'
        
        return levels
    
    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> PortfolioResult:
        """Run backtest with volatility-level conditioning."""
        vol_levels = self.classify_vol_level()
        
        # Compute effective returns
        effective = pd.DataFrame(index=self.factor_returns.index)
        
        for factor in self.factor_names:
            if factor in self.exposures:
                exposure = vol_levels.map(self.exposures[factor])
            else:
                exposure = 1.0
            
            effective[factor] = self.factor_returns[factor] * exposure
        
        # Filter dates
        if start:
            effective = effective.loc[start:]
        if end:
            effective = effective.loc[:end]
        
        # Equal-weighted portfolio
        portfolio_returns = effective.mean(axis=1)
        
        # Estimate turnover
        turnover = pd.Series(0.15, index=effective.index)
        
        # Net returns
        net_returns = portfolio_returns - self.transaction_cost * turnover
        
        return PortfolioResult(
            returns=pd.DataFrame({
                'gross': portfolio_returns,
                'net': net_returns,
            }),
            turnover=turnover,
        )
