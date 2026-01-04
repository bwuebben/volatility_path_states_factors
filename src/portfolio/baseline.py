"""
Baseline (unconditional) factor portfolio implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """
    Container for portfolio backtest results.
    
    Attributes
    ----------
    returns : pd.DataFrame
        Portfolio returns.
    weights : pd.DataFrame
        Portfolio weights over time.
    turnover : pd.Series
        Monthly turnover.
    exposures : pd.DataFrame
        Factor exposures over time.
    """
    returns: pd.DataFrame
    weights: Optional[pd.DataFrame] = None
    turnover: Optional[pd.Series] = None
    exposures: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'returns': self.returns,
            'weights': self.weights,
            'turnover': self.turnover,
            'exposures': self.exposures,
        }


class BaselinePortfolio:
    """
    Baseline (unconditional) factor portfolio.
    
    This class implements standard long-short factor portfolios
    without any state conditioning.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor return time series.
    target_volatility : float, optional
        Target annualized volatility for scaling. If None, no scaling.
    transaction_cost : float, default 0.002
        One-way transaction cost (20 bps).
        
    Attributes
    ----------
    factor_names : list
        Names of factors in the portfolio.
    scaled_returns : pd.DataFrame
        Volatility-scaled factor returns.
        
    Examples
    --------
    >>> portfolio = BaselinePortfolio(factor_returns, target_volatility=0.10)
    >>> result = portfolio.backtest(start='2000-01-01')
    >>> print(result.returns.mean() * 12)  # Annualized returns
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        target_volatility: Optional[float] = None,
        transaction_cost: float = 0.002,
    ):
        """Initialize baseline portfolio."""
        self.factor_returns = factor_returns.copy()
        self.target_volatility = target_volatility
        self.transaction_cost = transaction_cost
        self.factor_names = list(factor_returns.columns)
        
        # Scale returns if target volatility specified
        if target_volatility is not None:
            self.scaled_returns = self._scale_to_target_vol()
        else:
            self.scaled_returns = self.factor_returns
            
    def _scale_to_target_vol(self) -> pd.DataFrame:
        """Scale factor returns to target volatility."""
        scaled = pd.DataFrame(index=self.factor_returns.index)
        
        for col in self.factor_returns.columns:
            returns = self.factor_returns[col]
            
            # Use 36-month trailing volatility
            trailing_vol = returns.rolling(36, min_periods=12).std() * np.sqrt(12)
            
            # Compute scaling factor
            scale = self.target_volatility / trailing_vol
            scale = scale.clip(0.5, 2.0)  # Limit scaling
            
            scaled[col] = returns * scale.shift(1)  # Use prior month's scale
        
        return scaled
    
    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """
        Run portfolio backtest.
        
        Parameters
        ----------
        start : str, optional
            Start date for backtest.
        end : str, optional
            End date for backtest.
        weights : dict, optional
            Factor weights. Default is equal weight.
            
        Returns
        -------
        PortfolioResult
            Backtest results.
        """
        # Filter dates
        returns = self.scaled_returns.copy()
        if start:
            returns = returns.loc[start:]
        if end:
            returns = returns.loc[:end]
        
        # Default to equal weights
        if weights is None:
            weights = {f: 1.0 / len(self.factor_names) for f in self.factor_names}
        
        # Compute portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for factor, weight in weights.items():
            if factor in returns.columns:
                portfolio_returns += weight * returns[factor]
        
        # Compute turnover (constant for baseline)
        turnover = self._compute_turnover(returns)
        
        # Apply transaction costs
        net_returns = portfolio_returns - self.transaction_cost * turnover
        
        # Create result
        result_df = pd.DataFrame({
            'gross': portfolio_returns,
            'net': net_returns,
        })
        
        for factor in self.factor_names:
            if factor in returns.columns:
                result_df[factor] = returns[factor]
        
        return PortfolioResult(
            returns=result_df,
            turnover=turnover,
        )
    
    def _compute_turnover(self, returns: pd.DataFrame) -> pd.Series:
        """Compute portfolio turnover."""
        # For baseline portfolio, turnover is just the average factor turnover
        # Estimate from factor return autocorrelation
        avg_turnover = 0.15  # Approximate monthly turnover
        return pd.Series(avg_turnover, index=returns.index)
    
    def compute_statistics(
        self,
        result: PortfolioResult,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute performance statistics.
        
        Parameters
        ----------
        result : PortfolioResult
            Backtest results.
        annualize : bool, default True
            Annualize statistics.
            
        Returns
        -------
        pd.DataFrame
            Performance statistics.
        """
        factor = 12 if annualize else 1
        sqrt_factor = np.sqrt(factor)
        
        stats = {}
        
        for col in ['gross', 'net'] + self.factor_names:
            if col not in result.returns.columns:
                continue
                
            ret = result.returns[col].dropna()
            
            stats[col] = {
                'mean_return': ret.mean() * factor,
                'volatility': ret.std() * sqrt_factor,
                'sharpe_ratio': ret.mean() / ret.std() * sqrt_factor if ret.std() > 0 else 0,
                'skewness': ret.skew(),
                'kurtosis': ret.kurtosis(),
                'min_return': ret.min(),
                'max_return': ret.max(),
                'max_drawdown': self._max_drawdown(ret),
                'hit_rate': (ret > 0).mean(),
            }
        
        return pd.DataFrame(stats).T
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()


class MultifactorPortfolio(BaselinePortfolio):
    """
    Multi-factor portfolio combining multiple factors.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Individual factor returns.
    weights : dict, optional
        Factor weights. Default is equal weight.
    rebalance_frequency : str, default 'M'
        Rebalancing frequency.
    target_volatility : float, optional
        Target portfolio volatility.
    transaction_cost : float, default 0.002
        Transaction cost per unit turnover.
    """
    
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        rebalance_frequency: str = 'M',
        target_volatility: Optional[float] = None,
        transaction_cost: float = 0.002,
    ):
        """Initialize multi-factor portfolio."""
        super().__init__(factor_returns, target_volatility, transaction_cost)
        
        self.weights = weights or {f: 1/len(self.factor_names) for f in self.factor_names}
        self.rebalance_frequency = rebalance_frequency
        
    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """Run multi-factor portfolio backtest."""
        weights = weights or self.weights
        return super().backtest(start, end, weights)
    
    def optimize_weights(
        self,
        method: str = 'equal_risk',
        lookback: int = 60,
    ) -> Dict[str, float]:
        """
        Optimize factor weights.
        
        Parameters
        ----------
        method : str
            Optimization method: 'equal_weight', 'equal_risk', 'min_variance'.
        lookback : int
            Lookback period for covariance estimation.
            
        Returns
        -------
        dict
            Optimized weights.
        """
        if method == 'equal_weight':
            return {f: 1/len(self.factor_names) for f in self.factor_names}
        
        # Estimate covariance
        cov = self.factor_returns.tail(lookback).cov()
        
        if method == 'equal_risk':
            # Risk parity weights
            vols = np.sqrt(np.diag(cov))
            inv_vols = 1 / vols
            weights = inv_vols / inv_vols.sum()
            return dict(zip(self.factor_names, weights))
        
        elif method == 'min_variance':
            # Minimum variance weights
            from scipy.optimize import minimize
            
            n = len(self.factor_names)
            
            def portfolio_variance(w):
                return w @ cov.values @ w
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n)]
            
            result = minimize(
                portfolio_variance,
                x0=np.ones(n) / n,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )
            
            return dict(zip(self.factor_names, result.x))
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
