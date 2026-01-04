"""
Exposure optimization for state-conditioned portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
import logging

logger = logging.getLogger(__name__)


class ExposureOptimizer:
    """
    Optimize state-conditional exposure values.
    
    This class implements various optimization methods for finding
    optimal exposure values that maximize risk-adjusted returns
    while incorporating regularization.
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        Factor return time series.
    regimes : pd.Series
        Regime classification.
    regularization : float, default 0.5
        Regularization strength (penalty for deviating from 1.0).
    min_exposure : float, default 0.0
        Minimum allowed exposure.
    max_exposure : float, default 1.0
        Maximum allowed exposure.
        
    Attributes
    ----------
    optimal_exposures : dict
        Optimized exposure values.
        
    Examples
    --------
    >>> optimizer = ExposureOptimizer(factor_returns, regimes)
    >>> exposures = optimizer.optimize(method='sharpe')
    >>> print(exposures['Momentum']['Crash-Spike'])
    0.0
    """
    
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
        regularization: float = 0.5,
        min_exposure: float = 0.0,
        max_exposure: float = 1.0,
    ):
        """Initialize optimizer."""
        self.factor_returns = factor_returns.copy()
        self.regimes = regimes.copy()
        self.regularization = regularization
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        
        self.factor_names = list(factor_returns.columns)
        self.optimal_exposures = None
        
        # Align data
        common_idx = factor_returns.index.intersection(regimes.index)
        self.factor_returns = factor_returns.loc[common_idx]
        self.regimes = regimes.loc[common_idx]
        
    def optimize(
        self,
        method: str = 'sharpe',
        by_factor: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize exposure values.
        
        Parameters
        ----------
        method : str, default 'sharpe'
            Optimization objective: 'sharpe', 'sortino', 'calmar'.
        by_factor : bool, default True
            Optimize each factor separately vs jointly.
            
        Returns
        -------
        dict
            Optimized exposures by factor and regime.
        """
        logger.info(f"Optimizing exposures using {method} objective")
        
        if by_factor:
            exposures = {}
            for factor in self.factor_names:
                exposures[factor] = self._optimize_factor(factor, method)
        else:
            exposures = self._optimize_joint(method)
        
        self.optimal_exposures = exposures
        return exposures
    
    def _optimize_factor(
        self,
        factor: str,
        method: str,
    ) -> Dict[str, float]:
        """Optimize exposures for a single factor."""
        returns = self.factor_returns[factor]
        
        exposures = {}
        
        for regime in self.REGIME_ORDER:
            regime_mask = self.regimes == regime
            regime_returns = returns.loc[regime_mask]
            
            if len(regime_returns) < 12:
                # Not enough data
                exposures[regime] = 1.0
                continue
            
            # Define objective
            def objective(g):
                g = g[0]
                adj_returns = g * regime_returns
                
                if method == 'sharpe':
                    if adj_returns.std() == 0:
                        metric = 0
                    else:
                        metric = adj_returns.mean() / adj_returns.std()
                elif method == 'sortino':
                    downside = adj_returns[adj_returns < 0].std()
                    if downside == 0:
                        metric = adj_returns.mean() / adj_returns.std() if adj_returns.std() > 0 else 0
                    else:
                        metric = adj_returns.mean() / downside
                elif method == 'calmar':
                    max_dd = self._max_drawdown(adj_returns)
                    if max_dd == 0:
                        metric = adj_returns.mean() * 12
                    else:
                        metric = adj_returns.mean() * 12 / max_dd
                else:
                    metric = adj_returns.mean() / adj_returns.std() if adj_returns.std() > 0 else 0
                
                # Regularization penalty
                penalty = self.regularization * (g - 1) ** 2
                
                return -metric + penalty
            
            # Optimize
            result = minimize(
                objective,
                x0=[1.0],
                bounds=[(self.min_exposure, self.max_exposure)],
                method='L-BFGS-B',
            )
            
            exposures[regime] = float(result.x[0])
        
        return exposures
    
    def _optimize_joint(self, method: str) -> Dict[str, Dict[str, float]]:
        """Optimize exposures for all factors jointly."""
        n_factors = len(self.factor_names)
        n_regimes = len(self.REGIME_ORDER)
        n_params = n_factors * n_regimes
        
        def objective(params):
            # Reshape parameters
            exposures = {}
            for i, factor in enumerate(self.factor_names):
                exposures[factor] = {}
                for j, regime in enumerate(self.REGIME_ORDER):
                    exposures[factor][regime] = params[i * n_regimes + j]
            
            # Compute portfolio return
            total_return = pd.Series(0.0, index=self.factor_returns.index)
            
            for factor in self.factor_names:
                exp_series = self.regimes.map(exposures[factor])
                total_return += self.factor_returns[factor] * exp_series / n_factors
            
            # Compute metric
            if method == 'sharpe':
                metric = total_return.mean() / total_return.std() if total_return.std() > 0 else 0
            else:
                metric = total_return.mean() / total_return.std() if total_return.std() > 0 else 0
            
            # Regularization
            penalty = self.regularization * np.sum((np.array(params) - 1) ** 2)
            
            return -metric + penalty
        
        # Initial values
        x0 = np.ones(n_params)
        
        # Bounds
        bounds = [(self.min_exposure, self.max_exposure)] * n_params
        
        # Optimize
        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
        )
        
        # Extract results
        exposures = {}
        for i, factor in enumerate(self.factor_names):
            exposures[factor] = {}
            for j, regime in enumerate(self.REGIME_ORDER):
                exposures[factor][regime] = float(result.x[i * n_regimes + j])
        
        return exposures
    
    def grid_search(
        self,
        factor: str,
        grid_points: int = 11,
    ) -> pd.DataFrame:
        """
        Perform grid search over exposure values.
        
        Parameters
        ----------
        factor : str
            Factor to analyze.
        grid_points : int, default 11
            Number of grid points per dimension.
            
        Returns
        -------
        pd.DataFrame
            Performance metrics at each grid point.
        """
        returns = self.factor_returns[factor]
        grid = np.linspace(self.min_exposure, self.max_exposure, grid_points)
        
        results = []
        
        for regime in self.REGIME_ORDER:
            regime_mask = self.regimes == regime
            regime_returns = returns.loc[regime_mask]
            
            if len(regime_returns) < 12:
                continue
            
            for g in grid:
                adj_returns = g * regime_returns
                
                results.append({
                    'regime': regime,
                    'exposure': g,
                    'mean_return': adj_returns.mean() * 12,
                    'volatility': adj_returns.std() * np.sqrt(12),
                    'sharpe': adj_returns.mean() / adj_returns.std() * np.sqrt(12) if adj_returns.std() > 0 else 0,
                    'n_obs': len(regime_returns),
                })
        
        return pd.DataFrame(results)
    
    def cross_validate(
        self,
        factor: str,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Cross-validate exposure optimization.
        
        Parameters
        ----------
        factor : str
            Factor to optimize.
        n_folds : int, default 5
            Number of cross-validation folds.
            
        Returns
        -------
        dict
            Cross-validated optimal exposures.
        """
        returns = self.factor_returns[factor]
        n = len(returns)
        fold_size = n // n_folds
        
        cv_exposures = {regime: [] for regime in self.REGIME_ORDER}
        
        for fold in range(n_folds):
            # Create train/test split
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n)
            
            train_idx = list(range(0, test_start)) + list(range(test_end, n))
            train_returns = returns.iloc[train_idx]
            train_regimes = self.regimes.iloc[train_idx]
            
            # Optimize on training data
            for regime in self.REGIME_ORDER:
                regime_mask = train_regimes == regime
                regime_returns = train_returns.loc[regime_mask]
                
                if len(regime_returns) < 12:
                    cv_exposures[regime].append(1.0)
                    continue
                
                def objective(g):
                    adj = g[0] * regime_returns
                    sharpe = adj.mean() / adj.std() if adj.std() > 0 else 0
                    return -sharpe + self.regularization * (g[0] - 1) ** 2
                
                result = minimize(
                    objective,
                    x0=[1.0],
                    bounds=[(self.min_exposure, self.max_exposure)],
                )
                
                cv_exposures[regime].append(float(result.x[0]))
        
        # Average across folds
        return {
            regime: np.mean(values)
            for regime, values in cv_exposures.items()
        }
    
    def sensitivity_analysis(
        self,
        factor: str,
        param_name: str = 'regularization',
        param_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to optimization parameters.
        
        Parameters
        ----------
        factor : str
            Factor to analyze.
        param_name : str
            Parameter to vary.
        param_range : list, optional
            Range of parameter values.
            
        Returns
        -------
        pd.DataFrame
            Optimal exposures at each parameter value.
        """
        if param_range is None:
            if param_name == 'regularization':
                param_range = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
            else:
                param_range = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        results = []
        
        for param_value in param_range:
            # Update parameter
            if param_name == 'regularization':
                orig_reg = self.regularization
                self.regularization = param_value
            
            # Optimize
            exposures = self._optimize_factor(factor, 'sharpe')
            
            # Record results
            row = {param_name: param_value}
            for regime, exp in exposures.items():
                row[regime] = exp
            results.append(row)
            
            # Restore parameter
            if param_name == 'regularization':
                self.regularization = orig_reg
        
        return pd.DataFrame(results).set_index(param_name)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()
    
    def report(self) -> str:
        """Generate optimization report."""
        if self.optimal_exposures is None:
            return "No optimization performed yet. Call optimize() first."
        
        lines = ["Optimal Exposures", "=" * 50]
        
        for factor in self.factor_names:
            if factor not in self.optimal_exposures:
                continue
            
            lines.append(f"\n{factor}:")
            for regime in self.REGIME_ORDER:
                exp = self.optimal_exposures[factor].get(regime, 1.0)
                lines.append(f"  {regime:20s}: {exp:.3f}")
        
        return "\n".join(lines)
