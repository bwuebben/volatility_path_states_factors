"""
Robustness tests for factor and regime analysis.

This module provides various robustness checks including:
- Subsample analysis
- Alternative regime definitions
- Parameter sensitivity
- Out-of-sample testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """
    Container for robustness test results.
    
    Attributes
    ----------
    test_name : str
        Name of the test.
    baseline_value : float
        Baseline metric value.
    test_values : dict
        Test metric values for each variation.
    passed : bool
        Whether the test passed.
    description : str
        Description of the test.
    """
    test_name: str
    baseline_value: float
    test_values: Dict
    passed: bool
    description: str


class RobustnessAnalyzer:
    """
    Perform robustness tests on factor and portfolio analysis.
    
    Parameters
    ----------
    significance_level : float, default 0.05
        Significance level for hypothesis tests.
    tolerance : float, default 0.2
        Tolerance for metric variation (20% default).
        
    Examples
    --------
    >>> analyzer = RobustnessAnalyzer()
    >>> results = analyzer.subsample_analysis(returns, regimes, n_subsamples=5)
    >>> sensitivity = analyzer.parameter_sensitivity(returns, regimes, param_grid)
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        tolerance: float = 0.2,
    ):
        """Initialize robustness analyzer."""
        self.significance_level = significance_level
        self.tolerance = tolerance
        
    def subsample_analysis(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        n_subsamples: int = 5,
        overlap: float = 0.0,
    ) -> RobustnessResult:
        """
        Test stability across time subsamples.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        n_subsamples : int, default 5
            Number of subsamples.
        overlap : float, default 0.0
            Fraction of overlap between subsamples.
            
        Returns
        -------
        RobustnessResult
            Test results.
        """
        # Align data
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        n = len(returns)
        subsample_size = int(n / n_subsamples * (1 + overlap))
        step = int(n / n_subsamples)
        
        # Compute baseline Sharpe
        baseline_sharpe = returns.mean() / returns.std() * np.sqrt(12)
        
        # Compute Sharpe for each subsample
        subsample_sharpes = {}
        
        for i in range(n_subsamples):
            start = i * step
            end = min(start + subsample_size, n)
            
            sub_returns = returns.iloc[start:end]
            sub_sharpe = sub_returns.mean() / sub_returns.std() * np.sqrt(12)
            
            subsample_sharpes[f'subsample_{i+1}'] = sub_sharpe
        
        # Check if all subsamples are within tolerance
        values = list(subsample_sharpes.values())
        passed = all(
            abs(v - baseline_sharpe) / abs(baseline_sharpe) < self.tolerance
            for v in values if baseline_sharpe != 0
        )
        
        return RobustnessResult(
            test_name='Subsample Analysis',
            baseline_value=baseline_sharpe,
            test_values=subsample_sharpes,
            passed=passed,
            description=f'Sharpe ratio stability across {n_subsamples} subsamples',
        )
    
    def expanding_window_analysis(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        min_window: int = 60,
        step: int = 12,
    ) -> pd.DataFrame:
        """
        Test with expanding window estimation.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        min_window : int, default 60
            Minimum estimation window.
        step : int, default 12
            Step size for expansion.
            
        Returns
        -------
        pd.DataFrame
            Results for each window size.
        """
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        n = len(returns)
        results = []
        
        for window_end in range(min_window, n, step):
            sub_returns = returns.iloc[:window_end]
            sub_regimes = regimes.iloc[:window_end]
            
            # Compute metrics
            sharpe = sub_returns.mean() / sub_returns.std() * np.sqrt(12)
            
            # Regime-specific Sharpe for Crash-Spike
            crash_mask = sub_regimes == 'Crash-Spike'
            if crash_mask.any():
                crash_sharpe = (
                    sub_returns[crash_mask].mean() / 
                    sub_returns[crash_mask].std() * np.sqrt(12)
                )
            else:
                crash_sharpe = np.nan
            
            results.append({
                'window_end': window_end,
                'date': returns.index[window_end - 1],
                'sharpe': sharpe,
                'crash_sharpe': crash_sharpe,
                'n_obs': window_end,
            })
        
        return pd.DataFrame(results)
    
    def alternative_regime_definitions(
        self,
        returns: pd.Series,
        volatility_data: pd.DataFrame,
        quantile_pairs: List[Tuple[float, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Test with alternative regime boundary definitions.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        volatility_data : pd.DataFrame
            Volatility data for regime classification.
        quantile_pairs : list, optional
            List of (low, high) quantile pairs to test.
            
        Returns
        -------
        dict
            Results for each quantile definition.
        """
        if quantile_pairs is None:
            quantile_pairs = [
                (0.25, 0.75),
                (0.30, 0.70),
                (0.33, 0.67),
                (0.40, 0.60),
            ]
        
        from ..regimes.regime_classifier import RegimeClassifier
        
        results = {}
        
        for low, high in quantile_pairs:
            classifier = RegimeClassifier(
                vol_quantile_low=low,
                vol_quantile_high=high,
            )
            
            regimes = classifier.classify(volatility_data)
            
            # Compute regime-specific statistics
            common_idx = returns.index.intersection(regimes.index)
            ret = returns.loc[common_idx]
            reg = regimes.loc[common_idx]
            
            stats = {}
            for regime in reg.unique():
                mask = reg == regime
                if mask.sum() > 10:
                    r = ret[mask]
                    stats[regime] = {
                        'mean': r.mean() * 12 * 100,
                        'std': r.std() * np.sqrt(12) * 100,
                        'sharpe': r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0,
                        'n_obs': mask.sum(),
                    }
            
            results[f'q_{low}_{high}'] = pd.DataFrame(stats).T
        
        return results
    
    def parameter_sensitivity(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        param_name: str,
        param_values: List,
        compute_metric: callable,
    ) -> pd.DataFrame:
        """
        Test sensitivity to a parameter.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        param_name : str
            Name of parameter to vary.
        param_values : list
            Values to test.
        compute_metric : callable
            Function to compute metric given returns, regimes, and param value.
            
        Returns
        -------
        pd.DataFrame
            Sensitivity results.
        """
        results = []
        
        for value in param_values:
            try:
                metric = compute_metric(returns, regimes, value)
                results.append({
                    param_name: value,
                    'metric': metric,
                })
            except Exception as e:
                logger.warning(f"Failed for {param_name}={value}: {e}")
        
        return pd.DataFrame(results)
    
    def bootstrap_confidence_interval(
        self,
        returns: pd.Series,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        metric_func : callable
            Function to compute metric.
        n_bootstrap : int, default 1000
            Number of bootstrap samples.
        confidence : float, default 0.95
            Confidence level.
            
        Returns
        -------
        tuple
            (point_estimate, lower_bound, upper_bound)
        """
        point_estimate = metric_func(returns)
        
        bootstrap_values = []
        n = len(returns)
        
        for _ in range(n_bootstrap):
            # Block bootstrap to preserve autocorrelation
            block_size = max(1, int(np.sqrt(n)))
            n_blocks = int(np.ceil(n / block_size))
            
            indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n - block_size + 1)
                indices.extend(range(start, min(start + block_size, n)))
            
            indices = indices[:n]
            boot_returns = returns.iloc[indices]
            
            try:
                bootstrap_values.append(metric_func(boot_returns))
            except:
                continue
        
        bootstrap_values = np.array(bootstrap_values)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
        
        return point_estimate, lower, upper
    
    def out_of_sample_test(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        training_end: str,
        exposures: Dict[str, float],
    ) -> Dict:
        """
        Test portfolio performance out-of-sample.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        training_end : str
            End of training period.
        exposures : dict
            Regime-specific exposures.
            
        Returns
        -------
        dict
            In-sample and out-of-sample performance.
        """
        # Align
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Split
        training_mask = returns.index <= training_end
        
        in_sample_ret = returns[training_mask]
        out_sample_ret = returns[~training_mask]
        in_sample_reg = regimes[training_mask]
        out_sample_reg = regimes[~training_mask]
        
        def compute_conditioned_perf(ret, reg):
            exposure = reg.map(exposures)
            cond_ret = ret * exposure
            
            return {
                'mean': cond_ret.mean() * 12 * 100,
                'vol': cond_ret.std() * np.sqrt(12) * 100,
                'sharpe': cond_ret.mean() / cond_ret.std() * np.sqrt(12) if cond_ret.std() > 0 else 0,
            }
        
        return {
            'in_sample': compute_conditioned_perf(in_sample_ret, in_sample_reg),
            'out_of_sample': compute_conditioned_perf(out_sample_ret, out_sample_reg),
            'n_in_sample': len(in_sample_ret),
            'n_out_sample': len(out_sample_ret),
        }
    
    def cross_validation(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Perform time-series cross-validation.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        n_folds : int, default 5
            Number of folds.
            
        Returns
        -------
        pd.DataFrame
            Cross-validation results.
        """
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        n = len(returns)
        fold_size = n // n_folds
        
        results = []
        
        for fold in range(n_folds):
            # Expanding window: train on all data before test fold
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n)
            
            if fold == 0:
                continue  # Skip first fold (no training data)
            
            train_ret = returns.iloc[:test_start]
            train_reg = regimes.iloc[:test_start]
            test_ret = returns.iloc[test_start:test_end]
            test_reg = regimes.iloc[test_start:test_end]
            
            # Compute training Sharpe
            train_sharpe = train_ret.mean() / train_ret.std() * np.sqrt(12)
            test_sharpe = test_ret.mean() / test_ret.std() * np.sqrt(12)
            
            results.append({
                'fold': fold,
                'train_end': returns.index[test_start - 1],
                'test_start': returns.index[test_start],
                'test_end': returns.index[test_end - 1],
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'n_train': len(train_ret),
                'n_test': len(test_ret),
            })
        
        return pd.DataFrame(results)
    
    def run_all_tests(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        volatility_data: pd.DataFrame = None,
    ) -> Dict[str, RobustnessResult]:
        """
        Run all robustness tests.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        volatility_data : pd.DataFrame, optional
            Volatility data for regime tests.
            
        Returns
        -------
        dict
            All test results.
        """
        results = {}
        
        # Subsample analysis
        results['subsample'] = self.subsample_analysis(returns, regimes)
        
        # Expanding window
        expanding = self.expanding_window_analysis(returns, regimes)
        sharpe_stability = expanding['sharpe'].std() / expanding['sharpe'].mean()
        results['expanding_window'] = RobustnessResult(
            test_name='Expanding Window',
            baseline_value=expanding['sharpe'].iloc[-1],
            test_values={'coefficient_of_variation': sharpe_stability},
            passed=sharpe_stability < 0.5,
            description='Sharpe ratio stability with expanding window',
        )
        
        # Cross-validation
        cv_results = self.cross_validation(returns, regimes)
        avg_test_sharpe = cv_results['test_sharpe'].mean()
        results['cross_validation'] = RobustnessResult(
            test_name='Cross-Validation',
            baseline_value=returns.mean() / returns.std() * np.sqrt(12),
            test_values={'avg_test_sharpe': avg_test_sharpe},
            passed=avg_test_sharpe > 0,
            description='Average test fold Sharpe ratio',
        )
        
        return results
