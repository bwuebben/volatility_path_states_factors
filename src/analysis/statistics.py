"""
Statistical tests for factor and regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Container for statistical test results.
    
    Attributes
    ----------
    statistic : float
        Test statistic.
    pvalue : float
        P-value.
    null_hypothesis : str
        Description of null hypothesis.
    reject_null : bool
        Whether to reject null at 5% level.
    """
    statistic: float
    pvalue: float
    null_hypothesis: str
    reject_null: bool
    
    def __repr__(self):
        result = "Reject" if self.reject_null else "Fail to reject"
        return (
            f"TestResult(statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4f}, {result} H0)"
        )


class StatisticalTests:
    """
    Statistical hypothesis tests for factor analysis.
    
    This class implements various tests used in the paper including
    tests for mean differences, Sharpe ratio differences, and
    regime transition probabilities.
    
    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for hypothesis tests.
        
    Examples
    --------
    >>> tests = StatisticalTests()
    >>> result = tests.test_mean_difference(returns_a, returns_b)
    >>> print(result.pvalue)
    """
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tests."""
        self.alpha = alpha
        
    def newey_west_tstat(
        self,
        returns: pd.Series,
        lags: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute t-statistic with Newey-West standard errors.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        lags : int, optional
            Number of lags. Default is floor(4 * (T/100)^(2/9)).
            
        Returns
        -------
        mean : float
            Sample mean.
        tstat : float
            T-statistic.
        pvalue : float
            P-value (two-sided).
        """
        returns = returns.dropna()
        T = len(returns)
        
        if T < 2:
            return 0, 0, 1
        
        # Default lags
        if lags is None:
            lags = int(np.floor(4 * (T / 100) ** (2/9)))
        
        mean = returns.mean()
        demeaned = returns - mean
        
        # Compute Newey-West variance
        gamma_0 = (demeaned ** 2).sum() / T
        
        nw_var = gamma_0
        for j in range(1, lags + 1):
            weight = 1 - j / (lags + 1)  # Bartlett kernel
            gamma_j = (demeaned.iloc[j:] * demeaned.iloc[:-j].values).sum() / T
            nw_var += 2 * weight * gamma_j
        
        se = np.sqrt(nw_var / T)
        
        if se == 0:
            return mean, 0, 1
        
        tstat = mean / se
        pvalue = 2 * (1 - stats.t.cdf(abs(tstat), T - 1))
        
        return mean, tstat, pvalue
    
    def test_mean_difference(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        paired: bool = True,
    ) -> TestResult:
        """
        Test for difference in means.
        
        Parameters
        ----------
        returns_a : pd.Series
            First return series.
        returns_b : pd.Series
            Second return series.
        paired : bool, default True
            Use paired test.
            
        Returns
        -------
        TestResult
            Test result.
        """
        if paired:
            # Align series
            common_idx = returns_a.index.intersection(returns_b.index)
            diff = returns_a.loc[common_idx] - returns_b.loc[common_idx]
            
            mean, tstat, pvalue = self.newey_west_tstat(diff)
            
        else:
            # Welch's t-test
            stat, pvalue = stats.ttest_ind(
                returns_a.dropna(), 
                returns_b.dropna(), 
                equal_var=False
            )
            tstat = stat
        
        return TestResult(
            statistic=tstat,
            pvalue=pvalue,
            null_hypothesis="Mean difference equals zero",
            reject_null=pvalue < self.alpha,
        )
    
    def test_sharpe_difference(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        method: str = 'jobson_korkie',
    ) -> TestResult:
        """
        Test for difference in Sharpe ratios.
        
        Parameters
        ----------
        returns_a : pd.Series
            First return series.
        returns_b : pd.Series
            Second return series.
        method : str, default 'jobson_korkie'
            Test method: 'jobson_korkie' or 'ledoit_wolf'.
            
        Returns
        -------
        TestResult
            Test result.
        """
        # Align series
        common_idx = returns_a.index.intersection(returns_b.index)
        a = returns_a.loc[common_idx].dropna()
        b = returns_b.loc[common_idx].dropna()
        
        # Use common observations
        valid_idx = a.index.intersection(b.index)
        a = a.loc[valid_idx]
        b = b.loc[valid_idx]
        
        n = len(a)
        
        if n < 10:
            return TestResult(0, 1, "Sharpe ratios are equal", False)
        
        # Compute Sharpe ratios
        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.std(), b.std()
        sr_a = mu_a / sig_a if sig_a > 0 else 0
        sr_b = mu_b / sig_b if sig_b > 0 else 0
        
        # Correlation
        rho = a.corr(b)
        
        if method == 'jobson_korkie':
            # Jobson-Korkie test with Memmel correction
            theta = (
                (1/n) * (2 * (1 - rho) + 0.5 * (sr_a**2 + sr_b**2 - 2*sr_a*sr_b*rho**2))
            )
            
            if theta <= 0:
                return TestResult(0, 1, "Sharpe ratios are equal", False)
            
            z = (sr_a - sr_b) / np.sqrt(theta)
            pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
            
        else:  # Ledoit-Wolf
            # Bootstrap-based test
            z, pvalue = self._bootstrap_sharpe_test(a, b)
        
        return TestResult(
            statistic=z,
            pvalue=pvalue,
            null_hypothesis="Sharpe ratios are equal",
            reject_null=pvalue < self.alpha,
        )
    
    def _bootstrap_sharpe_test(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """Bootstrap test for Sharpe ratio difference."""
        n = len(returns_a)
        
        # Observed difference
        sr_a = returns_a.mean() / returns_a.std()
        sr_b = returns_b.mean() / returns_b.std()
        observed_diff = sr_a - sr_b
        
        # Bootstrap under null (centered)
        combined = pd.concat([returns_a, returns_b])
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(combined), size=2*n, replace=True)
            boot_a = combined.iloc[idx[:n]]
            boot_b = combined.iloc[idx[n:]]
            
            sr_boot_a = boot_a.mean() / boot_a.std()
            sr_boot_b = boot_b.mean() / boot_b.std()
            bootstrap_diffs.append(sr_boot_a - sr_boot_b)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute p-value
        pvalue = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Z-score
        z = observed_diff / np.std(bootstrap_diffs) if np.std(bootstrap_diffs) > 0 else 0
        
        return z, pvalue
    
    def test_regime_difference(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        regime_a: str,
        regime_b: str,
    ) -> TestResult:
        """
        Test for difference in returns between regimes.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        regime_a : str
            First regime.
        regime_b : str
            Second regime.
            
        Returns
        -------
        TestResult
            Test result.
        """
        # Align
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Extract regime returns
        returns_a = returns.loc[regimes == regime_a]
        returns_b = returns.loc[regimes == regime_b]
        
        # Welch's t-test (unpaired)
        stat, pvalue = stats.ttest_ind(
            returns_a.dropna(),
            returns_b.dropna(),
            equal_var=False,
        )
        
        return TestResult(
            statistic=stat,
            pvalue=pvalue,
            null_hypothesis=f"Mean returns equal in {regime_a} and {regime_b}",
            reject_null=pvalue < self.alpha,
        )
    
    def test_all_regimes_equal(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> TestResult:
        """
        Test that returns are equal across all regimes (ANOVA).
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
            
        Returns
        -------
        TestResult
            Test result.
        """
        # Align
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Group by regime
        groups = [returns.loc[regimes == r].dropna().values 
                  for r in regimes.unique()]
        
        # Filter empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return TestResult(0, 1, "Returns are equal across regimes", False)
        
        # ANOVA
        stat, pvalue = stats.f_oneway(*groups)
        
        return TestResult(
            statistic=stat,
            pvalue=pvalue,
            null_hypothesis="Returns are equal across all regimes",
            reject_null=pvalue < self.alpha,
        )
    
    def test_crash_concentration(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        crash_regime: str = 'Crash-Spike',
        crash_percentile: float = 5,
    ) -> TestResult:
        """
        Test whether crashes are concentrated in a specific regime.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
        crash_regime : str
            Regime where crashes are expected to concentrate.
        crash_percentile : float
            Percentile defining crashes.
            
        Returns
        -------
        TestResult
            Test result.
        """
        # Align
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Identify crashes
        threshold = returns.quantile(crash_percentile / 100)
        is_crash = returns <= threshold
        
        # Contingency table
        in_regime = regimes == crash_regime
        
        # Count
        crash_in_regime = (is_crash & in_regime).sum()
        crash_out_regime = (is_crash & ~in_regime).sum()
        no_crash_in_regime = (~is_crash & in_regime).sum()
        no_crash_out_regime = (~is_crash & ~in_regime).sum()
        
        contingency = np.array([
            [crash_in_regime, crash_out_regime],
            [no_crash_in_regime, no_crash_out_regime],
        ])
        
        # Chi-squared test
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        
        return TestResult(
            statistic=chi2,
            pvalue=pvalue,
            null_hypothesis=f"Crashes are not concentrated in {crash_regime}",
            reject_null=pvalue < self.alpha,
        )
    
    def test_transition_probability(
        self,
        regimes: pd.Series,
        from_regime: str,
        to_regime: str,
        null_prob: float,
    ) -> TestResult:
        """
        Test whether transition probability differs from null.
        
        Parameters
        ----------
        regimes : pd.Series
            Regime classification.
        from_regime : str
            Origin regime.
        to_regime : str
            Destination regime.
        null_prob : float
            Null hypothesis probability.
            
        Returns
        -------
        TestResult
            Test result.
        """
        # Count transitions
        n_from = (regimes.shift(1) == from_regime).sum()
        n_transition = ((regimes.shift(1) == from_regime) & 
                        (regimes == to_regime)).sum()
        
        if n_from == 0:
            return TestResult(0, 1, f"Transition prob equals {null_prob}", False)
        
        # Observed probability
        p_hat = n_transition / n_from
        
        # Binomial test
        pvalue = stats.binom_test(n_transition, n_from, null_prob, alternative='two-sided')
        
        # Z-score
        se = np.sqrt(null_prob * (1 - null_prob) / n_from)
        z = (p_hat - null_prob) / se if se > 0 else 0
        
        return TestResult(
            statistic=z,
            pvalue=pvalue,
            null_hypothesis=f"Transition probability equals {null_prob}",
            reject_null=pvalue < self.alpha,
        )
    
    def test_autocorrelation(
        self,
        returns: pd.Series,
        lags: int = 1,
    ) -> TestResult:
        """
        Test for autocorrelation in returns.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        lags : int, default 1
            Number of lags to test.
            
        Returns
        -------
        TestResult
            Test result (Ljung-Box test).
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(returns.dropna(), lags=lags, return_df=True)
        
        lb_stat = result['lb_stat'].iloc[-1]
        pvalue = result['lb_pvalue'].iloc[-1]
        
        return TestResult(
            statistic=lb_stat,
            pvalue=pvalue,
            null_hypothesis="No autocorrelation",
            reject_null=pvalue < self.alpha,
        )
    
    def run_all_tests(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Run all relevant tests.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
            
        Returns
        -------
        pd.DataFrame
            Summary of all test results.
        """
        results = []
        
        # Test mean significance
        mean, tstat, pvalue = self.newey_west_tstat(returns)
        results.append({
            'test': 'Mean Return',
            'statistic': tstat,
            'pvalue': pvalue,
            'reject_null': pvalue < self.alpha,
        })
        
        # Test regime equality
        result = self.test_all_regimes_equal(returns, regimes)
        results.append({
            'test': 'Regime Equality (ANOVA)',
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'reject_null': result.reject_null,
        })
        
        # Test Calm vs Crash-Spike
        if 'Calm Trend' in regimes.values and 'Crash-Spike' in regimes.values:
            result = self.test_regime_difference(
                returns, regimes, 'Calm Trend', 'Crash-Spike'
            )
            results.append({
                'test': 'Calm vs Crash-Spike',
                'statistic': result.statistic,
                'pvalue': result.pvalue,
                'reject_null': result.reject_null,
            })
        
        # Test crash concentration
        if 'Crash-Spike' in regimes.values:
            result = self.test_crash_concentration(returns, regimes)
            results.append({
                'test': 'Crash Concentration',
                'statistic': result.statistic,
                'pvalue': result.pvalue,
                'reject_null': result.reject_null,
            })
        
        return pd.DataFrame(results)
