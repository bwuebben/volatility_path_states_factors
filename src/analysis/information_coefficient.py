"""
Information coefficient analysis.

This module provides functionality for computing and analyzing
information coefficients (ICs) for factor signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class InformationCoefficientAnalyzer:
    """
    Analyze information coefficients for factor signals.
    
    Information coefficients measure the cross-sectional correlation
    between a signal and subsequent returns.
    
    Parameters
    ----------
    method : str, default 'spearman'
        Correlation method: 'spearman' or 'pearson'.
        
    Examples
    --------
    >>> analyzer = InformationCoefficientAnalyzer()
    >>> ic_series = analyzer.compute_ic(signals, returns)
    >>> ic_by_regime = analyzer.analyze_by_regime(ic_series, regimes)
    """
    
    def __init__(self, method: str = 'spearman'):
        """Initialize IC analyzer."""
        self.method = method
        
    def compute_ic(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> pd.Series:
        """
        Compute time series of information coefficients.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Cross-sectional signals (stocks x time).
        returns : pd.DataFrame
            Subsequent returns (stocks x time).
        lag : int, default 1
            Return lag (1 = next period return).
            
        Returns
        -------
        pd.Series
            Time series of ICs.
        """
        # Align signals with lagged returns
        returns_lagged = returns.shift(-lag)
        
        ics = []
        dates = signals.columns
        
        for date in dates:
            if date not in returns_lagged.columns:
                continue
                
            sig = signals[date].dropna()
            ret = returns_lagged[date].dropna()
            
            # Common stocks
            common = sig.index.intersection(ret.index)
            
            if len(common) < 10:
                ics.append({'date': date, 'ic': np.nan})
                continue
            
            # Compute correlation
            if self.method == 'spearman':
                ic, _ = stats.spearmanr(sig.loc[common], ret.loc[common])
            else:
                ic, _ = stats.pearsonr(sig.loc[common], ret.loc[common])
            
            ics.append({'date': date, 'ic': ic})
        
        ic_df = pd.DataFrame(ics).set_index('date')
        return ic_df['ic']
    
    def compute_factor_ic(
        self,
        factor_returns: pd.Series,
        factor_volatility: pd.Series,
        window: int = 12,
    ) -> pd.Series:
        """
        Compute IC proxy from factor returns and volatility.
        
        For factor portfolios without stock-level data, estimate
        IC from the relationship between factor returns and volatility.
        
        Parameters
        ----------
        factor_returns : pd.Series
            Factor return series.
        factor_volatility : pd.Series
            Factor volatility series.
        window : int, default 12
            Rolling window for estimation.
            
        Returns
        -------
        pd.Series
            Estimated IC series.
        """
        # Normalize returns by volatility
        normalized = factor_returns / factor_volatility.shift(1)
        
        # Rolling mean as IC proxy
        ic_proxy = normalized.rolling(window).mean()
        
        # Scale to typical IC range
        ic_scaled = ic_proxy * 0.05  # Typical IC magnitude
        
        return ic_scaled.clip(-0.15, 0.15)
    
    def compute_ic_statistics(
        self,
        ic_series: pd.Series,
    ) -> Dict:
        """
        Compute IC summary statistics.
        
        Parameters
        ----------
        ic_series : pd.Series
            Time series of ICs.
            
        Returns
        -------
        dict
            IC statistics.
        """
        ic = ic_series.dropna()
        
        if len(ic) == 0:
            return {
                'mean_ic': 0,
                'std_ic': 0,
                'ir': 0,
                't_stat': 0,
                'hit_rate': 0,
                'positive_months': 0,
                'negative_months': 0,
            }
        
        mean_ic = ic.mean()
        std_ic = ic.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic))) if std_ic > 0 else 0
        hit_rate = (ic > 0).mean()
        
        return {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ir': ir,  # IC Information Ratio
            't_stat': t_stat,
            'hit_rate': hit_rate,
            'positive_months': (ic > 0).sum(),
            'negative_months': (ic < 0).sum(),
            'max_ic': ic.max(),
            'min_ic': ic.min(),
            'skewness': ic.skew(),
        }
    
    def analyze_by_regime(
        self,
        ic_series: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Analyze ICs by regime.
        
        Parameters
        ----------
        ic_series : pd.Series
            Time series of ICs.
        regimes : pd.Series
            Regime classification.
            
        Returns
        -------
        pd.DataFrame
            IC statistics by regime.
        """
        # Align
        common_idx = ic_series.index.intersection(regimes.index)
        ic_series = ic_series.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        results = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_ic = ic_series.loc[mask]
            
            stats = self.compute_ic_statistics(regime_ic)
            stats['n_obs'] = len(regime_ic)
            stats['frequency'] = len(regime_ic) / len(ic_series) * 100
            
            results[regime] = stats
        
        return pd.DataFrame(results).T
    
    def compute_ic_decay(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 12,
    ) -> pd.DataFrame:
        """
        Compute IC decay over multiple holding periods.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Cross-sectional signals.
        returns : pd.DataFrame
            Returns.
        max_lag : int, default 12
            Maximum holding period.
            
        Returns
        -------
        pd.DataFrame
            IC statistics by lag.
        """
        results = []
        
        for lag in range(1, max_lag + 1):
            ic_series = self.compute_ic(signals, returns, lag=lag)
            stats = self.compute_ic_statistics(ic_series)
            stats['lag'] = lag
            results.append(stats)
        
        return pd.DataFrame(results).set_index('lag')
    
    def compute_rolling_ic(
        self,
        ic_series: pd.Series,
        window: int = 12,
    ) -> pd.DataFrame:
        """
        Compute rolling IC statistics.
        
        Parameters
        ----------
        ic_series : pd.Series
            Time series of ICs.
        window : int, default 12
            Rolling window size.
            
        Returns
        -------
        pd.DataFrame
            Rolling IC statistics.
        """
        rolling = pd.DataFrame(index=ic_series.index)
        
        rolling['ic_mean'] = ic_series.rolling(window).mean()
        rolling['ic_std'] = ic_series.rolling(window).std()
        rolling['ir'] = rolling['ic_mean'] / rolling['ic_std']
        rolling['hit_rate'] = ic_series.rolling(window).apply(lambda x: (x > 0).mean())
        
        return rolling
    
    def test_ic_significance(
        self,
        ic_series: pd.Series,
        null_mean: float = 0,
    ) -> Dict:
        """
        Test if IC is significantly different from null.
        
        Parameters
        ----------
        ic_series : pd.Series
            Time series of ICs.
        null_mean : float, default 0
            Null hypothesis mean.
            
        Returns
        -------
        dict
            Test results.
        """
        ic = ic_series.dropna()
        
        if len(ic) < 10:
            return {
                't_stat': 0,
                'pvalue': 1,
                'significant': False,
            }
        
        # T-test
        t_stat, pvalue = stats.ttest_1samp(ic, null_mean)
        
        return {
            't_stat': t_stat,
            'pvalue': pvalue,
            'significant': pvalue < 0.05,
            'mean_ic': ic.mean(),
            'se_ic': ic.std() / np.sqrt(len(ic)),
        }
    
    def compare_ic_across_factors(
        self,
        factor_ics: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compare IC statistics across factors.
        
        Parameters
        ----------
        factor_ics : dict
            Dictionary of factor name to IC series.
            
        Returns
        -------
        pd.DataFrame
            Comparison table.
        """
        results = {}
        
        for factor, ic_series in factor_ics.items():
            stats = self.compute_ic_statistics(ic_series)
            results[factor] = stats
        
        return pd.DataFrame(results).T
    
    def compute_ic_correlation(
        self,
        factor_ics: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compute correlation between factor ICs.
        
        Parameters
        ----------
        factor_ics : dict
            Dictionary of factor name to IC series.
            
        Returns
        -------
        pd.DataFrame
            IC correlation matrix.
        """
        ic_df = pd.DataFrame(factor_ics)
        return ic_df.corr()
    
    def analyze_ic_reversals(
        self,
        ic_series: pd.Series,
        threshold: float = 0.02,
    ) -> Dict:
        """
        Analyze IC sign reversals.
        
        Parameters
        ----------
        ic_series : pd.Series
            Time series of ICs.
        threshold : float, default 0.02
            Minimum IC magnitude for significance.
            
        Returns
        -------
        dict
            Reversal statistics.
        """
        ic = ic_series.dropna()
        
        # Significant ICs
        significant = ic.abs() > threshold
        
        # Sign changes
        signs = np.sign(ic)
        sign_changes = (signs != signs.shift(1)) & (signs != 0) & (signs.shift(1) != 0)
        
        # Streaks
        positive_streak = 0
        negative_streak = 0
        current_streak = 0
        last_sign = 0
        max_positive = 0
        max_negative = 0
        
        for val in ic:
            if np.isnan(val):
                continue
            
            current_sign = np.sign(val)
            
            if current_sign == last_sign:
                current_streak += 1
            else:
                if last_sign > 0:
                    max_positive = max(max_positive, current_streak)
                elif last_sign < 0:
                    max_negative = max(max_negative, current_streak)
                current_streak = 1
            
            last_sign = current_sign
        
        return {
            'n_reversals': sign_changes.sum(),
            'reversal_frequency': sign_changes.mean(),
            'significant_fraction': significant.mean(),
            'max_positive_streak': max_positive,
            'max_negative_streak': max_negative,
        }


def compute_information_coefficient(
    signals: pd.Series,
    returns: pd.Series,
    method: str = 'spearman',
) -> float:
    """
    Compute single-period information coefficient.
    
    Parameters
    ----------
    signals : pd.Series
        Cross-sectional signals for one period.
    returns : pd.Series
        Cross-sectional returns for subsequent period.
    method : str, default 'spearman'
        Correlation method.
        
    Returns
    -------
    float
        Information coefficient.
    """
    # Align
    common = signals.dropna().index.intersection(returns.dropna().index)
    
    if len(common) < 10:
        return np.nan
    
    sig = signals.loc[common]
    ret = returns.loc[common]
    
    if method == 'spearman':
        ic, _ = stats.spearmanr(sig, ret)
    else:
        ic, _ = stats.pearsonr(sig, ret)
    
    return ic
