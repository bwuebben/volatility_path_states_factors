"""
Performance analysis and metrics computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    
    Attributes
    ----------
    mean_return : float
        Annualized mean return.
    volatility : float
        Annualized volatility.
    sharpe_ratio : float
        Sharpe ratio.
    sortino_ratio : float
        Sortino ratio.
    max_drawdown : float
        Maximum drawdown.
    calmar_ratio : float
        Calmar ratio (return / max drawdown).
    skewness : float
        Return skewness.
    kurtosis : float
        Return kurtosis.
    var_5 : float
        5th percentile return (Value at Risk).
    cvar_5 : float
        Conditional VaR (Expected Shortfall).
    hit_rate : float
        Fraction of positive returns.
    avg_win : float
        Average winning return.
    avg_loss : float
        Average losing return.
    win_loss_ratio : float
        Ratio of average win to average loss.
    best_month : float
        Best monthly return.
    worst_month : float
        Worst monthly return.
    """
    mean_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    var_5: float
    cvar_5: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    best_month: float
    worst_month: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.to_dict())
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  mean_return={self.mean_return:.4f},\n"
            f"  volatility={self.volatility:.4f},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.4f},\n"
            f"  max_drawdown={self.max_drawdown:.4f}\n"
            f")"
        )


class PerformanceAnalyzer:
    """
    Analyze portfolio and factor performance.
    
    Parameters
    ----------
    annualization_factor : int, default 12
        Factor for annualization (12 for monthly data).
    risk_free_rate : float, default 0.0
        Risk-free rate for Sharpe ratio calculation.
        
    Examples
    --------
    >>> analyzer = PerformanceAnalyzer()
    >>> metrics = analyzer.compute_metrics(returns)
    >>> print(metrics.sharpe_ratio)
    0.85
    
    >>> comparison = analyzer.compare_strategies(
    ...     {'Baseline': baseline_returns, 'Conditioned': cond_returns}
    ... )
    """
    
    def __init__(
        self,
        annualization_factor: int = 12,
        risk_free_rate: float = 0.0,
    ):
        """Initialize performance analyzer."""
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.sqrt_factor = np.sqrt(annualization_factor)
        
    def compute_metrics(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        column: Optional[str] = None,
    ) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.
        
        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Return series.
        column : str, optional
            Column to use if DataFrame is passed.
            
        Returns
        -------
        PerformanceMetrics
            Computed performance metrics.
        """
        # Extract series
        if isinstance(returns, pd.DataFrame):
            if column:
                ret = returns[column]
            else:
                ret = returns.iloc[:, 0]
        else:
            ret = returns
        
        ret = ret.dropna()
        
        if len(ret) == 0:
            return self._empty_metrics()
        
        # Basic statistics
        mean_ret = ret.mean() * self.annualization_factor
        vol = ret.std() * self.sqrt_factor
        
        # Risk-adjusted returns
        excess_ret = ret - self.risk_free_rate / self.annualization_factor
        sharpe = excess_ret.mean() / ret.std() * self.sqrt_factor if ret.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_ret = ret[ret < 0]
        downside_std = downside_ret.std() * self.sqrt_factor if len(downside_ret) > 0 else vol
        sortino = mean_ret / downside_std if downside_std > 0 else 0
        
        # Drawdown analysis
        max_dd = self._compute_max_drawdown(ret)
        calmar = mean_ret / max_dd if max_dd > 0 else 0
        
        # Distribution metrics
        skew = ret.skew()
        kurt = ret.kurtosis()
        
        # VaR and CVaR
        var_5 = ret.quantile(0.05)
        cvar_5 = ret[ret <= var_5].mean() if len(ret[ret <= var_5]) > 0 else var_5
        
        # Win/loss analysis
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        hit_rate = len(wins) / len(ret) if len(ret) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        return PerformanceMetrics(
            mean_return=mean_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            skewness=skew,
            kurtosis=kurt,
            var_5=var_5,
            cvar_5=cvar_5,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            best_month=ret.max(),
            worst_month=ret.min(),
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for no data."""
        return PerformanceMetrics(
            mean_return=0, volatility=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, calmar_ratio=0, skewness=0, kurtosis=0,
            var_5=0, cvar_5=0, hit_rate=0, avg_win=0, avg_loss=0,
            win_loss_ratio=0, best_month=0, worst_month=0,
        )
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown from return series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()
    
    def compute_drawdown_series(
        self,
        returns: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute drawdown time series.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
            
        Returns
        -------
        drawdown : pd.Series
            Drawdown series.
        duration : pd.Series
            Drawdown duration in periods.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        
        # Compute duration
        is_underwater = drawdown > 0
        duration = pd.Series(0, index=returns.index)
        
        current_duration = 0
        for i, underwater in enumerate(is_underwater):
            if underwater:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return drawdown, duration
    
    def compute_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 36,
    ) -> pd.DataFrame:
        """
        Compute rolling performance metrics.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        window : int, default 36
            Rolling window size.
            
        Returns
        -------
        pd.DataFrame
            Rolling metrics.
        """
        rolling = pd.DataFrame(index=returns.index)
        
        # Rolling mean and volatility
        rolling['mean'] = returns.rolling(window).mean() * self.annualization_factor
        rolling['volatility'] = returns.rolling(window).std() * self.sqrt_factor
        
        # Rolling Sharpe
        rolling['sharpe'] = (
            returns.rolling(window).mean() / 
            returns.rolling(window).std() * self.sqrt_factor
        )
        
        # Rolling skewness
        rolling['skewness'] = returns.rolling(window).skew()
        
        return rolling
    
    def compare_strategies(
        self,
        strategies: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Parameters
        ----------
        strategies : dict
            Dictionary of strategy name to return series.
            
        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each strategy.
        """
        results = {}
        
        for name, returns in strategies.items():
            metrics = self.compute_metrics(returns)
            results[name] = metrics.to_dict()
        
        df = pd.DataFrame(results).T
        
        # Format percentages
        pct_cols = ['mean_return', 'volatility', 'max_drawdown', 'var_5', 
                    'cvar_5', 'hit_rate', 'best_month', 'worst_month']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col] * 100
        
        return df
    
    def analyze_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Analyze performance by regime.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        regimes : pd.Series
            Regime classification.
            
        Returns
        -------
        pd.DataFrame
            Performance metrics by regime.
        """
        # Align data
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        results = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns = returns.loc[mask]
            
            if len(regime_returns) < 2:
                continue
            
            metrics = self.compute_metrics(regime_returns)
            results[regime] = {
                'n_obs': len(regime_returns),
                'frequency': len(regime_returns) / len(returns) * 100,
                'mean_return': metrics.mean_return * 100,
                'volatility': metrics.volatility * 100,
                'sharpe_ratio': metrics.sharpe_ratio,
                'skewness': metrics.skewness,
                'hit_rate': metrics.hit_rate * 100,
                'worst_month': metrics.worst_month * 100,
            }
        
        return pd.DataFrame(results).T
    
    def compute_crash_statistics(
        self,
        returns: pd.Series,
        threshold_percentile: float = 5,
    ) -> Dict:
        """
        Compute crash statistics.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
        threshold_percentile : float, default 5
            Percentile to define crashes.
            
        Returns
        -------
        dict
            Crash statistics.
        """
        threshold = returns.quantile(threshold_percentile / 100)
        crashes = returns[returns <= threshold]
        
        return {
            'threshold': threshold * 100,
            'n_crashes': len(crashes),
            'crash_frequency': len(crashes) / len(returns) * 100,
            'avg_crash_return': crashes.mean() * 100,
            'worst_crash': crashes.min() * 100,
            'crash_dates': crashes.index.tolist(),
        }
    
    def compute_correlation_matrix(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for multiple return series.
        
        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame with multiple return series.
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        return returns.corr()
    
    def compute_information_ratio(
        self,
        returns: pd.Series,
        benchmark: pd.Series,
    ) -> float:
        """
        Compute information ratio vs benchmark.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns.
        benchmark : pd.Series
            Benchmark returns.
            
        Returns
        -------
        float
            Information ratio.
        """
        # Align
        common_idx = returns.index.intersection(benchmark.index)
        excess = returns.loc[common_idx] - benchmark.loc[common_idx]
        
        if excess.std() == 0:
            return 0
        
        return excess.mean() / excess.std() * self.sqrt_factor
    
    def summary_table(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        format_pct: bool = True,
    ) -> pd.DataFrame:
        """
        Generate summary table of performance metrics.
        
        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Return series or DataFrame with multiple series.
        format_pct : bool, default True
            Format percentages.
            
        Returns
        -------
        pd.DataFrame
            Summary table.
        """
        if isinstance(returns, pd.Series):
            returns = returns.to_frame('returns')
        
        results = {}
        for col in returns.columns:
            metrics = self.compute_metrics(returns[col])
            results[col] = metrics.to_dict()
        
        df = pd.DataFrame(results)
        
        if format_pct:
            pct_rows = ['mean_return', 'volatility', 'max_drawdown', 
                        'var_5', 'cvar_5', 'hit_rate', 'avg_win', 'avg_loss']
            for row in pct_rows:
                if row in df.index:
                    df.loc[row] = df.loc[row] * 100
        
        return df


def compute_performance_metrics(
    returns: pd.Series,
    annualize: bool = True,
) -> Dict:
    """
    Convenience function to compute basic performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    annualize : bool, default True
        Annualize metrics.
        
    Returns
    -------
    dict
        Performance metrics.
    """
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.compute_metrics(returns)
    return metrics.to_dict()
