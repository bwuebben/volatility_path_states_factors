"""
Return calculation utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def compute_returns(
    prices: pd.Series,
    method: str = 'simple',
) -> pd.Series:
    """
    Compute returns from prices.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    method : str, default 'simple'
        Return method: 'simple' or 'log'.
        
    Returns
    -------
    pd.Series
        Return series.
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")


def compute_cumulative_returns(
    returns: pd.Series,
    method: str = 'simple',
) -> pd.Series:
    """
    Compute cumulative returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    method : str, default 'simple'
        'simple' for compounding, 'log' for sum of log returns.
        
    Returns
    -------
    pd.Series
        Cumulative return series.
    """
    if method == 'simple':
        return (1 + returns).cumprod() - 1
    elif method == 'log':
        return returns.cumsum()
    else:
        raise ValueError(f"Unknown method: {method}")


def annualize_returns(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Annualize average returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    periods_per_year : int, default 12
        Number of periods per year.
        
    Returns
    -------
    float
        Annualized return.
    """
    mean_return = returns.mean()
    return (1 + mean_return) ** periods_per_year - 1


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Annualize volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    periods_per_year : int, default 12
        Number of periods per year.
        
    Returns
    -------
    float
        Annualized volatility.
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_excess_returns(
    returns: pd.Series,
    risk_free: Union[float, pd.Series],
) -> pd.Series:
    """
    Compute excess returns over risk-free rate.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    risk_free : float or pd.Series
        Risk-free rate.
        
    Returns
    -------
    pd.Series
        Excess returns.
    """
    return returns - risk_free


def compute_rolling_returns(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling cumulative returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Rolling window.
        
    Returns
    -------
    pd.Series
        Rolling cumulative returns.
    """
    return returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1,
        raw=False
    )


def compute_drawdown(
    returns: pd.Series,
) -> pd.Series:
    """
    Compute drawdown series.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
        
    Returns
    -------
    pd.Series
        Drawdown series (positive values indicate drawdown).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max
    return drawdown


def compute_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Compute maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
        
    Returns
    -------
    float
        Maximum drawdown.
    """
    return compute_drawdown(returns).max()


def compound_returns(
    returns: pd.Series,
) -> float:
    """
    Compute total compounded return.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
        
    Returns
    -------
    float
        Total compounded return.
    """
    return (1 + returns).prod() - 1


def winsorize_returns(
    returns: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Winsorize returns at specified percentiles.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    lower : float, default 0.01
        Lower percentile.
    upper : float, default 0.99
        Upper percentile.
        
    Returns
    -------
    pd.Series
        Winsorized returns.
    """
    lower_bound = returns.quantile(lower)
    upper_bound = returns.quantile(upper)
    return returns.clip(lower_bound, upper_bound)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0,
    periods_per_year: int = 12,
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    risk_free : float, default 0
        Risk-free rate (annualized).
    periods_per_year : int, default 12
        Number of periods per year.
        
    Returns
    -------
    float
        Sharpe ratio.
    """
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def compute_sortino_ratio(
    returns: pd.Series,
    target: float = 0,
    periods_per_year: int = 12,
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    target : float, default 0
        Target return.
    periods_per_year : int, default 12
        Number of periods per year.
        
    Returns
    -------
    float
        Sortino ratio.
    """
    excess = returns - target / periods_per_year
    downside = returns[returns < target / periods_per_year]
    
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0
    
    downside_std = downside.std() * np.sqrt(periods_per_year)
    annual_excess = excess.mean() * periods_per_year
    
    return annual_excess / downside_std


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Compute Calmar ratio (return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    periods_per_year : int, default 12
        Number of periods per year.
        
    Returns
    -------
    float
        Calmar ratio.
    """
    annual_return = annualize_returns(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if annual_return > 0 else 0
    
    return annual_return / max_dd
