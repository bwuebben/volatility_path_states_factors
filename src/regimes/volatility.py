"""
Volatility calculation utilities.

This module provides functions for computing realized volatility
at multiple time horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """
    Calculate realized volatility at multiple time horizons.
    
    This class computes various volatility measures from daily returns,
    including short-term, medium-term, and long-term realized volatility,
    as well as the volatility ratio used for regime classification.
    
    Parameters
    ----------
    horizons : dict, optional
        Dictionary mapping horizon names to number of trading days.
        Default: {'1w': 5, '1m': 21, '3m': 63, '6m': 126}
    annualization_factor : int, default 252
        Number of trading days per year for annualization.
        
    Attributes
    ----------
    horizons : dict
        Volatility calculation horizons.
    annualization_factor : int
        Annualization factor.
        
    Examples
    --------
    >>> calc = VolatilityCalculator()
    >>> vol = calc.compute(daily_returns)
    >>> print(vol.columns)
    Index(['sigma_1w', 'sigma_1m', 'sigma_3m', 'sigma_6m', 'rho_sigma'], ...)
    """
    
    DEFAULT_HORIZONS = {
        '1w': 5,
        '1m': 21,
        '3m': 63,
        '6m': 126,
    }
    
    def __init__(
        self,
        horizons: Optional[Dict[str, int]] = None,
        annualization_factor: int = 252,
    ):
        """Initialize volatility calculator."""
        self.horizons = horizons or self.DEFAULT_HORIZONS.copy()
        self.annualization_factor = annualization_factor
        
    def compute(
        self,
        daily_returns: Union[pd.Series, pd.DataFrame],
        return_daily: bool = False,
    ) -> pd.DataFrame:
        """
        Compute realized volatility at multiple horizons.
        
        Parameters
        ----------
        daily_returns : pd.Series or pd.DataFrame
            Daily returns. If DataFrame, uses first column or 'market_return'.
        return_daily : bool, default False
            If True, return daily frequency. Otherwise, resample to monthly.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with volatility measures:
            - sigma_{horizon} for each horizon
            - rho_sigma: volatility ratio (short/long)
        """
        # Extract return series
        if isinstance(daily_returns, pd.DataFrame):
            if 'market_return' in daily_returns.columns:
                returns = daily_returns['market_return']
            else:
                returns = daily_returns.iloc[:, 0]
        else:
            returns = daily_returns
            
        # Compute volatility at each horizon
        vol_data = {}
        for name, days in self.horizons.items():
            vol = self._realized_volatility(returns, days)
            vol_data[f'sigma_{name}'] = vol
        
        # Compute volatility ratio
        short_key = f'sigma_{min(self.horizons.keys(), key=lambda k: self.horizons[k])}'
        long_key = f'sigma_{max(self.horizons.keys(), key=lambda k: self.horizons[k])}'
        
        # Use 1w and 3m for ratio by default
        if 'sigma_1w' in vol_data and 'sigma_3m' in vol_data:
            vol_data['rho_sigma'] = vol_data['sigma_1w'] / vol_data['sigma_3m']
        else:
            vol_data['rho_sigma'] = vol_data[short_key] / vol_data[long_key]
        
        # Create DataFrame
        vol_df = pd.DataFrame(vol_data)
        
        # Resample to monthly if requested
        if not return_daily:
            vol_df = vol_df.resample('M').last()
            
        return vol_df
    
    def _realized_volatility(
        self,
        returns: pd.Series,
        window: int,
    ) -> pd.Series:
        """
        Compute realized volatility over a rolling window.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns.
        window : int
            Number of days in the window.
            
        Returns
        -------
        pd.Series
            Annualized realized volatility.
        """
        # Compute rolling standard deviation
        rolling_std = returns.rolling(window=window, min_periods=max(1, window // 2)).std()
        
        # Annualize
        annualized_vol = rolling_std * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def compute_drawdown(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        lookback: int = 126,
    ) -> pd.DataFrame:
        """
        Compute drawdown measures.
        
        Parameters
        ----------
        prices : pd.Series or pd.DataFrame
            Price series or index level.
        lookback : int, default 126
            Lookback period for computing running maximum.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with:
            - drawdown: current drawdown from high
            - drawdown_speed: rate of drawdown
            - days_from_high: days since last high
        """
        # Extract price series
        if isinstance(prices, pd.DataFrame):
            if 'price' in prices.columns:
                price = prices['price']
            elif 'close' in prices.columns:
                price = prices['close']
            else:
                price = prices.iloc[:, 0]
        else:
            price = prices
        
        # Compute running maximum
        running_max = price.rolling(window=lookback, min_periods=1).max()
        
        # Compute drawdown
        drawdown = (running_max - price) / running_max
        
        # Compute days from high
        is_high = price >= running_max
        days_from_high = (~is_high).groupby((is_high).cumsum()).cumsum()
        
        # Compute drawdown speed
        drawdown_speed = drawdown / (days_from_high.clip(lower=1)) * 21  # Per month
        
        return pd.DataFrame({
            'drawdown': drawdown,
            'drawdown_speed': drawdown_speed,
            'days_from_high': days_from_high,
        })
    
    def compute_returns(
        self,
        prices: pd.Series,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute returns at multiple horizons.
        
        Parameters
        ----------
        prices : pd.Series
            Price series.
        horizons : list, optional
            List of horizons in trading days. Default: [21, 63] (1m, 3m).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with return columns for each horizon.
        """
        horizons = horizons or [21, 63]
        
        returns = {}
        for h in horizons:
            ret = prices.pct_change(periods=h)
            if h == 21:
                returns['ret_1m'] = ret
            elif h == 63:
                returns['ret_3m'] = ret
            else:
                returns[f'ret_{h}d'] = ret
        
        return pd.DataFrame(returns)


def compute_realized_volatility(
    daily_returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Convenience function to compute realized volatility.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    window : int, default 21
        Rolling window size.
    annualize : bool, default True
        Whether to annualize the volatility.
        
    Returns
    -------
    pd.Series
        Realized volatility series.
    """
    vol = daily_returns.rolling(window=window, min_periods=max(1, window // 2)).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def compute_volatility_ratio(
    short_vol: pd.Series,
    long_vol: pd.Series,
) -> pd.Series:
    """
    Compute volatility ratio.
    
    Parameters
    ----------
    short_vol : pd.Series
        Short-horizon volatility.
    long_vol : pd.Series
        Long-horizon volatility.
        
    Returns
    -------
    pd.Series
        Volatility ratio (short / long).
    """
    return short_vol / long_vol


def compute_all_volatility_measures(
    daily_returns: pd.Series,
    horizons: Dict[str, int] = None,
) -> pd.DataFrame:
    """
    Compute all volatility measures used in the paper.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    horizons : dict, optional
        Volatility horizons.
        
    Returns
    -------
    pd.DataFrame
        All volatility measures.
    """
    calculator = VolatilityCalculator(horizons=horizons)
    return calculator.compute(daily_returns)
