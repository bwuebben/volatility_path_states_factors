"""
Momentum factor construction.

Implements the standard momentum factor (12-1) as well as
alternative momentum measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumFactor:
    """
    Construct momentum factor portfolios.
    
    The standard momentum factor uses returns from month t-12 to t-2,
    skipping the most recent month to avoid short-term reversal.
    
    Parameters
    ----------
    formation_start : int, default 12
        Start of formation period (months ago).
    formation_end : int, default 2
        End of formation period (months ago, skipping recent).
    holding_period : int, default 1
        Holding period in months.
    n_portfolios : int, default 10
        Number of portfolios for decile sorts.
        
    Examples
    --------
    >>> mom = MomentumFactor()
    >>> signals = mom.compute_signals(returns)
    >>> portfolios = mom.construct_portfolios(returns, signals, market_cap)
    """
    
    def __init__(
        self,
        formation_start: int = 12,
        formation_end: int = 2,
        holding_period: int = 1,
        n_portfolios: int = 10,
    ):
        """Initialize momentum factor."""
        self.formation_start = formation_start
        self.formation_end = formation_end
        self.holding_period = holding_period
        self.n_portfolios = n_portfolios
        
    def compute_signals(
        self,
        returns: pd.DataFrame,
        method: str = 'cumulative',
    ) -> pd.DataFrame:
        """
        Compute momentum signals for each stock.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns (stocks as columns, dates as rows).
        method : str, default 'cumulative'
            Signal computation method:
            - 'cumulative': Cumulative return over formation period
            - 'average': Average monthly return
            - 'geometric': Geometric mean return
            
        Returns
        -------
        pd.DataFrame
            Momentum signals (same shape as returns).
        """
        logger.info("Computing momentum signals...")
        
        if method == 'cumulative':
            # Rolling cumulative return from t-12 to t-2
            # (1+r_t-12) * (1+r_t-11) * ... * (1+r_t-2) - 1
            window = self.formation_start - self.formation_end + 1
            
            cum_ret = (1 + returns).rolling(window=window).apply(
                lambda x: x.prod() - 1, raw=True
            )
            
            # Shift to align with signal date (skip recent month)
            signals = cum_ret.shift(self.formation_end)
            
        elif method == 'average':
            window = self.formation_start - self.formation_end + 1
            signals = returns.rolling(window=window).mean().shift(self.formation_end)
            
        elif method == 'geometric':
            window = self.formation_start - self.formation_end + 1
            signals = returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() ** (1/len(x)) - 1, raw=True
            ).shift(self.formation_end)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return signals
    
    def construct_portfolios(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        weighting: str = 'equal',
    ) -> pd.DataFrame:
        """
        Construct momentum portfolios.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        signals : pd.DataFrame
            Momentum signals.
        market_cap : pd.DataFrame, optional
            Market capitalization for value weighting.
        weighting : str, default 'equal'
            Portfolio weighting: 'equal' or 'value'.
            
        Returns
        -------
        pd.DataFrame
            Portfolio returns with columns for each decile and WML.
        """
        logger.info("Constructing momentum portfolios...")
        
        portfolio_returns = []
        
        for date in returns.index:
            if date not in signals.index:
                continue
            
            # Get signals for this date
            sig = signals.loc[date].dropna()
            ret = returns.loc[date].reindex(sig.index).dropna()
            
            if len(ret) < self.n_portfolios * 5:  # Need enough stocks
                continue
            
            # Assign to portfolios based on signal ranking
            ranks = sig.rank(pct=True)
            portfolios = pd.cut(ranks, bins=self.n_portfolios, labels=False) + 1
            
            # Compute portfolio returns
            port_ret = {}
            
            for p in range(1, self.n_portfolios + 1):
                mask = portfolios == p
                stocks = ret[mask]
                
                if len(stocks) == 0:
                    continue
                
                if weighting == 'value' and market_cap is not None:
                    weights = market_cap.loc[date].reindex(stocks.index)
                    weights = weights / weights.sum()
                    port_ret[f'P{p}'] = (stocks * weights).sum()
                else:
                    port_ret[f'P{p}'] = stocks.mean()
            
            # WML (Winners minus Losers)
            if f'P{self.n_portfolios}' in port_ret and 'P1' in port_ret:
                port_ret['WML'] = port_ret[f'P{self.n_portfolios}'] - port_ret['P1']
            
            port_ret['date'] = date
            portfolio_returns.append(port_ret)
        
        df = pd.DataFrame(portfolio_returns).set_index('date')
        
        return df
    
    def compute_wml_returns(
        self,
        returns: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        weighting: str = 'equal',
    ) -> pd.Series:
        """
        Compute WML (Winners minus Losers) return series.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        market_cap : pd.DataFrame, optional
            Market capitalization.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.Series
            WML return series.
        """
        signals = self.compute_signals(returns)
        portfolios = self.construct_portfolios(
            returns, signals, market_cap, weighting
        )
        
        return portfolios['WML']
    
    def compute_industry_neutral_momentum(
        self,
        returns: pd.DataFrame,
        industries: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute industry-neutral momentum signals.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        industries : pd.DataFrame
            Industry classifications.
            
        Returns
        -------
        pd.DataFrame
            Industry-adjusted momentum signals.
        """
        raw_signals = self.compute_signals(returns)
        
        # Subtract industry mean
        adjusted_signals = pd.DataFrame(index=raw_signals.index,
                                        columns=raw_signals.columns)
        
        for date in raw_signals.index:
            if date not in industries.index:
                continue
            
            sig = raw_signals.loc[date]
            ind = industries.loc[date]
            
            # Industry means
            ind_means = sig.groupby(ind).transform('mean')
            adjusted_signals.loc[date] = sig - ind_means
        
        return adjusted_signals


def compute_momentum_returns(
    returns: pd.DataFrame,
    formation_period: Tuple[int, int] = (12, 2),
    n_portfolios: int = 10,
) -> pd.Series:
    """
    Convenience function to compute momentum factor returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns.
    formation_period : tuple, default (12, 2)
        Formation period (start_month, end_month).
    n_portfolios : int, default 10
        Number of portfolios.
        
    Returns
    -------
    pd.Series
        Momentum factor returns (WML).
    """
    mom = MomentumFactor(
        formation_start=formation_period[0],
        formation_end=formation_period[1],
        n_portfolios=n_portfolios,
    )
    
    return mom.compute_wml_returns(returns)
