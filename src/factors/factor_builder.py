"""
Factor portfolio builder.

Unified interface for constructing multiple factor portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

from .momentum import MomentumFactor
from .value import ValueFactor
from .quality import QualityFactor
from .low_risk import LowRiskFactor

logger = logging.getLogger(__name__)


class FactorBuilder:
    """
    Build multiple factor portfolios from stock-level data.
    
    This class provides a unified interface for constructing standard
    equity factors (momentum, value, quality, low-risk) from CRSP
    and Compustat data.
    
    Parameters
    ----------
    n_portfolios : int, default 10
        Number of portfolios for decile sorts.
    weighting : str, default 'equal'
        Portfolio weighting: 'equal' or 'value'.
        
    Examples
    --------
    >>> builder = FactorBuilder()
    >>> factors = builder.build_all_factors(data)
    >>> momentum = builder.build_momentum(returns)
    """
    
    FACTOR_NAMES = ['Momentum', 'Value', 'Quality', 'Low-Risk']
    
    def __init__(
        self,
        n_portfolios: int = 10,
        weighting: str = 'equal',
    ):
        """Initialize factor builder."""
        self.n_portfolios = n_portfolios
        self.weighting = weighting
        
        # Initialize factor constructors
        self.momentum = MomentumFactor(n_portfolios=n_portfolios)
        self.value = ValueFactor(n_portfolios=n_portfolios)
        self.quality = QualityFactor(n_portfolios=n_portfolios)
        self.low_risk = LowRiskFactor(n_portfolios=n_portfolios)
    
    def build_all_factors(
        self,
        data: Dict[str, pd.DataFrame],
        factors_to_build: List[str] = None,
    ) -> pd.DataFrame:
        """
        Build all factor portfolios.
        
        Parameters
        ----------
        data : dict
            Dictionary containing:
            - 'returns': Stock returns (stocks x time)
            - 'market_returns': Market returns
            - 'market_cap': Market capitalization
            - 'book_equity': Book equity
            - 'gross_profit': Gross profit
            - 'total_assets': Total assets
        factors_to_build : list, optional
            Factors to build. Default is all.
            
        Returns
        -------
        pd.DataFrame
            Factor returns with columns for each factor.
        """
        logger.info("Building factor portfolios...")
        
        if factors_to_build is None:
            factors_to_build = self.FACTOR_NAMES
        
        factor_returns = {}
        
        # Required data
        returns = data.get('returns')
        if returns is None:
            raise ValueError("'returns' required in data")
        
        market_cap = data.get('market_cap')
        
        # Momentum
        if 'Momentum' in factors_to_build:
            logger.info("Building momentum factor...")
            mom_returns = self._build_momentum(returns, market_cap)
            factor_returns['Momentum'] = mom_returns
        
        # Value
        if 'Value' in factors_to_build:
            book_equity = data.get('book_equity')
            market_equity = data.get('market_equity', market_cap)
            
            if book_equity is not None and market_equity is not None:
                logger.info("Building value factor...")
                val_returns = self._build_value(returns, book_equity, 
                                                market_equity, market_cap)
                factor_returns['Value'] = val_returns
            else:
                logger.warning("Skipping value factor: missing book_equity or market_equity")
        
        # Quality
        if 'Quality' in factors_to_build:
            gross_profit = data.get('gross_profit')
            total_assets = data.get('total_assets')
            
            if gross_profit is not None and total_assets is not None:
                logger.info("Building quality factor...")
                qual_returns = self._build_quality(returns, gross_profit,
                                                   total_assets, market_cap)
                factor_returns['Quality'] = qual_returns
            else:
                logger.warning("Skipping quality factor: missing gross_profit or total_assets")
        
        # Low-Risk
        if 'Low-Risk' in factors_to_build:
            market_returns = data.get('market_returns')
            
            if market_returns is not None:
                logger.info("Building low-risk factor...")
                lr_returns = self._build_low_risk(returns, market_returns, market_cap)
                factor_returns['Low-Risk'] = lr_returns
            else:
                logger.warning("Skipping low-risk factor: missing market_returns")
        
        # Combine into DataFrame
        df = pd.DataFrame(factor_returns)
        df = df.sort_index()
        
        logger.info(f"Built {len(df.columns)} factors with {len(df)} observations")
        
        return df
    
    def _build_momentum(
        self,
        returns: pd.DataFrame,
        market_cap: pd.DataFrame = None,
    ) -> pd.Series:
        """Build momentum factor."""
        signals = self.momentum.compute_signals(returns)
        portfolios = self.momentum.construct_portfolios(
            returns, signals, market_cap, self.weighting
        )
        return portfolios['WML']
    
    def _build_value(
        self,
        returns: pd.DataFrame,
        book_equity: pd.DataFrame,
        market_equity: pd.DataFrame,
        market_cap: pd.DataFrame = None,
    ) -> pd.Series:
        """Build value factor."""
        signals = self.value.compute_signals(
            book_equity=book_equity,
            market_equity=market_equity,
        )
        portfolios = self.value.construct_portfolios(
            returns, signals, market_cap, self.weighting
        )
        return portfolios['HML']
    
    def _build_quality(
        self,
        returns: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        market_cap: pd.DataFrame = None,
    ) -> pd.Series:
        """Build quality factor."""
        signals = self.quality.compute_signals(
            gross_profit=gross_profit,
            total_assets=total_assets,
        )
        portfolios = self.quality.construct_portfolios(
            returns, signals, market_cap, self.weighting
        )
        return portfolios['RMW']
    
    def _build_low_risk(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        market_cap: pd.DataFrame = None,
    ) -> pd.Series:
        """Build low-risk factor."""
        signals = self.low_risk.compute_signals(returns, market_returns)
        portfolios = self.low_risk.construct_portfolios(
            returns, signals, market_cap, self.weighting
        )
        return portfolios['BAB']
    
    def build_factor_from_signals(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        long_short: bool = True,
    ) -> pd.Series:
        """
        Build factor from pre-computed signals.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        signals : pd.DataFrame
            Pre-computed signals.
        market_cap : pd.DataFrame, optional
            Market capitalization.
        long_short : bool, default True
            Return long-short portfolio.
            
        Returns
        -------
        pd.Series
            Factor returns.
        """
        portfolio_returns = []
        
        for date in returns.index:
            if date not in signals.index:
                continue
            
            sig = signals.loc[date].dropna()
            ret = returns.loc[date].reindex(sig.index).dropna()
            
            if len(ret) < self.n_portfolios * 5:
                continue
            
            # Assign to portfolios
            ranks = sig.rank(pct=True)
            portfolios = pd.cut(ranks, bins=self.n_portfolios, labels=False) + 1
            
            # Long and short portfolios
            long_mask = portfolios == self.n_portfolios
            short_mask = portfolios == 1
            
            long_ret = ret[long_mask]
            short_ret = ret[short_mask]
            
            if len(long_ret) == 0 or len(short_ret) == 0:
                continue
            
            if self.weighting == 'value' and market_cap is not None:
                long_weights = market_cap.loc[date].reindex(long_ret.index)
                long_weights = long_weights / long_weights.sum()
                long_return = (long_ret * long_weights).sum()
                
                short_weights = market_cap.loc[date].reindex(short_ret.index)
                short_weights = short_weights / short_weights.sum()
                short_return = (short_ret * short_weights).sum()
            else:
                long_return = long_ret.mean()
                short_return = short_ret.mean()
            
            if long_short:
                factor_ret = long_return - short_return
            else:
                factor_ret = long_return
            
            portfolio_returns.append({'date': date, 'return': factor_ret})
        
        df = pd.DataFrame(portfolio_returns).set_index('date')
        return df['return']
    
    def compute_factor_statistics(
        self,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute summary statistics for factors.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics.
        """
        stats = {}
        
        for col in factor_returns.columns:
            ret = factor_returns[col].dropna()
            
            stats[col] = {
                'mean': ret.mean() * 12,  # Annualized
                'std': ret.std() * np.sqrt(12),  # Annualized
                'sharpe': ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0,
                'skew': ret.skew(),
                'kurtosis': ret.kurtosis(),
                't_stat': ret.mean() / (ret.std() / np.sqrt(len(ret))) if ret.std() > 0 else 0,
                'min': ret.min(),
                'max': ret.max(),
                'n_obs': len(ret),
            }
        
        return pd.DataFrame(stats).T
    
    def compute_factor_correlations(
        self,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute factor correlation matrix.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        return factor_returns.corr()


def build_factors(
    data: Dict[str, pd.DataFrame],
    n_portfolios: int = 10,
    weighting: str = 'equal',
) -> pd.DataFrame:
    """
    Convenience function to build all factors.
    
    Parameters
    ----------
    data : dict
        Data dictionary.
    n_portfolios : int, default 10
        Number of portfolios.
    weighting : str, default 'equal'
        Portfolio weighting.
        
    Returns
    -------
    pd.DataFrame
        Factor returns.
    """
    builder = FactorBuilder(n_portfolios=n_portfolios, weighting=weighting)
    return builder.build_all_factors(data)
