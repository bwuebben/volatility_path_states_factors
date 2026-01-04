"""
Value factor construction.

Implements book-to-market and related value measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ValueFactor:
    """
    Construct value factor portfolios.
    
    The standard value factor uses book-to-market equity ratio,
    with book equity lagged to ensure availability.
    
    Parameters
    ----------
    signal_type : str, default 'bm'
        Signal type: 'bm' (book-to-market), 'ep' (earnings-to-price),
        'cf' (cash flow-to-price), or 'composite'.
    lag_months : int, default 6
        Months to lag book values for availability.
    n_portfolios : int, default 10
        Number of portfolios for sorts.
        
    Examples
    --------
    >>> value = ValueFactor()
    >>> signals = value.compute_signals(book_equity, market_equity)
    >>> portfolios = value.construct_portfolios(returns, signals)
    """
    
    def __init__(
        self,
        signal_type: str = 'bm',
        lag_months: int = 6,
        n_portfolios: int = 10,
    ):
        """Initialize value factor."""
        self.signal_type = signal_type
        self.lag_months = lag_months
        self.n_portfolios = n_portfolios
    
    def compute_signals(
        self,
        book_equity: pd.DataFrame = None,
        market_equity: pd.DataFrame = None,
        earnings: pd.DataFrame = None,
        cash_flow: pd.DataFrame = None,
        price: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute value signals.
        
        Parameters
        ----------
        book_equity : pd.DataFrame
            Book equity values.
        market_equity : pd.DataFrame
            Market equity values.
        earnings : pd.DataFrame
            Earnings values (for E/P).
        cash_flow : pd.DataFrame
            Cash flow values (for CF/P).
        price : pd.DataFrame
            Stock prices.
            
        Returns
        -------
        pd.DataFrame
            Value signals.
        """
        logger.info(f"Computing {self.signal_type} value signals...")
        
        if self.signal_type == 'bm':
            # Book-to-market
            if book_equity is None or market_equity is None:
                raise ValueError("book_equity and market_equity required for B/M")
            
            # Lag book equity
            book_lagged = book_equity.shift(self.lag_months)
            signals = book_lagged / market_equity
            
        elif self.signal_type == 'ep':
            # Earnings-to-price
            if earnings is None or market_equity is None:
                raise ValueError("earnings and market_equity required for E/P")
            
            earnings_lagged = earnings.shift(self.lag_months)
            signals = earnings_lagged / market_equity
            
        elif self.signal_type == 'cf':
            # Cash flow-to-price
            if cash_flow is None or market_equity is None:
                raise ValueError("cash_flow and market_equity required for CF/P")
            
            cf_lagged = cash_flow.shift(self.lag_months)
            signals = cf_lagged / market_equity
            
        elif self.signal_type == 'composite':
            # Composite value (average of normalized signals)
            signals_list = []
            
            if book_equity is not None and market_equity is not None:
                bm = book_equity.shift(self.lag_months) / market_equity
                signals_list.append(self._normalize_cross_section(bm))
            
            if earnings is not None and market_equity is not None:
                ep = earnings.shift(self.lag_months) / market_equity
                signals_list.append(self._normalize_cross_section(ep))
            
            if cash_flow is not None and market_equity is not None:
                cfp = cash_flow.shift(self.lag_months) / market_equity
                signals_list.append(self._normalize_cross_section(cfp))
            
            if len(signals_list) == 0:
                raise ValueError("At least one value measure required")
            
            signals = pd.concat(signals_list).groupby(level=0).mean()
            
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
        
        # Remove extreme values
        signals = self._winsorize(signals, 0.01, 0.99)
        
        return signals
    
    def _normalize_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize signals cross-sectionally."""
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def _winsorize(
        self,
        df: pd.DataFrame,
        lower: float,
        upper: float,
    ) -> pd.DataFrame:
        """Winsorize at percentiles."""
        return df.apply(
            lambda x: x.clip(x.quantile(lower), x.quantile(upper)),
            axis=1
        )
    
    def construct_portfolios(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        weighting: str = 'equal',
    ) -> pd.DataFrame:
        """
        Construct value portfolios.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        signals : pd.DataFrame
            Value signals.
        market_cap : pd.DataFrame, optional
            Market capitalization for value weighting.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.DataFrame
            Portfolio returns with HML (High minus Low).
        """
        logger.info("Constructing value portfolios...")
        
        portfolio_returns = []
        
        for date in returns.index:
            if date not in signals.index:
                continue
            
            sig = signals.loc[date].dropna()
            ret = returns.loc[date].reindex(sig.index).dropna()
            
            if len(ret) < self.n_portfolios * 5:
                continue
            
            # Positive B/M only (exclude negative book equity)
            mask = sig > 0
            sig = sig[mask]
            ret = ret.reindex(sig.index).dropna()
            
            if len(ret) < self.n_portfolios * 5:
                continue
            
            # Assign to portfolios
            ranks = sig.rank(pct=True)
            portfolios = pd.cut(ranks, bins=self.n_portfolios, labels=False) + 1
            
            port_ret = {}
            
            for p in range(1, self.n_portfolios + 1):
                port_mask = portfolios == p
                stocks = ret[port_mask]
                
                if len(stocks) == 0:
                    continue
                
                if weighting == 'value' and market_cap is not None:
                    weights = market_cap.loc[date].reindex(stocks.index)
                    weights = weights / weights.sum()
                    port_ret[f'P{p}'] = (stocks * weights).sum()
                else:
                    port_ret[f'P{p}'] = stocks.mean()
            
            # HML (High B/M minus Low B/M)
            if f'P{self.n_portfolios}' in port_ret and 'P1' in port_ret:
                port_ret['HML'] = port_ret[f'P{self.n_portfolios}'] - port_ret['P1']
            
            port_ret['date'] = date
            portfolio_returns.append(port_ret)
        
        df = pd.DataFrame(portfolio_returns).set_index('date')
        
        return df
    
    def compute_hml_returns(
        self,
        returns: pd.DataFrame,
        book_equity: pd.DataFrame,
        market_equity: pd.DataFrame,
        weighting: str = 'equal',
    ) -> pd.Series:
        """
        Compute HML return series.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        book_equity : pd.DataFrame
            Book equity.
        market_equity : pd.DataFrame
            Market equity.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.Series
            HML return series.
        """
        signals = self.compute_signals(
            book_equity=book_equity,
            market_equity=market_equity,
        )
        portfolios = self.construct_portfolios(returns, signals, weighting=weighting)
        
        return portfolios['HML']


def compute_value_returns(
    returns: pd.DataFrame,
    book_equity: pd.DataFrame,
    market_equity: pd.DataFrame,
    lag_months: int = 6,
) -> pd.Series:
    """
    Convenience function to compute value factor returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns.
    book_equity : pd.DataFrame
        Book equity values.
    market_equity : pd.DataFrame
        Market equity values.
    lag_months : int, default 6
        Lag for book equity.
        
    Returns
    -------
    pd.Series
        Value factor returns (HML).
    """
    value = ValueFactor(lag_months=lag_months)
    return value.compute_hml_returns(returns, book_equity, market_equity)
