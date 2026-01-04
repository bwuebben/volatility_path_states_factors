"""
Quality factor construction.

Implements profitability, investment, and composite quality measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityFactor:
    """
    Construct quality factor portfolios.
    
    Quality is measured using profitability metrics (ROE, ROA, gross profit),
    investment patterns, and earnings quality measures.
    
    Parameters
    ----------
    signal_type : str, default 'profitability'
        Signal type: 'profitability', 'investment', 'composite'.
    exclude_financials : bool, default True
        Whether to exclude financial firms (SIC 6000-6999).
    lag_months : int, default 6
        Months to lag fundamental data.
    n_portfolios : int, default 10
        Number of portfolios for sorts.
        
    Examples
    --------
    >>> quality = QualityFactor()
    >>> signals = quality.compute_signals(fundamentals)
    >>> portfolios = quality.construct_portfolios(returns, signals)
    """
    
    # SIC codes for financials
    FINANCIAL_SICS = range(6000, 7000)
    
    def __init__(
        self,
        signal_type: str = 'profitability',
        exclude_financials: bool = True,
        lag_months: int = 6,
        n_portfolios: int = 10,
    ):
        """Initialize quality factor."""
        self.signal_type = signal_type
        self.exclude_financials = exclude_financials
        self.lag_months = lag_months
        self.n_portfolios = n_portfolios
    
    def compute_signals(
        self,
        gross_profit: pd.DataFrame = None,
        total_assets: pd.DataFrame = None,
        net_income: pd.DataFrame = None,
        book_equity: pd.DataFrame = None,
        operating_profit: pd.DataFrame = None,
        revenue: pd.DataFrame = None,
        total_assets_lag: pd.DataFrame = None,
        sic_codes: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute quality signals.
        
        Parameters
        ----------
        gross_profit : pd.DataFrame
            Gross profit (revenue - COGS).
        total_assets : pd.DataFrame
            Total assets.
        net_income : pd.DataFrame
            Net income.
        book_equity : pd.DataFrame
            Book equity.
        operating_profit : pd.DataFrame
            Operating profitability.
        revenue : pd.DataFrame
            Revenue.
        total_assets_lag : pd.DataFrame
            Lagged total assets (for investment).
        sic_codes : pd.DataFrame
            SIC codes for industry filtering.
            
        Returns
        -------
        pd.DataFrame
            Quality signals.
        """
        logger.info(f"Computing {self.signal_type} quality signals...")
        
        if self.signal_type == 'profitability':
            signals = self._compute_profitability_signals(
                gross_profit, total_assets, net_income, book_equity
            )
            
        elif self.signal_type == 'investment':
            signals = self._compute_investment_signals(
                total_assets, total_assets_lag
            )
            
        elif self.signal_type == 'composite':
            signals = self._compute_composite_signals(
                gross_profit, total_assets, net_income, book_equity,
                total_assets_lag
            )
            
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
        
        # Exclude financials if requested
        if self.exclude_financials and sic_codes is not None:
            signals = self._exclude_financials(signals, sic_codes)
        
        # Lag signals
        signals = signals.shift(self.lag_months)
        
        return signals
    
    def _compute_profitability_signals(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame = None,
        book_equity: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Compute profitability signals."""
        # Gross profitability (GP/AT) - Novy-Marx
        if gross_profit is not None and total_assets is not None:
            gp_at = gross_profit / total_assets
            return self._winsorize(gp_at, 0.01, 0.99)
        
        # ROE as fallback
        if net_income is not None and book_equity is not None:
            roe = net_income / book_equity
            return self._winsorize(roe, 0.01, 0.99)
        
        raise ValueError("Need gross_profit/total_assets or net_income/book_equity")
    
    def _compute_investment_signals(
        self,
        total_assets: pd.DataFrame,
        total_assets_lag: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Compute investment signals (low investment = high quality)."""
        if total_assets_lag is None:
            total_assets_lag = total_assets.shift(12)  # Annual lag
        
        # Asset growth (lower is better for quality)
        asset_growth = (total_assets - total_assets_lag) / total_assets_lag
        
        # Negative because low investment is high quality
        signals = -self._winsorize(asset_growth, 0.01, 0.99)
        
        return signals
    
    def _compute_composite_signals(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        total_assets_lag: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute composite quality signals."""
        signals_list = []
        
        # Profitability
        if gross_profit is not None and total_assets is not None:
            gp_at = gross_profit / total_assets
            signals_list.append(self._normalize_cross_section(gp_at))
        
        # ROE
        if net_income is not None and book_equity is not None:
            roe = net_income / book_equity
            signals_list.append(self._normalize_cross_section(roe))
        
        # Investment (negative)
        if total_assets is not None:
            if total_assets_lag is None:
                total_assets_lag = total_assets.shift(12)
            asset_growth = (total_assets - total_assets_lag) / total_assets_lag
            signals_list.append(-self._normalize_cross_section(asset_growth))
        
        if len(signals_list) == 0:
            raise ValueError("No valid signals to compute")
        
        # Average normalized signals
        composite = pd.concat(signals_list).groupby(level=0).mean()
        
        return composite
    
    def _normalize_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize signals cross-sectionally."""
        return df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)
    
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
    
    def _exclude_financials(
        self,
        signals: pd.DataFrame,
        sic_codes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Exclude financial firms."""
        mask = ~sic_codes.isin(self.FINANCIAL_SICS)
        return signals.where(mask)
    
    def construct_portfolios(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        weighting: str = 'equal',
    ) -> pd.DataFrame:
        """
        Construct quality portfolios.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        signals : pd.DataFrame
            Quality signals.
        market_cap : pd.DataFrame, optional
            Market capitalization for value weighting.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.DataFrame
            Portfolio returns with RMW (Robust minus Weak).
        """
        logger.info("Constructing quality portfolios...")
        
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
            
            # RMW (Robust minus Weak)
            if f'P{self.n_portfolios}' in port_ret and 'P1' in port_ret:
                port_ret['RMW'] = port_ret[f'P{self.n_portfolios}'] - port_ret['P1']
            
            port_ret['date'] = date
            portfolio_returns.append(port_ret)
        
        df = pd.DataFrame(portfolio_returns).set_index('date')
        
        return df
    
    def compute_rmw_returns(
        self,
        returns: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        weighting: str = 'equal',
    ) -> pd.Series:
        """
        Compute RMW return series.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        gross_profit : pd.DataFrame
            Gross profit.
        total_assets : pd.DataFrame
            Total assets.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.Series
            RMW return series.
        """
        signals = self.compute_signals(
            gross_profit=gross_profit,
            total_assets=total_assets,
        )
        portfolios = self.construct_portfolios(returns, signals, weighting=weighting)
        
        return portfolios['RMW']


def compute_quality_returns(
    returns: pd.DataFrame,
    gross_profit: pd.DataFrame,
    total_assets: pd.DataFrame,
    exclude_financials: bool = True,
) -> pd.Series:
    """
    Convenience function to compute quality factor returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns.
    gross_profit : pd.DataFrame
        Gross profit.
    total_assets : pd.DataFrame
        Total assets.
    exclude_financials : bool, default True
        Exclude financial firms.
        
    Returns
    -------
    pd.Series
        Quality factor returns (RMW).
    """
    quality = QualityFactor(exclude_financials=exclude_financials)
    return quality.compute_rmw_returns(returns, gross_profit, total_assets)
