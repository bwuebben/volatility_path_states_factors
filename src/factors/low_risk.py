"""
Low-risk/beta factor construction.

Implements betting-against-beta and low volatility factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LowRiskFactor:
    """
    Construct low-risk factor portfolios.
    
    Implements betting-against-beta (BAB) and low volatility strategies.
    The factor goes long low-beta/low-vol stocks and short high-beta/high-vol stocks.
    
    Parameters
    ----------
    signal_type : str, default 'beta'
        Signal type: 'beta', 'volatility', 'idio_vol'.
    beta_window : int, default 60
        Window for beta estimation (months).
    vol_window : int, default 12
        Window for volatility estimation (months).
    shrinkage : float, default 0.6
        Shrinkage toward cross-sectional mean beta.
    n_portfolios : int, default 10
        Number of portfolios for sorts.
        
    Examples
    --------
    >>> low_risk = LowRiskFactor()
    >>> signals = low_risk.compute_signals(returns, market_returns)
    >>> portfolios = low_risk.construct_portfolios(returns, signals)
    """
    
    def __init__(
        self,
        signal_type: str = 'beta',
        beta_window: int = 60,
        vol_window: int = 12,
        shrinkage: float = 0.6,
        n_portfolios: int = 10,
    ):
        """Initialize low-risk factor."""
        self.signal_type = signal_type
        self.beta_window = beta_window
        self.vol_window = vol_window
        self.shrinkage = shrinkage
        self.n_portfolios = n_portfolios
    
    def compute_signals(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series = None,
        rf: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Compute low-risk signals (lower = better).
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        market_returns : pd.Series
            Market returns (required for beta).
        rf : pd.Series, optional
            Risk-free rate.
            
        Returns
        -------
        pd.DataFrame
            Low-risk signals (inverted so lower risk = higher signal).
        """
        logger.info(f"Computing {self.signal_type} low-risk signals...")
        
        if self.signal_type == 'beta':
            raw_signals = self._compute_beta_signals(returns, market_returns, rf)
            
        elif self.signal_type == 'volatility':
            raw_signals = self._compute_volatility_signals(returns)
            
        elif self.signal_type == 'idio_vol':
            raw_signals = self._compute_idio_vol_signals(returns, market_returns)
            
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
        
        # Invert signals (low risk = high signal for long position)
        signals = -raw_signals
        
        return signals
    
    def _compute_beta_signals(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        rf: pd.Series = None,
    ) -> pd.DataFrame:
        """Compute beta signals."""
        if market_returns is None:
            raise ValueError("market_returns required for beta estimation")
        
        # Excess returns
        if rf is not None:
            excess_returns = returns.sub(rf, axis=0)
            excess_market = market_returns - rf
        else:
            excess_returns = returns
            excess_market = market_returns
        
        betas = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for date in returns.index[self.beta_window:]:
            # Get historical window
            end_idx = returns.index.get_loc(date)
            start_idx = end_idx - self.beta_window
            
            hist_ret = excess_returns.iloc[start_idx:end_idx]
            hist_mkt = excess_market.iloc[start_idx:end_idx]
            
            # Compute betas
            mkt_var = hist_mkt.var()
            if mkt_var == 0:
                continue
            
            for col in returns.columns:
                stock_ret = hist_ret[col].dropna()
                if len(stock_ret) < self.beta_window // 2:
                    continue
                
                # Align with market
                common_idx = stock_ret.index.intersection(hist_mkt.dropna().index)
                if len(common_idx) < self.beta_window // 2:
                    continue
                
                cov = np.cov(stock_ret.loc[common_idx], hist_mkt.loc[common_idx])[0, 1]
                beta = cov / mkt_var
                
                # Shrink toward 1
                beta_shrunk = self.shrinkage * 1.0 + (1 - self.shrinkage) * beta
                
                betas.loc[date, col] = beta_shrunk
        
        return betas.astype(float)
    
    def _compute_volatility_signals(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute total volatility signals."""
        vol = returns.rolling(window=self.vol_window).std()
        return vol
    
    def _compute_idio_vol_signals(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute idiosyncratic volatility signals."""
        if market_returns is None:
            raise ValueError("market_returns required for idio_vol")
        
        # Compute residual volatility from market model
        residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for col in returns.columns:
            stock_ret = returns[col].dropna()
            
            if len(stock_ret) < self.beta_window:
                continue
            
            # Rolling regression residuals
            for i in range(self.beta_window, len(stock_ret)):
                window_ret = stock_ret.iloc[i-self.beta_window:i]
                window_mkt = market_returns.reindex(window_ret.index).dropna()
                
                if len(window_mkt) < self.beta_window // 2:
                    continue
                
                common_idx = window_ret.index.intersection(window_mkt.index)
                y = window_ret.loc[common_idx]
                x = window_mkt.loc[common_idx]
                
                # OLS
                beta = np.cov(y, x)[0, 1] / x.var() if x.var() > 0 else 0
                alpha = y.mean() - beta * x.mean()
                
                # Current residual
                current_date = stock_ret.index[i]
                if current_date in market_returns.index:
                    pred = alpha + beta * market_returns.loc[current_date]
                    residuals.loc[current_date, col] = stock_ret.iloc[i] - pred
        
        # Rolling volatility of residuals
        idio_vol = residuals.rolling(window=self.vol_window).std()
        
        return idio_vol.astype(float)
    
    def construct_portfolios(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_cap: pd.DataFrame = None,
        weighting: str = 'equal',
        leverage_target: float = 1.0,
    ) -> pd.DataFrame:
        """
        Construct low-risk portfolios.
        
        For BAB, portfolios are leverage-adjusted to have equal beta
        on long and short sides.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns.
        signals : pd.DataFrame
            Low-risk signals (already inverted).
        market_cap : pd.DataFrame, optional
            Market capitalization.
        weighting : str, default 'equal'
            Portfolio weighting.
        leverage_target : float, default 1.0
            Target leverage.
            
        Returns
        -------
        pd.DataFrame
            Portfolio returns with BAB.
        """
        logger.info("Constructing low-risk portfolios...")
        
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
            
            # BAB (Low risk minus High risk)
            # Note: signals are already inverted, so high signal = low risk
            if f'P{self.n_portfolios}' in port_ret and 'P1' in port_ret:
                port_ret['BAB'] = port_ret[f'P{self.n_portfolios}'] - port_ret['P1']
            
            port_ret['date'] = date
            portfolio_returns.append(port_ret)
        
        df = pd.DataFrame(portfolio_returns).set_index('date')
        
        return df
    
    def construct_bab_portfolio(
        self,
        returns: pd.DataFrame,
        betas: pd.DataFrame,
        market_cap: pd.DataFrame = None,
    ) -> pd.Series:
        """
        Construct proper BAB portfolio with leverage adjustment.
        
        This implements the Frazzini-Pedersen BAB factor with
        leverage to equalize beta on long and short sides.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        betas : pd.DataFrame
            Stock betas.
        market_cap : pd.DataFrame, optional
            Market capitalization.
            
        Returns
        -------
        pd.Series
            BAB returns.
        """
        logger.info("Constructing BAB portfolio with leverage...")
        
        bab_returns = []
        
        for date in returns.index:
            if date not in betas.index:
                continue
            
            beta = betas.loc[date].dropna()
            ret = returns.loc[date].reindex(beta.index).dropna()
            
            if len(ret) < 20:
                continue
            
            # Median beta split
            median_beta = beta.median()
            
            low_beta_mask = beta <= median_beta
            high_beta_mask = beta > median_beta
            
            low_beta_stocks = ret[low_beta_mask]
            high_beta_stocks = ret[high_beta_mask]
            
            if len(low_beta_stocks) == 0 or len(high_beta_stocks) == 0:
                continue
            
            # Portfolio betas
            beta_low = beta[low_beta_mask].mean()
            beta_high = beta[high_beta_mask].mean()
            
            # Portfolio returns
            ret_low = low_beta_stocks.mean()
            ret_high = high_beta_stocks.mean()
            
            # Leverage to equalize beta
            if beta_low > 0 and beta_high > 0:
                # BAB = 1/beta_low * ret_low - 1/beta_high * ret_high
                bab = (1 / beta_low) * ret_low - (1 / beta_high) * ret_high
            else:
                bab = ret_low - ret_high
            
            bab_returns.append({'date': date, 'BAB': bab})
        
        df = pd.DataFrame(bab_returns).set_index('date')
        
        return df['BAB']
    
    def compute_bab_returns(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        weighting: str = 'equal',
    ) -> pd.Series:
        """
        Compute BAB return series.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns.
        market_returns : pd.Series
            Market returns.
        weighting : str, default 'equal'
            Portfolio weighting.
            
        Returns
        -------
        pd.Series
            BAB return series.
        """
        # Compute beta signals (not inverted)
        betas = self._compute_beta_signals(returns, market_returns)
        
        # Use proper BAB construction
        return self.construct_bab_portfolio(returns, betas)


def compute_low_risk_returns(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    signal_type: str = 'beta',
    beta_window: int = 60,
) -> pd.Series:
    """
    Convenience function to compute low-risk factor returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns.
    market_returns : pd.Series
        Market returns.
    signal_type : str, default 'beta'
        Signal type.
    beta_window : int, default 60
        Beta estimation window.
        
    Returns
    -------
    pd.Series
        Low-risk factor returns (BAB).
    """
    low_risk = LowRiskFactor(
        signal_type=signal_type,
        beta_window=beta_window,
    )
    
    return low_risk.compute_bab_returns(returns, market_returns)
