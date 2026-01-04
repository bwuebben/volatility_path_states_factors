"""
Synthetic data generator for testing and demonstration.

This module generates synthetic market and factor data that matches
the statistical properties described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic data matching the statistical properties from the paper.
    
    This generator creates realistic market data, factor returns, and 
    volatility dynamics for testing the methodology without requiring
    access to actual market data.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    regime_names : list
        Names of the five regimes.
    return_params : dict
        State-conditional return parameters for each factor.
    ic_params : dict
        State-conditional IC parameters for each factor.
        
    Examples
    --------
    >>> generator = SyntheticDataGenerator(seed=42)
    >>> data = generator.generate(n_months=732)
    >>> print(data['factors'].columns)
    Index(['Momentum', 'Value', 'Quality', 'Low-Risk'], dtype='object')
    """
    
    # Regime names
    REGIME_NAMES = [
        'Calm Trend',
        'Choppy Transition',
        'Slow-Burn Stress',
        'Crash-Spike',
        'Recovery'
    ]
    
    # State-conditional return parameters (mean, std) for each factor
    RETURN_PARAMS = {
        'Momentum': {
            'Calm Trend': (0.0142, 0.04),
            'Choppy Transition': (0.0078, 0.05),
            'Slow-Burn Stress': (0.0022, 0.06),
            'Crash-Spike': (-0.0385, 0.10),
            'Recovery': (0.0095, 0.055)
        },
        'Value': {
            'Calm Trend': (0.0038, 0.03),
            'Choppy Transition': (0.0042, 0.035),
            'Slow-Burn Stress': (0.0055, 0.04),
            'Crash-Spike': (-0.0082, 0.07),
            'Recovery': (0.0128, 0.045)
        },
        'Quality': {
            'Calm Trend': (0.0035, 0.02),
            'Choppy Transition': (0.0028, 0.022),
            'Slow-Burn Stress': (0.0048, 0.025),
            'Crash-Spike': (0.0072, 0.05),
            'Recovery': (0.0018, 0.028)
        },
        'Low-Risk': {
            'Calm Trend': (0.0022, 0.018),
            'Choppy Transition': (0.0018, 0.02),
            'Slow-Burn Stress': (0.0035, 0.025),
            'Crash-Spike': (0.0045, 0.045),
            'Recovery': (-0.0015, 0.025)
        }
    }
    
    # State-conditional IC parameters
    IC_PARAMS = {
        'Momentum': {
            'Calm Trend': (0.048, 0.02),
            'Choppy Transition': (0.032, 0.025),
            'Slow-Burn Stress': (0.015, 0.03),
            'Crash-Spike': (-0.052, 0.05),
            'Recovery': (0.028, 0.035)
        },
        'Value': {
            'Calm Trend': (0.022, 0.015),
            'Choppy Transition': (0.020, 0.018),
            'Slow-Burn Stress': (0.025, 0.02),
            'Crash-Spike': (0.012, 0.04),
            'Recovery': (0.038, 0.025)
        },
        'Quality': {
            'Calm Trend': (0.028, 0.012),
            'Choppy Transition': (0.025, 0.015),
            'Slow-Burn Stress': (0.032, 0.018),
            'Crash-Spike': (0.038, 0.03),
            'Recovery': (0.022, 0.02)
        },
        'Low-Risk': {
            'Calm Trend': (0.015, 0.012),
            'Choppy Transition': (0.012, 0.015),
            'Slow-Burn Stress': (0.018, 0.018),
            'Crash-Spike': (0.025, 0.03),
            'Recovery': (0.008, 0.02)
        }
    }
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize the generator."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def generate(
        self,
        n_months: int = 732,
        start_date: str = '1963-01-01',
        n_daily_per_month: int = 21,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for the full sample period.
        
        Parameters
        ----------
        n_months : int, default 732
            Number of months to generate (732 = Jan 1963 to Dec 2023).
        start_date : str, default '1963-01-01'
            Start date for the sample.
        n_daily_per_month : int, default 21
            Number of trading days per month.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'market': Market-level data (monthly)
            - 'factors': Factor returns (monthly)
            - 'daily': Daily market returns
            - 'regimes': Regime classifications
            - 'volatility': Volatility measures
            - 'ic': Information coefficients
        """
        logger.info(f"Generating synthetic data: {n_months} months from {start_date}")
        
        # Generate dates
        dates = self._generate_dates(start_date, n_months)
        daily_dates = self._generate_daily_dates(start_date, n_months, n_daily_per_month)
        
        # Generate volatility process
        vol_data = self._generate_volatility(n_months, n_daily_per_month)
        
        # Classify regimes
        regimes = self._classify_regimes(vol_data)
        
        # Generate factor returns
        factor_returns = self._generate_factor_returns(n_months, regimes)
        
        # Generate market returns
        market_data = self._generate_market_returns(n_months, vol_data, regimes)
        
        # Generate ICs
        ic_data = self._generate_information_coefficients(n_months, regimes)
        
        # Generate daily returns
        daily_returns = self._generate_daily_returns(vol_data, n_daily_per_month)
        
        # Create DataFrames
        market_df = pd.DataFrame(market_data, index=dates)
        market_df.index.name = 'date'
        
        factors_df = pd.DataFrame(factor_returns, index=dates)
        factors_df.index.name = 'date'
        
        daily_df = pd.DataFrame({'market_return': daily_returns}, index=daily_dates)
        daily_df.index.name = 'date'
        
        regimes_df = pd.DataFrame({
            'regime': regimes,
            'sigma_1w': vol_data['sigma_1w'],
            'sigma_1m': vol_data['sigma_1m'],
            'sigma_3m': vol_data['sigma_3m'],
            'rho_sigma': vol_data['rho_sigma'],
        }, index=dates)
        regimes_df.index.name = 'date'
        
        vol_df = pd.DataFrame(vol_data, index=dates)
        vol_df.index.name = 'date'
        
        ic_df = pd.DataFrame(ic_data, index=dates)
        ic_df.index.name = 'date'
        
        logger.info("Synthetic data generation complete")
        
        return {
            'market': market_df,
            'factors': factors_df,
            'daily': daily_df,
            'regimes': regimes_df,
            'volatility': vol_df,
            'ic': ic_df,
        }
    
    def _generate_dates(self, start_date: str, n_months: int) -> pd.DatetimeIndex:
        """Generate monthly dates."""
        start = pd.Timestamp(start_date)
        dates = pd.date_range(start=start, periods=n_months, freq='M')
        return dates
    
    def _generate_daily_dates(
        self,
        start_date: str,
        n_months: int,
        n_daily_per_month: int,
    ) -> pd.DatetimeIndex:
        """Generate daily dates."""
        n_days = n_months * n_daily_per_month
        start = pd.Timestamp(start_date)
        dates = pd.date_range(start=start, periods=n_days, freq='B')
        return dates
    
    def _generate_volatility(
        self,
        n_months: int,
        n_daily_per_month: int,
    ) -> Dict[str, np.ndarray]:
        """Generate volatility process with regime structure."""
        # Base volatility process (AR(1) with jumps)
        vol_base = np.zeros(n_months)
        vol_base[0] = 0.15
        
        for t in range(1, n_months):
            shock = np.random.normal(0, 0.02)
            jump = np.random.choice([0, 0.15], p=[0.95, 0.05])
            vol_base[t] = 0.9 * vol_base[t-1] + 0.1 * 0.15 + shock + jump
            vol_base[t] = np.clip(vol_base[t], 0.08, 0.50)
        
        # Compute multi-horizon volatilities
        sigma_1w = vol_base + np.random.normal(0, 0.03, n_months)
        sigma_1m = vol_base + np.random.normal(0, 0.02, n_months)
        sigma_3m = pd.Series(vol_base).rolling(3, min_periods=1).mean().values
        sigma_3m = sigma_3m + np.random.normal(0, 0.01, n_months)
        sigma_6m = pd.Series(vol_base).rolling(6, min_periods=1).mean().values
        sigma_6m = sigma_6m + np.random.normal(0, 0.008, n_months)
        
        # Clip to reasonable ranges
        sigma_1w = np.clip(sigma_1w, 0.05, 0.60)
        sigma_1m = np.clip(sigma_1m, 0.05, 0.55)
        sigma_3m = np.clip(sigma_3m, 0.05, 0.50)
        sigma_6m = np.clip(sigma_6m, 0.05, 0.45)
        
        # Volatility ratio
        rho_sigma = sigma_1w / sigma_3m
        
        # Compute returns for drawdown calculation
        cumret = np.cumsum(np.random.normal(0.005, 0.04, n_months))
        price = 100 * np.exp(cumret)
        
        # Compute drawdown
        running_max = np.maximum.accumulate(price)
        drawdown = (running_max - price) / running_max
        
        # Compute drawdown speed (simplified)
        drawdown_speed = np.zeros(n_months)
        in_drawdown = False
        dd_start = 0
        
        for t in range(n_months):
            if drawdown[t] > 0.02:  # In a drawdown
                if not in_drawdown:
                    dd_start = t
                    in_drawdown = True
                duration = max(t - dd_start, 1)
                drawdown_speed[t] = drawdown[t] / duration * 21  # Per month
            else:
                in_drawdown = False
                drawdown_speed[t] = 0
        
        return {
            'vol_base': vol_base,
            'sigma_1w': sigma_1w,
            'sigma_1m': sigma_1m,
            'sigma_3m': sigma_3m,
            'sigma_6m': sigma_6m,
            'rho_sigma': rho_sigma,
            'drawdown': drawdown,
            'drawdown_speed': drawdown_speed,
            'price': price,
        }
    
    def _classify_regimes(self, vol_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Classify regimes based on volatility measures."""
        sigma_1m = vol_data['sigma_1m']
        rho_sigma = vol_data['rho_sigma']
        
        # Volatility level thresholds
        vol_33 = np.percentile(sigma_1m, 33)
        vol_67 = np.percentile(sigma_1m, 67)
        
        # Classify
        n = len(sigma_1m)
        regimes = np.empty(n, dtype=object)
        
        for t in range(n):
            if sigma_1m[t] <= vol_33:
                regimes[t] = 'Calm Trend'
            elif sigma_1m[t] <= vol_67:
                regimes[t] = 'Choppy Transition'
            else:
                if rho_sigma[t] > 1.5:
                    regimes[t] = 'Crash-Spike'
                elif rho_sigma[t] < 0.8:
                    regimes[t] = 'Recovery'
                else:
                    regimes[t] = 'Slow-Burn Stress'
        
        return regimes
    
    def _generate_factor_returns(
        self,
        n_months: int,
        regimes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate factor returns based on regime."""
        factors = ['Momentum', 'Value', 'Quality', 'Low-Risk']
        returns = {f: np.zeros(n_months) for f in factors}
        
        for t in range(n_months):
            regime = regimes[t]
            for factor in factors:
                mean, std = self.RETURN_PARAMS[factor][regime]
                
                # Add negative skewness for momentum in crash states
                if factor == 'Momentum' and regime == 'Crash-Spike':
                    returns[factor][t] = mean + std * (np.random.normal(0, 1) - 0.5)
                else:
                    returns[factor][t] = np.random.normal(mean, std)
        
        return returns
    
    def _generate_market_returns(
        self,
        n_months: int,
        vol_data: Dict[str, np.ndarray],
        regimes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate market returns."""
        sigma_1m = vol_data['sigma_1m']
        
        # Mean return depends slightly on regime
        mean_returns = {
            'Calm Trend': 0.01,
            'Choppy Transition': 0.005,
            'Slow-Burn Stress': 0.002,
            'Crash-Spike': -0.05,
            'Recovery': 0.025,
        }
        
        market_return = np.zeros(n_months)
        for t in range(n_months):
            mean = mean_returns[regimes[t]]
            std = sigma_1m[t] / np.sqrt(12)  # Monthly vol
            market_return[t] = np.random.normal(mean, std)
        
        return {
            'market_return': market_return,
            'market_excess_return': market_return - 0.003,  # Approx risk-free
            'risk_free_rate': np.full(n_months, 0.003),
        }
    
    def _generate_information_coefficients(
        self,
        n_months: int,
        regimes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate information coefficients for each factor."""
        factors = ['Momentum', 'Value', 'Quality', 'Low-Risk']
        ics = {f: np.zeros(n_months) for f in factors}
        
        for t in range(n_months):
            regime = regimes[t]
            for factor in factors:
                mean, std = self.IC_PARAMS[factor][regime]
                ics[factor][t] = np.random.normal(mean, std)
        
        return ics
    
    def _generate_daily_returns(
        self,
        vol_data: Dict[str, np.ndarray],
        n_daily_per_month: int,
    ) -> np.ndarray:
        """Generate daily returns consistent with monthly volatility."""
        n_months = len(vol_data['sigma_1m'])
        n_days = n_months * n_daily_per_month
        
        daily_returns = np.zeros(n_days)
        
        for m in range(n_months):
            monthly_vol = vol_data['sigma_1m'][m]
            daily_vol = monthly_vol / np.sqrt(252)  # Convert to daily
            
            start_idx = m * n_daily_per_month
            end_idx = start_idx + n_daily_per_month
            
            daily_returns[start_idx:end_idx] = np.random.normal(
                0.0003,  # Small positive drift
                daily_vol,
                n_daily_per_month
            )
        
        return daily_returns
    
    def generate_cross_sectional_data(
        self,
        n_months: int = 732,
        n_stocks: int = 2000,
        start_date: str = '1963-01-01',
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate cross-sectional stock data for factor construction.
        
        This method generates a panel of stock-level data including
        returns, market cap, book-to-market, profitability, and beta.
        
        Parameters
        ----------
        n_months : int
            Number of months.
        n_stocks : int
            Number of stocks per month.
        start_date : str
            Start date.
            
        Returns
        -------
        dict
            Dictionary with stock-level data.
        """
        logger.info(f"Generating cross-sectional data: {n_stocks} stocks, {n_months} months")
        
        dates = self._generate_dates(start_date, n_months)
        
        # Generate base data
        data = {
            'date': [],
            'permno': [],
            'ret': [],
            'mktcap': [],
            'bm': [],
            'gp': [],
            'beta': [],
            'exchange': [],
        }
        
        for t, date in enumerate(dates):
            for i in range(n_stocks):
                permno = 10000 + i
                
                # Generate characteristics
                mktcap = np.exp(np.random.normal(6, 2))  # Log-normal market cap
                bm = np.exp(np.random.normal(-0.5, 0.6))  # Book-to-market
                gp = np.random.beta(3, 7)  # Gross profitability (0-1)
                beta = np.random.normal(1.0, 0.5)
                
                # Generate return with factor exposure
                base_ret = np.random.normal(0.01, 0.08)
                
                # Exchange (1=NYSE, 2=AMEX, 3=NASDAQ)
                exchange = np.random.choice([1, 2, 3], p=[0.3, 0.1, 0.6])
                
                data['date'].append(date)
                data['permno'].append(permno)
                data['ret'].append(base_ret)
                data['mktcap'].append(mktcap)
                data['bm'].append(bm)
                data['gp'].append(gp)
                data['beta'].append(beta)
                data['exchange'].append(exchange)
        
        df = pd.DataFrame(data)
        
        return {'stocks': df}


def generate_synthetic_data(
    n_months: int = 732,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate synthetic data.
    
    Parameters
    ----------
    n_months : int
        Number of months to generate.
    seed : int
        Random seed.
        
    Returns
    -------
    dict
        Dictionary with all generated data.
    """
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate(n_months=n_months)
