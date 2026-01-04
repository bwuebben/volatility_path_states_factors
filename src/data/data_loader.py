"""
Base data loader class and utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loading.
    
    This class defines the interface for loading market data, factor returns,
    and other data required for the analysis.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching downloaded data.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize data loader."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def load_market_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load market-level data.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - date : datetime index
            - market_return : market return
            - risk_free_rate : risk-free rate
        """
        pass
    
    @abstractmethod
    def load_factor_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load factor returns.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with factor returns as columns.
        """
        pass
    
    def load_daily_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load daily market returns for volatility calculation.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with daily returns.
        """
        raise NotImplementedError("Subclass must implement load_daily_returns")
    
    def _cache_path(self, name: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{name}.parquet"
    
    def _load_from_cache(self, name: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._cache_path(name)
        if cache_path.exists():
            logger.info(f"Loading {name} from cache")
            return pd.read_parquet(cache_path)
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, name: str) -> None:
        """Save data to cache."""
        cache_path = self._cache_path(name)
        data.to_parquet(cache_path)
        logger.info(f"Saved {name} to cache")


class FrenchDataLoader(DataLoader):
    """
    Load data from Kenneth French's data library.
    
    This loader downloads factor returns and market data from the
    Fama-French data library.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching downloaded data.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize French data loader."""
        super().__init__(cache_dir)
        
    def load_market_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load market data from French library."""
        try:
            import pandas_datareader as pdr
        except ImportError:
            raise ImportError("pandas_datareader required. Install with: pip install pandas-datareader")
        
        # Check cache
        cached = self._load_from_cache('french_market')
        if cached is not None:
            return cached.loc[start_date:end_date]
        
        # Download Fama-French factors
        ff = pdr.get_data_famafrench('F-F_Research_Data_Factors', start=start_date)[0]
        ff.index = pd.to_datetime(ff.index.astype(str), format='%Y%m')
        ff = ff / 100  # Convert from percentage
        
        # Construct market data
        market_data = pd.DataFrame({
            'market_return': ff['Mkt-RF'] + ff['RF'],
            'market_excess_return': ff['Mkt-RF'],
            'risk_free_rate': ff['RF'],
        })
        
        # Cache and return
        self._save_to_cache(market_data, 'french_market')
        return market_data.loc[start_date:end_date]
    
    def load_factor_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load factor returns from French library."""
        try:
            import pandas_datareader as pdr
        except ImportError:
            raise ImportError("pandas_datareader required. Install with: pip install pandas-datareader")
        
        # Check cache
        cached = self._load_from_cache('french_factors')
        if cached is not None:
            return cached.loc[start_date:end_date]
        
        # Download factors
        ff3 = pdr.get_data_famafrench('F-F_Research_Data_Factors', start='1926')[0]
        ff5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1963')[0]
        mom = pdr.get_data_famafrench('F-F_Momentum_Factor', start='1926')[0]
        
        # Process indices
        ff3.index = pd.to_datetime(ff3.index.astype(str), format='%Y%m')
        ff5.index = pd.to_datetime(ff5.index.astype(str), format='%Y%m')
        mom.index = pd.to_datetime(mom.index.astype(str), format='%Y%m')
        
        # Convert to decimal
        ff3 = ff3 / 100
        ff5 = ff5 / 100
        mom = mom / 100
        
        # Combine factors
        factors = pd.DataFrame({
            'MKT': ff3['Mkt-RF'],
            'SMB': ff3['SMB'],
            'HML': ff3['HML'],
            'MOM': mom['Mom   '] if 'Mom   ' in mom.columns else mom.iloc[:, 0],
            'RMW': ff5['RMW'],
            'CMA': ff5['CMA'],
            'RF': ff3['RF'],
        })
        
        # Cache and return
        self._save_to_cache(factors, 'french_factors')
        return factors.loc[start_date:end_date]
    
    def load_daily_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load daily market returns."""
        try:
            import pandas_datareader as pdr
        except ImportError:
            raise ImportError("pandas_datareader required. Install with: pip install pandas-datareader")
        
        # Check cache
        cached = self._load_from_cache('french_daily')
        if cached is not None:
            return cached.loc[start_date:end_date]
        
        # Download daily data
        ff_daily = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1926')[0]
        ff_daily.index = pd.to_datetime(ff_daily.index)
        ff_daily = ff_daily / 100
        
        daily_data = pd.DataFrame({
            'market_return': ff_daily['Mkt-RF'] + ff_daily['RF'],
            'market_excess_return': ff_daily['Mkt-RF'],
        })
        
        # Cache and return
        self._save_to_cache(daily_data, 'french_daily')
        return daily_data.loc[start_date:end_date]


class YahooDataLoader(DataLoader):
    """
    Load data from Yahoo Finance.
    
    This loader downloads price data from Yahoo Finance and computes returns.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching downloaded data.
    market_ticker : str, default '^GSPC'
        Ticker symbol for market index.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        market_ticker: str = '^GSPC',
    ):
        """Initialize Yahoo data loader."""
        super().__init__(cache_dir)
        self.market_ticker = market_ticker
        
    def load_market_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load market data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        # Download data
        ticker = yf.Ticker(self.market_ticker)
        hist = ticker.history(start=start_date, end=end_date)
        
        # Compute returns
        returns = hist['Close'].pct_change().dropna()
        
        # Monthly aggregation
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        market_data = pd.DataFrame({
            'market_return': monthly_returns,
            'risk_free_rate': 0.0,  # Placeholder
        })
        
        return market_data
    
    def load_factor_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load factor returns.
        
        Note: Yahoo Finance doesn't provide factor returns directly.
        This method falls back to French data library.
        """
        french_loader = FrenchDataLoader(str(self.cache_dir))
        return french_loader.load_factor_returns(start_date, end_date)
    
    def load_daily_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load daily market returns from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        # Download data
        ticker = yf.Ticker(self.market_ticker)
        hist = ticker.history(start=start_date, end=end_date)
        
        # Compute returns
        returns = hist['Close'].pct_change().dropna()
        
        daily_data = pd.DataFrame({
            'market_return': returns,
        })
        
        return daily_data


def load_data(
    source: str = 'french',
    start_date: str = '1963-01-01',
    end_date: str = '2023-12-31',
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all required data.
    
    Parameters
    ----------
    source : str
        Data source: 'french', 'yahoo', or 'synthetic'.
    start_date : str
        Start date.
    end_date : str
        End date.
    cache_dir : str, optional
        Cache directory.
        
    Returns
    -------
    market_data : pd.DataFrame
        Market-level data.
    factor_returns : pd.DataFrame
        Factor returns.
    daily_returns : pd.DataFrame
        Daily returns for volatility calculation.
    """
    if source == 'french':
        loader = FrenchDataLoader(cache_dir)
    elif source == 'yahoo':
        loader = YahooDataLoader(cache_dir)
    elif source == 'synthetic':
        from .synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        data = generator.generate()
        return data['market'], data['factors'], data['daily']
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    market_data = loader.load_market_data(start_date, end_date)
    factor_returns = loader.load_factor_returns(start_date, end_date)
    daily_returns = loader.load_daily_returns(start_date, end_date)
    
    return market_data, factor_returns, daily_returns
