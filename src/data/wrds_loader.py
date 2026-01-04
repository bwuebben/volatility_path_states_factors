"""
WRDS/CRSP data loader.

This module provides functionality for loading data from WRDS
(Wharton Research Data Services), including CRSP stock data
and Compustat fundamentals.

Note: Requires WRDS subscription and wrds Python package.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WRDSLoader:
    """
    Load data from WRDS database.
    
    Parameters
    ----------
    username : str, optional
        WRDS username. If not provided, will prompt.
        
    Examples
    --------
    >>> loader = WRDSLoader(username='myusername')
    >>> crsp = loader.load_crsp_monthly('1963-01-01', '2023-12-31')
    >>> compustat = loader.load_compustat_annual('1963-01-01', '2023-12-31')
    """
    
    def __init__(self, username: Optional[str] = None):
        """Initialize WRDS connection."""
        self.username = username
        self._connection = None
        
    def connect(self):
        """Establish WRDS connection."""
        try:
            import wrds
            self._connection = wrds.Connection(wrds_username=self.username)
            logger.info("Connected to WRDS")
        except ImportError:
            raise ImportError(
                "wrds package not installed. Install with: pip install wrds"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to WRDS: {e}")
    
    @property
    def connection(self):
        """Get or create WRDS connection."""
        if self._connection is None:
            self.connect()
        return self._connection
    
    def close(self):
        """Close WRDS connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
    
    def load_crsp_monthly(
        self,
        start_date: str,
        end_date: str,
        share_codes: List[int] = [10, 11],
        exchange_codes: List[int] = [1, 2, 3],
    ) -> pd.DataFrame:
        """
        Load CRSP monthly stock data.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str
            End date (YYYY-MM-DD).
        share_codes : list, default [10, 11]
            CRSP share codes (10, 11 = common stock).
        exchange_codes : list, default [1, 2, 3]
            Exchange codes (1=NYSE, 2=AMEX, 3=NASDAQ).
            
        Returns
        -------
        pd.DataFrame
            CRSP monthly data with columns:
            - permno: CRSP permanent identifier
            - date: Month end date
            - ret: Monthly return
            - retx: Return excluding dividends
            - prc: Price
            - shrout: Shares outstanding
            - vol: Trading volume
            - mktcap: Market capitalization
        """
        logger.info(f"Loading CRSP monthly data: {start_date} to {end_date}")
        
        shrcd_str = ','.join(map(str, share_codes))
        exchcd_str = ','.join(map(str, exchange_codes))
        
        query = f"""
            SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol,
                   abs(a.prc) * a.shrout as mktcap,
                   b.shrcd, b.exchcd, b.siccd
            FROM crsp.msf as a
            LEFT JOIN crsp.msenames as b
                ON a.permno = b.permno
                AND a.date >= b.namedt
                AND a.date <= b.nameendt
            WHERE a.date >= '{start_date}'
                AND a.date <= '{end_date}'
                AND b.shrcd IN ({shrcd_str})
                AND b.exchcd IN ({exchcd_str})
        """
        
        df = self.connection.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} CRSP records")
        return df
    
    def load_crsp_daily(
        self,
        start_date: str,
        end_date: str,
        permnos: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Load CRSP daily stock data.
        
        Parameters
        ----------
        start_date : str
            Start date.
        end_date : str
            End date.
        permnos : list, optional
            Specific PERMNOs to load. If None, loads all.
            
        Returns
        -------
        pd.DataFrame
            CRSP daily data.
        """
        logger.info(f"Loading CRSP daily data: {start_date} to {end_date}")
        
        query = f"""
            SELECT permno, date, ret, prc, vol
            FROM crsp.dsf
            WHERE date >= '{start_date}'
                AND date <= '{end_date}'
        """
        
        if permnos is not None:
            permno_str = ','.join(map(str, permnos))
            query += f" AND permno IN ({permno_str})"
        
        df = self.connection.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} daily records")
        return df
    
    def load_compustat_annual(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load Compustat annual fundamental data.
        
        Parameters
        ----------
        start_date : str
            Start date.
        end_date : str
            End date.
            
        Returns
        -------
        pd.DataFrame
            Compustat annual data with key fundamentals.
        """
        logger.info(f"Loading Compustat annual data: {start_date} to {end_date}")
        
        query = f"""
            SELECT gvkey, datadate, fyear,
                   -- Income statement
                   sale, revt, cogs, xsga, oibdp, oiadp, ni, ib,
                   -- Balance sheet
                   at, lt, ceq, seq, che, invt, rect, act, lct,
                   dltt, dlc, txditc,
                   -- Cash flow
                   oancf, capx, dp,
                   -- Shares
                   csho, prcc_f,
                   -- Other
                   sic, naics
            FROM comp.funda
            WHERE datadate >= '{start_date}'
                AND datadate <= '{end_date}'
                AND indfmt = 'INDL'
                AND datafmt = 'STD'
                AND popsrc = 'D'
                AND consol = 'C'
        """
        
        df = self.connection.raw_sql(query)
        df['datadate'] = pd.to_datetime(df['datadate'])
        
        logger.info(f"Loaded {len(df)} Compustat records")
        return df
    
    def load_compustat_quarterly(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load Compustat quarterly data.
        
        Parameters
        ----------
        start_date : str
            Start date.
        end_date : str
            End date.
            
        Returns
        -------
        pd.DataFrame
            Compustat quarterly data.
        """
        logger.info(f"Loading Compustat quarterly data")
        
        query = f"""
            SELECT gvkey, datadate, fqtr, fyearq,
                   saleq, revtq, cogsq, oibdpq, niq, ibq,
                   atq, ltq, ceqq, cheq
            FROM comp.fundq
            WHERE datadate >= '{start_date}'
                AND datadate <= '{end_date}'
                AND indfmt = 'INDL'
                AND datafmt = 'STD'
                AND popsrc = 'D'
                AND consol = 'C'
        """
        
        df = self.connection.raw_sql(query)
        df['datadate'] = pd.to_datetime(df['datadate'])
        
        return df
    
    def load_linking_table(self) -> pd.DataFrame:
        """
        Load CRSP-Compustat linking table.
        
        Returns
        -------
        pd.DataFrame
            Linking table with PERMNO-GVKEY mappings.
        """
        query = """
            SELECT gvkey, lpermno as permno, linktype, linkprim,
                   linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU', 'LC')
                AND linkprim IN ('P', 'C')
        """
        
        df = self.connection.raw_sql(query)
        df['linkdt'] = pd.to_datetime(df['linkdt'])
        df['linkenddt'] = pd.to_datetime(df['linkenddt'])
        
        # Handle missing end dates
        df['linkenddt'] = df['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
        
        return df
    
    def merge_crsp_compustat(
        self,
        crsp: pd.DataFrame,
        compustat: pd.DataFrame,
        link: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Merge CRSP and Compustat data.
        
        Parameters
        ----------
        crsp : pd.DataFrame
            CRSP data with permno and date columns.
        compustat : pd.DataFrame
            Compustat data with gvkey and datadate columns.
        link : pd.DataFrame, optional
            Linking table. If None, will load.
            
        Returns
        -------
        pd.DataFrame
            Merged dataset.
        """
        if link is None:
            link = self.load_linking_table()
        
        # Add gvkey to CRSP
        crsp_link = crsp.merge(link, on='permno', how='inner')
        
        # Filter to valid link dates
        crsp_link = crsp_link[
            (crsp_link['date'] >= crsp_link['linkdt']) &
            (crsp_link['date'] <= crsp_link['linkenddt'])
        ]
        
        # Merge with Compustat (use most recent fiscal year end)
        compustat['year'] = compustat['datadate'].dt.year
        crsp_link['year'] = crsp_link['date'].dt.year - 1  # Use lagged fundamentals
        
        merged = crsp_link.merge(
            compustat,
            on=['gvkey', 'year'],
            how='inner',
            suffixes=('', '_comp')
        )
        
        return merged
    
    def load_fama_french_factors(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load Fama-French factors from WRDS.
        
        Parameters
        ----------
        start_date : str
            Start date.
        end_date : str
            End date.
            
        Returns
        -------
        pd.DataFrame
            Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA, RF).
        """
        query = f"""
            SELECT date, mktrf, smb, hml, rmw, cma, rf
            FROM ff.fivefactors_monthly
            WHERE date >= '{start_date}'
                AND date <= '{end_date}'
        """
        
        df = self.connection.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Convert from percent to decimal
        for col in df.columns:
            df[col] = df[col] / 100
        
        return df
    
    def load_market_returns(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'monthly',
    ) -> pd.DataFrame:
        """
        Load market index returns.
        
        Parameters
        ----------
        start_date : str
            Start date.
        end_date : str
            End date.
        frequency : str, default 'monthly'
            'monthly' or 'daily'.
            
        Returns
        -------
        pd.DataFrame
            Market returns.
        """
        if frequency == 'monthly':
            table = 'crsp.msi'
        else:
            table = 'crsp.dsi'
        
        query = f"""
            SELECT date, vwretd, vwretx, ewretd, ewretx, sprtrn
            FROM {table}
            WHERE date >= '{start_date}'
                AND date <= '{end_date}'
        """
        
        df = self.connection.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df


def load_wrds_data(
    start_date: str = '1963-01-01',
    end_date: str = '2023-12-31',
    username: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load standard WRDS data.
    
    Parameters
    ----------
    start_date : str
        Start date.
    end_date : str
        End date.
    username : str, optional
        WRDS username.
        
    Returns
    -------
    dict
        Dictionary with:
        - 'crsp': CRSP monthly data
        - 'compustat': Compustat annual data
        - 'factors': Fama-French factors
        - 'market': Market returns
    """
    loader = WRDSLoader(username=username)
    
    try:
        data = {
            'crsp': loader.load_crsp_monthly(start_date, end_date),
            'compustat': loader.load_compustat_annual(start_date, end_date),
            'factors': loader.load_fama_french_factors(start_date, end_date),
            'market': loader.load_market_returns(start_date, end_date),
        }
        
        return data
        
    finally:
        loader.close()
