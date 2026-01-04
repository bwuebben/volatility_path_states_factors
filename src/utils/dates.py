"""
Date handling utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from datetime import datetime, timedelta


def to_datetime(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Convert various date formats to pandas Timestamp."""
    return pd.Timestamp(date)


def get_month_end(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Get the last day of the month for a given date."""
    ts = to_datetime(date)
    return ts + pd.offsets.MonthEnd(0)


def get_month_start(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Get the first day of the month for a given date."""
    ts = to_datetime(date)
    return ts - pd.offsets.MonthBegin(1) + pd.offsets.MonthBegin(0)


def generate_monthly_dates(
    start: str,
    end: str,
    freq: str = 'M',
) -> pd.DatetimeIndex:
    """
    Generate monthly date range.
    
    Parameters
    ----------
    start : str
        Start date.
    end : str
        End date.
    freq : str, default 'M'
        Frequency: 'M' for month-end, 'MS' for month-start.
        
    Returns
    -------
    pd.DatetimeIndex
        Monthly dates.
    """
    return pd.date_range(start=start, end=end, freq=freq)


def get_trading_days(
    start: str,
    end: str,
    calendar: str = 'NYSE',
) -> pd.DatetimeIndex:
    """
    Get trading days between two dates.
    
    Parameters
    ----------
    start : str
        Start date.
    end : str
        End date.
    calendar : str, default 'NYSE'
        Trading calendar to use.
        
    Returns
    -------
    pd.DatetimeIndex
        Trading days.
    """
    # Simple approximation using business days
    return pd.date_range(start=start, end=end, freq='B')


def count_trading_days(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> int:
    """Count trading days between two dates."""
    days = get_trading_days(str(start), str(end))
    return len(days)


def align_dates(
    *series: pd.Series,
    how: str = 'inner',
) -> List[pd.Series]:
    """
    Align multiple series to common dates.
    
    Parameters
    ----------
    *series : pd.Series
        Series to align.
    how : str, default 'inner'
        Alignment method: 'inner', 'outer', 'left', 'right'.
        
    Returns
    -------
    list
        Aligned series.
    """
    if len(series) == 0:
        return []
    
    if len(series) == 1:
        return [series[0]]
    
    # Get common index
    if how == 'inner':
        common_idx = series[0].index
        for s in series[1:]:
            common_idx = common_idx.intersection(s.index)
    elif how == 'outer':
        common_idx = series[0].index
        for s in series[1:]:
            common_idx = common_idx.union(s.index)
    else:
        common_idx = series[0].index
    
    return [s.reindex(common_idx) for s in series]


def split_by_date(
    data: pd.DataFrame,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into before and after a date.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to split.
    split_date : str
        Split date.
        
    Returns
    -------
    before : pd.DataFrame
        Data before split date.
    after : pd.DataFrame
        Data after split date.
    """
    split = to_datetime(split_date)
    return data.loc[:split], data.loc[split:]


def resample_to_monthly(
    daily_data: pd.DataFrame,
    agg_func: str = 'last',
) -> pd.DataFrame:
    """
    Resample daily data to monthly frequency.
    
    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily data.
    agg_func : str, default 'last'
        Aggregation function: 'last', 'first', 'mean', 'sum'.
        
    Returns
    -------
    pd.DataFrame
        Monthly data.
    """
    if agg_func == 'last':
        return daily_data.resample('M').last()
    elif agg_func == 'first':
        return daily_data.resample('M').first()
    elif agg_func == 'mean':
        return daily_data.resample('M').mean()
    elif agg_func == 'sum':
        return daily_data.resample('M').sum()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


def get_year(date: Union[str, pd.Timestamp]) -> int:
    """Get year from date."""
    return to_datetime(date).year


def get_month(date: Union[str, pd.Timestamp]) -> int:
    """Get month from date."""
    return to_datetime(date).month


def is_month_end(date: Union[str, pd.Timestamp]) -> bool:
    """Check if date is month end."""
    ts = to_datetime(date)
    return ts == get_month_end(ts)


def get_quarters(
    start: str,
    end: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get list of quarters between two dates.
    
    Returns
    -------
    list
        List of (quarter_start, quarter_end) tuples.
    """
    dates = pd.date_range(start=start, end=end, freq='Q')
    quarters = []
    
    for date in dates:
        q_end = date
        q_start = date - pd.offsets.QuarterBegin(1)
        quarters.append((q_start, q_end))
    
    return quarters
