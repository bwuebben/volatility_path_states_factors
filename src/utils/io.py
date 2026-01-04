"""
Input/output utilities.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = 'parquet',
):
    """
    Save DataFrame to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str
        Output path.
    format : str, default 'parquet'
        Output format: 'parquet', 'csv', 'pickle'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(path)
    elif format == 'csv':
        df.to_csv(path)
    elif format == 'pickle':
        df.to_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved DataFrame to {path}")


def load_dataframe(
    path: str,
    format: str = None,
) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Parameters
    ----------
    path : str
        Input path.
    format : str, optional
        Input format. If None, inferred from extension.
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lstrip('.')
    
    if format == 'parquet':
        return pd.read_parquet(path)
    elif format == 'csv':
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif format in ['pickle', 'pkl']:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_results(
    results: Dict,
    path: str,
    format: str = 'json',
):
    """
    Save results dictionary to file.
    
    Parameters
    ----------
    results : dict
        Results to save.
    path : str
        Output path.
    format : str, default 'json'
        Output format: 'json', 'pickle'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy/pandas types to native Python
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2, default=str)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved results to {path}")


def load_results(
    path: str,
    format: str = None,
) -> Dict:
    """
    Load results from file.
    
    Parameters
    ----------
    path : str
        Input path.
    format : str, optional
        Input format. If None, inferred from extension.
        
    Returns
    -------
    dict
        Loaded results.
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lstrip('.')
    
    if format == 'json':
        with open(path, 'r') as f:
            return json.load(f)
    elif format in ['pickle', 'pkl']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists.
    
    Parameters
    ----------
    path : str
        Directory path.
        
    Returns
    -------
    Path
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: str,
    pattern: str = '*',
    recursive: bool = False,
) -> list:
    """
    List files in directory.
    
    Parameters
    ----------
    directory : str
        Directory path.
    pattern : str, default '*'
        Glob pattern.
    recursive : bool, default False
        Search recursively.
        
    Returns
    -------
    list
        List of file paths.
    """
    path = Path(directory)
    
    if recursive:
        return list(path.rglob(pattern))
    return list(path.glob(pattern))


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    
    # Navigate up to find project root (contains setup.py)
    for parent in current.parents:
        if (parent / 'setup.py').exists():
            return parent
    
    return current.parent.parent.parent


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / 'data'


def get_output_dir() -> Path:
    """Get output directory."""
    return get_project_root() / 'output'
