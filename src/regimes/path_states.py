"""
Multi-scale path state construction.

This module implements the path state methodology from the paper,
constructing a state vector that captures multi-horizon volatility
dynamics and regime transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .volatility import VolatilityCalculator

logger = logging.getLogger(__name__)


@dataclass
class PathState:
    """
    Container for path state variables at a point in time.
    
    Attributes
    ----------
    ret_1m : float
        One-month cumulative return.
    ret_3m : float
        Three-month cumulative return.
    sigma_1w : float
        One-week realized volatility (annualized).
    sigma_1m : float
        One-month realized volatility (annualized).
    sigma_3m : float
        Three-month realized volatility (annualized).
    sigma_6m : float
        Six-month realized volatility (annualized).
    rho_sigma : float
        Volatility ratio (sigma_1w / sigma_3m).
    drawdown : float
        Current drawdown from 6-month high.
    drawdown_speed : float
        Rate of drawdown (% per month).
    """
    ret_1m: float
    ret_3m: float
    sigma_1w: float
    sigma_1m: float
    sigma_3m: float
    sigma_6m: float
    rho_sigma: float
    drawdown: float
    drawdown_speed: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.ret_1m,
            self.ret_3m,
            self.sigma_1w,
            self.sigma_1m,
            self.sigma_3m,
            self.sigma_6m,
            self.rho_sigma,
            self.drawdown,
            self.drawdown_speed,
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PathState':
        """Create from numpy array."""
        return cls(
            ret_1m=arr[0],
            ret_3m=arr[1],
            sigma_1w=arr[2],
            sigma_1m=arr[3],
            sigma_3m=arr[4],
            sigma_6m=arr[5],
            rho_sigma=arr[6],
            drawdown=arr[7],
            drawdown_speed=arr[8],
        )


class PathStateClassifier:
    """
    Construct multi-scale path states from market data.
    
    This class computes the full state vector described in the paper,
    which captures:
    - Multi-horizon returns
    - Multi-horizon realized volatility
    - Volatility acceleration/deceleration
    - Drawdown measures
    
    Parameters
    ----------
    vol_horizons : dict, optional
        Volatility calculation horizons in trading days.
    return_horizons : list, optional
        Return calculation horizons in trading days.
    drawdown_lookback : int, default 126
        Lookback for drawdown calculation (6 months).
        
    Examples
    --------
    >>> classifier = PathStateClassifier()
    >>> states = classifier.compute_states(daily_returns)
    >>> print(states.columns)
    Index(['ret_1m', 'ret_3m', 'sigma_1w', ..., 'rho_sigma', 'drawdown'], ...)
    """
    
    STATE_VARIABLES = [
        'ret_1m', 'ret_3m',
        'sigma_1w', 'sigma_1m', 'sigma_3m', 'sigma_6m',
        'rho_sigma', 'drawdown', 'drawdown_speed'
    ]
    
    def __init__(
        self,
        vol_horizons: Optional[Dict[str, int]] = None,
        return_horizons: Optional[List[int]] = None,
        drawdown_lookback: int = 126,
    ):
        """Initialize path state classifier."""
        self.vol_horizons = vol_horizons or {
            '1w': 5,
            '1m': 21,
            '3m': 63,
            '6m': 126,
        }
        self.return_horizons = return_horizons or [21, 63]
        self.drawdown_lookback = drawdown_lookback
        
        self.vol_calculator = VolatilityCalculator(horizons=self.vol_horizons)
        
    def compute_states(
        self,
        daily_returns: Union[pd.Series, pd.DataFrame],
        prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute path state variables from daily data.
        
        Parameters
        ----------
        daily_returns : pd.Series or pd.DataFrame
            Daily market returns.
        prices : pd.Series, optional
            Price series for drawdown calculation. If not provided,
            will be computed from returns.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all state variables at daily frequency.
        """
        logger.info("Computing path state variables")
        
        # Extract returns
        if isinstance(daily_returns, pd.DataFrame):
            if 'market_return' in daily_returns.columns:
                returns = daily_returns['market_return']
            else:
                returns = daily_returns.iloc[:, 0]
        else:
            returns = daily_returns
        
        # Compute prices if not provided
        if prices is None:
            prices = (1 + returns).cumprod() * 100
        
        # Compute volatility measures
        vol_df = self.vol_calculator.compute(returns, return_daily=True)
        
        # Compute returns at multiple horizons
        ret_1m = returns.rolling(21).apply(lambda x: (1 + x).prod() - 1, raw=False)
        ret_3m = returns.rolling(63).apply(lambda x: (1 + x).prod() - 1, raw=False)
        
        # Compute drawdown measures
        dd_df = self.vol_calculator.compute_drawdown(prices, self.drawdown_lookback)
        
        # Combine all state variables
        states = pd.DataFrame({
            'ret_1m': ret_1m,
            'ret_3m': ret_3m,
            'sigma_1w': vol_df['sigma_1w'],
            'sigma_1m': vol_df['sigma_1m'],
            'sigma_3m': vol_df['sigma_3m'],
            'sigma_6m': vol_df['sigma_6m'] if 'sigma_6m' in vol_df.columns else vol_df['sigma_3m'],
            'rho_sigma': vol_df['rho_sigma'],
            'drawdown': dd_df['drawdown'],
            'drawdown_speed': dd_df['drawdown_speed'],
        })
        
        logger.info(f"Computed {len(states)} daily state observations")
        
        return states
    
    def compute_monthly_states(
        self,
        daily_returns: Union[pd.Series, pd.DataFrame],
        prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute path state variables at monthly frequency.
        
        Parameters
        ----------
        daily_returns : pd.Series or pd.DataFrame
            Daily market returns.
        prices : pd.Series, optional
            Price series for drawdown calculation.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with state variables at monthly frequency.
        """
        # Compute daily states
        daily_states = self.compute_states(daily_returns, prices)
        
        # Resample to monthly (end of month values)
        monthly_states = daily_states.resample('M').last()
        
        return monthly_states
    
    def get_state_at(
        self,
        states: pd.DataFrame,
        date: Union[str, pd.Timestamp],
    ) -> PathState:
        """
        Get path state at a specific date.
        
        Parameters
        ----------
        states : pd.DataFrame
            DataFrame of state variables.
        date : str or Timestamp
            Date to query.
            
        Returns
        -------
        PathState
            State at the specified date.
        """
        date = pd.Timestamp(date)
        
        # Find nearest date
        if date in states.index:
            row = states.loc[date]
        else:
            idx = states.index.get_indexer([date], method='nearest')[0]
            row = states.iloc[idx]
        
        return PathState(
            ret_1m=row['ret_1m'],
            ret_3m=row['ret_3m'],
            sigma_1w=row['sigma_1w'],
            sigma_1m=row['sigma_1m'],
            sigma_3m=row['sigma_3m'],
            sigma_6m=row['sigma_6m'],
            rho_sigma=row['rho_sigma'],
            drawdown=row['drawdown'],
            drawdown_speed=row['drawdown_speed'],
        )
    
    def standardize_states(
        self,
        states: pd.DataFrame,
        training_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardize state variables using expanding window statistics.
        
        Parameters
        ----------
        states : pd.DataFrame
            Raw state variables.
        training_end : str, optional
            End of training period. Statistics after this date
            use only prior observations (no look-ahead).
            
        Returns
        -------
        pd.DataFrame
            Standardized state variables.
        """
        standardized = states.copy()
        
        if training_end is not None:
            training_end = pd.Timestamp(training_end)
            
            for col in states.columns:
                # Training period: use expanding window
                train_mask = states.index <= training_end
                expanding_mean = states.loc[train_mask, col].expanding().mean()
                expanding_std = states.loc[train_mask, col].expanding().std()
                
                standardized.loc[train_mask, col] = (
                    (states.loc[train_mask, col] - expanding_mean) / expanding_std
                )
                
                # Test period: use expanding window from training
                test_mask = states.index > training_end
                if test_mask.any():
                    for date in states.index[test_mask]:
                        prior_data = states.loc[:date, col].iloc[:-1]
                        mean = prior_data.mean()
                        std = prior_data.std()
                        standardized.loc[date, col] = (states.loc[date, col] - mean) / std
        else:
            # Simple expanding window standardization
            for col in states.columns:
                expanding_mean = states[col].expanding().mean()
                expanding_std = states[col].expanding().std()
                standardized[col] = (states[col] - expanding_mean) / expanding_std
        
        return standardized
    
    def compute_state_similarity(
        self,
        state1: PathState,
        state2: PathState,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute similarity between two path states.
        
        Parameters
        ----------
        state1 : PathState
            First state.
        state2 : PathState
            Second state.
        weights : np.ndarray, optional
            Weights for each state variable.
            
        Returns
        -------
        float
            Similarity score (0 to 1).
        """
        arr1 = state1.to_array()
        arr2 = state2.to_array()
        
        if weights is None:
            weights = np.ones(len(arr1))
        
        # Weighted Euclidean distance
        diff = arr1 - arr2
        weighted_dist = np.sqrt(np.sum(weights * diff ** 2))
        
        # Convert to similarity (using Gaussian kernel)
        similarity = np.exp(-weighted_dist ** 2 / 2)
        
        return similarity
    
    def find_similar_states(
        self,
        current_state: PathState,
        historical_states: pd.DataFrame,
        n_neighbors: int = 10,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Find historical states most similar to current state.
        
        Parameters
        ----------
        current_state : PathState
            Current path state.
        historical_states : pd.DataFrame
            Historical state data.
        n_neighbors : int, default 10
            Number of similar states to return.
        weights : np.ndarray, optional
            Weights for state variables.
            
        Returns
        -------
        pd.DataFrame
            Most similar historical states with similarity scores.
        """
        current_arr = current_state.to_array()
        
        if weights is None:
            weights = np.ones(len(current_arr))
        
        # Compute similarity to all historical states
        similarities = []
        for idx, row in historical_states.iterrows():
            hist_state = PathState.from_array(row[self.STATE_VARIABLES].values)
            sim = self.compute_state_similarity(current_state, hist_state, weights)
            similarities.append({'date': idx, 'similarity': sim})
        
        sim_df = pd.DataFrame(similarities)
        sim_df = sim_df.sort_values('similarity', ascending=False).head(n_neighbors)
        
        # Add state data
        result = historical_states.loc[sim_df['date']].copy()
        result['similarity'] = sim_df.set_index('date')['similarity']
        
        return result


def compute_path_states(
    daily_returns: pd.Series,
    monthly: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to compute path states.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily market returns.
    monthly : bool, default True
        Return monthly frequency data.
        
    Returns
    -------
    pd.DataFrame
        Path state variables.
    """
    classifier = PathStateClassifier()
    
    if monthly:
        return classifier.compute_monthly_states(daily_returns)
    else:
        return classifier.compute_states(daily_returns)
