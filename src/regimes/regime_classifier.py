"""
Regime classification based on path states.

This module classifies multi-scale path states into discrete regimes
as described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Enumeration of path state regimes."""
    CALM_TREND = "Calm Trend"
    CHOPPY_TRANSITION = "Choppy Transition"
    SLOW_BURN_STRESS = "Slow-Burn Stress"
    CRASH_SPIKE = "Crash-Spike"
    RECOVERY = "Recovery"
    
    @classmethod
    def from_string(cls, s: str) -> 'Regime':
        """Create regime from string."""
        mapping = {
            'calm_trend': cls.CALM_TREND,
            'calm trend': cls.CALM_TREND,
            'choppy_transition': cls.CHOPPY_TRANSITION,
            'choppy transition': cls.CHOPPY_TRANSITION,
            'slow_burn_stress': cls.SLOW_BURN_STRESS,
            'slow-burn stress': cls.SLOW_BURN_STRESS,
            'slow burn stress': cls.SLOW_BURN_STRESS,
            'crash_spike': cls.CRASH_SPIKE,
            'crash-spike': cls.CRASH_SPIKE,
            'crash spike': cls.CRASH_SPIKE,
            'recovery': cls.RECOVERY,
        }
        return mapping.get(s.lower(), cls.CALM_TREND)


class RegimeClassifier:
    """
    Classify path states into discrete regimes.
    
    The classification uses a hierarchical approach:
    1. Primary: Volatility level (low/medium/high)
    2. Secondary: Volatility dynamics (spike/sustained/decay) for high-vol states
    
    Parameters
    ----------
    vol_quantile_low : float, default 0.33
        Percentile threshold for low volatility.
    vol_quantile_high : float, default 0.67
        Percentile threshold for high volatility.
    ratio_spike : float, default 1.5
        Volatility ratio threshold for spike classification.
    ratio_decay : float, default 0.8
        Volatility ratio threshold for decay classification.
    expanding_window : bool, default True
        Use expanding window for threshold estimation (no look-ahead).
        
    Attributes
    ----------
    thresholds : dict
        Current threshold values.
        
    Examples
    --------
    >>> classifier = RegimeClassifier()
    >>> regimes = classifier.classify(states)
    >>> print(regimes.value_counts())
    Calm Trend           245
    Choppy Transition    246
    Slow-Burn Stress     116
    Crash-Spike           53
    Recovery              76
    """
    
    REGIME_ORDER = [
        'Calm Trend',
        'Choppy Transition',
        'Slow-Burn Stress',
        'Crash-Spike',
        'Recovery'
    ]
    
    def __init__(
        self,
        vol_quantile_low: float = 0.33,
        vol_quantile_high: float = 0.67,
        ratio_spike: float = 1.5,
        ratio_decay: float = 0.8,
        expanding_window: bool = True,
    ):
        """Initialize regime classifier."""
        self.vol_quantile_low = vol_quantile_low
        self.vol_quantile_high = vol_quantile_high
        self.ratio_spike = ratio_spike
        self.ratio_decay = ratio_decay
        self.expanding_window = expanding_window
        
        self.thresholds = {}
        self._fitted = False
        
    def fit(
        self,
        states: pd.DataFrame,
        training_end: Optional[str] = None,
    ) -> 'RegimeClassifier':
        """
        Fit the classifier by estimating volatility thresholds.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data with 'sigma_1m' column.
        training_end : str, optional
            End of training period for threshold estimation.
            
        Returns
        -------
        self
        """
        if training_end is not None:
            training_data = states.loc[:training_end, 'sigma_1m']
        else:
            training_data = states['sigma_1m']
        
        self.thresholds = {
            'vol_low': np.percentile(training_data.dropna(), self.vol_quantile_low * 100),
            'vol_high': np.percentile(training_data.dropna(), self.vol_quantile_high * 100),
            'ratio_spike': self.ratio_spike,
            'ratio_decay': self.ratio_decay,
        }
        
        self._fitted = True
        logger.info(f"Fitted thresholds: {self.thresholds}")
        
        return self
    
    def classify(
        self,
        states: pd.DataFrame,
        training_end: Optional[str] = None,
    ) -> pd.Series:
        """
        Classify path states into regimes.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data with 'sigma_1m' and 'rho_sigma' columns.
        training_end : str, optional
            End of training period. If expanding_window is True,
            thresholds are estimated using only prior data.
            
        Returns
        -------
        pd.Series
            Regime classification for each observation.
        """
        if self.expanding_window:
            return self._classify_expanding(states, training_end)
        else:
            if not self._fitted:
                self.fit(states, training_end)
            return self._classify_fixed(states)
    
    def _classify_fixed(self, states: pd.DataFrame) -> pd.Series:
        """Classify using fixed thresholds."""
        n = len(states)
        regimes = pd.Series(index=states.index, dtype=object)
        
        sigma = states['sigma_1m'].values
        rho = states['rho_sigma'].values
        
        for t in range(n):
            regimes.iloc[t] = self._classify_single(
                sigma[t], rho[t],
                self.thresholds['vol_low'],
                self.thresholds['vol_high'],
            )
        
        return regimes
    
    def _classify_expanding(
        self,
        states: pd.DataFrame,
        training_end: Optional[str] = None,
    ) -> pd.Series:
        """Classify using expanding window thresholds."""
        n = len(states)
        regimes = pd.Series(index=states.index, dtype=object)
        
        sigma = states['sigma_1m'].values
        rho = states['rho_sigma'].values
        
        # Minimum observations for threshold estimation
        min_obs = 24  # 2 years
        
        for t in range(n):
            if t < min_obs:
                # Not enough data; use default classification
                regimes.iloc[t] = 'Calm Trend'
                continue
            
            # Estimate thresholds from prior data
            prior_sigma = sigma[:t]
            vol_low = np.percentile(prior_sigma[~np.isnan(prior_sigma)], 
                                    self.vol_quantile_low * 100)
            vol_high = np.percentile(prior_sigma[~np.isnan(prior_sigma)], 
                                     self.vol_quantile_high * 100)
            
            regimes.iloc[t] = self._classify_single(
                sigma[t], rho[t], vol_low, vol_high
            )
        
        return regimes
    
    def _classify_single(
        self,
        sigma: float,
        rho: float,
        vol_low: float,
        vol_high: float,
    ) -> str:
        """Classify a single observation."""
        if np.isnan(sigma) or np.isnan(rho):
            return 'Calm Trend'  # Default for missing data
        
        # Primary classification: volatility level
        if sigma <= vol_low:
            return 'Calm Trend'
        elif sigma <= vol_high:
            return 'Choppy Transition'
        else:
            # Secondary classification: volatility dynamics
            if rho > self.ratio_spike:
                return 'Crash-Spike'
            elif rho < self.ratio_decay:
                return 'Recovery'
            else:
                return 'Slow-Burn Stress'
    
    def get_regime_at(
        self,
        states: pd.DataFrame,
        date: Union[str, pd.Timestamp],
    ) -> str:
        """
        Get regime classification at a specific date.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data.
        date : str or Timestamp
            Date to query.
            
        Returns
        -------
        str
            Regime name.
        """
        regimes = self.classify(states)
        date = pd.Timestamp(date)
        
        if date in regimes.index:
            return regimes.loc[date]
        else:
            # Find nearest date
            idx = regimes.index.get_indexer([date], method='nearest')[0]
            return regimes.iloc[idx]
    
    def compute_regime_statistics(
        self,
        states: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute summary statistics by regime.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data.
        factor_returns : pd.DataFrame, optional
            Factor returns to include in statistics.
            
        Returns
        -------
        pd.DataFrame
            Statistics by regime.
        """
        regimes = self.classify(states)
        
        # Basic statistics
        stats = []
        for regime in self.REGIME_ORDER:
            mask = regimes == regime
            n_obs = mask.sum()
            freq = n_obs / len(regimes) * 100
            
            regime_states = states.loc[mask]
            
            stat = {
                'regime': regime,
                'observations': n_obs,
                'frequency': freq,
                'avg_sigma_1m': regime_states['sigma_1m'].mean(),
                'avg_rho_sigma': regime_states['rho_sigma'].mean(),
            }
            
            # Add factor return statistics if provided
            if factor_returns is not None:
                aligned_returns = factor_returns.loc[mask]
                for col in aligned_returns.columns:
                    stat[f'{col}_mean'] = aligned_returns[col].mean() * 100
                    stat[f'{col}_sharpe'] = (
                        aligned_returns[col].mean() / aligned_returns[col].std() * np.sqrt(12)
                    )
            
            stats.append(stat)
        
        return pd.DataFrame(stats).set_index('regime')
    
    def compute_transition_matrix(
        self,
        states: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute regime transition probabilities.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data.
            
        Returns
        -------
        pd.DataFrame
            Transition probability matrix.
        """
        regimes = self.classify(states)
        
        # Initialize transition counts
        n_regimes = len(self.REGIME_ORDER)
        transitions = np.zeros((n_regimes, n_regimes))
        
        regime_to_idx = {r: i for i, r in enumerate(self.REGIME_ORDER)}
        
        # Count transitions
        for t in range(1, len(regimes)):
            from_regime = regimes.iloc[t-1]
            to_regime = regimes.iloc[t]
            
            i = regime_to_idx[from_regime]
            j = regime_to_idx[to_regime]
            transitions[i, j] += 1
        
        # Convert to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums
        
        return pd.DataFrame(
            transition_probs,
            index=self.REGIME_ORDER,
            columns=self.REGIME_ORDER,
        )
    
    def identify_regime_dates(
        self,
        states: pd.DataFrame,
        regime: str,
    ) -> pd.DatetimeIndex:
        """
        Get dates when a specific regime occurred.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data.
        regime : str
            Regime name to find.
            
        Returns
        -------
        pd.DatetimeIndex
            Dates in the specified regime.
        """
        regimes = self.classify(states)
        return regimes.index[regimes == regime]
    
    def identify_regime_episodes(
        self,
        states: pd.DataFrame,
        regime: str,
        min_duration: int = 1,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Identify contiguous episodes of a regime.
        
        Parameters
        ----------
        states : pd.DataFrame
            Path state data.
        regime : str
            Regime name.
        min_duration : int, default 1
            Minimum episode duration (periods).
            
        Returns
        -------
        list
            List of (start_date, end_date) tuples for each episode.
        """
        regimes = self.classify(states)
        is_regime = regimes == regime
        
        # Find episode boundaries
        episodes = []
        in_episode = False
        start_date = None
        
        for date, is_match in is_regime.items():
            if is_match and not in_episode:
                # Start of episode
                in_episode = True
                start_date = date
            elif not is_match and in_episode:
                # End of episode
                in_episode = False
                episodes.append((start_date, prev_date))
            prev_date = date
        
        # Handle episode that extends to end of sample
        if in_episode:
            episodes.append((start_date, prev_date))
        
        # Filter by minimum duration
        if min_duration > 1:
            episodes = [
                (s, e) for s, e in episodes
                if len(regimes.loc[s:e]) >= min_duration
            ]
        
        return episodes


def classify_regimes(
    states: pd.DataFrame,
    expanding_window: bool = True,
) -> pd.Series:
    """
    Convenience function to classify regimes.
    
    Parameters
    ----------
    states : pd.DataFrame
        Path state data.
    expanding_window : bool, default True
        Use expanding window for thresholds.
        
    Returns
    -------
    pd.Series
        Regime classifications.
    """
    classifier = RegimeClassifier(expanding_window=expanding_window)
    return classifier.classify(states)
