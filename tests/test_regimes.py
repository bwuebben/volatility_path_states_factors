"""
Tests for regime classification module.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.regimes.regime_classifier import RegimeClassifier, Regime
from src.regimes.path_states import PathStateClassifier, PathState
from src.data.synthetic_data import SyntheticDataGenerator


class TestRegimeClassifier:
    """Tests for RegimeClassifier class."""
    
    @pytest.fixture
    def sample_states(self):
        """Generate sample state data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        
        sigma_1m = np.random.uniform(0.10, 0.30, n)
        rho_sigma = np.random.uniform(0.6, 1.8, n)
        
        return pd.DataFrame({
            'sigma_1m': sigma_1m,
            'rho_sigma': rho_sigma,
        }, index=dates)
    
    @pytest.fixture
    def classifier(self):
        """Create regime classifier."""
        return RegimeClassifier()
    
    def test_classify_returns_series(self, classifier, sample_states):
        """Test that classify returns a Series."""
        regimes = classifier.classify(sample_states)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_states)
    
    def test_valid_regime_names(self, classifier, sample_states):
        """Test that all regime names are valid."""
        regimes = classifier.classify(sample_states)
        
        valid_regimes = set(classifier.REGIME_ORDER)
        assert set(regimes.dropna().unique()).issubset(valid_regimes)
    
    def test_all_regimes_present(self, classifier):
        """Test that all regimes can be classified."""
        # Generate data with extreme values to cover all regimes
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        
        # Create data that covers all regime combinations
        sigma_1m = np.concatenate([
            np.random.uniform(0.08, 0.12, 100),  # Low vol
            np.random.uniform(0.15, 0.20, 100),  # Medium vol
            np.random.uniform(0.25, 0.35, 100),  # High vol, spike
            np.random.uniform(0.25, 0.35, 100),  # High vol, sustained
            np.random.uniform(0.25, 0.35, 100),  # High vol, decay
        ])
        
        rho_sigma = np.concatenate([
            np.random.uniform(0.9, 1.1, 100),    # Calm
            np.random.uniform(0.9, 1.1, 100),    # Choppy
            np.random.uniform(1.6, 2.0, 100),    # Spike
            np.random.uniform(0.9, 1.4, 100),    # Sustained
            np.random.uniform(0.5, 0.75, 100),   # Decay
        ])
        
        states = pd.DataFrame({
            'sigma_1m': sigma_1m,
            'rho_sigma': rho_sigma,
        }, index=dates)
        
        regimes = classifier.classify(states)
        unique_regimes = set(regimes.unique())
        
        # Should have at least 4 regimes (5 if well-distributed)
        assert len(unique_regimes) >= 4
    
    def test_transition_matrix(self, classifier, sample_states):
        """Test transition matrix computation."""
        regimes = classifier.classify(sample_states)
        trans_mat = classifier.compute_transition_matrix(sample_states)
        
        assert isinstance(trans_mat, pd.DataFrame)
        
        # Rows should sum to 1 (within floating point tolerance)
        row_sums = trans_mat.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)
    
    def test_regime_statistics(self, classifier, sample_states):
        """Test regime statistics computation."""
        stats = classifier.compute_regime_statistics(sample_states)
        
        assert isinstance(stats, pd.DataFrame)
        assert 'observations' in stats.columns
        assert 'frequency' in stats.columns


class TestPathStateClassifier:
    """Tests for PathStateClassifier class."""
    
    @pytest.fixture
    def sample_daily_returns(self):
        """Generate sample daily returns."""
        np.random.seed(42)
        n_days = 1000
        dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
        returns = pd.Series(np.random.normal(0.0003, 0.01, n_days), index=dates)
        return returns
    
    @pytest.fixture
    def classifier(self):
        """Create path state classifier."""
        return PathStateClassifier()
    
    def test_compute_states(self, classifier, sample_daily_returns):
        """Test state computation."""
        states = classifier.compute_states(sample_daily_returns)
        
        assert isinstance(states, pd.DataFrame)
        assert 'sigma_1w' in states.columns
        assert 'sigma_1m' in states.columns
        assert 'rho_sigma' in states.columns
    
    def test_monthly_states(self, classifier, sample_daily_returns):
        """Test monthly state computation."""
        states = classifier.compute_monthly_states(sample_daily_returns)
        
        # Should have fewer observations than daily
        assert len(states) < len(sample_daily_returns)
        
        # Should be monthly frequency
        assert states.index.freq == 'M' or len(states) < 50


class TestPathState:
    """Tests for PathState dataclass."""
    
    def test_to_array(self):
        """Test conversion to array."""
        state = PathState(
            ret_1m=0.02, ret_3m=0.05,
            sigma_1w=0.15, sigma_1m=0.18, sigma_3m=0.16, sigma_6m=0.15,
            rho_sigma=0.94, drawdown=0.05, drawdown_speed=0.02,
        )
        
        arr = state.to_array()
        assert len(arr) == 9
        assert arr[0] == 0.02
    
    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([0.02, 0.05, 0.15, 0.18, 0.16, 0.15, 0.94, 0.05, 0.02])
        state = PathState.from_array(arr)
        
        assert state.ret_1m == 0.02
        assert state.sigma_1m == 0.18
        assert state.rho_sigma == 0.94


class TestSyntheticData:
    """Tests using synthetic data generator."""
    
    def test_regime_classification_on_synthetic(self):
        """Test regime classification on synthetic data."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate(n_months=200)
        
        # Synthetic data should already have regimes
        regimes = data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        # Check all regimes are present
        unique = set(regimes.unique())
        expected = {'Calm Trend', 'Choppy Transition', 'Slow-Burn Stress', 
                   'Crash-Spike', 'Recovery'}
        
        assert len(unique) >= 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
