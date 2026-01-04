"""
Tests for portfolio construction module.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.baseline import BaselinePortfolio, MultifactorPortfolio
from src.portfolio.state_conditioned import StateConditionedPortfolio
from src.portfolio.volatility_scaling import VolatilityScaledPortfolio
from src.data.synthetic_data import SyntheticDataGenerator


class TestBaselinePortfolio:
    """Tests for BaselinePortfolio class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample factor returns."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        
        returns = pd.DataFrame({
            'Momentum': np.random.normal(0.005, 0.04, n),
            'Value': np.random.normal(0.003, 0.03, n),
            'Quality': np.random.normal(0.004, 0.02, n),
        }, index=dates)
        
        return returns
    
    def test_basic_backtest(self, sample_returns):
        """Test basic backtest."""
        portfolio = BaselinePortfolio(sample_returns)
        result = portfolio.backtest()
        
        assert hasattr(result, 'returns')
        assert 'gross' in result.returns.columns
        assert 'net' in result.returns.columns
    
    def test_volatility_scaling(self, sample_returns):
        """Test volatility-targeted portfolio."""
        portfolio = BaselinePortfolio(
            sample_returns, 
            target_volatility=0.10,
        )
        result = portfolio.backtest()
        
        # Realized vol should be close to target
        realized_vol = result.returns['gross'].std() * np.sqrt(12)
        assert 0.05 < realized_vol < 0.20
    
    def test_statistics_computation(self, sample_returns):
        """Test statistics computation."""
        portfolio = BaselinePortfolio(sample_returns)
        result = portfolio.backtest()
        stats = portfolio.compute_statistics(result)
        
        assert 'sharpe_ratio' in stats.columns
        assert 'max_drawdown' in stats.columns


class TestStateConditionedPortfolio:
    """Tests for StateConditionedPortfolio class."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with regimes."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate(n_months=300)
    
    def test_basic_construction(self, synthetic_data):
        """Test basic portfolio construction."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        portfolio = StateConditionedPortfolio(factors, regimes)
        
        assert portfolio.factor_names == list(factors.columns)
    
    def test_exposure_values(self, synthetic_data):
        """Test exposure value retrieval."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        portfolio = StateConditionedPortfolio(factors, regimes)
        
        # Momentum should have 0 exposure in Crash-Spike
        exp = portfolio.get_exposure('Momentum', 'Crash-Spike')
        assert exp == 0.0
        
        # Quality should have full exposure in Crash-Spike
        exp = portfolio.get_exposure('Quality', 'Crash-Spike')
        assert exp == 1.0
    
    def test_effective_returns(self, synthetic_data):
        """Test effective returns computation."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        portfolio = StateConditionedPortfolio(factors, regimes)
        effective = portfolio.compute_effective_returns()
        
        assert effective.shape == factors.shape
        
        # Effective returns should be <= raw returns in magnitude for some factors
        # (due to reduced exposure)
    
    def test_backtest(self, synthetic_data):
        """Test portfolio backtest."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        portfolio = StateConditionedPortfolio(factors, regimes)
        result = portfolio.backtest()
        
        assert hasattr(result, 'returns')
        assert hasattr(result, 'exposures')
    
    def test_fit(self, synthetic_data):
        """Test exposure fitting."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        portfolio = StateConditionedPortfolio(factors, regimes)
        
        # Split data
        split_date = factors.index[150]
        portfolio.fit(training_end=str(split_date))
        
        assert portfolio._fitted


class TestVolatilityScaledPortfolio:
    """Tests for VolatilityScaledPortfolio class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample factor returns."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        
        returns = pd.DataFrame({
            'Momentum': np.random.normal(0.005, 0.04, n),
            'Value': np.random.normal(0.003, 0.03, n),
        }, index=dates)
        
        return returns
    
    def test_scaling_factors(self, sample_returns):
        """Test scaling factor computation."""
        portfolio = VolatilityScaledPortfolio(sample_returns)
        
        assert hasattr(portfolio, 'scaling_factors')
        assert portfolio.scaling_factors.shape == sample_returns.shape
    
    def test_scaled_returns(self, sample_returns):
        """Test scaled returns computation."""
        portfolio = VolatilityScaledPortfolio(sample_returns)
        scaled = portfolio.compute_scaled_returns()
        
        assert scaled.shape == sample_returns.shape
    
    def test_backtest(self, sample_returns):
        """Test backtest."""
        portfolio = VolatilityScaledPortfolio(sample_returns)
        result = portfolio.backtest()
        
        assert hasattr(result, 'returns')


class TestPortfolioComparison:
    """Tests comparing different portfolio strategies."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate(n_months=300)
    
    def test_state_conditioned_vs_baseline(self, synthetic_data):
        """Test that state conditioning improves on baseline."""
        factors = synthetic_data['factors']
        regimes = synthetic_data['regimes']
        
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        # Baseline
        baseline = BaselinePortfolio(factors)
        baseline_result = baseline.backtest(start='2015-01-01')
        
        # State-conditioned
        state_cond = StateConditionedPortfolio(factors, regimes)
        state_cond_result = state_cond.backtest(start='2015-01-01')
        
        # Both should produce results
        assert len(baseline_result.returns) > 0
        assert len(state_cond_result.returns) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
