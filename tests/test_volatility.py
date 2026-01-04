"""
Tests for volatility calculation module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.regimes.volatility import VolatilityCalculator, compute_realized_volatility


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample daily returns."""
        np.random.seed(42)
        n_days = 500
        dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
        returns = pd.Series(np.random.normal(0.0005, 0.01, n_days), index=dates)
        return returns
    
    @pytest.fixture
    def calculator(self):
        """Create volatility calculator."""
        return VolatilityCalculator()
    
    def test_compute_basic(self, calculator, sample_returns):
        """Test basic volatility computation."""
        vol = calculator.compute(sample_returns)
        
        assert isinstance(vol, pd.DataFrame)
        assert 'sigma_1w' in vol.columns
        assert 'sigma_1m' in vol.columns
        assert 'sigma_3m' in vol.columns
        assert 'rho_sigma' in vol.columns
        
    def test_volatility_positive(self, calculator, sample_returns):
        """Test that volatility is always positive."""
        vol = calculator.compute(sample_returns)
        
        assert (vol['sigma_1m'].dropna() > 0).all()
        assert (vol['sigma_3m'].dropna() > 0).all()
    
    def test_volatility_ratio_reasonable(self, calculator, sample_returns):
        """Test that volatility ratio is in reasonable range."""
        vol = calculator.compute(sample_returns)
        
        rho = vol['rho_sigma'].dropna()
        assert (rho > 0).all()
        assert (rho < 10).all()  # Should not be extremely high
    
    def test_annualization(self, calculator, sample_returns):
        """Test volatility annualization."""
        vol = calculator.compute(sample_returns)
        
        # Annualized vol should be roughly in 10-30% range for normal data
        avg_vol = vol['sigma_1m'].dropna().mean()
        assert 0.05 < avg_vol < 0.50
    
    def test_drawdown_computation(self, calculator, sample_returns):
        """Test drawdown computation."""
        prices = (1 + sample_returns).cumprod() * 100
        dd = calculator.compute_drawdown(prices)
        
        assert 'drawdown' in dd.columns
        assert 'drawdown_speed' in dd.columns
        assert (dd['drawdown'] >= 0).all()
        assert (dd['drawdown'] <= 1).all()


class TestComputeRealizedVolatility:
    """Tests for compute_realized_volatility function."""
    
    def test_basic(self):
        """Test basic computation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        
        vol = compute_realized_volatility(returns, window=21)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(returns)
    
    def test_annualization(self):
        """Test annualization option."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        
        vol_ann = compute_realized_volatility(returns, annualize=True)
        vol_not_ann = compute_realized_volatility(returns, annualize=False)
        
        # Annualized should be higher
        assert vol_ann.dropna().mean() > vol_not_ann.dropna().mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
