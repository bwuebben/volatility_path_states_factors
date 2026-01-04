"""
Tests for factor-related functionality.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_data import SyntheticDataGenerator
from src.analysis.performance import PerformanceAnalyzer, PerformanceMetrics
from src.analysis.statistics import StatisticalTests
from src.analysis.information_coefficient import InformationCoefficientAnalyzer


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='M')
        returns = pd.Series(np.random.normal(0.005, 0.04, n), index=dates)
        return returns
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return PerformanceAnalyzer()
    
    def test_compute_metrics(self, analyzer, sample_returns):
        """Test metrics computation."""
        metrics = analyzer.compute_metrics(sample_returns)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
    
    def test_metrics_to_dict(self, analyzer, sample_returns):
        """Test metrics conversion to dict."""
        metrics = analyzer.compute_metrics(sample_returns)
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert 'sharpe_ratio' in d
    
    def test_drawdown_series(self, analyzer, sample_returns):
        """Test drawdown series computation."""
        dd, duration = analyzer.compute_drawdown_series(sample_returns)
        
        assert len(dd) == len(sample_returns)
        assert (dd >= 0).all()
        assert (dd <= 1).all()
    
    def test_rolling_metrics(self, analyzer, sample_returns):
        """Test rolling metrics."""
        rolling = analyzer.compute_rolling_metrics(sample_returns, window=24)
        
        assert 'mean' in rolling.columns
        assert 'sharpe' in rolling.columns
    
    def test_compare_strategies(self, analyzer):
        """Test strategy comparison."""
        np.random.seed(42)
        n = 100
        
        strategies = {
            'A': pd.Series(np.random.normal(0.01, 0.03, n)),
            'B': pd.Series(np.random.normal(0.005, 0.04, n)),
        }
        
        comparison = analyzer.compare_strategies(strategies)
        
        assert 'A' in comparison.index
        assert 'B' in comparison.index


class TestStatisticalTests:
    """Tests for StatisticalTests class."""
    
    @pytest.fixture
    def tests(self):
        """Create tests object."""
        return StatisticalTests()
    
    def test_newey_west_tstat(self, tests):
        """Test Newey-West t-statistic."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.03, 100))
        
        mean, tstat, pvalue = tests.newey_west_tstat(returns)
        
        assert mean > 0
        assert isinstance(tstat, float)
        assert 0 <= pvalue <= 1
    
    def test_mean_difference(self, tests):
        """Test mean difference test."""
        np.random.seed(42)
        a = pd.Series(np.random.normal(0.02, 0.03, 100))
        b = pd.Series(np.random.normal(0.01, 0.03, 100))
        
        result = tests.test_mean_difference(a, b)
        
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'reject_null')
    
    def test_regime_difference(self, tests):
        """Test regime difference test."""
        np.random.seed(42)
        n = 200
        
        returns = pd.Series(np.random.normal(0.005, 0.04, n))
        regimes = pd.Series(
            np.random.choice(['A', 'B'], n),
            index=returns.index
        )
        
        result = tests.test_regime_difference(returns, regimes, 'A', 'B')
        
        assert hasattr(result, 'pvalue')


class TestInformationCoefficientAnalyzer:
    """Tests for IC analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create IC analyzer."""
        return InformationCoefficientAnalyzer()
    
    def test_ic_statistics(self, analyzer):
        """Test IC statistics computation."""
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.03, 0.02, 100))
        
        stats = analyzer.compute_ic_statistics(ic_series)
        
        assert 'mean_ic' in stats
        assert 'ir' in stats
        assert 'hit_rate' in stats
    
    def test_rolling_ic(self, analyzer):
        """Test rolling IC computation."""
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.03, 0.02, 100))
        
        rolling = analyzer.compute_rolling_ic(ic_series, window=12)
        
        assert 'ic_mean' in rolling.columns
        assert 'ir' in rolling.columns


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation."""
    
    def test_generate_basic(self):
        """Test basic data generation."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate(n_months=100)
        
        assert 'market' in data
        assert 'factors' in data
        assert 'regimes' in data
    
    def test_factor_returns_reasonable(self):
        """Test that factor returns are reasonable."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate(n_months=200)
        
        factors = data['factors']
        
        for col in factors.columns:
            mean = factors[col].mean()
            std = factors[col].std()
            
            # Mean should be small (monthly)
            assert abs(mean) < 0.05
            
            # Std should be reasonable
            assert 0.01 < std < 0.15
    
    def test_regime_distribution(self):
        """Test regime distribution."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate(n_months=500)
        
        regimes = data['regimes']
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        # Check distribution is reasonable
        freq = regimes.value_counts(normalize=True)
        
        # Calm Trend should be most common
        assert freq.get('Calm Trend', 0) > 0.2
        
        # Crash-Spike should be rare
        assert freq.get('Crash-Spike', 0) < 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
