"""
Configuration management for the volatility path states project.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration settings."""
    start_date: str = '1963-01-01'
    end_date: str = '2023-12-31'
    training_end: str = '1999-12-31'
    evaluation_start: str = '2000-01-01'
    min_price: float = 1.0
    min_market_cap_percentile: int = 20
    min_return_history: int = 12
    source: str = 'synthetic'


@dataclass
class RegimeConfig:
    """Regime classification settings."""
    vol_quantile_low: float = 0.33
    vol_quantile_high: float = 0.67
    ratio_spike_threshold: float = 1.5
    ratio_decay_threshold: float = 0.8
    horizon_short: int = 5
    horizon_medium: int = 21
    horizon_long: int = 63
    horizon_extended: int = 126
    expanding_window: bool = True


@dataclass
class FactorConfig:
    """Factor construction settings."""
    momentum_lookback: int = 12
    momentum_skip: int = 1
    momentum_min_obs: int = 8
    value_lag_months: int = 6
    quality_exclude_financials: bool = True
    low_risk_lookback: int = 60
    low_risk_min_obs: int = 36
    n_quantiles: int = 10
    weighting: str = 'value'
    breakpoints: str = 'nyse'


@dataclass
class PortfolioConfig:
    """Portfolio construction settings."""
    regularization: float = 0.5
    min_exposure: float = 0.0
    max_exposure: float = 1.0
    transaction_cost: float = 0.002
    target_volatility: float = 0.10


@dataclass
class AnalysisConfig:
    """Analysis settings."""
    newey_west_lags: int = 12
    bootstrap_replications: int = 1000
    crash_percentile: int = 5
    annualization_factor: int = 12


class Config:
    """
    Main configuration class that loads and manages all settings.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML configuration file. If None, uses defaults.
        
    Examples
    --------
    >>> config = Config('config.yaml')
    >>> print(config.data.start_date)
    '1963-01-01'
    >>> print(config.regimes.ratio_spike_threshold)
    1.5
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self._config_dict: Dict[str, Any] = {}
        
        # Load from file if provided
        if config_path is not None:
            self.load(config_path)
        
        # Initialize component configs
        self._init_components()
        
    def _init_components(self):
        """Initialize component configuration objects."""
        data_dict = self._config_dict.get('data', {})
        regime_dict = self._config_dict.get('regimes', {})
        factor_dict = self._config_dict.get('factors', {})
        portfolio_dict = self._config_dict.get('portfolio', {})
        analysis_dict = self._config_dict.get('analysis', {})
        
        # Data config
        self.data = DataConfig(
            start_date=data_dict.get('start_date', '1963-01-01'),
            end_date=data_dict.get('end_date', '2023-12-31'),
            training_end=data_dict.get('training_end', '1999-12-31'),
            evaluation_start=data_dict.get('evaluation_start', '2000-01-01'),
            min_price=data_dict.get('min_price', 1.0),
            min_market_cap_percentile=data_dict.get('min_market_cap_percentile', 20),
            min_return_history=data_dict.get('min_return_history', 12),
            source=data_dict.get('source', 'synthetic'),
        )
        
        # Regime config
        vol_quantiles = regime_dict.get('vol_quantiles', {})
        ratio_thresholds = regime_dict.get('ratio_thresholds', {})
        horizons = regime_dict.get('horizons', {})
        
        self.regimes = RegimeConfig(
            vol_quantile_low=vol_quantiles.get('low', 0.33),
            vol_quantile_high=vol_quantiles.get('high', 0.67),
            ratio_spike_threshold=ratio_thresholds.get('spike', 1.5),
            ratio_decay_threshold=ratio_thresholds.get('decay', 0.8),
            horizon_short=horizons.get('short', 5),
            horizon_medium=horizons.get('medium', 21),
            horizon_long=horizons.get('long', 63),
            horizon_extended=horizons.get('extended', 126),
            expanding_window=regime_dict.get('expanding_window', True),
        )
        
        # Factor config
        momentum = factor_dict.get('momentum', {})
        value = factor_dict.get('value', {})
        quality = factor_dict.get('quality', {})
        low_risk = factor_dict.get('low_risk', {})
        portfolio_construction = factor_dict.get('portfolio', {})
        
        self.factors = FactorConfig(
            momentum_lookback=momentum.get('lookback', 12),
            momentum_skip=momentum.get('skip', 1),
            momentum_min_obs=momentum.get('min_observations', 8),
            value_lag_months=value.get('lag_months', 6),
            quality_exclude_financials=quality.get('exclude_financials', True),
            low_risk_lookback=low_risk.get('lookback', 60),
            low_risk_min_obs=low_risk.get('min_observations', 36),
            n_quantiles=portfolio_construction.get('n_quantiles', 10),
            weighting=portfolio_construction.get('weighting', 'value'),
            breakpoints=portfolio_construction.get('breakpoints', 'nyse'),
        )
        
        # Portfolio config
        tc = portfolio_dict.get('transaction_costs', {})
        
        self.portfolio = PortfolioConfig(
            regularization=portfolio_dict.get('regularization', 0.5),
            min_exposure=portfolio_dict.get('min_exposure', 0.0),
            max_exposure=portfolio_dict.get('max_exposure', 1.0),
            transaction_cost=tc.get('baseline', 0.002),
            target_volatility=portfolio_dict.get('target_volatility', 0.10),
        )
        
        # Analysis config
        self.analysis = AnalysisConfig(
            newey_west_lags=analysis_dict.get('newey_west_lags', 12),
            bootstrap_replications=analysis_dict.get('bootstrap_replications', 1000),
            crash_percentile=analysis_dict.get('crash_percentile', 5),
            annualization_factor=analysis_dict.get('annualization_factor', 12),
        )
        
        # Output settings
        output_dict = self._config_dict.get('output', {})
        self.output_dir = output_dict.get('results_dir', 'output/results')
        self.figures_dir = output_dict.get('figures_dir', 'output/figures')
        self.tables_dir = output_dict.get('tables_dir', 'output/tables')
        
        # Random seed
        self.random_seed = self._config_dict.get('random_seed', 42)
        
        # Default exposures
        default_exp = portfolio_dict.get('default_exposures', {})
        self.default_exposures = self._parse_default_exposures(default_exp)
        
    def _parse_default_exposures(self, exp_dict: Dict) -> Dict[str, Dict[str, float]]:
        """Parse default exposure values from config."""
        defaults = {
            'momentum': {
                'calm_trend': 1.0,
                'choppy_transition': 0.7,
                'slow_burn_stress': 0.5,
                'crash_spike': 0.0,
                'recovery': 0.7,
            },
            'value': {
                'calm_trend': 1.0,
                'choppy_transition': 1.0,
                'slow_burn_stress': 0.8,
                'crash_spike': 0.4,
                'recovery': 1.0,
            },
            'quality': {
                'calm_trend': 1.0,
                'choppy_transition': 1.0,
                'slow_burn_stress': 1.0,
                'crash_spike': 1.0,
                'recovery': 0.85,
            },
            'low_risk': {
                'calm_trend': 1.0,
                'choppy_transition': 1.0,
                'slow_burn_stress': 1.0,
                'crash_spike': 1.0,
                'recovery': 0.75,
            },
        }
        
        # Override with config values
        for factor, regimes in exp_dict.items():
            if factor in defaults and isinstance(regimes, dict):
                for regime, value in regimes.items():
                    if regime in defaults[factor]:
                        defaults[factor][regime] = value
                        
        return defaults
    
    def load(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(path, 'r') as f:
            self._config_dict = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        self._init_components()
        
    def save(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to save configuration file.
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._config_dict, f, default_flow_style=False)
            
        logger.info(f"Saved configuration to {config_path}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config_dict.copy()
    
    def __repr__(self) -> str:
        return f"Config(data={self.data}, regimes={self.regimes}, ...)"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file.
        
    Returns
    -------
    Config
        Configuration object.
    """
    return Config(config_path)
