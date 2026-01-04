#!/usr/bin/env python
"""
Main analysis script for the volatility path states paper.

This script runs the full analysis pipeline including:
1. Data loading/generation
2. Regime classification
3. Factor performance analysis
4. Portfolio construction and backtesting
5. Statistical tests
6. Figure and table generation

Usage:
    python run_analysis.py --config config.yaml
    python run_analysis.py --synthetic
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import Config
from src.data.synthetic_data import SyntheticDataGenerator
from src.regimes.path_states import PathStateClassifier
from src.regimes.regime_classifier import RegimeClassifier
from src.portfolio.baseline import BaselinePortfolio
from src.portfolio.state_conditioned import StateConditionedPortfolio
from src.portfolio.volatility_scaling import VolatilityScaledPortfolio
from src.analysis.performance import PerformanceAnalyzer
from src.analysis.statistics import StatisticalTests
from src.visualization.figures import FigureGenerator
from src.visualization.tables import TableGenerator
from src.utils.io import save_results, ensure_dir


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run volatility path states analysis'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of real data',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory',
    )
    parser.add_argument(
        '--figures-only',
        action='store_true',
        help='Only generate figures',
    )
    parser.add_argument(
        '--tables-only',
        action='store_true',
        help='Only generate tables',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    return parser.parse_args()


def load_data(config: Config, synthetic: bool = False, seed: int = 42):
    """Load or generate data."""
    logger.info("Loading data...")
    
    if synthetic or config.data.source == 'synthetic':
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator(seed=seed)
        data = generator.generate(n_months=732)
        return data
    else:
        # Load real data
        from src.data.data_loader import load_data as load_real_data
        market, factors, daily = load_real_data(
            source=config.data.source,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
        )
        return {
            'market': market,
            'factors': factors,
            'daily': daily,
        }


def classify_regimes(data: dict, config: Config):
    """Classify path state regimes."""
    logger.info("Classifying regimes...")
    
    # Get volatility data
    if 'volatility' in data:
        vol_data = data['volatility']
    elif 'regimes' in data and 'sigma_1m' in data['regimes'].columns:
        vol_data = data['regimes']
    else:
        # Compute from daily returns
        classifier = PathStateClassifier()
        vol_data = classifier.compute_monthly_states(data['daily'])
    
    # Classify regimes
    regime_classifier = RegimeClassifier(
        vol_quantile_low=config.regimes.vol_quantile_low,
        vol_quantile_high=config.regimes.vol_quantile_high,
        ratio_spike=config.regimes.ratio_spike_threshold,
        ratio_decay=config.regimes.ratio_decay_threshold,
        expanding_window=config.regimes.expanding_window,
    )
    
    regimes = regime_classifier.classify(vol_data)
    
    # Compute statistics
    stats = regime_classifier.compute_regime_statistics(vol_data, data.get('factors'))
    logger.info(f"Regime statistics:\n{stats}")
    
    return regimes, vol_data


def run_portfolio_analysis(data: dict, regimes: pd.Series, config: Config):
    """Run portfolio analysis."""
    logger.info("Running portfolio analysis...")
    
    factors = data['factors']
    
    results = {}
    
    # Baseline portfolios
    logger.info("Building baseline portfolios...")
    baseline = BaselinePortfolio(
        factors,
        target_volatility=config.portfolio.target_volatility,
        transaction_cost=config.portfolio.transaction_cost,
    )
    baseline_result = baseline.backtest(start=config.data.evaluation_start)
    results['baseline'] = baseline_result
    
    # State-conditioned portfolios
    logger.info("Building state-conditioned portfolios...")
    state_cond = StateConditionedPortfolio(
        factors,
        regimes,
        target_volatility=config.portfolio.target_volatility,
        transaction_cost=config.portfolio.transaction_cost,
    )
    state_cond.fit(
        training_end=config.data.training_end,
        regularization=config.portfolio.regularization,
    )
    state_cond_result = state_cond.backtest(start=config.data.evaluation_start)
    results['state_conditioned'] = state_cond_result
    
    # Volatility-scaled portfolios
    logger.info("Building volatility-scaled portfolios...")
    vol_scaled = VolatilityScaledPortfolio(
        factors,
        target_volatility=config.portfolio.target_volatility,
        transaction_cost=config.portfolio.transaction_cost,
    )
    vol_scaled_result = vol_scaled.backtest(start=config.data.evaluation_start)
    results['vol_scaled'] = vol_scaled_result
    
    # Performance comparison
    logger.info("Comparing performance...")
    analyzer = PerformanceAnalyzer()
    
    comparison = analyzer.compare_strategies({
        'Baseline': baseline_result.returns['net'],
        'State-Conditioned': state_cond_result.returns['net'],
        'Vol-Scaled': vol_scaled_result.returns['net'],
    })
    logger.info(f"Performance comparison:\n{comparison}")
    
    results['comparison'] = comparison
    results['exposures'] = state_cond.exposures
    
    return results


def run_statistical_tests(data: dict, regimes: pd.Series, results: dict):
    """Run statistical tests."""
    logger.info("Running statistical tests...")
    
    factors = data['factors']
    tests = StatisticalTests()
    
    test_results = {}
    
    # Test regime differences for each factor
    for factor in factors.columns:
        logger.info(f"Testing {factor}...")
        
        # ANOVA
        anova = tests.test_all_regimes_equal(factors[factor], regimes)
        test_results[f'{factor}_anova'] = {
            'statistic': anova.statistic,
            'pvalue': anova.pvalue,
            'reject_null': anova.reject_null,
        }
        
        # Calm vs Crash-Spike
        if 'Calm Trend' in regimes.values and 'Crash-Spike' in regimes.values:
            diff_test = tests.test_regime_difference(
                factors[factor], regimes, 'Calm Trend', 'Crash-Spike'
            )
            test_results[f'{factor}_calm_vs_crash'] = {
                'statistic': diff_test.statistic,
                'pvalue': diff_test.pvalue,
                'reject_null': diff_test.reject_null,
            }
    
    # Test Sharpe ratio differences
    baseline_ret = results['baseline'].returns['net']
    cond_ret = results['state_conditioned'].returns['net']
    
    sharpe_test = tests.test_sharpe_difference(cond_ret, baseline_ret)
    test_results['sharpe_difference'] = {
        'statistic': sharpe_test.statistic,
        'pvalue': sharpe_test.pvalue,
        'reject_null': sharpe_test.reject_null,
    }
    
    # Test crash concentration
    if 'Momentum' in factors.columns:
        crash_test = tests.test_crash_concentration(
            factors['Momentum'], regimes, 'Crash-Spike'
        )
        test_results['crash_concentration'] = {
            'statistic': crash_test.statistic,
            'pvalue': crash_test.pvalue,
            'reject_null': crash_test.reject_null,
        }
    
    logger.info(f"Statistical tests complete")
    
    return test_results


def generate_outputs(data: dict, regimes: pd.Series, vol_data: pd.DataFrame,
                    results: dict, output_dir: str):
    """Generate figures and tables."""
    logger.info("Generating outputs...")
    
    output_path = Path(output_dir)
    
    # Prepare data for figure generator
    fig_data = {
        'factors': data['factors'],
        'regimes': regimes,
        'states': vol_data,
        'volatility': vol_data,
        'ic': data.get('ic'),
    }
    
    # Generate figures
    logger.info("Generating figures...")
    fig_gen = FigureGenerator(output_dir=str(output_path / 'figures'))
    fig_gen.generate_all_figures(fig_data)
    
    # Generate tables
    logger.info("Generating tables...")
    table_gen = TableGenerator(output_dir=str(output_path / 'tables'))
    table_gen.generate_all_tables(fig_data)
    
    # Save results
    logger.info("Saving results...")
    results_path = output_path / 'results'
    ensure_dir(results_path)
    
    # Performance comparison
    if 'comparison' in results:
        results['comparison'].to_csv(results_path / 'performance_comparison.csv')
    
    # Exposures
    if 'exposures' in results:
        exp_df = pd.DataFrame(results['exposures'])
        exp_df.to_csv(results_path / 'optimal_exposures.csv')
    
    logger.info("Output generation complete")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Volatility Path States Analysis")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = Config(str(config_path))
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = Config()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = ensure_dir(args.output)
    
    try:
        # Load data
        data = load_data(config, synthetic=args.synthetic, seed=args.seed)
        
        # Classify regimes
        regimes, vol_data = classify_regimes(data, config)
        
        if args.figures_only:
            # Only generate figures
            generate_outputs(data, regimes, vol_data, {}, str(output_dir))
            return
        
        if args.tables_only:
            # Only generate tables
            table_gen = TableGenerator(output_dir=str(output_dir / 'tables'))
            table_gen.generate_all_tables({
                'factors': data['factors'],
                'regimes': regimes,
            })
            return
        
        # Run portfolio analysis
        results = run_portfolio_analysis(data, regimes, config)
        
        # Run statistical tests
        test_results = run_statistical_tests(data, regimes, results)
        results['tests'] = test_results
        
        # Generate outputs
        generate_outputs(data, regimes, vol_data, results, str(output_dir))
        
        # Save all results
        save_results(
            {
                'test_results': test_results,
                'exposures': results.get('exposures', {}),
            },
            str(output_dir / 'results' / 'analysis_results.json'),
        )
        
        logger.info("=" * 60)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()
