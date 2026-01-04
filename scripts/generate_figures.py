#!/usr/bin/env python
"""
Generate all figures for the paper.

Usage:
    python generate_figures.py --output output/figures
    python generate_figures.py --synthetic --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.synthetic_data import SyntheticDataGenerator
from src.visualization.figures import FigureGenerator
from src.visualization.styles import set_publication_style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output', '-o', type=str, default='output/figures',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'svg'],
                       help='Output format')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Generating figures...")
    
    # Set style
    set_publication_style()
    
    # Generate or load data
    np.random.seed(args.seed)
    generator = SyntheticDataGenerator(seed=args.seed)
    data = generator.generate(n_months=732)
    
    # Get regimes
    if 'regimes' in data:
        regimes = data['regimes']
        if isinstance(regimes, dict):
            regimes = regimes['regime']
        elif hasattr(regimes, 'columns') and 'regime' in regimes.columns:
            regimes = regimes['regime']
    
    # Prepare data
    fig_data = {
        'factors': data['factors'],
        'regimes': regimes,
        'states': data['volatility'],
        'volatility': data['volatility'],
        'ic': data.get('ic'),
    }
    
    # Generate figures
    fig_gen = FigureGenerator(
        output_dir=args.output,
        formats=[args.format],
    )
    fig_gen.generate_all_figures(fig_data)
    
    logger.info(f"Figures saved to {args.output}")


if __name__ == '__main__':
    main()
