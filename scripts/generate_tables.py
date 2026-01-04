#!/usr/bin/env python
"""
Generate all LaTeX tables for the paper.

Usage:
    python generate_tables.py --output output/tables
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.synthetic_data import SyntheticDataGenerator
from src.visualization.tables import TableGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper tables')
    parser.add_argument('--output', '-o', type=str, default='output/tables',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Generating tables...")
    
    # Generate data
    np.random.seed(args.seed)
    generator = SyntheticDataGenerator(seed=args.seed)
    data = generator.generate(n_months=732)
    
    # Get regimes
    regimes = data['regimes']
    if hasattr(regimes, 'columns') and 'regime' in regimes.columns:
        regimes = regimes['regime']
    
    # Prepare data
    table_data = {
        'factors': data['factors'],
        'regimes': regimes,
    }
    
    # Generate tables
    table_gen = TableGenerator(output_dir=args.output)
    table_gen.generate_all_tables(table_data)
    
    logger.info(f"Tables saved to {args.output}")


if __name__ == '__main__':
    main()
