# Volatility Path States and Factor Performance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and analysis for the paper:

**"Path-Dependent Volatility and the Conditional Performance of Equity Factors"**

The paper demonstrates that equity factor performance depends not only on the level of market volatility but on the multi-scale path by which that volatility formed. By distinguishing crash-driven volatility spikes from gradually accumulating stress, investors can identify market environments with qualitatively different implications for factor returns.

## Key Findings

1. **Factor returns vary dramatically across path states**: Momentum earns 1.4% monthly in calm states but loses 3.9% in crash-spike states
2. **Signal efficacy is regime-dependent**: Momentum ICs turn negative during rapid volatility expansions
3. **Factor crashes are concentrated**: Nearly 50% of momentum crashes occur in crash-spike states (7% of sample)
4. **State conditioning improves performance**: 15-40% Sharpe ratio improvement, 25-50% drawdown reduction

## Project Structure

```
volatility_path_states/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── config.yaml                 # Configuration parameters
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Load market and factor data
│   │   ├── wrds_loader.py     # WRDS/CRSP data access
│   │   └── synthetic_data.py  # Generate synthetic data for testing
│   │
│   ├── factors/               # Factor construction
│   │   ├── __init__.py
│   │   ├── factor_builder.py  # Build factor portfolios
│   │   ├── momentum.py        # Momentum factor
│   │   ├── value.py           # Value factor
│   │   ├── quality.py         # Quality factor
│   │   └── low_risk.py        # Low-risk/beta factor
│   │
│   ├── regimes/               # Regime classification
│   │   ├── __init__.py
│   │   ├── volatility.py      # Volatility calculations
│   │   ├── path_states.py     # Multi-scale path state construction
│   │   └── regime_classifier.py  # Regime classification
│   │
│   ├── portfolio/             # Portfolio construction
│   │   ├── __init__.py
│   │   ├── baseline.py        # Baseline factor portfolios
│   │   ├── state_conditioned.py  # State-conditioned portfolios
│   │   ├── volatility_scaling.py # Volatility-managed portfolios
│   │   └── optimizer.py       # Exposure optimization
│   │
│   ├── analysis/              # Analysis and statistics
│   │   ├── __init__.py
│   │   ├── performance.py     # Performance metrics
│   │   ├── statistics.py      # Statistical tests
│   │   ├── information_coefficient.py  # IC analysis
│   │   └── robustness.py      # Robustness tests
│   │
│   ├── visualization/         # Plotting and figures
│   │   ├── __init__.py
│   │   ├── figures.py         # Generate paper figures
│   │   ├── tables.py          # Generate LaTeX tables
│   │   └── styles.py          # Plot styling
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── dates.py           # Date handling
│       ├── returns.py         # Return calculations
│       └── io.py              # Input/output helpers
│
├── data/                      # Data directory
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Processed datasets
│   └── cache/                 # Cached computations
│
├── output/                    # Output directory
│   ├── figures/               # Generated figures
│   ├── tables/                # Generated tables
│   └── results/               # Analysis results
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   ├── 03_regime_classification.ipynb
│   ├── 04_portfolio_construction.ipynb
│   └── 05_results_visualization.ipynb
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_volatility.py
│   ├── test_regimes.py
│   ├── test_factors.py
│   └── test_portfolio.py
│
└── scripts/                   # Executable scripts
    ├── run_analysis.py        # Main analysis script
    ├── generate_figures.py    # Generate all figures
    └── generate_tables.py     # Generate all tables
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/volatility_path_states.git
cd volatility_path_states
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode

```bash
pip install -e .
```

## Data Sources

The analysis can use either:

1. **Real Data** (requires access):
   - CRSP Monthly Stock File (via WRDS)
   - Compustat North America Fundamentals
   - Kenneth French Data Library (freely available)

2. **Synthetic Data** (included):
   - Simulated data matching the statistical properties described in the paper
   - Useful for testing and demonstration

## Quick Start

### Using Synthetic Data

```python
from src.data.synthetic_data import SyntheticDataGenerator
from src.regimes.path_states import PathStateClassifier
from src.portfolio.state_conditioned import StateConditionedPortfolio
from src.analysis.performance import PerformanceAnalyzer

# Generate synthetic data
generator = SyntheticDataGenerator(seed=42)
data = generator.generate(n_months=732)

# Classify path states
classifier = PathStateClassifier()
regimes = classifier.classify(data['market'])

# Build state-conditioned portfolio
portfolio = StateConditionedPortfolio(
    factor_returns=data['factors'],
    regimes=regimes
)
portfolio.fit(training_end='1999-12-31')

# Analyze performance
analyzer = PerformanceAnalyzer()
results = analyzer.analyze(portfolio, start='2000-01-01')
print(results.summary())
```

### Using Real Data (WRDS)

```python
from src.data.wrds_loader import WRDSDataLoader
from src.factors.factor_builder import FactorBuilder

# Load data from WRDS (requires credentials)
loader = WRDSDataLoader(username='your_username')
crsp_data = loader.load_crsp(start='1963-01-01', end='2023-12-31')
compustat_data = loader.load_compustat(start='1962-01-01', end='2023-12-31')

# Build factors
builder = FactorBuilder(crsp_data, compustat_data)
factors = builder.build_all_factors()
```

## Running the Analysis

### Full Analysis Pipeline

```bash
python scripts/run_analysis.py --config config.yaml
```

### Generate Figures Only

```bash
python scripts/generate_figures.py --output output/figures/
```

### Generate Tables Only

```bash
python scripts/generate_tables.py --output output/tables/
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Sample configuration
data:
  start_date: '1963-01-01'
  end_date: '2023-12-31'
  training_end: '1999-12-31'
  
regimes:
  vol_quantiles: [0.33, 0.67]
  ratio_thresholds: [0.8, 1.5]
  
portfolio:
  regularization: 0.5
  transaction_cost: 0.002
  
factors:
  momentum_lookback: 12
  momentum_skip: 1
  beta_lookback: 60
```

## Key Components

### Path State Variables

The path state vector consists of:
- Multi-horizon returns: 1-month, 3-month
- Multi-horizon realized volatility: 1-week, 1-month, 3-month, 6-month
- Volatility ratio: σ^(1w) / σ^(3m)
- Drawdown magnitude and speed

### Regime Classification

Five regimes are identified:
1. **Calm Trend**: Low volatility
2. **Choppy Transition**: Medium volatility
3. **Slow-Burn Stress**: High volatility, sustained dynamics
4. **Crash-Spike**: High volatility, rapid acceleration
5. **Recovery**: High volatility, decaying dynamics

### State-Conditioned Portfolio

Exposure adjustment based on regime:
```
g(s_t) = { 1.0  if Calm Trend
         { 0.7  if Choppy Transition
         { 0.5  if Slow-Burn Stress
         { 0.0  if Crash-Spike
         { 0.7  if Recovery
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2026volatility,
  title={Path-Dependent Volatility and the Conditional Performance of Equity Factors},
  author={Bernd J. Wuebben},
  journal={Journal of Finance},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please open an issue or contact [wuebben@gmail.com].
