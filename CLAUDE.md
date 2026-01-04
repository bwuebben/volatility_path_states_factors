# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance research project analyzing how equity factor performance depends on multi-scale volatility path states. The project implements regime classification based on volatility dynamics and constructs state-conditioned factor portfolios.

**Key Concept**: Factor returns vary dramatically across different volatility "path states" - not just the level of volatility, but how that volatility formed (crash-driven spike vs. gradual accumulation).

## Common Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev,data,notebooks]"
```

### Running Analysis
```bash
# Full analysis pipeline with synthetic data
python scripts/run_analysis.py --config config.yaml --synthetic

# Generate only figures
python scripts/generate_figures.py --output output/figures/

# Generate only tables
python scripts/generate_tables.py --output output/tables/
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run specific test module
pytest tests/test_regimes.py -v

# Run single test
pytest tests/test_portfolio.py::test_state_conditioned_backtest -v
```

### Development Tools
```bash
# Code formatting
black src/ tests/ scripts/

# Linting
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

## Architecture

### Data Flow Pipeline

The analysis follows this sequence:

1. **Data Loading** (`src/data/`)
   - Real data via WRDS/CRSP or synthetic data generation
   - Daily returns → Monthly factor returns
   - Market returns used for regime classification

2. **Path State Construction** (`src/regimes/path_states.py`)
   - Computes multi-horizon volatility (1w, 1m, 3m, 6m)
   - Calculates volatility ratio: ρ_σ = σ^(1w) / σ^(3m)
   - Tracks drawdowns and returns
   - Output: 9-dimensional state vector per time period

3. **Regime Classification** (`src/regimes/regime_classifier.py`)
   - Hierarchical classification based on volatility level + dynamics
   - Five regimes: Calm Trend, Choppy Transition, Slow-Burn Stress, Crash-Spike, Recovery
   - Uses **expanding window** for threshold estimation (no look-ahead bias)
   - Primary split: volatility level (low/medium/high)
   - Secondary split (high vol): dynamics via volatility ratio

4. **Portfolio Construction** (`src/portfolio/`)
   - `baseline.py`: Standard factor portfolios
   - `state_conditioned.py`: Adjust exposures based on current regime
   - `volatility_scaling.py`: Vol-targeting approach
   - Exposure optimization uses training period only

5. **Analysis** (`src/analysis/`)
   - Performance metrics, statistical tests, IC analysis
   - Robustness checks and regime statistics

### Key Design Patterns

**Expanding Window**: Critical for avoiding look-ahead bias. Volatility thresholds and state standardization use only data available at each point in time.

**State-Conditioned Exposures**: The core methodology adjusts factor exposure g(s_t) based on regime:
- Calm Trend: full exposure (1.0)
- Crash-Spike: zero exposure for momentum (0.0), reduced for others
- Other regimes: intermediate exposures (0.4-1.0)

**Training/Test Split**: Default training period ends 1999-12-31, evaluation starts 2000-01-01. Exposures are fitted on training data and held constant out-of-sample.

### Important Classes

**PathStateClassifier** (`src/regimes/path_states.py:86`):
- `compute_states()`: Daily path state variables from returns
- `compute_monthly_states()`: Month-end state observations
- `standardize_states()`: Expanding window standardization
- Returns DataFrame with columns: ret_1m, ret_3m, sigma_1w, sigma_1m, sigma_3m, sigma_6m, rho_sigma, drawdown, drawdown_speed

**RegimeClassifier** (`src/regimes/regime_classifier.py:44`):
- `classify()`: Assigns regime labels to each time period
- `_classify_single()`: Core classification logic for single observation
- Uses expanding window thresholds when `expanding_window=True`
- Returns pd.Series of regime labels

**StateConditionedPortfolio** (`src/portfolio/state_conditioned.py:19`):
- `fit()`: Optimize exposures on training data
- `backtest()`: Run out-of-sample backtest with state conditioning
- `compute_effective_returns()`: Factor returns × exposure(regime)
- `_compute_turnover()`: Account for transaction costs from exposure changes

### Configuration

All parameters controlled via `config.yaml`:
- Data periods and sources
- Volatility horizons and thresholds
- Factor construction parameters
- Portfolio settings (regularization, transaction costs, default exposures)
- Analysis settings (Newey-West lags, bootstrap replications)

The Config class (`src/config.py`) handles loading and validation.

### Factor Construction

Factors are built in `src/factors/`:
- **Momentum**: 12-month return (skip most recent month)
- **Value**: Book-to-market ratio
- **Quality**: Gross profitability
- **Low-Risk**: Beta (60-month estimation)

Each factor uses long-short decile portfolios with NYSE breakpoints.

### Regime Classification Logic

The hierarchical classification in `RegimeClassifier._classify_single()`:
```
if sigma <= vol_low:
    return 'Calm Trend'
elif sigma <= vol_high:
    return 'Choppy Transition'
else:  # High volatility
    if rho > ratio_spike:  # Rapid acceleration
        return 'Crash-Spike'
    elif rho < ratio_decay:  # Decaying volatility
        return 'Recovery'
    else:  # Sustained high vol
        return 'Slow-Burn Stress'
```

Where:
- `vol_low`, `vol_high`: Volatility level thresholds (33rd, 67th percentiles)
- `rho`: Volatility ratio σ^(1w) / σ^(3m)
- `ratio_spike`: Spike threshold (default 1.5)
- `ratio_decay`: Decay threshold (default 0.8)

## Development Notes

### Data Sources
- **Synthetic data**: Default for testing/demos, generated via `SyntheticDataGenerator`
- **Real data**: Requires WRDS access (CRSP + Compustat) or Yahoo Finance
- Set `source: 'synthetic'` in config.yaml to use synthetic data

### Notebooks
Exploratory notebooks in `notebooks/`:
1. Data exploration
2. Factor analysis
3. Regime classification
4. Portfolio construction
5. Results visualization

### Output Structure
```
output/
├── figures/     # PDF/PNG plots
├── tables/      # LaTeX tables
└── results/     # CSV/JSON results
```

### Testing Strategy
- `test_volatility.py`: Volatility calculations, drawdowns
- `test_regimes.py`: Path states, regime classification, transitions
- `test_portfolio.py`: Portfolio construction, backtesting, exposure optimization
- `test_factors.py`: Factor construction and signals
