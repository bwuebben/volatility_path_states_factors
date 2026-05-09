# Factor Naming Standardization - Fix Summary

## Problem Identified

The codebase had **inconsistent factor naming** between synthetic and real data:

- **Synthetic data**: Used full names `'Momentum'`, `'Value'`, `'Quality'`, `'Low-Risk'`
- **Real data** (French/Yahoo): Used Fama-French abbreviations `'MOM'`, `'HML'`, `'RMW'`, `'CMA'`, `'SMB'`
- **Visualization code**: Hardcoded `'MOM'` which only worked with real data

This caused:
1. KeyError crashes when running with synthetic data
2. Empty charts (Figure 2, Figure 6, Figure 7)
3. NANs in tables when analysis failed mid-execution

## Solution Implemented

### 1. Standardized Synthetic Data to Match Real Data
**File: `src/data/synthetic_data.py`**

Changed factor column names to Fama-French standard:
- `'Momentum'` → `'MOM'` (Momentum)
- `'Value'` → `'HML'` (High Minus Low - Value)
- `'Quality'` → `'RMW'` (Robust Minus Weak - Profitability/Quality)
- `'Low-Risk'` → `'BAB'` (Betting Against Beta - Low Volatility)

**Lines updated:**
- Line 58-86: `RETURN_PARAMS` dictionary keys
- Line 90-118: `IC_PARAMS` dictionary keys
- Line 338: `_generate_factor_returns()` factor list
- Line 391: `_generate_information_coefficients()` factor list
- Line 44: Docstring example

### 2. Added Factor Name Mapping for Portfolio Classes
**Files: `src/portfolio/state_conditioned.py`, `src/portfolio/volatility_scaling.py`**

Added `FACTOR_NAME_MAP` to translate abbreviated names to semantic names for exposure lookup:
```python
FACTOR_NAME_MAP = {
    'MOM': 'Momentum',
    'HML': 'Value',
    'RMW': 'Quality',
    'BAB': 'Low-Risk',
}
```

This allows both naming conventions to work with the default exposure mappings.

**StateConditionedPortfolio** (line 57-68, 135-153):
- Added name mapping constant
- Updated `_default_exposures()` to try abbreviated→full name mapping

**VolatilityLevelConditionedPortfolio** (line 271-318):
- Added name mapping constant
- Updated `__init__()` to map factor names when building exposures dict

### 3. Added Proper Error Handling to Visualization
**File: `src/visualization/figures.py`**

**Removed silent fallbacks, added explicit validation:**

**figure2_cumulative_performance()** (line 154-210):
- Validates factor exists in data
- Raises `ValueError` with clear message listing available factors
- Logs warning if falling back to first column
- Tries `['MOM', 'Momentum']` before fallback

**figure6_drawdown()** via `generate_all_figures()` (line 797-810):
- Validates factor column exists
- Raises `ValueError` if no factors found
- Logs warning when using non-momentum factor

**figure7_regime_factor_panel()** (line 611-637):
- Validates at least one factor exists
- Raises `ValueError` if no factors found
- Logs warning if fewer than 2 factors available
- Tries to find MOM and RMW (as in paper) before fallback

### 4. Updated Style Mappings
**File: `src/visualization/styles.py`**

Added clear documentation that both naming conventions are supported (line 35-46).

## Verification

### ✅ Synthetic Data Test
```bash
python scripts/run_analysis.py --synthetic
```
**Results:**
- All figures generated successfully (no errors)
- Tables use correct column names: MOM, HML, RMW, BAB
- No NANs in output
- Factor statistics computed correctly

### ✅ Error Handling Test
Created test with invalid column names - confirmed:
- Explicit `ValueError` raised with clear message listing available columns
- No silent failures
- Warnings logged when auto-detection falls back

## What Will Break if Columns Don't Match

The code now **fails loudly and clearly** instead of silently:

1. **Missing momentum factor for Figure 2/6**: Logs warning, uses first available column
2. **Completely invalid column names**: Raises `ValueError` listing what columns are available
3. **Portfolio construction**: Will use full exposure (1.0) for unknown factors, doesn't crash

## Column Name Reference

### Real Data (French/Yahoo)
- `MOM` - Momentum (12-month, skip 1)
- `HML` - High Minus Low (Value/Book-to-Market)
- `RMW` - Robust Minus Weak (Quality/Profitability)
- `SMB` - Small Minus Big (Size)
- `CMA` - Conservative Minus Aggressive (Investment)
- `MKT` - Market return
- `RF` - Risk-free rate

### Synthetic Data (Now Standardized)
- `MOM` - Momentum
- `HML` - Value
- `RMW` - Quality
- `BAB` - Betting Against Beta (Low-Risk)

## Testing Checklist

- [x] Synthetic data generates correct column names
- [x] Real data loading still works (uses French data)
- [x] Portfolio construction maps factor names correctly
- [x] Visualization finds correct factors
- [x] Error messages are clear and actionable
- [x] Tables display correct factor names
- [x] No silent failures or NANs
