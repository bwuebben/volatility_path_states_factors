# Data Frequency, Timestamps, and Look-Ahead Bias - Complete Analysis

## Executive Summary

**CRITICAL FINDINGS**:
1. ✅ BAB (4th factor) removed - warnings added
2. ⚠️ **POTENTIAL LOOK-AHEAD BIAS DETECTED** in current implementation
3. French factor timestamps are MISLEADING (labeled month-start but represent month-end data)
4. Current alignment may use future information depending on research intent

---

## Raw Data Sources - Exact Frequencies and Timestamps

### 1. French Factor Library Data

**Source**: Ken French Data Library via `pandas_datareader`
**Factors**: Fama-French 3-Factor, 5-Factor, Momentum

**Frequency**: MONTHLY
**Original Format**: Period index with format `YYYYMM` (e.g., `196301` for January 1963)
**Converted To** (in our code): `DatetimeIndex` with first day of month (e.g., `1963-01-01`)

**CRITICAL - What the timestamp means**:
```
Index: 1963-01-01 (after conversion from period 196301)
Represents: Return EARNED during January 1-31, 1963
Available: ONLY AFTER January 31, 1963 market close
```

**The Misleading Part**:
- The `1963-01-01` timestamp suggests this data is available on January 1
- **THIS IS FALSE** - it's actually available on January 31
- This is a common convention in financial data but creates confusion

**Code Location**: `src/data/data_loader.py` lines 198-200
```python
ff3.index = pd.to_datetime(ff3.index.astype(str), format='%Y-%m')
```

### 2. Yahoo Finance Market Data

**Source**: Yahoo Finance via `yfinance`
**Frequency**: DAILY
**Original Format**: DatetimeIndex with timezone (e.g., `1963-01-03 00:00:00-05:00`)
**Day of Week**: Trading days only (Mon-Fri, excluding holidays)

**After Processing**:
- Timezone stripped (lines 295, 337 in `data_loader.py`)
- Used to compute monthly volatility via `.resample('ME').last()`
- Results in month-END timestamps (e.g., `1963-01-31`)

**What the timestamp means**:
```
Index: 1963-01-31
Represents: Volatility calculated using data from Jan 1-31
Available: After January 31 market close
Correct: Timestamp accurately reflects data availability
```

---

## Downsampling and Reindexing Operations

### Operation 1: Daily → Monthly Volatility

**File**: `src/regimes/path_states.py` line 225
```python
monthly_states = daily_states.resample('ME').last()
```

**What it does**:
- Takes LAST value of each month from daily volatility
- `'ME'` = Month End
- Results in timestamps: `1963-01-31`, `1963-02-28`, etc.

**Is this correct?** ✅ YES
- Uses data THROUGH month end
- Timestamp reflects data availability
- No look-ahead bias in THIS step

### Operation 2: Month-Period Alignment

**File**: `src/regimes/regime_classifier.py` lines 321-322
```python
factor_period_idx = factor_returns.index.to_period('M')  # 1963-01
states_period_idx = states.index.to_period('M')          # 1963-01
```

**What it does**:
- Converts `1963-01-01` → `1963-01` (period)
- Converts `1963-01-31` → `1963-01` (period)
- Aligns data from same calendar month

**The Problem**:
```
Before alignment:
- Factor: 1963-01-01 (misleading - actually available 1963-01-31)
- Regime: 1963-01-31 (correct)

After to_period('M'):
- Factor: 1963-01
- Regime: 1963-01

Both treated as "January 1963" - looks aligned, but...
```

### Operation 3: Regime-Conditioned Returns

**File**: `src/portfolio/state_conditioned.py` line 289
```python
effective[factor] = self.factor_returns[factor] * exposure_series
```

**What it does**:
- For each month t: `return_t * exposure(regime_t)`
- Uses January's regime to scale January's return

**Timeline of what happens**:
```
Month: January 1963

Current Implementation:
Jan 1:    Market opens, we trade
Jan 1-31: Earn return R_jan, volatility evolves
Jan 31:   Calculate vol_jan → classify regime_jan
Jan 31:   Apply exposure based on regime_jan to R_jan  ← PROBLEM!

We're using information from Jan 31 (regime) to adjust
the return we earned Jan 1-31. This is look-ahead bias
if interpreted as a trading strategy.
```

---

## Look-Ahead Bias Assessment

### Scenario A: This is **Conditional Statistics** (Descriptive Research)

**Research Question**: "How do factors perform IN different volatility regimes?"

**Current Approach**: ✅ ACCEPTABLE
- We're analyzing realized relationships
- Not claiming this is tradeable
- Just documenting that momentum crashes in high-vol regimes

**No fix needed** - this is a valid research design

### Scenario B: This is a **Trading Strategy** (Prescriptive)

**Research Question**: "Can we USE regime information to improve returns?"

**Current Approach**: ❌ LOOK-AHEAD BIAS
- We cannot know January's regime on January 1
- We only know it on January 31
- Using it to adjust January's return is peeking at future info

**Required Fix**: LAG the regime by 1 month
```python
# Use PREVIOUS month's regime for current month's exposure
exposure_series = self.regimes.shift(1).map(...)
effective[factor] = self.factor_returns[factor] * exposure_series
```

**Correct Timeline**:
```
Dec 1-31: Earn R_dec, calculate vol_dec → regime_dec
Jan 1:    Know regime_dec, set exposure for January
Jan 1-31: Trade with exposure_dec, earn R_jan
Jan 31:   Calculate regime_jan (for use in February)
```

---

## Expanding Window - Avoiding Look-Ahead in Thresholds

**File**: `src/regimes/regime_classifier.py`

**Question**: When classifying January 1963, what data do we use for thresholds?

**Check the code** (line 165 in `run_analysis.py`):
```python
regime_classifier = RegimeClassifier(
    expanding_window=config.regimes.expanding_window,  # Set to True
)
```

**Config** (`config.yaml` line 44):
```yaml
expanding_window: true
```

**Implementation** (check `regime_classifier.py`):
- With `expanding_window=True`: Uses data through time t-1 to set thresholds for time t
- ✅ This is CORRECT - no look-ahead bias in threshold estimation

---

## Summary of Data Flow

```
RAW DATA:
┌─────────────────────────────────────────────────────────────┐
│ Yahoo Daily:     1963-01-03, 1963-01-04, ..., 1963-01-31    │
│                  (timezone-aware, trading days only)          │
│ French Factors:  Period(1963-01) → converted to 1963-01-01  │
│                  (misleading timestamp!)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
PROCESSING:
┌─────────────────────────────────────────────────────────────┐
│ 1. Strip timezone from Yahoo data                            │
│ 2. Calculate daily volatility                                 │
│ 3. Resample to month-end: .resample('ME').last()            │
│    → Index: 1963-01-31 ✅ CORRECT                            │
│ 4. Classify regime from vol                                  │
│ 5. Convert to period for alignment                           │
│    French 1963-01-01 → Period(1963-01)                       │
│    Vol    1963-01-31 → Period(1963-01)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
USAGE:
┌─────────────────────────────────────────────────────────────┐
│ return[Jan] * exposure(regime[Jan])                          │
│                                                               │
│ ⚠️  Both are from January, both available Jan 31             │
│ ⚠️  If trading strategy: LOOK-AHEAD BIAS                     │
│ ✅  If conditional stats: OK                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### IMMEDIATE (Required):
1. ✅ **DONE**: Added warnings about BAB removal
2. ✅ **DONE**: Added warnings about potential look-ahead bias
3. ✅ **DONE**: Created this documentation

### SHORT-TERM (Before Publishing):
1. **Determine research intent**: Is this conditional statistics or a trading strategy?
2. **If trading strategy**: Add lag parameter and default to lagged implementation
3. **Document clearly**: Make explicit whether results include look-ahead bias

### LONG-TERM (Data Quality):
1. **Find BAB factor**:
   - AQR Capital publishes BAB factor data
   - Could construct from CRSP/Compustat individual stocks
   - Alternative: Use published low-vol factor from providers

2. **Standardize timestamps**:
   - Consider using month-end convention for ALL data
   - Or use Period index throughout (clearer semantics)

---

## Testing for Look-Ahead Bias

### Test 1: Check alignment dates
```python
print("Factor index:", factors.index[:5])
print("Regime index:", regimes.index[:5])
# Should see month-start vs month-end difference
```

### Test 2: Verify expanding window
```python
# Check that thresholds use only past data
# Add assertions in regime_classifier.py
```

### Test 3: Compare lagged vs unlagged
```python
# Run backtest with regime.shift(1)
# Performance should be LOWER if there's look-ahead bias
```

---

## Questions to Answer

1. **What does the original paper do?**
   - Check methodology section
   - Do they lag regimes or not?
   - Is it described as a trading strategy or conditional analysis?

2. **What's the performance impact?**
   - Run with `lag=0` (current) vs `lag=1` (correct for trading)
   - If performance drops significantly: look-ahead bias was helping

3. **What's the research goal?**
   - Understanding regime-return relationships? (lag not needed)
   - Implementable strategy? (lag required)
