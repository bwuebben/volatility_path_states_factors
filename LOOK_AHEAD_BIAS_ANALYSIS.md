# CRITICAL: Look-Ahead Bias Analysis

## Data Timestamps and Meaning

### French Factor Data
- **Index Format**: `1963-01-01`, `1963-02-01`, etc. (first day of month)
- **What it represents**: Return EARNED DURING that month (Jan 1-31)
- **When available**: At END of month (known only after Jan 31)
- **Critical Issue**: The timestamp `1963-01-01` is MISLEADING - this return is only known on `1963-01-31`

### Volatility Data (from `compute_monthly_states`)
- **Computed via**: `.resample('ME').last()` on daily volatility
- **Index Format**: `1963-01-31`, `1963-02-28`, etc. (month-end)
- **What it represents**: Volatility calculated using ALL data through month-end
- **When available**: At END of month (Jan 31)
- **Correct**: Timestamp accurately reflects when information is available

## Current Alignment Logic

In `regime_classifier.py` lines 319-351:
```python
# Convert both to month period
factor_period_idx = factor_returns.index.to_period('M')  # 1963-01
states_period_idx = states.index.to_period('M')          # 1963-01
```

This aligns:
- January factor return (labeled `1963-01-01` but actually available `1963-01-31`)
- January regime (from volatility through `1963-01-31`)

## The Look-Ahead Bias Problem

In `state_conditioned.py` line 289:
```python
effective[factor] = self.factor_returns[factor] * exposure_series
```

This multiplies:
- **January's return** (earned Jan 1-31)
- **January's exposure** (based on regime classified using volatility through Jan 31)

### The Issue
If this is meant to be a **trading strategy**:
- On Jan 1, we need to set our exposure for the month
- But we're using volatility/regime calculated using data through Jan 31
- This is **using future information** - we're peeking at the full month to decide exposure for that month

### Correct Approach for Trading
```python
# Use PREVIOUS month's regime to set THIS month's exposure
effective[factor] = self.factor_returns[factor] * exposure_series.shift(1)
```

This would:
- Use January's regime (known Jan 31) to set February's exposure
- Set February exposure on Feb 1 using information available through Jan 31
- Earn February's return (Feb 1-28) with that exposure

## Current vs Correct Timeline

### CURRENT (Look-Ahead Bias):
```
Jan 1-31: Market moves, earn return R_Jan
Jan 31:   Calculate vol_Jan, classify regime_Jan
Jan 31:   Apply exposure based on regime_Jan to R_Jan  ❌ WRONG
```

### CORRECT (No Look-Ahead):
```
Jan 1-31: Market moves, earn return R_Jan
Jan 31:   Calculate vol_Jan, classify regime_Jan
Feb 1:    Set exposure for Feb based on regime_Jan
Feb 1-28: Earn return R_Feb with exposure_Jan  ✓ CORRECT
```

## Is This Actually a Problem?

It depends on the **research question**:

### 1. If studying "regime-conditional statistics" (descriptive)
- Question: "What was the average return when volatility was high?"
- Current approach: **FINE** - we're conditioning on realized regimes
- No need to lag - we're analyzing relationships, not trading

### 2. If implementing a trading strategy (prescriptive)
- Question: "Can we use regime information to improve returns?"
- Current approach: **WRONG** - look-ahead bias inflates performance
- Must lag regime by 1 month

## Recommendation

**URGENT**: Determine the paper's intent:

1. **Check the paper methodology**: Does it describe this as:
   - Conditional statistics? (OK as-is)
   - A trading strategy? (MUST fix with lag)

2. **Add explicit lag option**:
```python
def backtest(self, lag_regime: bool = True, ...):
    if lag_regime:
        # Shift regimes by 1 month to avoid look-ahead bias
        regimes_to_use = self.regimes.shift(1)
    else:
        regimes_to_use = self.regimes
```

3. **Document clearly**:
   - Current behavior: In-sample conditioning (look-ahead)
   - With lag: Out-of-sample forecasting (no look-ahead)

## Additional Concerns

### Expanding Window in Regime Classification
In `regime_classifier.py`, check if threshold calculation uses expanding window:
- If thresholds are calculated using ALL data: look-ahead bias
- If using expanding window through time t-1 only: no bias

### Factor Construction
Check if factor portfolios are:
- Constructed using information available at month-start
- Or using information from the full month

## Action Items

1. [ ] Add warning to run_analysis.py about BAB removal
2. [ ] Document exact timestamps and what they represent
3. [ ] Add `lag_regime` parameter to portfolio classes
4. [ ] Default to `lag_regime=True` for trading strategies
5. [ ] Add tests to verify no future information is used
6. [ ] Check expanding window logic in regime classification
