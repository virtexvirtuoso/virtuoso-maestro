# Maestro UX Analysis - Detailed Findings

## Executive Summary

The current Maestro frontend suffers from a disconnected user experience that makes the platform feel like a developer prototype rather than a professional quantitative trading tool. The primary issues center around:

1. **Misleading visual states** - Chart shows data before user selects anything
2. **Form overload** - 10+ fields presented simultaneously without guidance
3. **Missing feedback loops** - No loading states, no data dependency indicators
4. **Lack of onboarding** - New users face a wall of empty dropdowns

---

## Issue 1: CandleStickChart Shows Hardcoded Data

### Current Behavior
**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/CandleStickChart.js`

```javascript
// Line 16 - Hardcoded fetch, no dependency on user selection
fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/bitmex/xbtusd/1d`)
```

**Problem:** The chart always displays XBTUSD daily data from BitMEX, regardless of:
- Whether the user has selected any provider
- Whether the user has selected any symbol
- Whether any data exists

**User Impact:**
- Creates false expectations ("I see a chart, data must exist")
- Misleading when user selects different provider/symbol
- No correlation between chart and form selections

### Recommended Fix
1. Remove chart from initial home page
2. Show chart only after user selects provider + symbol
3. Or: Show chart that updates based on form selections

```javascript
// Proposed: Accept props for dynamic data loading
function CandleStickChart({ provider, symbol, timeframe }) {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!provider || !symbol) {
      setLoading(false);
      return; // Show empty state
    }
    // Fetch based on actual selections
    fetch(`${API_URL}/datasource/${provider}/${symbol}/${timeframe}`)
    // ...
  }, [provider, symbol, timeframe]);

  if (!provider || !symbol) {
    return <EmptyChartState />;
  }
}
```

---

## Issue 2: Empty Dropdown States

### Current Behavior
**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationForm.js`

```javascript
// Lines 27-33 - All start empty, populate async
const [provider, setProvider] = useState('');
const [providers, setProviders] = useState([]);
const [symbol, setSymbol] = useState('');
const [symbols, setSymbols] = useState([]);
const [strategy, setStrategy] = useState('');
const [strategies, setStrategies] = useState([]);
```

**Problem:**
- Dropdowns render as empty on initial load
- No loading indicators while fetching options
- User sees empty `<Select>` elements that look broken

**User Impact:**
- Confusion about whether app is loading or broken
- No indication of async operations
- May try to interact before data loads

### Recommended Fix
Add loading states and skeleton UI:

```javascript
const [loadingProviders, setLoadingProviders] = useState(true);
const [loadingSymbols, setLoadingSymbols] = useState(false);
const [loadingStrategies, setLoadingStrategies] = useState(true);

// In render:
{loadingProviders ? (
  <Skeleton variant="rectangular" height={56} />
) : (
  <Select value={provider} onChange={handleProviderChange}>
    {providers.map(p => <MenuItem key={p} value={p}>{p}</MenuItem>)}
  </Select>
)}
```

---

## Issue 3: Data Dependency Chain Not Visible

### Current Behavior
```javascript
// Lines 66-75 - Symbol depends on Provider, but UI doesn't show this
const updateSymbolsAvailable = useCallback((providerName) => {
  if (!providerName) return;
  fetch(`${API_URL}/datasource/${providerName}/symbols`)
    .then((response) => response.json())
    .then((data) => {
      setSymbols(data);
      setSymbol(data.length > 0 ? data[0] : '');
    })
```

**Problem:**
- Symbol dropdown is enabled even when no provider selected
- User doesn't know symbols will change when provider changes
- No visual indication of dependency relationship

**User Impact:**
- May select symbol before provider (fails silently)
- Confusion when symbol list changes unexpectedly
- No progressive disclosure of related fields

### Recommended Fix
Disable dependent fields until parent selected:

```jsx
<FormControl disabled={!provider}>
  <InputLabel>Symbol</InputLabel>
  <Select
    value={symbol}
    disabled={!provider || loadingSymbols}
  >
    {loadingSymbols && <MenuItem disabled>Loading...</MenuItem>}
    {symbols.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
  </Select>
  {!provider && (
    <FormHelperText>Select a provider first</FormHelperText>
  )}
</FormControl>
```

---

## Issue 4: Form Overwhelm (Cognitive Load)

### Current Behavior
All fields presented in a flat layout:
- Test Name
- From/To dates
- Provider
- Symbol
- Time Frame
- Cash
- Commissions
- Strategy
- Dynamic Strategy Parameters (variable count)
- Optimization Type (3 radio buttons)

**Problem:**
- 10+ fields visible simultaneously
- No logical grouping
- No guidance on what values to use
- Expert-only interface

**User Impact:**
- High cognitive load for new users
- No sense of progress
- Easy to miss required fields
- Anxiety about "getting it right"

### Recommended Fix
Implement wizard-style progressive disclosure:

**Step 1: Data Source** (Test name, Provider, Symbol, Timeframe, Date range)
**Step 2: Strategy** (Strategy selection, Dynamic parameters)
**Step 3: Settings** (Optimization type, Cash, Commissions, WFO settings)
**Step 4: Review** (Summary of all selections, Run button)

Benefits:
- One concern per step
- Clear progress indicator
- Can validate per step
- Review before submit

---

## Issue 5: No Empty State for Home Page

### Current Behavior
**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/Home.js`

```javascript
// Home always shows chart + form, regardless of data availability
export default function Home() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper><CandleStickChart /></Paper>  // Always renders
      </Grid>
      <Grid item xs={12}>
        <Paper><OptimizationForm /></Paper>  // Always renders
      </Grid>
    </Grid>
  );
}
```

**Problem:**
- No check for whether data sources exist
- New users see empty dropdowns
- No guidance to Data Management page

**User Impact:**
- Lost on first visit
- May not discover Data page
- Frustrating onboarding

### Recommended Fix
Check data availability and show appropriate state:

```javascript
export default function Home() {
  const [hasData, setHasData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/datasource/available`)
      .then(res => res.json())
      .then(providers => {
        setHasData(providers.length > 0);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSkeleton />;

  if (!hasData) {
    return <WelcomeEmptyState
      title="Welcome to Maestro"
      action="Go to Data Management"
      onAction={() => navigate('/data')}
    />;
  }

  return <DashboardWithData />;
}
```

---

## Issue 6: No Guidance on Field Values

### Current Behavior
```javascript
// Lines 39-40 - Default values with no explanation
const [cash, setCash] = useState(10000);
const [commissions, setCommissions] = useState(0.01);
```

**Problem:**
- Magic numbers without context
- Users don't know if these are good defaults
- No validation or range hints
- Commission format unclear (0.01 = 1%? 0.01%?)

**User Impact:**
- Uncertainty about correct values
- May use unrealistic parameters
- Results may be meaningless

### Recommended Fix
Add tooltips, validation, and contextual help:

```jsx
<TextField
  label="Starting Cash"
  value={cash}
  type="number"
  InputProps={{
    startAdornment: <InputAdornment position="start">$</InputAdornment>,
  }}
  helperText="Initial portfolio value for simulation"
/>

<Tooltip title="Trading fee per transaction. 0.1% is typical for crypto exchanges.">
  <TextField
    label="Commission Rate"
    value={commissions}
    type="number"
    InputProps={{
      endAdornment: <InputAdornment position="end">%</InputAdornment>,
    }}
    inputProps={{ min: 0, max: 1, step: 0.01 }}
    helperText="Typical: 0.05-0.20%"
  />
</Tooltip>
```

---

## Issue 7: Strategy Parameters UX

### Current Behavior
```javascript
// Lines 311-321 - Dynamic parameters rendered inline
<FormControl sx={formControlSx}>
  {strategiesParameters.map((row) => (
    <FormControl key={row + '-form-control'} sx={formControlSx}>
      <TextField
        id={row + '-text'}
        label={row}
        type="number"
        value={paramValues[row] || ''}
        onChange={(e) => handleParamChange(row, Number(e.target.value))}
      />
    </FormControl>
  ))}
</FormControl>
```

**Problem:**
- Nested FormControls (unusual pattern)
- No parameter descriptions
- No default value hints
- No validation per strategy

**User Impact:**
- Don't know what parameters mean
- No idea what ranges are valid
- Must guess at values

### Recommended Fix
Fetch parameter metadata from backend:

```javascript
// Backend should return:
{
  "fast_period": {
    "value": 12,
    "min": 1,
    "max": 200,
    "description": "Fast EMA period for crossover signal"
  },
  "slow_period": {
    "value": 26,
    "min": 1,
    "max": 500,
    "description": "Slow EMA period for crossover signal",
    "must_be_greater_than": "fast_period"
  }
}
```

---

## Issue 8: Form Submission UX

### Current Behavior
```javascript
// Lines 336-342 - Simple icon button, no disabled state management
<Grid container direction="row" justifyContent="flex-end" alignItems="center">
  <Box>
    <IconButton aria-label="Run new test" onClick={submitTest} sx={{ m: 0.5 }}>
      <PlayCircleFilledWhiteIcon color="primary" fontSize="large" />
    </IconButton>
  </Box>
</Grid>
```

**Problem:**
- Just an icon (no text label)
- No disabled state during running
- No form validation before submit
- Easy to miss

**User Impact:**
- May not realize this is the submit button
- Can accidentally re-submit
- No feedback on form completeness

### Recommended Fix

```jsx
<Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 3 }}>
  <Button variant="outlined" onClick={handleCancel}>
    Cancel
  </Button>
  <Button
    variant="contained"
    color="primary"
    onClick={submitTest}
    disabled={!isFormValid || isRunning}
    startIcon={isRunning ? <CircularProgress size={20} /> : <PlayArrowIcon />}
  >
    {isRunning ? 'Running...' : 'Run Optimization'}
  </Button>
</Box>
```

---

## Summary: Priority Matrix

| Issue | Severity | Effort | Priority |
|-------|----------|--------|----------|
| Hardcoded chart data | High | Low | P1 |
| Empty dropdown states | High | Low | P1 |
| No empty state for Home | High | Medium | P1 |
| Data dependency chain | Medium | Medium | P2 |
| Form overwhelm (wizard) | High | High | P2 |
| Field value guidance | Medium | Low | P2 |
| Strategy params UX | Medium | Medium | P3 |
| Form submission UX | Low | Low | P3 |

---

## Quick Wins (Can implement in hours)

1. Add loading states to dropdowns
2. Disable symbol dropdown until provider selected
3. Add helper text to Cash and Commissions fields
4. Replace icon-only submit with proper button
5. Fix chart to show empty state when no data

## Larger Refactors (Days/weeks)

1. Wizard-style form (multi-step)
2. Home page data-aware empty state
3. Strategy parameter metadata from backend
4. Chart synchronized with form selections
