# Maestro UX Redesign - Implementation Plan

## Phase 1: Quick Wins (1-2 days)

### 1.1 Add Loading States to Dropdowns

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationForm.js`

**Changes:**
```javascript
// Add loading state variables (around line 35)
const [loadingProviders, setLoadingProviders] = useState(true);
const [loadingSymbols, setLoadingSymbols] = useState(false);
const [loadingStrategies, setLoadingStrategies] = useState(true);
const [loadingParams, setLoadingParams] = useState(false);

// Wrap fetch calls with loading states
useEffect(() => {
  setLoadingStrategies(true);
  fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`)
    .then((response) => response.json())
    .then((data) => {
      setStrategies(data);
      if (data.length > 0) {
        setStrategy(data[0]);
        updateStrategyParams(data[0]);
      }
    })
    .catch((e) => notify(`Failed to load strategies: ${e}`, 'error'))
    .finally(() => setLoadingStrategies(false));
  // ... similar for providers
}, []);

// Add loading indicator to Select components
<FormControl sx={formControlSx}>
  <InputLabel id="provider-select-label">Provider</InputLabel>
  <Select
    labelId="provider-select-label"
    value={provider}
    label="Provider"
    onChange={handleProviderChange}
    disabled={loadingProviders}
    endAdornment={loadingProviders ? <CircularProgress size={20} /> : null}
  >
    {providers.map((p) => (
      <MenuItem key={p} value={p}>{p}</MenuItem>
    ))}
  </Select>
</FormControl>
```

### 1.2 Disable Symbol Dropdown Until Provider Selected

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationForm.js`

**Changes:**
```javascript
<FormControl sx={formControlSx} disabled={!provider || loadingSymbols}>
  <InputLabel id="symbol-select-label">Symbol</InputLabel>
  <Select
    labelId="symbol-select-label"
    value={symbol}
    label="Symbol"
    onChange={(e) => setSymbol(e.target.value)}
  >
    {symbols.length === 0 && !loadingSymbols && (
      <MenuItem disabled value="">
        {provider ? 'No symbols available' : 'Select provider first'}
      </MenuItem>
    )}
    {symbols.map((p) => (
      <MenuItem key={p} value={p}>{p}</MenuItem>
    ))}
  </Select>
  {!provider && (
    <FormHelperText>Select a data provider first</FormHelperText>
  )}
</FormControl>
```

### 1.3 Add Helper Text to Cash and Commissions

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationForm.js`

**Changes:**
```javascript
import InputAdornment from '@mui/material/InputAdornment';
import Tooltip from '@mui/material/Tooltip';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

// Cash field (around line 276)
<FormControl sx={formControlSx}>
  <TextField
    id="cash-text"
    label="Starting Cash"
    type="number"
    value={cash}
    onChange={(e) => setCash(Number(e.target.value))}
    InputProps={{
      startAdornment: <InputAdornment position="start">$</InputAdornment>,
    }}
    helperText="Initial portfolio value for simulation"
    inputProps={{ min: 100, step: 1000 }}
  />
</FormControl>

// Commissions field (around line 285)
<FormControl sx={formControlSx}>
  <TextField
    id="commissions-text"
    label="Commission Rate"
    type="number"
    value={commissions}
    onChange={(e) => setCommissions(Number(e.target.value))}
    InputProps={{
      endAdornment: (
        <InputAdornment position="end">
          <Tooltip title="Per-trade fee. Typical crypto: 0.05-0.20%">
            <HelpOutlineIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          </Tooltip>
        </InputAdornment>
      ),
    }}
    helperText="Trading fee per transaction (%)"
    inputProps={{ min: 0, max: 1, step: 0.01 }}
  />
</FormControl>
```

### 1.4 Replace Icon-Only Submit with Proper Button

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationForm.js`

**Changes:**
```javascript
import Button from '@mui/material/Button';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CircularProgress from '@mui/material/CircularProgress';

// Add form validation function
const isFormValid = useMemo(() => {
  return (
    testName.trim() !== '' &&
    provider !== '' &&
    symbol !== '' &&
    strategy !== '' &&
    cash > 0 &&
    startDate < endDate
  );
}, [testName, provider, symbol, strategy, cash, startDate, endDate]);

// Replace the submit button section (around line 336)
<Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 3, mb: 1 }}>
  <Button
    variant="contained"
    color="primary"
    onClick={submitTest}
    disabled={!isFormValid || isRunning}
    startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
    sx={{ minWidth: 180 }}
  >
    {isRunning ? 'Running...' : 'Run Optimization'}
  </Button>
</Box>
```

### 1.5 Fix Chart to Show Empty State

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/CandleStickChart.js`

**Changes:**
```javascript
import EmptyState from './EmptyState';
import ShowChartIcon from '@mui/icons-material/ShowChart';

export default function CandleStickChart({ provider, symbol, binSize }) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);
  const [ohlcvData, setOhlcvData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Default to bitmex/xbtusd/1d if no props provided (for dashboard view)
  const dataProvider = provider || 'bitmex';
  const dataSymbol = symbol || 'xbtusd';
  const dataBinSize = binSize || '1d';

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${dataProvider}/${dataSymbol}/${dataBinSize}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!data.data || data.data.length === 0) {
          throw new Error('No data available');
        }
        // ... existing data processing
      })
      .catch((err) => {
        console.error('Chart data fetch failed:', err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [dataProvider, dataSymbol, dataBinSize, theme.palette.secondary.light]);

  if (loading) {
    return (
      <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || ohlcvData.length === 0) {
    return (
      <EmptyState
        icon={ShowChartIcon}
        title="No Chart Data"
        description={error || "No OHLCV data available for the selected symbol."}
      />
    );
  }

  // ... rest of chart rendering
}
```

---

## Phase 2: Data-Aware Home Page (2-3 days)

### 2.1 Create DataSourceContext for Global State

**New File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/context/DataSourceContext.js`

```javascript
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const DataSourceContext = createContext(null);

export function DataSourceProvider({ children }) {
  const [providers, setProviders] = useState([]);
  const [symbolsByProvider, setSymbolsByProvider] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDataSources = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`);
      const providerList = await response.json();
      setProviders(providerList);

      // Fetch symbols for each provider
      const symbolsMap = {};
      for (const provider of providerList) {
        const symbolsRes = await fetch(
          `${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/symbols`
        );
        symbolsMap[provider] = await symbolsRes.json();
      }
      setSymbolsByProvider(symbolsMap);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDataSources();
  }, [fetchDataSources]);

  const hasData = providers.length > 0 &&
    Object.values(symbolsByProvider).some(symbols => symbols.length > 0);

  return (
    <DataSourceContext.Provider value={{
      providers,
      symbolsByProvider,
      loading,
      error,
      hasData,
      refresh: fetchDataSources,
    }}>
      {children}
    </DataSourceContext.Provider>
  );
}

export function useDataSources() {
  const context = useContext(DataSourceContext);
  if (!context) {
    throw new Error('useDataSources must be used within DataSourceProvider');
  }
  return context;
}
```

### 2.2 Update Home Component

**File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/Home.js`

```javascript
import React from 'react';
import { useNavigate } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Skeleton from '@mui/material/Skeleton';
import { useDataSources } from '../context/DataSourceContext';
import EmptyState from './EmptyState';
import StorageIcon from '@mui/icons-material/Storage';
import DataSourceCards from './DataSourceCards';
import RecentOptimizations from './RecentOptimizations';

export default function Home() {
  const navigate = useNavigate();
  const { hasData, loading } = useDataSources();

  if (loading) {
    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Skeleton variant="text" width={200} height={32} />
            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Skeleton variant="rectangular" width={300} height={200} />
              <Skeleton variant="rectangular" width={300} height={200} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    );
  }

  if (!hasData) {
    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 4 }}>
            <EmptyState
              icon={StorageIcon}
              title="Welcome to Maestro"
              description="Walk-Forward Optimization Engine for quantitative trading strategy research. To get started, download historical data from a supported provider."
              actionLabel="Go to Data Management"
              onAction={() => navigate('/data')}
            />
          </Paper>
        </Grid>
      </Grid>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <DataSourceCards />
      </Grid>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <RecentOptimizations limit={5} />
        </Paper>
      </Grid>
    </Grid>
  );
}
```

---

## Phase 3: Wizard-Style Form (1 week)

### 3.1 Create OptimizationWizard Component Structure

**New File:** `/Users/ffv_macmini/Desktop/maestro/frontend/maestro-ui/src/components/OptimizationWizard/index.js`

```javascript
import React, { useState, useCallback } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import WizardStepper from './WizardStepper';
import StepDataSource from './StepDataSource';
import StepStrategy from './StepStrategy';
import StepSettings from './StepSettings';
import StepReview from './StepReview';

const STEPS = ['Data Source', 'Strategy', 'Settings', 'Review'];

export default function OptimizationWizard({ open, onClose, onSubmit }) {
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    testName: '',
    provider: '',
    symbol: '',
    binSize: '1d',
    startDate: new Date(2020, 0, 1),
    endDate: new Date(),
    strategy: '',
    strategyParams: {},
    optType: 'WALKFORWARD',
    cash: 10000,
    commissions: 0.1,
    numSplits: 10,
    trainRatio: 0.8,
  });

  const updateFormData = useCallback((updates) => {
    setFormData(prev => ({ ...prev, ...updates }));
  }, []);

  const handleNext = () => setActiveStep(prev => Math.min(prev + 1, STEPS.length - 1));
  const handleBack = () => setActiveStep(prev => Math.max(prev - 1, 0));

  const handleSubmit = () => {
    onSubmit(formData);
    onClose();
  };

  const renderStep = () => {
    switch (activeStep) {
      case 0: return <StepDataSource formData={formData} updateFormData={updateFormData} />;
      case 1: return <StepStrategy formData={formData} updateFormData={updateFormData} />;
      case 2: return <StepSettings formData={formData} updateFormData={updateFormData} />;
      case 3: return <StepReview formData={formData} onEdit={setActiveStep} />;
      default: return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <WizardStepper steps={STEPS} activeStep={activeStep} />
      <DialogContent>
        {renderStep()}
      </DialogContent>
      <WizardActions
        activeStep={activeStep}
        totalSteps={STEPS.length}
        onBack={handleBack}
        onNext={handleNext}
        onSubmit={handleSubmit}
        onCancel={onClose}
      />
    </Dialog>
  );
}
```

### 3.2 Step Components

Each step component validates its own fields and exposes validation state:

```javascript
// StepDataSource.js
export default function StepDataSource({ formData, updateFormData }) {
  const { providers, symbolsByProvider, loading } = useDataSources();

  const isValid = formData.testName && formData.provider && formData.symbol;

  return (
    <Box>
      <TextField
        label="Test Name"
        value={formData.testName}
        onChange={(e) => updateFormData({ testName: e.target.value })}
        fullWidth
        required
      />

      <ProviderSelector
        providers={providers}
        selected={formData.provider}
        onSelect={(provider) => updateFormData({
          provider,
          symbol: '', // Reset symbol when provider changes
        })}
        loading={loading}
      />

      <SymbolSelector
        symbols={symbolsByProvider[formData.provider] || []}
        selected={formData.symbol}
        onSelect={(symbol) => updateFormData({ symbol })}
        disabled={!formData.provider}
      />

      <DateRangePicker
        startDate={formData.startDate}
        endDate={formData.endDate}
        onChange={(start, end) => updateFormData({ startDate: start, endDate: end })}
      />
    </Box>
  );
}
```

---

## Phase 4: Backend Enhancements (Optional, enables better UX)

### 4.1 Strategy Parameter Metadata Endpoint

**Endpoint:** `GET /strategy/{name}/params/metadata`

**Response:**
```json
{
  "fast_period": {
    "value": 12,
    "type": "integer",
    "min": 1,
    "max": 200,
    "description": "Fast EMA period for crossover signal",
    "group": "Moving Averages"
  },
  "slow_period": {
    "value": 26,
    "type": "integer",
    "min": 1,
    "max": 500,
    "description": "Slow EMA period for crossover signal",
    "group": "Moving Averages",
    "constraints": {
      "must_be_greater_than": "fast_period"
    }
  }
}
```

### 4.2 Data Availability Endpoint

**Endpoint:** `GET /datasource/{provider}/{symbol}/availability`

**Response:**
```json
{
  "provider": "binance",
  "symbol": "BTCUSDT",
  "timeframes": {
    "1d": {
      "first_date": "2017-08-17T00:00:00Z",
      "last_date": "2024-12-31T00:00:00Z",
      "candle_count": 2698,
      "has_gaps": false
    },
    "1h": {
      "first_date": "2019-01-01T00:00:00Z",
      "last_date": "2024-12-31T23:00:00Z",
      "candle_count": 52584,
      "has_gaps": true,
      "gap_count": 3
    }
  }
}
```

---

## Component File Structure

```
src/components/
├── optimization/
│   ├── OptimizationWizard/
│   │   ├── index.js
│   │   ├── WizardStepper.js
│   │   ├── WizardActions.js
│   │   ├── StepDataSource.js
│   │   ├── StepStrategy.js
│   │   ├── StepSettings.js
│   │   └── StepReview.js
│   ├── ProviderCard.js
│   ├── SymbolSelector.js
│   ├── StrategyCard.js
│   └── ParameterInput.js
├── dashboard/
│   ├── DataSourceCards.js
│   ├── RecentOptimizations.js
│   └── QuickStats.js
├── common/
│   ├── EmptyState.js (existing)
│   ├── LoadingSkeleton.js
│   └── DateRangePicker.js
└── charts/
    ├── CandleStickChart.js (updated)
    └── MiniChart.js (for cards)
```

---

## Testing Checklist

### Unit Tests
- [ ] DataSourceContext fetches and caches data correctly
- [ ] Wizard step validation functions work
- [ ] Form data persists across step navigation

### Integration Tests
- [ ] Empty state shows when no data sources
- [ ] Dropdowns populate correctly after API calls
- [ ] Symbol dropdown resets when provider changes
- [ ] Wizard completes and submits correctly

### E2E Tests
- [ ] New user flow: Empty state -> Data Management -> Download -> Home with data
- [ ] Complete optimization wizard flow
- [ ] Cancel wizard preserves no state
- [ ] Error states display correctly

---

## Migration Strategy

1. **Week 1:** Implement Phase 1 quick wins (non-breaking)
2. **Week 2:** Add DataSourceContext, update Home (non-breaking)
3. **Week 3-4:** Build OptimizationWizard, keep old form as fallback
4. **Week 5:** Switch to wizard as default, deprecate old form
5. **Week 6:** Remove old form code, cleanup
