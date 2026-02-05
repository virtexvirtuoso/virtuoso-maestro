# Maestro Frontend Implementation Plan

*Comprehensive roadmap for modernizing the Maestro UI*

**Created:** 2026-02-05
**Status:** Planning
**Stack:** React 18, MUI v5, React Query, TradingView Lightweight Charts

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Implementation Phases](#implementation-phases)
4. [Phase 0: Foundation Fixes](#phase-0-foundation-fixes)
5. [Phase 1: Real-Time Progress](#phase-1-real-time-progress)
6. [Phase 2: Strategy Comparison](#phase-2-strategy-comparison)
7. [Phase 3: Walk-Forward Visualization](#phase-3-walk-forward-visualization)
8. [Phase 4: Optuna Integration](#phase-4-optuna-integration)
9. [Phase 5: Analytics Enhancement](#phase-5-analytics-enhancement)
10. [Phase 6: Data Management](#phase-6-data-management)
11. [Breaking Changes & Mitigations](#breaking-changes--mitigations)
12. [Testing Strategy](#testing-strategy)
13. [Rollback Procedures](#rollback-procedures)

---

## Executive Summary

The Maestro frontend requires significant enhancements to support the modernized backend (FastAPI, QuestDB, VectorBT, Optuna). This document outlines a phased implementation approach that minimizes risk while delivering incremental value.

### Key Objectives

1. **Wire up existing unused code** (WebSocket hook, TradingChart)
2. **Add strategy comparison** for side-by-side analysis
3. **Visualize walk-forward optimization** with timeline and fold details
4. **Integrate Optuna dashboard** for hyperparameter tuning
5. **Improve UX** with better progress feedback, error states, and empty states

### Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Backend API changes break frontend | High | High | Version API endpoints, feature flags |
| Highcharts → TradingView migration | Medium | Medium | Gradual migration, keep both temporarily |
| WebSocket connection failures | Medium | Low | Polling fallback already implemented |
| Bundle size increase | Low | Low | Code splitting, lazy loading |

---

## Current State Analysis

### Existing Components

```
src/
├── components/
│   ├── Dashboard.js          ✅ Themed (Virtuoso dark)
│   ├── Home.js               ✅ Working
│   ├── OptimizationForm.js   ⚠️  Uses HTTP polling, not WebSocket
│   ├── OptimizationResults.js ✅ Working
│   ├── Evaluation.js         ⚠️  Single test only, no comparison
│   ├── CandleStickChart.js   ⚠️  Uses Highcharts
│   ├── StrategyChart.js      ⚠️  Uses Highcharts
│   ├── TradingChart.tsx      🔴 EXISTS BUT UNUSED
│   ├── PnLChart.js           ⚠️  Basic implementation
│   ├── HeatMap.js            ✅ Working
│   ├── ParametersDistributionPlot.js ✅ Working
│   ├── WalkForwardMetrics.js ✅ Working
│   └── LinearProgressWithLabel.js ✅ Basic
├── hooks/
│   ├── useOptimizationProgress.ts 🔴 EXISTS BUT UNUSED
│   └── index.ts
└── theme.js                  ✅ Virtuoso dark theme
```

### Unused Assets (Already Built)

1. **`useOptimizationProgress.ts`** - Full WebSocket hook with polling fallback
2. **`TradingChart.tsx`** - TradingView Lightweight Charts implementation
3. **React Query** - Configured but underutilized

### API Endpoints (Current Flask)

| Endpoint | Method | Used By |
|----------|--------|---------|
| `/strategy/available` | GET | OptimizationForm |
| `/strategy/<name>/params` | GET | OptimizationForm |
| `/optimization/new/` | POST | OptimizationForm |
| `/optimization/progress/<tid>/` | GET | OptimizationForm (polling) |
| `/optimization/results` | GET | OptimizationResults |
| `/optimization/results/<tid>` | GET/DELETE | Evaluation |
| `/optimization/results/<tid>/correlation` | GET | HeatMap |
| `/optimization/results/<tid>/report` | GET | Evaluation (export) |
| `/datasource/available` | GET | OptimizationForm |
| `/datasource/<provider>/symbols` | GET | OptimizationForm |
| `/datasource/<provider>/<symbol>/<timeframe>/<start>/<end>` | GET | StrategyChart |

### API Endpoints (Future FastAPI v2)

| Endpoint | Method | New Feature |
|----------|--------|-------------|
| `/api/v2/optimization/<tid>/ws` | WebSocket | Real-time progress |
| `/api/v2/optimization/<tid>/progress` | GET | Polling fallback |
| `/api/v2/optuna/studies` | GET | Optuna integration |
| `/api/v2/optuna/studies/<id>/trials` | GET | Trial history |
| `/api/v2/compare` | POST | Strategy comparison |
| `/api/v2/analytics/quantstats/<tid>` | GET | QuantStats metrics |

---

## Implementation Phases

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
Foundation  Progress   Compare    WF Visual   Optuna     Analytics   Data Mgmt
   │           │          │          │          │           │           │
   └── 2 days ─┴── 3 days ┴── 4 days ┴── 3 days ┴── 5 days ─┴── 4 days ─┴── 3 days
```

**Total Estimated Effort:** 24 days

---

## Phase 0: Foundation Fixes

**Goal:** Fix existing issues and prepare codebase for new features

### 0.1 Wire Up WebSocket Progress Hook

**File:** `src/components/OptimizationForm.js`

**Current Implementation (Lines 95-135):**
```javascript
// Current: HTTP polling every 800ms
timerRef.current = setInterval(() => {
  fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/progress/${data['tid']}`)
    .then(...)
}, 800);
```

**New Implementation:**
```javascript
import { useOptimizationProgress } from '../hooks/useOptimizationProgress';

// Inside component:
const { progress, isConnected, isComplete, error, status } = useOptimizationProgress(
  runningTid,
  isRunning
);

useEffect(() => {
  if (isComplete) {
    setIsRunning(false);
    // Show success toast
  }
}, [isComplete]);
```

**What Could Break:**
- WebSocket URL mismatch with backend
- CORS issues on WebSocket upgrade
- Missing `REACT_APP_WS_HOST` env variable

**Fix:**
```bash
# Add to .env
REACT_APP_WS_HOST=localhost:8000
REACT_APP_REST_API_V2_URL=http://localhost:8000
```

**Fallback:** Hook already has polling fallback built-in.

---

### 0.2 Replace Highcharts with TradingView

**Files to Modify:**
- `src/components/CandleStickChart.js`
- `src/components/StrategyChart.js`

**Current (Highcharts):**
```javascript
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
```

**New (TradingView):**
```javascript
import TradingChart, { convertOHLCVData, convertTradeMarkers } from './TradingChart';
```

**Migration Strategy:**
1. Keep both implementations temporarily
2. Add feature flag: `REACT_APP_USE_TRADINGVIEW=true`
3. Gradually migrate each chart
4. Remove Highcharts after validation

**What Could Break:**
- Data format differences (Highcharts uses arrays, TradingView uses objects)
- Missing indicator support in TradingView component
- Time zone handling differences

**Fix:**
```typescript
// Use existing converter functions in TradingChart.tsx
const ohlcvData = convertOHLCVData(apiResponse.data);
const markers = convertTradeMarkers(data.buy, data.sell);
```

---

### 0.3 Add Error Boundaries and Empty States

**New File:** `src/components/ErrorBoundary.js`
```javascript
import React from 'react';
import { Alert, Button, Box } from '@mui/material';

export class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box p={3}>
          <Alert
            severity="error"
            action={
              <Button onClick={() => window.location.reload()}>
                Reload
              </Button>
            }
          >
            Something went wrong: {this.state.error?.message}
          </Alert>
        </Box>
      );
    }
    return this.props.children;
  }
}
```

**New File:** `src/components/EmptyState.js`
```javascript
import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';

export default function EmptyState({
  icon: Icon = ScienceIcon,
  title = 'No Data Yet',
  description = 'Run your first optimization to see results here.',
  actionLabel = 'Run Test',
  onAction
}) {
  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      py={6}
      sx={{ opacity: 0.7 }}
    >
      <Icon sx={{ fontSize: 64, mb: 2, color: 'primary.main' }} />
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        {description}
      </Typography>
      {onAction && (
        <Button variant="outlined" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </Box>
  );
}
```

---

### 0.4 Add Toast Notifications

**Install:** Already have MUI, use Snackbar

**New File:** `src/context/NotificationContext.js`
```javascript
import React, { createContext, useContext, useState, useCallback } from 'react';
import { Snackbar, Alert } from '@mui/material';

const NotificationContext = createContext();

export function NotificationProvider({ children }) {
  const [notification, setNotification] = useState(null);

  const notify = useCallback((message, severity = 'info') => {
    setNotification({ message, severity });
  }, []);

  const handleClose = () => setNotification(null);

  return (
    <NotificationContext.Provider value={{ notify }}>
      {children}
      <Snackbar
        open={!!notification}
        autoHideDuration={6000}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        {notification && (
          <Alert severity={notification.severity} onClose={handleClose}>
            {notification.message}
          </Alert>
        )}
      </Snackbar>
    </NotificationContext.Provider>
  );
}

export const useNotification = () => useContext(NotificationContext);
```

---

## Phase 1: Real-Time Progress

**Goal:** Rich progress feedback during optimization runs

### 1.1 Enhanced Progress Component

**New File:** `src/components/OptimizationProgress.js`
```javascript
import React from 'react';
import {
  Box, Card, CardContent, Typography, LinearProgress,
  Chip, IconButton, Collapse
} from '@mui/material';
import {
  PlayArrow, Pause, Stop, ExpandMore, CheckCircle, Error
} from '@mui/icons-material';
import { useOptimizationProgress } from '../hooks/useOptimizationProgress';

export default function OptimizationProgress({ tid, onComplete, onCancel }) {
  const { progress, isConnected, isComplete, error, status } = useOptimizationProgress(tid);
  const [expanded, setExpanded] = React.useState(true);

  const statusColors = {
    pending: 'default',
    running: 'primary',
    completed: 'success',
    failed: 'error',
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={status}
              color={statusColors[status]}
              size="small"
            />
            <Chip
              label={isConnected ? 'Live' : 'Polling'}
              color={isConnected ? 'success' : 'warning'}
              size="small"
              variant="outlined"
            />
          </Box>
          <Box>
            {status === 'running' && (
              <>
                <IconButton size="small" onClick={onCancel}>
                  <Stop />
                </IconButton>
              </>
            )}
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
              sx={{ transform: expanded ? 'rotate(180deg)' : 'none' }}
            >
              <ExpandMore />
            </IconButton>
          </Box>
        </Box>

        <Box mt={2}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2">
              {progress.message || `Processing...`}
            </Typography>
            <Typography variant="body2" fontFamily="monospace">
              {progress.percent.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={progress.percent}
            sx={{
              height: 8,
              borderRadius: 4,
              '& .MuiLinearProgress-bar': {
                transition: 'transform 0.3s ease',
              }
            }}
          />
        </Box>

        <Collapse in={expanded}>
          <Box mt={2} p={2} bgcolor="background.paper" borderRadius={1}>
            <Typography variant="caption" component="div" gutterBottom>
              Fold Progress
            </Typography>
            <Box display="flex" gap={0.5} flexWrap="wrap">
              {Array.from({ length: progress.total || 10 }, (_, i) => (
                <Box
                  key={i}
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: i < progress.current
                      ? 'success.main'
                      : i === progress.current
                        ? 'primary.main'
                        : 'grey.800',
                    fontSize: 10,
                    fontFamily: 'monospace',
                  }}
                >
                  {i + 1}
                </Box>
              ))}
            </Box>
          </Box>
        </Collapse>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
```

### 1.2 Integration with OptimizationForm

**Modify:** `src/components/OptimizationForm.js`

```javascript
// Add state for tracking running optimization
const [runningTid, setRunningTid] = useState(null);

// In submitTest success handler:
setRunningTid(data['tid']);
setIsRunning(true);

// Replace LinearProgressWithLabel with:
{isRunning && runningTid && (
  <OptimizationProgress
    tid={runningTid}
    onComplete={(result) => {
      setIsRunning(false);
      setRunningTid(null);
      notify('Optimization completed!', 'success');
    }}
    onCancel={async () => {
      await fetch(`${API_URL}/optimization/${runningTid}/cancel`, { method: 'POST' });
      setIsRunning(false);
      setRunningTid(null);
    }}
  />
)}
```

**What Could Break:**
- Cancel endpoint doesn't exist in Flask backend
- WebSocket connection refused on different port

**Fix:**
1. Add cancel endpoint to backend (or disable button until FastAPI migration)
2. Ensure CORS allows WebSocket upgrade
3. Use `REACT_APP_WS_HOST` for WebSocket URL

---

## Phase 2: Strategy Comparison

**Goal:** Compare multiple strategies side-by-side

### 2.1 New Route and Component

**Add Route:** `src/components/Dashboard.js`
```javascript
import StrategyComparison from './StrategyComparison';

// In Routes:
<Route path="/compare" element={<StrategyComparison />} />
```

**New File:** `src/components/StrategyComparison.js`
```javascript
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Grid, Paper, Typography, Chip, Box, Table, TableBody,
  TableCell, TableHead, TableRow, TableContainer, IconButton,
  Autocomplete, TextField, Button
} from '@mui/material';
import { Add, Remove, TrendingUp, TrendingDown } from '@mui/icons-material';
import Title from './Title';
import TradingChart from './TradingChart';

const COMPARISON_METRICS = [
  { key: 'sharpe', label: 'Sharpe Ratio', format: 'decimal', higherBetter: true },
  { key: 'annual_return', label: 'Annual Return', format: 'percent', higherBetter: true },
  { key: 'max_drawdown', label: 'Max Drawdown', format: 'percent', higherBetter: false },
  { key: 'win_rate', label: 'Win Rate', format: 'percent', higherBetter: true },
  { key: 'profit_factor', label: 'Profit Factor', format: 'decimal', higherBetter: true },
  { key: 'total_trades', label: 'Total Trades', format: 'number', higherBetter: null },
  { key: 'calmar', label: 'Calmar Ratio', format: 'decimal', higherBetter: true },
  { key: 'sortino', label: 'Sortino Ratio', format: 'decimal', higherBetter: true },
];

export default function StrategyComparison() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [availableTests, setAvailableTests] = useState([]);
  const [selectedTests, setSelectedTests] = useState([]);
  const [comparisonData, setComparisonData] = useState({});
  const [loading, setLoading] = useState(false);

  // Parse initial tids from URL
  useEffect(() => {
    const tids = searchParams.get('tids')?.split(',').filter(Boolean) || [];
    if (tids.length > 0) {
      loadTests(tids);
    }
  }, []);

  // Load available tests
  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/available`)
      .then(res => res.json())
      .then(setAvailableTests);
  }, []);

  const loadTests = async (tids) => {
    setLoading(true);
    const results = {};

    for (const tid of tids) {
      try {
        const res = await fetch(
          `${process.env.REACT_APP_REST_API_URL}/optimization/results/${tid}`
        );
        results[tid] = await res.json();
      } catch (e) {
        console.error(`Failed to load test ${tid}:`, e);
      }
    }

    setComparisonData(results);
    setSelectedTests(tids);
    setSearchParams({ tids: tids.join(',') });
    setLoading(false);
  };

  const addTest = (test) => {
    if (test && !selectedTests.includes(test.tid)) {
      const newTids = [...selectedTests, test.tid];
      loadTests(newTids);
    }
  };

  const removeTest = (tid) => {
    const newTids = selectedTests.filter(t => t !== tid);
    loadTests(newTids);
  };

  const extractMetric = (data, metricKey) => {
    // Extract from PyFolio analyzers
    const bt = data?.optimizations?.BACKTESTING?.[0];
    const wf = data?.optimizations?.WALKFORWARD;

    const metricMap = {
      sharpe: bt?.analyzers?.PyFolio?.['Sharpe ratio'],
      annual_return: bt?.analyzers?.PyFolio?.['Annual return'],
      max_drawdown: bt?.analyzers?.PyFolio?.['Max drawdown'],
      // Add more mappings
    };

    return metricMap[metricKey] ?? null;
  };

  const formatValue = (value, format) => {
    if (value === null || value === undefined) return 'N/A';
    switch (format) {
      case 'percent': return `${(value * 100).toFixed(2)}%`;
      case 'decimal': return value.toFixed(3);
      case 'number': return value.toLocaleString();
      default: return value;
    }
  };

  const getBestValue = (metricKey, higherBetter) => {
    const values = selectedTests
      .map(tid => extractMetric(comparisonData[tid], metricKey))
      .filter(v => v !== null);

    if (values.length === 0) return null;
    return higherBetter ? Math.max(...values) : Math.min(...values);
  };

  return (
    <Grid container spacing={3}>
      {/* Test Selector */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Title>Compare Strategies</Title>
          <Box display="flex" gap={2} alignItems="center" flexWrap="wrap">
            <Autocomplete
              options={availableTests.filter(t => !selectedTests.includes(t.tid))}
              getOptionLabel={(opt) => `${opt.test_name} (${opt.strategy})`}
              onChange={(_, value) => addTest(value)}
              renderInput={(params) => (
                <TextField {...params} label="Add Strategy" size="small" />
              )}
              sx={{ minWidth: 300 }}
            />
            {selectedTests.map((tid, idx) => (
              <Chip
                key={tid}
                label={comparisonData[tid]?.test_name || tid}
                onDelete={() => removeTest(tid)}
                color={['primary', 'secondary', 'success', 'warning'][idx % 4]}
              />
            ))}
          </Box>
        </Paper>
      </Grid>

      {/* Comparison Table */}
      {selectedTests.length > 0 && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Title>Performance Comparison</Title>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    {selectedTests.map((tid, idx) => (
                      <TableCell key={tid} align="right">
                        <Chip
                          label={comparisonData[tid]?.test_name || 'Loading...'}
                          size="small"
                          color={['primary', 'secondary', 'success', 'warning'][idx % 4]}
                        />
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {COMPARISON_METRICS.map(({ key, label, format, higherBetter }) => {
                    const bestValue = getBestValue(key, higherBetter);

                    return (
                      <TableRow key={key} hover>
                        <TableCell>
                          <Box display="flex" alignItems="center" gap={1}>
                            {label}
                            {higherBetter !== null && (
                              higherBetter
                                ? <TrendingUp fontSize="small" color="success" />
                                : <TrendingDown fontSize="small" color="error" />
                            )}
                          </Box>
                        </TableCell>
                        {selectedTests.map(tid => {
                          const value = extractMetric(comparisonData[tid], key);
                          const isBest = value === bestValue && bestValue !== null;

                          return (
                            <TableCell
                              key={tid}
                              align="right"
                              sx={{
                                fontWeight: isBest ? 700 : 400,
                                color: isBest ? 'primary.main' : 'inherit',
                                fontFamily: 'monospace',
                              }}
                            >
                              {formatValue(value, format)}
                              {isBest && ' ★'}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      )}

      {/* Equity Curve Overlay */}
      {selectedTests.length > 1 && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Title>Equity Curves</Title>
            {/* TODO: Overlay equity curves from all selected strategies */}
            <Typography color="text.secondary">
              Equity curve overlay coming in Phase 3
            </Typography>
          </Paper>
        </Grid>
      )}
    </Grid>
  );
}
```

### 2.2 Add Navigation Link

**Modify:** `src/components/ListItems.js`
```javascript
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

// Add to list items:
<ListItemButton component={Link} to="/compare">
  <ListItemIcon>
    <CompareArrowsIcon />
  </ListItemIcon>
  <ListItemText primary="Compare" />
</ListItemButton>
```

**What Could Break:**
- PyFolio metric keys may differ from expected
- Large number of comparisons may cause performance issues

**Fix:**
1. Add metric key mapping layer
2. Limit to max 5 strategies
3. Use React Query for caching

---

## Phase 3: Walk-Forward Visualization

**Goal:** Visual timeline of WFO splits with parameter drift analysis

### 3.1 Timeline Component

**New File:** `src/components/WalkForwardTimeline.js`
```javascript
import React from 'react';
import { Box, Typography, Tooltip } from '@mui/material';

export default function WalkForwardTimeline({ splits, totalPeriod }) {
  if (!splits || splits.length === 0) return null;

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Walk-Forward Timeline
      </Typography>
      <Box
        display="flex"
        height={60}
        border={1}
        borderColor="divider"
        borderRadius={1}
        overflow="hidden"
      >
        {splits.map((split, idx) => {
          const trainWidth = (split.train_end - split.train_start) / totalPeriod * 100;
          const testWidth = (split.test_end - split.test_start) / totalPeriod * 100;

          return (
            <React.Fragment key={idx}>
              <Tooltip title={`Train ${idx + 1}: ${new Date(split.train_start).toLocaleDateString()}`}>
                <Box
                  sx={{
                    width: `${trainWidth}%`,
                    bgcolor: 'primary.dark',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRight: 1,
                    borderColor: 'divider',
                  }}
                >
                  <Typography variant="caption" color="white">
                    T{idx + 1}
                  </Typography>
                </Box>
              </Tooltip>
              <Tooltip title={`Test ${idx + 1}: ${new Date(split.test_start).toLocaleDateString()}`}>
                <Box
                  sx={{
                    width: `${testWidth}%`,
                    bgcolor: 'success.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRight: 1,
                    borderColor: 'divider',
                  }}
                >
                  <Typography variant="caption" color="white">
                    V{idx + 1}
                  </Typography>
                </Box>
              </Tooltip>
            </React.Fragment>
          );
        })}
      </Box>
      <Box display="flex" justifyContent="space-between" mt={0.5}>
        <Typography variant="caption" color="text.secondary">
          {new Date(splits[0]?.train_start).toLocaleDateString()}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {new Date(splits[splits.length - 1]?.test_end).toLocaleDateString()}
        </Typography>
      </Box>
    </Box>
  );
}
```

### 3.2 Parameter Stability Chart

**New File:** `src/components/ParameterStabilityChart.js`
```javascript
import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine
} from 'recharts';

export default function ParameterStabilityChart({ walkforwardResults, parameters }) {
  const chartData = useMemo(() => {
    if (!walkforwardResults || !parameters) return [];

    return walkforwardResults
      .sort((a, b) => a.num_split - b.num_split)
      .map((result, idx) => ({
        fold: `Fold ${idx + 1}`,
        ...Object.fromEntries(
          parameters.map(p => [p, result.parameters[p]])
        ),
      }));
  }, [walkforwardResults, parameters]);

  const colors = ['#fbbf24', '#06B6D4', '#10b981', '#ef4444', '#9333ea'];

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Parameter Stability Across Folds
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="fold" stroke="#9ca3af" />
          <YAxis stroke="#9ca3af" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#111',
              border: '1px solid #333'
            }}
          />
          <Legend />
          {parameters.map((param, idx) => (
            <Line
              key={param}
              type="monotone"
              dataKey={param}
              stroke={colors[idx % colors.length]}
              strokeWidth={2}
              dot={{ fill: colors[idx % colors.length] }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}
```

**What Could Break:**
- Split timestamps may be in different format
- Missing `num_split` field in old data

**Fix:**
1. Normalize timestamps at API response level
2. Add fallback sorting by array index

---

## Phase 4: Optuna Integration

**Goal:** Embed or replicate Optuna dashboard functionality

### 4.1 Optuna Dashboard Embed (Quick Win)

**New File:** `src/components/OptunaDashboard.js`
```javascript
import React from 'react';
import { Box, Typography, Alert } from '@mui/material';

export default function OptunaDashboard({ studyId }) {
  const dashboardUrl = process.env.REACT_APP_OPTUNA_DASHBOARD_URL;

  if (!dashboardUrl) {
    return (
      <Alert severity="info">
        Optuna Dashboard not configured. Set REACT_APP_OPTUNA_DASHBOARD_URL.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Hyperparameter Optimization
      </Typography>
      <Box
        component="iframe"
        src={`${dashboardUrl}/studies/${studyId}`}
        sx={{
          width: '100%',
          height: 600,
          border: 'none',
          borderRadius: 1,
          bgcolor: 'background.paper',
        }}
        title="Optuna Dashboard"
      />
    </Box>
  );
}
```

### 4.2 Native Optuna Visualizations (Better UX)

**New File:** `src/components/OptunaVisualization.js`
```javascript
import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Paper, Typography, Table, TableBody,
  TableCell, TableHead, TableRow, Chip
} from '@mui/material';
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell
} from 'recharts';

export default function OptunaVisualization({ studyId }) {
  const [study, setStudy] = useState(null);
  const [trials, setTrials] = useState([]);

  useEffect(() => {
    // Fetch from FastAPI Optuna endpoints
    const apiUrl = process.env.REACT_APP_REST_API_V2_URL;

    Promise.all([
      fetch(`${apiUrl}/api/v2/optuna/studies/${studyId}`).then(r => r.json()),
      fetch(`${apiUrl}/api/v2/optuna/studies/${studyId}/trials`).then(r => r.json()),
    ]).then(([studyData, trialsData]) => {
      setStudy(studyData);
      setTrials(trialsData);
    });
  }, [studyId]);

  if (!study) return <Typography>Loading...</Typography>;

  const bestTrial = trials.reduce((best, trial) =>
    trial.value > (best?.value ?? -Infinity) ? trial : best
  , null);

  return (
    <Grid container spacing={3}>
      {/* Study Info */}
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Study Info</Typography>
          <Box component="dl">
            <Typography variant="body2" color="text.secondary">
              Study Name
            </Typography>
            <Typography variant="body1" gutterBottom>
              {study.study_name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Trials
            </Typography>
            <Typography variant="body1" gutterBottom>
              {trials.length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Best Value
            </Typography>
            <Typography variant="body1" color="primary.main" fontWeight={700}>
              {bestTrial?.value?.toFixed(4)}
            </Typography>
          </Box>
        </Paper>
      </Grid>

      {/* Optimization History */}
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Optimization History</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="number"
                name="Trial"
                stroke="#9ca3af"
              />
              <YAxis
                dataKey="value"
                name="Objective"
                stroke="#9ca3af"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#111',
                  border: '1px solid #333'
                }}
              />
              <Scatter data={trials} fill="#fbbf24">
                {trials.map((trial, idx) => (
                  <Cell
                    key={idx}
                    fill={trial.number === bestTrial?.number ? '#10b981' : '#fbbf24'}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Best Parameters */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Best Parameters</Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell align="right">Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {bestTrial && Object.entries(bestTrial.params).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell>{key}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      </Grid>
    </Grid>
  );
}
```

**What Could Break:**
- Optuna dashboard CORS issues in iframe
- Missing Optuna API endpoints in backend

**Fix:**
1. Configure Optuna dashboard for CORS
2. Ensure FastAPI backend has Optuna endpoints
3. Use native visualizations as fallback

---

## Phase 5: Analytics Enhancement

**Goal:** Integrate QuantStats and add advanced analytics

### 5.1 QuantStats Integration

**New File:** `src/components/QuantStatsReport.js`
```javascript
import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, Skeleton } from '@mui/material';

export default function QuantStatsReport({ tid }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const apiUrl = process.env.REACT_APP_REST_API_V2_URL;
    fetch(`${apiUrl}/api/v2/analytics/quantstats/${tid}`)
      .then(res => res.json())
      .then(data => {
        setMetrics(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [tid]);

  if (loading) {
    return <Skeleton variant="rectangular" height={400} />;
  }

  if (!metrics) {
    return <Typography color="text.secondary">QuantStats not available</Typography>;
  }

  const sections = [
    {
      title: 'Returns',
      metrics: [
        { label: 'Total Return', value: metrics.total_return, format: 'percent' },
        { label: 'CAGR', value: metrics.cagr, format: 'percent' },
        { label: 'Best Day', value: metrics.best_day, format: 'percent' },
        { label: 'Worst Day', value: metrics.worst_day, format: 'percent' },
      ],
    },
    {
      title: 'Risk',
      metrics: [
        { label: 'Volatility', value: metrics.volatility, format: 'percent' },
        { label: 'Max Drawdown', value: metrics.max_drawdown, format: 'percent' },
        { label: 'VaR 95%', value: metrics.var_95, format: 'percent' },
        { label: 'CVaR 95%', value: metrics.cvar_95, format: 'percent' },
      ],
    },
    {
      title: 'Ratios',
      metrics: [
        { label: 'Sharpe', value: metrics.sharpe, format: 'decimal' },
        { label: 'Sortino', value: metrics.sortino, format: 'decimal' },
        { label: 'Calmar', value: metrics.calmar, format: 'decimal' },
        { label: 'Omega', value: metrics.omega, format: 'decimal' },
      ],
    },
  ];

  return (
    <Grid container spacing={2}>
      {sections.map(section => (
        <Grid item xs={12} md={4} key={section.title}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="primary.main" gutterBottom>
              {section.title}
            </Typography>
            {section.metrics.map(({ label, value, format }) => (
              <Box
                key={label}
                display="flex"
                justifyContent="space-between"
                py={0.5}
              >
                <Typography variant="body2" color="text.secondary">
                  {label}
                </Typography>
                <Typography variant="body2" fontFamily="monospace">
                  {format === 'percent'
                    ? `${(value * 100).toFixed(2)}%`
                    : value?.toFixed(3) ?? 'N/A'}
                </Typography>
              </Box>
            ))}
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
}
```

### 5.2 Monthly Returns Heatmap

**New File:** `src/components/MonthlyReturnsHeatmap.js`
```javascript
import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export default function MonthlyReturnsHeatmap({ returns }) {
  const heatmapData = useMemo(() => {
    if (!returns) return {};

    // Group returns by year and month
    const grouped = {};
    returns.forEach(({ date, value }) => {
      const d = new Date(date);
      const year = d.getFullYear();
      const month = d.getMonth();

      if (!grouped[year]) grouped[year] = {};
      grouped[year][month] = (grouped[year][month] || 0) + value;
    });

    return grouped;
  }, [returns]);

  const years = Object.keys(heatmapData).sort();

  const getColor = (value) => {
    if (value === undefined) return '#1a1a1a';
    if (value >= 0.1) return '#166534';
    if (value >= 0.05) return '#22c55e';
    if (value >= 0) return '#4ade80';
    if (value >= -0.05) return '#fca5a5';
    if (value >= -0.1) return '#ef4444';
    return '#991b1b';
  };

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Monthly Returns
      </Typography>
      <Box display="flex" gap={0.5}>
        <Box width={60} />
        {MONTHS.map(m => (
          <Box
            key={m}
            width={40}
            textAlign="center"
            fontSize={10}
            color="text.secondary"
          >
            {m}
          </Box>
        ))}
      </Box>
      {years.map(year => (
        <Box key={year} display="flex" gap={0.5} mt={0.5}>
          <Box width={60} fontSize={12} color="text.secondary">
            {year}
          </Box>
          {Array.from({ length: 12 }, (_, month) => {
            const value = heatmapData[year]?.[month];
            return (
              <Box
                key={month}
                width={40}
                height={30}
                bgcolor={getColor(value)}
                borderRadius={0.5}
                display="flex"
                alignItems="center"
                justifyContent="center"
                fontSize={9}
                fontFamily="monospace"
                color={value !== undefined ? 'white' : 'transparent'}
                title={value !== undefined ? `${(value * 100).toFixed(1)}%` : 'No data'}
              >
                {value !== undefined && `${(value * 100).toFixed(0)}%`}
              </Box>
            );
          })}
        </Box>
      ))}
    </Box>
  );
}
```

**What Could Break:**
- QuantStats endpoint not implemented
- Return data format mismatch

**Fix:**
1. Implement QuantStats endpoint in FastAPI
2. Add data transformation layer

---

## Phase 6: Data Management

**Goal:** UI for managing OHLCV data downloads

### 6.1 Data Management Page

**New Route:** `/data`

**New File:** `src/components/DataManagement.js`
```javascript
import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, Typography, Table, TableBody, TableCell,
  TableHead, TableRow, IconButton, Button, Dialog,
  DialogTitle, DialogContent, DialogActions, TextField,
  FormControl, InputLabel, Select, MenuItem, Chip, Box,
  LinearProgress
} from '@mui/material';
import { Download, Refresh, CheckCircle, Warning, Add } from '@mui/icons-material';
import Title from './Title';

export default function DataManagement() {
  const [dataSources, setDataSources] = useState([]);
  const [downloading, setDownloading] = useState({});
  const [addDialog, setAddDialog] = useState(false);
  const [newSymbol, setNewSymbol] = useState({ provider: '', symbol: '', timeframe: '1d' });

  useEffect(() => {
    fetchDataSources();
  }, []);

  const fetchDataSources = async () => {
    // Fetch available data and their status
    const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/status`);
    const data = await response.json();
    setDataSources(data);
  };

  const triggerDownload = async (provider, symbol, timeframe) => {
    const key = `${provider}-${symbol}-${timeframe}`;
    setDownloading(prev => ({ ...prev, [key]: true }));

    try {
      await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, symbol, timeframe }),
      });
      // Poll for completion or use WebSocket
      setTimeout(fetchDataSources, 5000);
    } finally {
      setDownloading(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Title>Data Sources</Title>
            <Box>
              <Button
                startIcon={<Refresh />}
                onClick={fetchDataSources}
                sx={{ mr: 1 }}
              >
                Refresh
              </Button>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setAddDialog(true)}
              >
                Add Symbol
              </Button>
            </Box>
          </Box>

          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Provider</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Timeframe</TableCell>
                <TableCell>Records</TableCell>
                <TableCell>Last Update</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataSources.map((ds) => {
                const key = `${ds.provider}-${ds.symbol}-${ds.timeframe}`;
                return (
                  <TableRow key={key} hover>
                    <TableCell>
                      <Chip label={ds.provider} size="small" />
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {ds.symbol.toUpperCase()}
                    </TableCell>
                    <TableCell>{ds.timeframe}</TableCell>
                    <TableCell>{ds.record_count?.toLocaleString() ?? 'N/A'}</TableCell>
                    <TableCell>
                      {ds.last_update
                        ? new Date(ds.last_update).toLocaleString()
                        : 'Never'}
                    </TableCell>
                    <TableCell>
                      {ds.has_gaps ? (
                        <Chip
                          icon={<Warning />}
                          label="Gaps"
                          color="warning"
                          size="small"
                        />
                      ) : (
                        <Chip
                          icon={<CheckCircle />}
                          label="OK"
                          color="success"
                          size="small"
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      <IconButton
                        onClick={() => triggerDownload(ds.provider, ds.symbol, ds.timeframe)}
                        disabled={downloading[key]}
                      >
                        {downloading[key] ? (
                          <LinearProgress sx={{ width: 24 }} />
                        ) : (
                          <Download />
                        )}
                      </IconButton>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </Paper>
      </Grid>

      {/* Add Symbol Dialog */}
      <Dialog open={addDialog} onClose={() => setAddDialog(false)}>
        <DialogTitle>Add New Symbol</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Provider</InputLabel>
            <Select
              value={newSymbol.provider}
              label="Provider"
              onChange={(e) => setNewSymbol(prev => ({ ...prev, provider: e.target.value }))}
            >
              <MenuItem value="binance">Binance</MenuItem>
              <MenuItem value="bitmex">BitMEX</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Symbol"
            value={newSymbol.symbol}
            onChange={(e) => setNewSymbol(prev => ({ ...prev, symbol: e.target.value }))}
            sx={{ mt: 2 }}
            placeholder="e.g., btcusdt"
          />
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={newSymbol.timeframe}
              label="Timeframe"
              onChange={(e) => setNewSymbol(prev => ({ ...prev, timeframe: e.target.value }))}
            >
              <MenuItem value="1m">1 Minute</MenuItem>
              <MenuItem value="5m">5 Minutes</MenuItem>
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="1d">1 Day</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => {
            triggerDownload(newSymbol.provider, newSymbol.symbol, newSymbol.timeframe);
            setAddDialog(false);
          }}>
            Download
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}
```

**What Could Break:**
- `/datasource/status` endpoint doesn't exist
- `/datasource/download` endpoint doesn't exist

**Fix:**
1. Add these endpoints to Flask backend (or wait for FastAPI)
2. Use existing endpoints with reduced functionality

---

## Breaking Changes & Mitigations

### API Contract Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| WebSocket URL format | Progress won't connect | Feature flag to disable WebSocket |
| Optuna endpoints missing | Dashboard shows errors | Graceful fallback to "not configured" |
| QuantStats endpoints missing | Analytics incomplete | Hide section if API returns 404 |
| Cancel endpoint missing | Cancel button non-functional | Disable button until implemented |

### Environment Variables Required

```bash
# .env.local
REACT_APP_REST_API_URL=http://localhost:5000       # Flask (existing)
REACT_APP_REST_API_V2_URL=http://localhost:8000    # FastAPI (new)
REACT_APP_WS_HOST=localhost:8000                   # WebSocket host
REACT_APP_OPTUNA_DASHBOARD_URL=http://localhost:8080  # Optional
REACT_APP_USE_TRADINGVIEW=true                     # Feature flag
```

### Migration Script

```bash
#!/bin/bash
# migrate-frontend.sh

# Backup current state
cp -r src src.backup

# Install new dependencies
npm install recharts lightweight-charts

# Update environment
cp .env.example .env.local

# Run type check
npx tsc --noEmit

# Run tests
npm test -- --watchAll=false

# Build
npm run build

echo "Migration complete. Review changes before deploying."
```

---

## Testing Strategy

### Unit Tests

```javascript
// src/components/__tests__/OptimizationProgress.test.js
import { render, screen, waitFor } from '@testing-library/react';
import OptimizationProgress from '../OptimizationProgress';

// Mock the hook
jest.mock('../../hooks/useOptimizationProgress', () => ({
  useOptimizationProgress: () => ({
    progress: { current: 5, total: 10, percent: 50, message: 'Processing...' },
    isConnected: true,
    isComplete: false,
    error: null,
    status: 'running',
  }),
}));

test('renders progress correctly', () => {
  render(<OptimizationProgress tid="test-123" />);
  expect(screen.getByText('50.0%')).toBeInTheDocument();
  expect(screen.getByText('Processing...')).toBeInTheDocument();
});
```

### Integration Tests

```javascript
// cypress/e2e/optimization.cy.js
describe('Optimization Flow', () => {
  it('runs optimization and shows progress', () => {
    cy.visit('/');
    cy.get('[data-testid="strategy-select"]').click();
    cy.contains('ema_cross').click();
    cy.get('[aria-label="Run new test"]').click();

    // Wait for progress
    cy.get('[data-testid="progress-bar"]', { timeout: 10000 })
      .should('be.visible');

    // Wait for completion
    cy.contains('completed', { timeout: 60000 });
  });
});
```

### Visual Regression

```bash
# Use Percy or Chromatic
npm install --save-dev @percy/cli @percy/cypress

# Run visual tests
npx percy exec -- cypress run
```

---

## Rollback Procedures

### Quick Rollback (< 5 minutes)

```bash
# Revert to last known good commit
git checkout HEAD~1 -- src/
npm run build

# Or restore from backup
rm -rf src
mv src.backup src
npm run build
```

### Feature Flag Rollback

```javascript
// src/config/featureFlags.js
export const FEATURES = {
  USE_WEBSOCKET: process.env.REACT_APP_USE_WEBSOCKET !== 'false',
  USE_TRADINGVIEW: process.env.REACT_APP_USE_TRADINGVIEW === 'true',
  SHOW_OPTUNA: process.env.REACT_APP_SHOW_OPTUNA === 'true',
  SHOW_QUANTSTATS: process.env.REACT_APP_SHOW_QUANTSTATS === 'true',
};

// Usage
import { FEATURES } from '../config/featureFlags';

{FEATURES.USE_WEBSOCKET && <OptimizationProgress tid={tid} />}
{!FEATURES.USE_WEBSOCKET && <LegacyProgress tid={tid} />}
```

### Database Rollback

N/A - Frontend is stateless, no database changes.

---

## Appendix: File Change Summary

### New Files

```
src/
├── components/
│   ├── EmptyState.js
│   ├── ErrorBoundary.js
│   ├── OptimizationProgress.js
│   ├── StrategyComparison.js
│   ├── WalkForwardTimeline.js
│   ├── ParameterStabilityChart.js
│   ├── OptunaDashboard.js
│   ├── OptunaVisualization.js
│   ├── QuantStatsReport.js
│   ├── MonthlyReturnsHeatmap.js
│   └── DataManagement.js
├── context/
│   └── NotificationContext.js
└── config/
    └── featureFlags.js
```

### Modified Files

```
src/
├── index.js                  # Add NotificationProvider
├── components/
│   ├── Dashboard.js          # Add new routes
│   ├── ListItems.js          # Add navigation links
│   ├── OptimizationForm.js   # Use WebSocket hook
│   ├── Evaluation.js         # Add QuantStats, WF timeline
│   ├── CandleStickChart.js   # Migrate to TradingView
│   └── StrategyChart.js      # Migrate to TradingView
```

### Deleted Files (After Migration)

```
# Remove after TradingView migration complete
src/components/CandleStickChart.js  # Replaced by TradingChart
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-05 | 0.1.0 | Initial implementation plan |

---

*Document authored by Claude Code using Ultrathink methodology*
