# Maestro Frontend UX Improvements

Prioritized UX enhancements for the walk-forward optimization platform.

| Status | Count |
|--------|-------|
| Completed | 6 |
| Planned | 6 |

## Tech Stack

- **Framework:** React 18
- **UI Library:** MUI v5
- **Charts:** TradingView Lightweight Charts, Highcharts
- **State:** React Query + Context

---

## Completed: Quick Wins

All quick wins shipped in commit `03ae680`.

### Results Page Improvements

| Change | Before | After |
|--------|--------|-------|
| Action tooltips | Plain icons, unclear purpose | "View Details" and "Delete" tooltips on hover |
| Date format | Browser-dependent `toLocaleString()` | Consistent `Jan 15, 2026 14:30` format |
| Test types display | Nested TreeView (unstable row heights) | Compact chips with tooltip for parameter details |

### Evaluation Page Improvements

| Change | Before | After |
|--------|--------|-------|
| Summary metrics | None — key metrics buried in tables | 6-card row: Sharpe, Annual Return, Max Drawdown, Volatility, Calmar, Sortino |
| Section navigation | 14+ sections in single scroll | 5 tabs: Overview, Charts, Metrics, Walk-Forward, Analysis |
| Metric colors | Plain text | Color-coded: green (good), red (bad), amber (neutral) |

### Already Integrated

- **OptimizationProgress component** — Real-time fold visualization with WebSocket updates
- **Cancel button** — Stop running optimizations
- **Connection indicator** — Live vs Polling status

---

## Planned: Medium Effort Improvements

### Sprint 1: Results Table Enhancement (2-3 days)

#### Bulk Select and Actions

**Page:** Results
**Impact:** High — eliminates repetitive clicking for batch operations

| Current | Proposed |
|---------|----------|
| Delete one at a time | Checkbox column + "Delete Selected" button |
| Compare via URL params | "Compare Selected" button (max 5) |
| No multi-select | Shift+click range selection |

**Implementation:**
```
- Add checkbox column with "Select All" header
- Track selectedTids in state
- Show floating action bar when selection > 0
- Batch DELETE endpoint: DELETE /optimization/results?tids=a,b,c
```

#### Status Filter Dropdown

**Page:** Results
**Impact:** Medium — quickly find failed runs for debugging

| Current | Proposed |
|---------|----------|
| Strategy filter only | Strategy + Status filters side-by-side |
| Can't filter by completed/failed | Dropdown: All, Completed, Failed, Running |

**Implementation:**
```
- Add status filter Autocomplete next to strategy filter
- Filter options derived from unique statuses in results
- Combine with existing strategy filter logic
```

---

### Sprint 2: Comparison and Validation (2-3 days)

#### Equity Curve Overlay Chart

**Page:** Compare
**Impact:** High — visual comparison is essential for traders

| Current | Proposed |
|---------|----------|
| Metrics table only | Equity curves overlaid on single chart |
| No temporal comparison | See divergence points between strategies |

**Implementation:**
```
- Fetch equity curve data from each strategy's backtest results
- Normalize to percentage returns (start at 0%)
- Use TradingView Lightweight Charts with multiple series
- Color-code lines to match strategy chips
- Add hover crosshair showing all values at timestamp
```

#### Data Availability Validation

**Page:** Optimization Wizard
**Impact:** High — prevents frustrating "no data" failures

| Current | Proposed |
|---------|----------|
| Submit and fail silently | Pre-flight check before submission |
| No date range feedback | Show data coverage indicator |

**Implementation:**
```
- On date range change, call: GET /datasource/{provider}/{symbol}/range
- Response: { start: "2020-01-01", end: "2024-12-31", gaps: [...] }
- Show warning chip if selected range exceeds available data
- Disable submit if no data overlap
```

---

### Sprint 3: Polish (1-2 days)

#### Collapsible Sections

**Page:** Evaluation
**Impact:** Medium — reduce visual noise within tabs

| Current | Proposed |
|---------|----------|
| All sections expanded | Accordion-style collapse/expand |
| Fixed section heights | Collapsed shows title + summary stat |

**Implementation:**
```
- Wrap each Paper section in MUI Accordion
- Default: first section expanded, others collapsed
- Persist expansion state in localStorage
- Show mini-stat in collapsed header (e.g., "Sharpe: 1.42")
```

#### Estimated Run Time

**Page:** Optimization Wizard (Review step)
**Impact:** Low — nice-to-have for expectation setting

| Current | Proposed |
|---------|----------|
| No time estimate | "Estimated time: ~5 minutes" |
| Unknown duration | Based on: data points × splits × strategy complexity |

**Implementation:**
```
- Calculate: (end_date - start_date) / timeframe_minutes = data_points
- Estimate: data_points × splits × 0.01 seconds (calibrate from historical runs)
- Display in Review step with disclaimer: "Actual time may vary"
```

---

## Priority Matrix

| Improvement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Bulk select/actions | High | Medium | 1 |
| Status filter | Medium | Low | 2 |
| Equity curve overlay | High | Medium | 3 |
| Data validation | High | Medium | 4 |
| Collapsible sections | Medium | Low | 5 |
| Run time estimate | Low | Low | 6 |

---

## Files to Modify

| Sprint | Files |
|--------|-------|
| Sprint 1 | `OptimizationResults.js` |
| Sprint 2 | `StrategyComparison.js`, `OptimizationWizard.js` |
| Sprint 3 | `Evaluation.js`, `OptimizationWizard.js` |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to delete 5 results | < 5 seconds (currently ~15s) |
| Time to find failed runs | < 3 seconds (currently ~10s) |
| Strategy comparison clarity | Visual + tabular (currently tabular only) |
| Failed optimization rate from bad dates | 0% (currently unknown) |
