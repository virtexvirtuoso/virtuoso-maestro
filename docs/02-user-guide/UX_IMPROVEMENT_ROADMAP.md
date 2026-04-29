# Maestro Frontend UX Improvement Roadmap

Cross-agent consensus analysis for prioritizing frontend improvements. Four specialized AI agents evaluated 10 potential improvements from different perspectives to eliminate single-viewpoint bias.

## Quick Reference: What To Do Next

| Priority | Task | Effort | Status |
|----------|------|--------|--------|
| **P0** | Fix wizard navigation bug | 1-2 days | Pending |
| **P1** | Standardize loading states | 1-2 days | Pending |
| **P2** | Export functionality (CSV/PDF) | 2-3 days | Blocked (needs backend) |

**Total estimated effort: 4-7 days**

---

## Completed Improvements (Sprint 1-3)

### Sprint 1: Results Page Enhancements
| Feature | Component | Status |
|---------|-----------|--------|
| Bulk select with checkboxes | `OptimizationResults.js` | ✅ Shipped |
| Floating action bar (N selected, Compare, Delete) | `OptimizationResults.js` | ✅ Shipped |
| Status filter dropdown (All/Completed/Failed/Running) | `OptimizationResults.js` | ✅ Shipped |
| Consistent date formatting | `OptimizationResults.js` | ✅ Shipped |

### Sprint 2: Wizard & Compare Improvements
| Feature | Component | Status |
|---------|-----------|--------|
| Data availability validation | `OptimizationWizard.js` | ✅ Shipped |
| Equity curve overlay chart | `StrategyComparison.js` | ✅ Shipped |
| Autocomplete dropdown filtering fix | `StrategyComparison.js` | ✅ Shipped |

### Sprint 3: Evaluation Page Enhancements
| Feature | Component | Status |
|---------|-----------|--------|
| Collapsible accordion sections | `Evaluation.js` | ✅ Shipped |
| Tab navigation (5 tabs) | `Evaluation.js` | ✅ Shipped |
| localStorage persistence for expand state | `Evaluation.js` | ✅ Shipped |
| Estimated run time in review step | `OptimizationWizard.js` | ✅ Shipped |

---

## Cross-Agent Consensus Matrix

Four specialized agents evaluated 10 potential improvements:

| Agent | Perspective | Focus Areas |
|-------|-------------|-------------|
| **Frontend Developer** | Component architecture, performance, UX polish |
| **Quant/Trader** | Trading workflow efficiency, professional requirements |
| **Tech Lead** | Technical debt, maintainability, architecture |
| **UX Researcher** | User experience, accessibility, task completion |

### Full Evaluation Results

| # | Item | Frontend | Quant | Tech Lead | UX | **Verdict** |
|---|------|:--------:|:-----:|:---------:|:--:|:-----------:|
| 4 | Wizard navigation bug | PRIORITY | PRIORITY | PRIORITY | PRIORITY | **DO NOW** |
| 3 | Loading states | PRIORITY | SKIP | PRIORITY | PRIORITY | **QUEUE** |
| 9 | Export functionality | DEFER | PRIORITY | DEFER | PRIORITY | **QUEUE** |
| 1 | Real-time progress | PRIORITY | SKIP | SKIP | SKIP | **DISCUSS** |
| 2 | Error handling | DEFER | SKIP | DEFER | DEFER | **BACKLOG** |
| 7 | Keyboard shortcuts | DEFER | DEFER | SKIP | DEFER | **BACKLOG** |
| 10 | Strategy templates | SKIP | DEFER | DEFER | SKIP | **BACKLOG** |
| 6 | Mobile responsiveness | DEFER | SKIP | DEFER | DEFER | **BACKLOG** |
| 5 | Empty states | SKIP | SKIP | SKIP | SKIP | **SKIP** |
| 8 | Dark/light toggle | SKIP | SKIP | SKIP | SKIP | **SKIP** |

### Decision Rules Applied

| Pattern | Action | Items |
|---------|--------|-------|
| All PRIORITY | **Do Now** | #4 Wizard bug |
| Majority PRIORITY | **Queue Next** | #3 Loading, #9 Export |
| Mixed votes | **Discuss** | #1 Real-time progress |
| Majority DEFER | **Backlog** | #2, #6, #7, #10 |
| All SKIP | **Skip** | #5, #8 |

---

## Priority Actions: Implementation Details

### P0: Fix Wizard Navigation Bug

**Problem:** Wizard closes unexpectedly after completing Step 2 (Strategy selection).

**Root Cause Analysis (Tech Lead):**
- `OptimizationWizard.js` is 958 lines — largest component in codebase
- `useEffect` with `initialPreset` dependency (lines 156-174) may trigger unexpected state resets
- No state machine or reducer pattern makes navigation logic hard to reason about

**Recommended Fix:**
```javascript
// Option 1: Refactor to useReducer
const [state, dispatch] = useReducer(wizardReducer, initialState);

// Option 2: Use form library
import { useForm, FormProvider } from 'react-hook-form';
// Separate form state from navigation state
```

**Files to Modify:**
- `frontend/maestro-ui/src/components/OptimizationWizard.js`

**Acceptance Criteria:**
- [ ] Wizard completes all 4 steps without unexpected closure
- [ ] Form data persists across step navigation
- [ ] Back button works correctly
- [ ] Submit triggers optimization and shows progress

---

### P1: Standardize Loading States

**Problem:** 50+ instances of `loading`/`isLoading` patterns with no shared abstraction.

**Current State (Tech Lead audit):**
```
DataManagement.js:    lines 44, 46, 73, 138
Home.js:              lines 138, 141, 203, 213, 217
OptimizationForm.js:  lines 56, 80, 89, 93
CandleStickChart.js:  lines 73, 85, 91, 144
```

**Recommended Fix:**
```javascript
// Option 1: Custom hook
function useAsyncData(fetchFn, deps) {
  const [state, setState] = useState({ loading: true, error: null, data: null });
  // ... implementation
  return state;
}

// Option 2: Leverage React Query (already installed but unused)
import { useQuery } from '@tanstack/react-query';

function useProviders() {
  return useQuery({
    queryKey: ['providers'],
    queryFn: () => fetch(`${API_URL}/datasource/available`).then(r => r.json()),
  });
}
```

**Files to Modify:**
- Create `frontend/maestro-ui/src/hooks/useAsyncData.js`
- Update components to use shared pattern

**Acceptance Criteria:**
- [ ] All async operations use consistent loading pattern
- [ ] Skeleton loaders appear during data fetches
- [ ] No layout shift when data loads
- [ ] Error states handled uniformly

---

### P2: Export Functionality

**Problem:** No way to export backtest results for external analysis.

**Quant Perspective (10/10 rating):**
> "This is a dealbreaker for professional use. Quants need to export results to CSV for further analysis in Python/R/Excel, generate PDF reports for trade journals, and document strategy performance."

**Implementation Sequence:**

1. **Backend First** (blocking):
   ```python
   # Add to backend/main/rest_api.py
   @app.route('/optimization/results/<tid>/export/<format>')
   def export_results(tid, format):
       # format: 'csv' | 'json' | 'pdf'
       results = get_optimization_results(tid)
       if format == 'csv':
           return Response(results_to_csv(results), mimetype='text/csv')
       # ...
   ```

2. **Frontend After Backend**:
   ```javascript
   // Add to Evaluation.js or OptimizationResults.js
   const handleExport = async (format) => {
     const response = await fetch(`${API_URL}/optimization/results/${tid}/export/${format}`);
     const blob = await response.blob();
     downloadBlob(blob, `${testName}.${format}`);
   };
   ```

**Acceptance Criteria:**
- [ ] Backend endpoint returns CSV with all metrics
- [ ] Export button in Evaluation page
- [ ] Export selected in Results page (bulk action)
- [ ] PDF report with charts (stretch goal)

---

## Backlog Items

### Real-time Progress Integration
**Current:** `useOptimizationProgress` hook exists with WebSocket + polling fallback.
**Gap:** Wizard closes after submission; users don't see progress inline.
**Recommendation:** Keep wizard open post-submission with progress visualization, OR add progress indicator to Recent Optimizations list.

### Error Handling Improvements
**Current:** `NotificationContext` + `ErrorBoundary` work.
**Gap:** No error recovery actions in toasts.
**Recommendation:** Add "Retry" action to error toasts where applicable.

### Keyboard Shortcuts
**Candidates:**
- `Ctrl+Enter` — Submit form / Start optimization
- `Ctrl+E` — Export results
- `Escape` — Close modal
- `Arrow keys` — Navigate results table

**Recommendation:** Implement after core UX is solid. Document shortcuts in help modal.

### Strategy Templates Expansion
**Current:** 4 presets in `WelcomeEmptyState.js` (RSI, MACD, Bollinger, EMA Cross).
**Recommendation:** Expand based on user feedback. Consider making templates API-driven.

### Mobile Responsiveness
**Current:** MUI Grid provides baseline responsiveness.
**Recommendation:** Low priority — quants use desktop with multiple monitors. Tablet view is sufficient.

---

## Skipped Items (No Action Needed)

### Empty States
**Reason:** Already well-implemented.
- `EmptyState.js` — Generic empty/no-data state with icon + message + action
- `WelcomeEmptyState.js` — First-time user onboarding with 4 preset templates

### Dark/Light Theme Toggle
**Reason:** Low ROI for trading application.
- Dark theme (340-line `theme.js`) is cohesive with neon amber/cyan palette
- Trading industry standard is dark mode
- Adding light mode doubles CSS maintenance burden
- All 4 agents rated SKIP

---

## Technical Context

### Codebase Metrics
| Metric | Value |
|--------|-------|
| Framework | React 18.2.0 + MUI v5.15 |
| Total Components | 33 |
| Total Lines | ~9,254 |
| TypeScript Coverage | ~30% (hooks + types) |
| Routes | 5 (Home, Results, Evaluation, Compare, Data) |

### Existing Infrastructure
| Feature | Implementation | Status |
|---------|----------------|--------|
| Real-time updates | `useOptimizationProgress.ts` (WebSocket + polling) | ✅ Working |
| Toast notifications | `NotificationContext.js` (MUI Snackbar) | ✅ Working |
| Error boundary | `ErrorBoundary.js` | ✅ Working |
| Dark theme | `theme.js` (340 lines) | ✅ Working |
| Loading skeletons | MUI `<Skeleton>` in Home.js | ✅ Partial |

### Technical Debt Identified
| Debt | Severity | Notes |
|------|----------|-------|
| 958-line OptimizationWizard.js | High | Needs refactor to useReducer or form library |
| Inconsistent loading patterns | Medium | 50+ instances, no shared hook |
| React Query unused | Low | Installed but not leveraged |
| Zustand unused | Low | Installed but not leveraged |
| No unit tests | High | Only `TradingChart.test.tsx` exists |

---

## Agent Insights: Key Disagreements

### Export Functionality (#9)
| Agent | Verdict | Reasoning |
|-------|---------|-----------|
| **Quant** | PRIORITY (10/10) | "Cannot share results, run external analysis, or document strategies" |
| **Frontend** | DEFER (2/10) | "Requires backend endpoints first" |
| **Tech Lead** | DEFER (6/10) | "Client-side export risks data integrity" |
| **UX** | PRIORITY (6/10) | "Workflow blocker for professional users" |

**Resolution:** High user value but blocked by backend. Sequence: Backend CSV endpoint → Frontend implementation.

### Real-time Progress (#1)
| Agent | Verdict | Reasoning |
|-------|---------|-----------|
| **Frontend** | PRIORITY (7/10) | "Wizard closes after submit, users see no progress" |
| **Others** | SKIP | "Already works, WebSocket + polling is solid" |

**Resolution:** Hook exists but wizard UX needs improvement. Lower priority than bug fix.

---

## Appendix: Agent Evaluation Criteria

### Frontend Developer
- Component architecture cleanliness
- Performance impact
- Code maintainability
- UX polish and consistency

### Quant/Trader
- Trading workflow efficiency
- Impact on analysis quality
- Time saved during backtesting
- Critical for professional trading

### Tech Lead
- Technical debt impact
- Code maintainability
- Architecture consistency
- Testing/reliability implications

### UX Researcher
- User friction and frustration
- Accessibility (WCAG compliance)
- Task completion efficiency
- User delight and professional feel

---

*Document generated: February 2026*
*Analysis method: Cross-Agent Consensus Matrix with 4 specialized AI agents*
