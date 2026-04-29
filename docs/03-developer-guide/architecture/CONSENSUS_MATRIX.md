# Maestro Modernization Plan - Cross-Agent Consensus Matrix

**Generated**: 2026-02-04
**Agents**: Plan Critic, Backend Architect, Quant Engineer, DevOps, Frontend Developer

---

## Executive Summary

5 specialized agents evaluated the Maestro modernization plan from different perspectives. The plan is **fundamentally sound** but requires **revisions before execution**.

| Overall Verdict | Status |
|-----------------|--------|
| Plan Critic | **NEEDS REVISION** |
| Consensus | **APPROVE WITH CONDITIONS** |

---

## Cross-Agent Consensus Matrix

| Phase | Backend | Quant | DevOps | Frontend | Consensus | Action |
|-------|---------|-------|--------|----------|-----------|--------|
| **Phase 1: Core Engine** | 8 APPROVE | 6 NEEDS_WORK | 7 APPROVE | 9 APPROVE | **7.5** | QUEUE - address quant concerns |
| **Phase 2: Data Layer** | 6 NEEDS_WORK | 8 APPROVE | 5 NEEDS_WORK | 8 APPROVE | **6.75** | REVISE - migration gaps |
| **Phase 3: API + Frontend** | 5 RISKY | 9 APPROVE | 6 NEEDS_WORK | 5 NEEDS_WORK | **6.25** | REVISE - major gaps |
| **Phase 4: Analytics** | 9 APPROVE | 7 APPROVE | 8 APPROVE | 7 APPROVE | **7.75** | APPROVE |

### Risk Heat Map

| Phase | Backend | Quant | DevOps | Frontend |
|-------|---------|-------|--------|----------|
| Phase 1 | LOW | **HIGH** | LOW | LOW |
| Phase 2 | MEDIUM | MEDIUM | **HIGH** | LOW |
| Phase 3 | **HIGH** | LOW | MEDIUM | **HIGH** |
| Phase 4 | LOW | MEDIUM | LOW | MEDIUM |

---

## Agreement Analysis

### Universal Agreement (All Approve)

**Phase 4: Analytics**
- Clean, additive enhancement
- QuantStats replaces dead PyFolio
- No breaking changes
- Only concern: annualization assumptions for crypto (365 vs 252 days)

### Strong Agreement (3+ Approve)

**Phase 1: Core Engine**
- Existing `engine_v2/` provides solid foundation
- VectorBT + Optuna architecture is correct
- Strategy abstraction pattern preserved

**Blocking Issue (Quant)**: Parity testing underspecified; some strategies CANNOT vectorize.

### Disagreement Zones

**Phase 2: Data Layer**
| Agent | Rating | Position |
|-------|--------|----------|
| Quant | 8 | Standard migration, test float precision |
| Frontend | 8 | Low impact if API maintains backward compat |
| Backend | 6 | Missing rollback strategy, schema mapping |
| DevOps | 5 | No checksums, no validation gate |

**Insight**: Data engineers (Backend/DevOps) see migration risk; consumers (Quant/Frontend) assume it "just works."

**Resolution**: Add Phase 2.0 pre-work for migration validation framework.

**Phase 3: API + Frontend**
| Agent | Rating | Position |
|-------|--------|----------|
| Quant | 9 | Pure infrastructure, low quant risk |
| DevOps | 6 | Port conflicts, no blue-green strategy |
| Backend | 5 | No API versioning, no auth |
| Frontend | 5 | New project approach wrong, charting underestimated |

**Insight**: Backend/Frontend see implementation complexity; Quant sees minimal domain risk.

**Resolution**: Phase 3 needs significant revision before starting.

---

## Critical Findings (All Agents)

### 1. Strategy Count Error
**Discovered by**: Plan Critic
- Plan claims "22 strategies, 18 remaining"
- Actual: **18 total, 14 remaining** (4 already converted)
- Impact: Timeline estimates off by ~25%

### 2. REST API Integration Gap
**Discovered by**: Plan Critic, Backend Architect
- `rest_api.py` has NO V2 engine routing
- No `engine_version` parameter exists
- `Optimizer` class needs modification
- This is **blocking work** not reflected in estimates

### 3. Non-Vectorizable Strategies
**Discovered by**: Quant Engineer

| Strategy | Blocker |
|----------|---------|
| `FernandoStrategy` | Bracket orders, stateful stop-loss, broker queries |
| `GridTradingStrategy` | Stateful `entry_price`, concurrent limit orders |
| `IchimokuStrategy` | `order_target_percent` semantics differ |
| `FundingRateArbitrage` | Requires external data feed |

**Impact**: These strategies need REWRITE, not conversion.

### 4. VWR Metric Missing
**Discovered by**: Quant Engineer
- V1 ranks by `['sharpe_ratio', 'vwr']`
- V2 ranks by `sharpe_ratio` only
- VWR prevents overfitting selection
- **Silent methodology degradation**

### 5. Parity Test Inadequacy
**Discovered by**: Plan Critic, Quant Engineer
- "5% tolerance" is too loose
- No trade-level alignment checks
- No equity curve correlation requirement
- Recommended: **1-2% tolerance, 95% trade alignment**

### 6. Charting Migration Underestimated
**Discovered by**: Frontend Developer
- Highcharts Stock features (flags, multi-pane, range selector) don't exist in Lightweight Charts
- Plan shows simplified example, reality needs custom overlays
- **Budget 2-3 weeks extra** for charting

### 7. No CI/CD or Monitoring
**Discovered by**: DevOps
- Zero CI/CD mentions in plan
- No health checks
- No observability (logging, metrics, tracing)
- Recommend **Phase 0: Infrastructure Preparation**

---

## Recommended Actions

### DO NOW (Before Phase 1)

| Item | Owner | Agents Agree |
|------|-------|--------------|
| Correct strategy count (18 total, 14 remaining) | Plan Update | All |
| Add REST API integration task to Phase 1 | Plan Update | Backend, Critic |
| Build parity test framework with proper criteria | Quant | Quant, Critic |
| Add VWR metric to V2 engine | Quant | Quant |
| Set up CI/CD pipeline | DevOps | DevOps |

### REVISE (Before Phase 2)

| Item | Owner | Agents Agree |
|------|-------|--------------|
| Add migration dry-run with checksums | DevOps | DevOps, Backend |
| Define explicit rollback procedures | DevOps | DevOps, Backend |
| Abstract RethinkDB access behind `StorageAdapter` | Backend | Backend |
| Add connection pooling for QuestDB | Backend | Backend |

### REVISE (Before Phase 3)

| Item | Owner | Agents Agree |
|------|-------|--------------|
| Incremental upgrade, NOT new project | Frontend | Frontend |
| Add nginx/traefik for API routing | DevOps | DevOps, Backend |
| Define WebSocket message protocol | Backend | Backend, Frontend |
| Budget 2-3 weeks for charting migration | Frontend | Frontend |
| Add auth to FastAPI endpoints | Backend | Backend |

### SKIP / DEFER

| Item | Reason | Agents Agree |
|------|--------|--------------|
| `FernandoStrategy` conversion | Requires rewrite, not conversion | Quant |
| `GridTradingStrategy` conversion | Stateful, cannot vectorize | Quant |
| TradingView Lightweight Charts | Consider full widget instead | Frontend |

---

## Insights from Disagreements

### Phase 2: Backend (6) vs Quant (8)

**Backend Architect**: "Missing rollback strategy, schema mapping undefined"
**Quant Engineer**: "Standard data migration, test float precision"

**Why they disagree**: Backend sees infrastructure complexity; Quant sees data correctness.

**Resolution**: Both are right. Add:
1. Schema mapping document (Backend concern)
2. Float precision validation (Quant concern)
3. Rollback runbook (Backend concern)

### Phase 3: Quant (9) vs Frontend (5)

**Quant Engineer**: "Pure infrastructure, low quant risk"
**Frontend Developer**: "Charting underestimated, new project approach wrong"

**Why they disagree**: Quant cares about methodology; Frontend cares about implementation.

**Resolution**: Quant is correct that quant risk is low. Frontend is correct that execution risk is high. Separate concerns:
- Quant validates: result schema compatibility
- Frontend executes: incremental migration with feature flags

### Phase 1: Quant (6) vs Frontend (9)

**Quant Engineer**: "Parity testing underspecified, strategies with order logic cannot vectorize"
**Frontend Developer**: "Minimal frontend impact, approve"

**Why they disagree**: Frontend sees no UI changes; Quant sees methodology risk.

**Resolution**: Both are right for their domain. Phase 1 is:
- LOW risk for Frontend (approve)
- HIGH risk for Quant methodology (needs work)

Add parity test framework as Phase 1 prerequisite.

---

## Revised Phase Plan

### Phase 0: Infrastructure Preparation (NEW)
- [ ] Set up CI/CD pipeline
- [ ] Add health checks to Docker services
- [ ] Create rollback runbooks
- [ ] Build parity test framework

### Phase 1: Core Engine (REVISED)
- [ ] Add REST API V2 routing
- [ ] Convert 14 remaining strategies (exclude FernandoStrategy, GridTradingStrategy)
- [ ] Add VWR metric to V2 ranking
- [ ] Run parity tests with 2% tolerance, 95% trade alignment
- [ ] Document non-vectorizable strategies for manual handling

### Phase 2: Data Layer (REVISED)
- [ ] Add migration dry-run script with checksums
- [ ] Define QuestDB schema mapping
- [ ] Implement connection pooling
- [ ] Abstract storage layer
- [ ] Keep RethinkDB for 30 days post-migration

### Phase 3: API + Frontend (REVISED)
- [ ] Run Flask + FastAPI on different ports with nginx routing
- [ ] Incremental frontend upgrade (NOT new project)
- [ ] Add WebSocket with polling fallback
- [ ] Charting: evaluate TradingView full widget vs Lightweight
- [ ] Add auth to new endpoints

### Phase 4: Analytics (APPROVED)
- [ ] QuantStats integration
- [ ] Optuna dashboard
- [ ] Use 365-day annualization for crypto

---

## Success Criteria (Revised)

### Phase 1 Complete When:
- [ ] 16 of 18 strategies converted (exclude 2 non-vectorizable)
- [ ] `/optimization/new/` accepts `engine_version: v2`
- [ ] Parity tests pass: **2% tolerance, 95% trade alignment**
- [ ] VWR metric available in V2 results
- [ ] CI pipeline passing

### Phase 2 Complete When:
- [ ] Migration dry-run validates **100% row match + checksum**
- [ ] QuestDB serving all queries
- [ ] RethinkDB still available as fallback
- [ ] Data queries **20x faster**

### Phase 3 Complete When:
- [ ] FastAPI serving `/api/v2/*` endpoints
- [ ] WebSocket progress with polling fallback
- [ ] Frontend upgraded in-place (React 18, MUI v5)
- [ ] Charting migrated with feature parity

---

## Agent Summary

| Agent | Key Contribution | Verdict |
|-------|------------------|---------|
| **Plan Critic** | Found strategy count error, REST API gap, VWR missing | NEEDS REVISION |
| **Backend Architect** | Identified state management conflicts, no circuit breaker | APPROVE WITH CONDITIONS |
| **Quant Engineer** | Identified non-vectorizable strategies, parity test criteria | NEEDS_WORK on Phase 1 |
| **DevOps** | Identified migration risk, no CI/CD, port conflicts | NEEDS_WORK on Phase 2 |
| **Frontend Developer** | Identified charting complexity, recommended incremental upgrade | NEEDS_WORK on Phase 3 |

---

## Final Verdict

```
╔═══════════════════════════════════════════════════════════════╗
║  CONSENSUS: APPROVE WITH CONDITIONS                           ║
║                                                               ║
║  The plan is architecturally sound but requires:              ║
║  1. Phase 0 (Infrastructure) added                            ║
║  2. Phase 1 parity tests strengthened                         ║
║  3. Phase 2 migration validation added                        ║
║  4. Phase 3 rewritten for incremental approach                ║
║                                                               ║
║  Proceed after addressing critical findings.                  ║
╚═══════════════════════════════════════════════════════════════╝
```
