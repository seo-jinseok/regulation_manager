# Implementation Plan: SPEC-RAG-EVAL-001

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-EVAL-001 |
| **Title** | RAG Quality Evaluator Agent Enhancement |
| **Methodology** | Hybrid (TDD for new code, DDD for modifications) |
| **Target Coverage** | 85%+ |

---

## Milestones

### Milestone 1: API Budget Optimization (Primary Goal)

**Priority**: P0 - Critical

**Objective**: Implement efficient parallel evaluation with API cost awareness

**Tasks**:

| Task ID | Description | Type | Dependencies |
|---------|-------------|------|--------------|
| TAG-EVAL-001 | BatchExecutor implementation | TDD | None |
| TAG-EVAL-002 | RateLimiter implementation | TDD | None |
| TAG-EVAL-003 | EvaluationCache implementation | DDD | None |

**Deliverables**:
- `src/rag/domain/evaluation/batch_executor.py`
- `src/rag/domain/evaluation/rate_limiter.py`
- Extended `EvaluationStore` with caching

**Acceptance Criteria**:
- [ ] BatchExecutor supports 1-10 queries per batch
- [ ] RateLimiter tracks per-provider limits
- [ ] Cache reduces redundant API calls by 50%+
- [ ] Cost estimation displayed before execution

### Milestone 2: Persona Coverage Enhancement

**Priority**: P0 - Critical

**Objective**: Achieve complete 6-persona test coverage with 150+ queries

**Tasks**:

| Task ID | Description | Type | Dependencies |
|---------|-------------|------|--------------|
| TAG-EVAL-004 | Fix parallel evaluation zip() bug | DDD | None |
| TAG-EVAL-005 | RegulationQueryGenerator implementation | TDD | TAG-EVAL-001 |
| TAG-EVAL-005a | Query distribution balancer | TDD | TAG-EVAL-005 |

**Deliverables**:
- Fixed `src/rag/domain/evaluation/parallel_evaluator.py`
- `src/rag/domain/evaluation/regulation_query_generator.py`
- `src/rag/domain/evaluation/query_balancer.py`

**Acceptance Criteria**:
- [ ] Zero errors in professor persona tests
- [ ] 25+ queries generated per persona
- [ ] Difficulty distribution: 40% easy, 40% medium, 20% hard
- [ ] All queries reference actual regulation content

### Milestone 3: Automation Pipeline

**Priority**: P1 - High

**Objective**: Enable fully automated evaluation with checkpoint/resume

**Tasks**:

| Task ID | Description | Type | Dependencies |
|---------|-------------|------|--------------|
| TAG-EVAL-006 | CheckpointManager implementation | TDD | None |
| TAG-EVAL-007 | ProgressReporter implementation | TDD | TAG-EVAL-006 |
| TAG-EVAL-008 | ResumeController implementation | TDD | TAG-EVAL-006 |

**Deliverables**:
- `src/rag/infrastructure/checkpoint_manager.py`
- `src/rag/interface/progress_reporter.py`
- `src/rag/application/resume_controller.py`

**Acceptance Criteria**:
- [ ] Checkpoints saved after each batch
- [ ] Progress bar displays real-time status
- [ ] Resume from interruption works correctly
- [ ] Graceful shutdown preserves state

### Milestone 4: Reporting Enhancement

**Priority**: P1 - High

**Objective**: Generate actionable improvement recommendations

**Tasks**:

| Task ID | Description | Type | Dependencies |
|---------|-------------|------|--------------|
| TAG-EVAL-009 | FailureClassifier implementation | TDD | None |
| TAG-EVAL-010 | RecommendationEngine implementation | TDD | TAG-EVAL-009 |
| TAG-EVAL-011 | SPECGenerator implementation | TDD | TAG-EVAL-009, TAG-EVAL-010 |

**Deliverables**:
- `src/rag/domain/evaluation/failure_classifier.py`
- `src/rag/domain/evaluation/recommendation_engine.py`
- `src/rag/domain/evaluation/spec_generator.py`

**Acceptance Criteria**:
- [ ] Failures classified into 5+ pattern types
- [ ] Recommendations include code examples
- [ ] Generated SPECs follow EARS format
- [ ] Priority scoring based on frequency

### Milestone 5: Interface Integration

**Priority**: P2 - Medium

**Objective**: Provide CLI and Gradio interfaces for all features

**Tasks**:

| Task ID | Description | Type | Dependencies |
|---------|-------------|------|--------------|
| TAG-EVAL-012 | CLI commands extension | DDD | TAG-EVAL-001 through TAG-EVAL-008 |
| TAG-EVAL-013 | Gradio dashboard enhancement | DDD | TAG-EVAL-007, TAG-EVAL-009 |

**Deliverables**:
- Extended `src/cli.py`
- Enhanced `src/rag/interface/web/quality_dashboard.py`

**Acceptance Criteria**:
- [ ] CLI supports --full, --persona, --resume, --budget flags
- [ ] Dashboard shows real-time progress
- [ ] Reports downloadable from dashboard

---

## Technical Approach

### File Structure

```
src/rag/
├── domain/
│   └── evaluation/
│       ├── batch_executor.py          # NEW
│       ├── rate_limiter.py            # NEW
│       ├── regulation_query_generator.py  # NEW
│       ├── query_balancer.py          # NEW
│       ├── failure_classifier.py      # NEW
│       ├── recommendation_engine.py   # NEW
│       ├── spec_generator.py          # NEW
│       ├── parallel_evaluator.py      # MODIFY
│       └── ...
├── infrastructure/
│   ├── checkpoint_manager.py          # NEW
│   └── storage/
│       └── evaluation_store.py        # MODIFY
├── application/
│   └── resume_controller.py           # NEW
├── interface/
│   ├── progress_reporter.py           # NEW
│   └── web/
│       └── quality_dashboard.py       # MODIFY
└── ...
```

### Dependencies

```toml
# Add to pyproject.toml
[project.dependencies]
rich = ">=13.0.0"
cachetools = ">=5.3.0"
tenacity = ">=8.2.0"
```

### API Design

**BatchExecutor**:
```python
class BatchExecutor:
    def __init__(self, batch_size: int = 5, rate_limiter: RateLimiter = None): ...
    async def execute_batch(self, queries: List[PersonaQuery]) -> List[JudgeResult]: ...
    def estimate_cost(self, queries: List[PersonaQuery]) -> Decimal: ...
```

**RateLimiter**:
```python
class RateLimiter:
    def __init__(self, rpm: int = 60, provider: str = "openai"): ...
    async def acquire(self): ...
    def get_stats(self) -> RateLimitStats: ...
```

**CheckpointManager**:
```python
class CheckpointManager:
    def save(self, state: EvaluationState) -> None: ...
    def load(self) -> Optional[EvaluationState]: ...
    def clear(self) -> None: ...
```

**FailureClassifier**:
```python
class FailureClassifier:
    def classify(self, result: JudgeResult) -> FailurePattern: ...
    def analyze_batch(self, results: List[JudgeResult]) -> List[FailurePattern]: ...
```

**SPECGenerator**:
```python
class SPECGenerator:
    def generate(self, patterns: List[FailurePattern]) -> SPEC: ...
    def to_ears_format(self, requirement: str) -> str: ...
```

---

## Testing Strategy

### Unit Tests

| Test File | Purpose | Coverage Target |
|-----------|---------|-----------------|
| `test_batch_executor.py` | Batch execution logic | 90% |
| `test_rate_limiter.py` | Rate limiting behavior | 90% |
| `test_checkpoint_manager.py` | State persistence | 85% |
| `test_failure_classifier.py` | Pattern classification | 85% |
| `test_spec_generator.py` | SPEC generation | 90% |
| `test_regulation_query_generator.py` | Query generation | 85% |

### Integration Tests

| Test File | Purpose |
|-----------|---------|
| `test_parallel_evaluation_integration.py` | Full evaluation pipeline |
| `test_resume_flow.py` | Checkpoint/resume flow |
| `test_spec_generation_flow.py` | Failure to SPEC flow |

### Characterization Tests

For modified files (`parallel_evaluator.py`, `evaluation_store.py`):
- Capture current behavior before changes
- Verify behavior preserved after modifications

---

## Execution Order

### Phase 1: Foundation (Milestone 1)
1. TAG-EVAL-001: BatchExecutor
2. TAG-EVAL-002: RateLimiter
3. TAG-EVAL-003: EvaluationCache

### Phase 2: Bug Fix + Enhancement (Milestone 2)
4. TAG-EVAL-004: Fix parallel evaluation bug
5. TAG-EVAL-005: RegulationQueryGenerator
6. TAG-EVAL-005a: Query distribution balancer

### Phase 3: Automation (Milestone 3)
7. TAG-EVAL-006: CheckpointManager
8. TAG-EVAL-007: ProgressReporter
9. TAG-EVAL-008: ResumeController

### Phase 4: Intelligence (Milestone 4)
10. TAG-EVAL-009: FailureClassifier
11. TAG-EVAL-010: RecommendationEngine
12. TAG-EVAL-011: SPECGenerator

### Phase 5: Integration (Milestone 5)
13. TAG-EVAL-012: CLI commands
14. TAG-EVAL-013: Gradio dashboard

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| API cost exceeds budget | Implement hard cost limit with early termination |
| Checkpoint corruption | Use atomic writes with backup |
| Query generation quality | Manual review of first 50 generated queries |
| SPEC template mismatch | Provide fallback generic template |

---

## Next Steps

1. **Immediate**: Fix parallel evaluation bug (TAG-EVAL-004)
2. **Short-term**: Implement BatchExecutor (TAG-EVAL-001)
3. **Medium-term**: Complete Milestones 1-3
4. **Long-term**: Complete Milestones 4-5

---

## References

- SPEC Document: `.moai/specs/SPEC-RAG-EVAL-001/spec.md`
- Acceptance Criteria: `.moai/specs/SPEC-RAG-EVAL-001/acceptance.md`
- Related SPEC: `.moai/specs/SPEC-RAG-QUALITY-001/spec.md`
