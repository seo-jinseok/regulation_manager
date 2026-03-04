# SPEC-RAG-EVAL-002 Implementation Plan

## Overview

| Field | Value |
|-------|-------|
| SPEC | SPEC-RAG-EVAL-002 |
| Title | Perpetual Quality Discovery Engine |
| Methodology | DDD (Analyze → Preserve → Improve) |
| Estimated Files | 10 new + 3 modified |
| Risk Level | Medium |

---

## Implementation Phases

### Phase 1: Query Synthesizer (Module 1)

**EARS Requirements:** EARS-U-001, EARS-U-002, EARS-U-003

#### New Files
- `src/rag/domain/evaluation/query_synthesizer.py`
  - `QuerySynthesizer` class
  - `generate_from_regulations(data_dir: str) -> List[GeneratedQuery]`
  - `generate_cross_regulation(regulations: List[dict]) -> List[GeneratedQuery]`
  - `generate_adversarial(count: int) -> List[GeneratedQuery]`
  - `GeneratedQuery` dataclass: query, expected_source, difficulty_tier, category, query_type

- `tests/rag/unit/evaluation/test_query_synthesizer.py`
  - Test query generation from sample regulation JSON
  - Test cross-regulation query creation
  - Test adversarial query creation
  - Test difficulty tier assignment
  - Test caching behavior

#### Modified Files
- `run_rag_quality_eval.py`
  - Add `--generate` flag for dynamic query generation
  - Add `--regenerate` flag to force regeneration
  - Integrate `QuerySynthesizer` into evaluation pipeline
  - Load generated queries from cache when available

#### Dependencies
- Regulation JSON files in `data/output/`
- Existing `QueryHandler` for query execution

#### Reference Implementations
- `PERSONA_TEST_QUERIES` dict in `run_rag_quality_eval.py:75-150` (current static queries pattern)
- `data/output/*.json` (regulation JSON structure for content extraction)

---

### Phase 2: Extended Metrics (Module 2)

**EARS Requirements:** EARS-U-004, EARS-U-005, EARS-U-006, EARS-U-007

#### New Files
- `src/rag/domain/evaluation/extended_metrics.py`
  - `LatencyTracker` class: measures and aggregates response times
  - `ConsistencyChecker` class: runs query N times, computes embedding similarity
  - `CitationVerifier` class: verifies cited articles exist in ChromaDB
  - `ReadabilityScorer` class: structural analysis of response formatting

- `tests/rag/unit/evaluation/test_extended_metrics.py`
  - Test latency percentile calculation
  - Test consistency detection with mock responses
  - Test citation verification against mock DB
  - Test readability scoring criteria

#### Modified Files
- `run_rag_quality_eval.py`
  - Add `--consistency` flag
  - Integrate latency tracking into evaluation loop
  - Add citation verification pass after evaluation
  - Include readability scoring in report

- `src/rag/domain/evaluation/llm_judge.py`
  - Extend `EvaluationResult` with new metric fields
  - Maintain backward compatibility with existing 4 metrics

#### Dependencies
- ChromaDB store access (for citation verification)
- Response embedding model (for consistency check)

---

### Phase 3: System Health Radar (Module 3)

**EARS Requirements:** EARS-U-008, EARS-U-009, EARS-U-010

#### New Files
- `src/rag/domain/evaluation/system_health.py`
  - `CodeQualityScanner` class: AST/regex-based source code analysis
  - `CoverageDeltaChecker` class: reads coverage.json, computes delta
  - `ConfigDriftDetector` class: validates env vars, config ranges
  - `SystemHealthReport` dataclass: aggregated health metrics

- `tests/rag/unit/evaluation/test_system_health.py`
  - Test bare except detection on sample code
  - Test TODO/FIXME counting
  - Test coverage delta calculation
  - Test env var validation

#### Modified Files
- `run_rag_quality_eval.py`
  - Add `--health` flag
  - Integrate health scan into `--full` mode
  - Include health results in JSON output

#### Dependencies
- `ast` module for Python source analysis
- `coverage.json` for coverage data
- `.env` or `os.environ` for config validation

#### Technical Notes
- CodeQualityScanner uses Python AST parsing for except:pass detection
- Regex fallback for non-AST patterns (TODO comments, hardcoded numbers)
- No external dependencies needed for this module

---

### Phase 4: Adaptive Difficulty (Module 4)

**EARS Requirements:** EARS-U-011, EARS-U-012

#### New Files
- `src/rag/domain/evaluation/difficulty_manager.py`
  - `DifficultyManager` class: manages tier state and progression
  - `DifficultyState` dataclass: per-tier mastery status
  - `assign_tier(query: GeneratedQuery) -> DifficultyTier`
  - `check_escalation(results: List[EvaluationResult]) -> Optional[EscalationAction]`
  - `get_mastery_display() -> str` (e.g., "L1 ✓ | L2 ✓ | L3 78% | L4 -- | L5 --")

- `tests/rag/unit/evaluation/test_difficulty_manager.py`
  - Test tier assignment logic
  - Test escalation trigger at 95% pass rate
  - Test mastery state persistence
  - Test display formatting

#### Modified Files
- `run_rag_quality_eval.py`
  - Add `--tier L3` flag for tier-specific testing
  - Integrate difficulty management into evaluation pipeline
  - Display mastery progress in report

#### State Files
- `data/evaluations/difficulty_state.json`: persisted mastery state

---

### Phase 5: Improvement Roadmap Engine (Module 5)

**EARS Requirements:** EARS-U-013, EARS-U-014, EARS-U-015

#### New Files
- `src/rag/domain/evaluation/improvement_radar.py`
  - `FailureClusterer` class: groups failures by root cause pattern
  - `RoadmapGenerator` class: produces prioritized improvement list
  - `TrendAnalyzer` class: computes quality trends over time
  - `ImprovementItem` dataclass: priority, description, location, complexity, suggested_spec

- `tests/rag/unit/evaluation/test_improvement_radar.py`
  - Test failure clustering with sample data
  - Test priority ranking algorithm
  - Test trend calculation with mock historical data
  - Test "never nothing to improve" guarantee

#### Modified Files
- `run_rag_quality_eval.py`
  - Add `--trend` flag
  - Generate improvement roadmap section in report
  - Ensure report always includes at least one recommendation

#### Dependencies
- Historical evaluation JSONs in `data/evaluations/`
- Failure data from current evaluation run

---

## Integration Phase

### Modified File: run_rag_quality_eval.py

**New CLI flags:**
```
--generate       Enable dynamic query generation
--regenerate     Force regeneration of cached queries
--consistency    Run answer consistency checks
--health         Run system health scans
--tier L1-L5     Test specific difficulty tier
--trend          Generate trend analysis
--full           Enable ALL new features
```

**Backward Compatibility:**
- Existing flags unchanged: `--quick`, `--personas`, `--queries`, `--summary`
- `--quick` runs only original 30 queries with original 4 metrics (no change)
- New features strictly opt-in via flags
- `--full` is the recommended new mode

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| LLM cost increase from query generation | Medium | Medium | Cache generated queries; regenerate only on data change |
| False positives in system health scan | Low | Low | Treat health issues as WARNING, not FAIL |
| Inconsistent difficulty tier assignment | Medium | Medium | Use clear heuristic rules; manual override via --tier |
| Evaluation timeout on --full mode | Medium | High | Implement query sampling; batch LLM calls; target 30 min |
| Breaking existing evaluation | Low | High | Backward compatibility is EARS-E-001 requirement |

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Query synthesis | LLM (existing OpenRouter) + template | Cost-effective generation with quality |
| Latency tracking | Python time.perf_counter | Precise wall-clock measurement |
| Consistency check | Existing embedding model | Reuse BGE-M3 for similarity |
| AST scanning | Python ast module | Built-in, no deps |
| Citation verification | ChromaDB query | Reuse existing store |
| Trend analysis | numpy/statistics | Minimal dependency |

---

## Implementation Order

```
Phase 1 (Query Synthesizer)  ← Foundation for all other modules
    ↓
Phase 2 (Extended Metrics)   ← Independent, can parallel with Phase 3
Phase 3 (System Health)      ← Independent, can parallel with Phase 2
    ↓
Phase 4 (Adaptive Difficulty) ← Depends on Phase 1 (query tiers)
    ↓
Phase 5 (Improvement Radar)   ← Depends on all previous phases (failure data)
    ↓
Integration                    ← Wire all modules into run_rag_quality_eval.py
```

Phases 2 and 3 are independent and can be implemented in parallel.
