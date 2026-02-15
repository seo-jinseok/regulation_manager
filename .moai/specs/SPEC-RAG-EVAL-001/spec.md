# SPEC-RAG-EVAL-001: RAG Quality Evaluator Agent Enhancement

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-EVAL-001 |
| **Title** | RAG Quality Evaluator Agent Enhancement |
| **Status** | Planned |
| **Priority** | High |
| **Created** | 2026-02-15 |
| **Source** | User Request - RAG Evaluator Improvement |
| **Target Agent** | rag-quality-evaluator (rag-quality-local skill) |

---

## Problem Analysis

### Current Performance

RAG 품질 평가 에이전트(`rag-quality-evaluator`)는 현재 다음과 같은 제약사항을 가지고 있습니다:

| Issue | Current State | Target State |
|-------|---------------|--------------|
| API Budget Constraints | Limited evaluation runs | Efficient batch processing |
| Persona Coverage | Professor tests incomplete (errors) | All 6 personas fully tested |
| Evaluation Pipeline | Manual intervention required | Fully automated |
| Reporting Quality | Basic metrics only | Actionable recommendations |
| Feedback Loop | No SPEC generation | Auto-generate improvement SPECs |
| Test Query Generation | Template-based only | Regulation-specific generation |
| Progress Tracking | None | Real-time with resume capability |

### Evidence from Evaluation

**Professor Persona Test Results** (`professor_persona_test_results.json`):
- Total queries: 13
- Successful: 9
- Errors: 4 (`zip() argument 2 is shorter than argument 1`)
- Average satisfaction: Very Satisfied (for successful queries)

**Current Metrics** (from SPEC-RAG-QUALITY-001):
- Pass Rate: 43.3% (Target: 80%+)
- Average Score: 0.795 (Target: 0.80+)
- Accuracy: 0.807
- Completeness: 0.773
- Citations: 0.842

### Root Cause Analysis

**Issue 1: API Budget Inefficiency (Priority: P0)**

**Symptom**: Limited evaluation runs due to API costs

**Root Cause**:
1. Sequential API calls instead of batched requests
2. No rate limiting optimization
3. Redundant LLM calls for similar queries
4. Missing cache for repeated evaluations

**Issue 2: Incomplete Persona Coverage (Priority: P0)**

**Symptom**: Professor persona tests failing with errors

**Root Cause**:
1. `zip()` mismatch in parallel evaluation code
2. Missing error handling for persona-specific query generation
3. Incomplete query template coverage for academic personas

**Issue 3: Manual Pipeline Intervention (Priority: P1)**

**Symptom**: Manual intervention required during evaluation

**Root Cause**:
1. No automatic error recovery
2. Missing checkpoint/resume mechanism
3. Incomplete automation in result aggregation

**Issue 4: Limited Reporting Quality (Priority: P1)**

**Symptom**: Reports lack actionable recommendations

**Root Cause**:
1. Basic metric aggregation only
2. No failure pattern analysis
3. Missing improvement priority scoring
4. No root cause mapping to SPEC templates

**Issue 5: No Feedback Loop (Priority: P0)**

**Symptom**: No automatic SPEC generation from failures

**Root Cause**:
1. Missing failure classification system
2. No SPEC template mapping
3. No automatic requirement generation

---

## Requirements (EARS Format)

### REQ-001: Efficient Parallel Evaluation with API Budget Awareness

**WHEN** evaluation is initiated

**THE SYSTEM SHALL** execute queries in parallel with configurable batch sizes

**SUCH THAT** API costs are minimized while maximizing throughput

**Acceptance Criteria**:
- [ ] Support configurable batch sizes (1-10 queries per batch)
- [ ] Implement rate limiting per API provider
- [ ] Cache repeated query evaluations
- [ ] Display estimated API cost before execution
- [ ] Support API budget limits with graceful degradation

### REQ-002: Complete 6-Persona Test Coverage

**WHEN** full evaluation is executed

**THE SYSTEM SHALL** test all 6 personas with 25+ queries each

**SUCH THAT** pass rate >= 80% across all personas

**Acceptance Criteria**:
- [ ] Fix `zip()` error in parallel evaluation
- [ ] Generate 25+ queries per persona (150+ total)
- [ ] Ensure regulation-specific query content
- [ ] Balance difficulty levels (easy/medium/hard)
- [ ] Cover all scenario categories per persona

### REQ-003: Automatic Failure Analysis and SPEC Generation

**WHEN** evaluation detects failing queries (score < 0.70)

**THE SYSTEM SHALL** analyze failure patterns and generate SPEC documents

**SUCH THAT** developers receive actionable improvement guidance

**Acceptance Criteria**:
- [ ] Classify failures by type (hallucination, retrieval, citation, etc.)
- [ ] Map failure patterns to SPEC templates
- [ ] Auto-generate SPEC requirements in EARS format
- [ ] Prioritize SPECs by impact (affected query count)
- [ ] Include reproduction steps in generated SPECs

### REQ-004: Regulation-Specific Test Query Generation

**WHEN** generating test queries for a persona

**THE SYSTEM SHALL** extract content from actual regulation documents

**SUCH THAT** queries reflect real-world usage patterns

**Acceptance Criteria**:
- [ ] Parse regulation JSON for query topic extraction
- [ ] Generate queries about specific articles (제N조)
- [ ] Create cross-reference queries (multiple regulations)
- [ ] Include temporal queries (deadlines, periods)
- [ ] Support procedure-based queries (step-by-step)

### REQ-005: Real-time Progress Tracking and Resumable Evaluation

**WHEN** evaluation is in progress or interrupted

**THE SYSTEM SHALL** track progress and support resume from checkpoint

**SUCH THAT** no work is lost on interruption

**Acceptance Criteria**:
- [ ] Save checkpoint after each batch completion
- [ ] Display real-time progress (X/Y queries, current persona)
- [ ] Support resume from last successful checkpoint
- [ ] Handle graceful shutdown (Ctrl+C) with state preservation
- [ ] Log all evaluation events for debugging

### REQ-006: Actionable Improvement Recommendations

**WHEN** evaluation report is generated

**THE SYSTEM SHALL** include specific, prioritized recommendations

**SUCH THAT** developers can immediately act on findings

**Acceptance Criteria**:
- [ ] Rank failures by frequency and severity
- [ ] Provide root cause analysis per failure type
- [ ] Suggest concrete code/prompt changes
- [ ] Include before/after examples
- [ ] Link to relevant SPEC documents

---

## Technical Approach

### Architecture Overview

```
RAG Quality Evaluator Enhancement
    |
    +-- Phase 1: API Optimization
    |       +-- BatchExecutor (configurable batch size)
    |       +-- RateLimiter (per-provider limits)
    |       +-- EvaluationCache (LRU with TTL)
    |
    +-- Phase 2: Persona Coverage
    |       +-- PersonaQueryGenerator (regulation-aware)
    |       +-- QueryBalancer (difficulty distribution)
    |       +-- ErrorRecovery (graceful handling)
    |
    +-- Phase 3: Automation Pipeline
    |       +-- CheckpointManager (state persistence)
    |       +-- ProgressReporter (real-time display)
    |       +-- ResumeController (interrupt recovery)
    |
    +-- Phase 4: Reporting Enhancement
    |       +-- FailureClassifier (pattern detection)
    |       +-- RecommendationEngine (actionable suggestions)
    |       +-- SPECGenerator (EARS format output)
    |
    +-- Phase 5: Integration
            +-- GradioDashboard (web interface)
            +-- CLICommands (terminal interface)
            +-- OutputFormats (JSON, Markdown, SPEC)
```

### Phase 1: API Budget Optimization (Primary Goal)

**1.1 BatchExecutor Implementation**

Create `src/rag/domain/evaluation/batch_executor.py`:
- Configure batch size (default: 5, max: 10)
- Group queries by persona for efficiency
- Implement exponential backoff on rate limits
- Support async execution with `asyncio.gather()`

**1.2 RateLimiter Implementation**

Create `src/rag/domain/evaluation/rate_limiter.py`:
- Per-provider rate tracking (OpenAI, Gemini, Ollama)
- Token bucket algorithm for smooth limiting
- Configurable requests per minute (RPM)
- Cost estimation before execution

**1.3 EvaluationCache Implementation**

Extend `EvaluationStore` with caching:
- LRU cache for repeated query evaluations
- TTL-based expiration (default: 24 hours)
- Cache hit/miss statistics
- Selective cache invalidation

### Phase 2: Persona Coverage Enhancement

**2.1 Fix Parallel Evaluation Bug**

Modify `src/rag/domain/evaluation/parallel_evaluator.py`:
- Fix `zip()` argument mismatch in `_evaluate_single_query`
- Add proper error handling for persona-specific generation
- Validate query count matches expected count

**2.2 Regulation-Aware Query Generator**

Create `src/rag/domain/evaluation/regulation_query_generator.py`:
- Parse `data/output/규정집.json` for content extraction
- Extract articles (제N조) for specific queries
- Generate cross-reference queries
- Create temporal queries (deadlines, periods)

**2.3 Query Balance and Distribution**

Create query distribution algorithm:
- Target: 40% easy, 40% medium, 20% hard per persona
- Ensure topic coverage across regulations
- Balance single-intent vs multi-intent queries

### Phase 3: Automation Pipeline

**3.1 Checkpoint Manager**

Create `src/rag/infrastructure/checkpoint_manager.py`:
- Save state after each batch
- Store: completed queries, pending queries, partial results
- Support JSON serialization
- Handle concurrent access safely

**3.2 Progress Reporter**

Create `src/rag/interface/progress_reporter.py`:
- Terminal progress bar with Rich/TQDM
- Web progress via Gradio callbacks
- Estimated time remaining
- Current operation display

**3.3 Resume Controller**

Create `src/rag/application/resume_controller.py`:
- Load checkpoint on evaluation start
- Skip completed queries
- Merge partial results
- Handle checkpoint corruption

### Phase 4: Reporting Enhancement

**4.1 Failure Classifier**

Create `src/rag/domain/evaluation/failure_classifier.py`:
- Pattern: Hallucination (fake contacts, wrong departments)
- Pattern: Retrieval Failure (missing relevant docs)
- Pattern: Citation Error (wrong article, missing reference)
- Pattern: Incomplete Answer (missing key info)
- Pattern: Ambiguity (unclear user intent)

**4.2 Recommendation Engine**

Create `src/rag/domain/evaluation/recommendation_engine.py`:
- Map failures to code changes
- Generate before/after examples
- Prioritize by impact score
- Link to existing SPECs when applicable

**4.3 SPEC Generator**

Create `src/rag/domain/evaluation/spec_generator.py`:
- EARS format requirement generation
- Problem analysis from failure data
- Acceptance criteria derivation
- Priority scoring based on frequency

### Phase 5: Interface Integration

**5.1 CLI Commands**

Extend `src/cli.py`:
- `regulation evaluate --full` (all personas, all scenarios)
- `regulation evaluate --persona professor` (targeted)
- `regulation evaluate --resume` (from checkpoint)
- `regulation evaluate --budget 10.00` (API cost limit)

**5.2 Gradio Dashboard Enhancement**

Extend `src/rag/interface/web/quality_dashboard.py`:
- Progress visualization
- Real-time metric updates
- Download reports
- SPEC preview

---

## Dependencies

### Python Packages

```toml
# Existing
ragas>=0.4.3
deepeval>=3.8.1
openai>=2.11.0

# New
rich>=13.0.0  # Progress bars
cachetools>=5.3.0  # LRU cache
tenacity>=8.2.0  # Retry logic
```

### Internal Modules

- `src/rag/domain/evaluation/` (existing evaluation framework)
- `src/rag/infrastructure/storage/` (evaluation storage)
- `src/rag/interface/web/` (Gradio dashboard)
- `data/output/규정집.json` (regulation data)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API costs exceed budget | Medium | High | Implement cost estimation and hard limits |
| Checkpoint corruption | Low | Medium | Use atomic writes, backup before overwrite |
| Query generation quality | Medium | Medium | Manual review of generated queries |
| SPEC template mismatch | Low | Low | Fallback to generic template |

---

## Traceability

| TAG | Description | REQ |
|-----|-------------|-----|
| TAG-EVAL-001 | BatchExecutor implementation | REQ-001 |
| TAG-EVAL-002 | RateLimiter implementation | REQ-001 |
| TAG-EVAL-003 | EvaluationCache implementation | REQ-001 |
| TAG-EVAL-004 | Parallel evaluation bug fix | REQ-002 |
| TAG-EVAL-005 | RegulationQueryGenerator | REQ-004 |
| TAG-EVAL-006 | CheckpointManager | REQ-005 |
| TAG-EVAL-007 | ProgressReporter | REQ-005 |
| TAG-EVAL-008 | ResumeController | REQ-005 |
| TAG-EVAL-009 | FailureClassifier | REQ-003 |
| TAG-EVAL-010 | RecommendationEngine | REQ-006 |
| TAG-EVAL-011 | SPECGenerator | REQ-003 |

---

## References

- Agent Definition: `.claude/agents/rag-quality-evaluator.md`
- Skill Definition: `.claude/skills/rag-quality-local/SKILL.md`
- Previous SPEC: `.moai/specs/SPEC-RAG-QUALITY-001/spec.md`
- Persona Test Results: `professor_persona_test_results.json`
- RAGAS Documentation: https://docs.ragas.io/

---

<moai>DONE</moai>
