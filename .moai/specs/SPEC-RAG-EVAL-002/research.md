# SPEC-RAG-EVAL-002 Research: Perpetual Quality Discovery Engine

## Research Date: 2026-03-02
## Method: Deep codebase exploration + UltraThink analysis

---

## 1. Current Evaluation System Architecture

### Entry Point
- `run_rag_quality_eval.py` (764 lines)
- 30 hardcoded queries: 6 personas × 5 queries each
- 4 metrics: Accuracy, Completeness, Citations, Context Relevance

### Evaluation Flow
```
PERSONA_TEST_QUERIES (30 static) 
  → QueryHandler.ask() per query
  → LLMJudge.evaluate() per response
  → Aggregate scores → Report
```

### Key Files
| File | Role |
|------|------|
| `run_rag_quality_eval.py` | CLI entry point, 30 hardcoded queries |
| `src/rag/domain/evaluation/llm_judge.py` | LLM-as-Judge, 4 metrics |
| `src/rag/domain/evaluation/parallel_evaluator.py` | Parallel persona runner |
| `src/rag/automation/` | Extended evaluation infrastructure |

### Historical Evaluation Results

| Date | Queries | Pass Rate | Notes |
|------|---------|-----------|-------|
| 2026-02-07 | 30 | 0% | Mock evaluator returning 0.5 (API failures) |
| 2026-02-15 | 150 | 0% | retrieval_quality_proxy, all scores ~0.5 |
| 2026-03-01 | 30 | 93.3% → 96.7% | SPEC-RAG-QUALITY-014 fixed last 2 failures |

**Critical Finding**: Multiple evaluation runs returned uniform 0.5 scores, indicating the LLM-as-Judge API calls were failing silently and defaulting to mock values. The "pass" verdicts may not reflect actual quality.

---

## 2. Why The System Says "Nothing To Improve"

### Root Cause Analysis

1. **Static Query Set**: Fixed 30 queries → once all pass, ceiling reached
2. **Narrow Metric Scope**: Only 4 RAG-specific metrics measured
3. **No Difficulty Progression**: No mechanism to generate harder tests
4. **No System Health Checks**: Code quality, performance, security not assessed
5. **No Comparative Baseline**: No gold-standard answer set for ground truth
6. **Mock Evaluator Masking**: API failures return 0.5 → still treated as data

### The Ceiling Effect
```
Improvement Trajectory:
  SPEC-RAG-QUALITY-001: 13.33% → Fixed core failures
  ...14 SPECs later...
  SPEC-RAG-QUALITY-014: 93.3% → 96.7% (30/30 queries pass)
  → "Nothing to improve" ← WE ARE HERE
  
Reality:
  Only 30/∞ possible queries tested
  Only 4/15+ quality dimensions measured
  Only RAG answers tested, not system health
```

---

## 3. Improvement Dimensions NOT Being Evaluated

### 3.1 RAG Quality (Beyond Current 4 Metrics)

| Dimension | Current State | Improvement Opportunity |
|-----------|--------------|------------------------|
| Response Latency | Not measured | Track p50/p95/p99 per query type |
| Answer Consistency | Not tested | Same query 3x → compare answers |
| Answer Readability | Not measured | Structure, formatting, organization |
| Cross-Regulation Synthesis | Not tested | Questions spanning multiple regulations |
| Temporal Awareness | Not tested | Queries about dates, deadlines, amendments |
| Disambiguation Quality | Not measured | How well ambiguous queries are handled |
| Multi-Turn Coherence | Not tested | Context preservation across conversation turns |
| Citation Existence Verification | Not done | Verify cited articles actually exist in DB |
| Answer Actionability | Not measured | Does answer tell user WHAT TO DO? |
| Regulatory Hierarchy Awareness | Not tested | 편 > 장 > 절 > 조 > 항 > 호 > 목 correctness |

### 3.2 Code Quality Issues (Found by Codebase Exploration)

| Issue | Location | Count |
|-------|----------|-------|
| Bare `except: pass` blocks | cache.py, entities.py, citation_enhancer.py, fact_checker.py, etc. | 15+ |
| Hardcoded configuration values | hyde.py (cache size 1000), search_usecase.py (MAX_REGENERATION=2) | 10+ |
| Code duplication | query_analyzer.py ↔ query_expander_v2.py ↔ colloquial_transformer.py | 3 clusters |
| TODO/FIXME comments | Scattered across infrastructure/ | 5+ |
| Commented-out code | query_analyzer.py:1159, others | 3+ |

### 3.3 Test Coverage Gaps

| Area | Current State |
|------|--------------|
| Edge cases (empty input, max length) | No tests |
| Concurrent query handling | No tests |
| Cache consistency across 3 layers | No tests |
| LLM fallback chain activation | No tests |
| Network timeout scenarios | No tests |
| Vector DB scaling behavior | No tests |
| Multi-turn conversation e2e | No tests |

### 3.4 Architecture & Performance

| Issue | Impact |
|-------|--------|
| N+1 query patterns in search pipeline | Slow responses for complex queries |
| 3 cache layers (memory/file/redis) without consistency | Stale data risk |
| Tokenizer cold-start (Kiwi/Komoran lazy load) | First query slow |
| RRF_K constant hardcoded (60) | Suboptimal fusion |
| No health check endpoint | Cannot verify readiness |

### 3.5 Security & Operations

| Issue | Risk Level |
|-------|-----------|
| No API rate limiting | DoS vulnerability |
| No input length validation on endpoints | Resource exhaustion |
| API keys not masked in logs | Credential leak |
| No env var validation at startup | Silent misconfiguration |
| No structured logging | Production debugging difficulty |

### 3.6 User Experience

| Issue | Impact |
|-------|--------|
| CLI/Web UI feature parity gap | Inconsistent user experience |
| Conversation context not persisted | Lost on restart |
| No proactive query suggestions | Lower engagement |
| No feedback mechanism (helpful/unhelpful) | No learning loop |

---

## 4. Reference Implementations Found

### Internal References
- `src/rag/automation/` - Extended evaluation infrastructure (domain/application/infrastructure pattern)
- `src/rag/domain/evaluation/llm_judge.py` - Current LLM-as-Judge with extensible metric framework
- `data/ground_truth/` - Existing ground truth data structure
- `src/rag/infrastructure/reranker_ab_test.py` - A/B testing framework (can be adapted)

### Pattern References
- The `automation/` subpackage already has a clean architecture pattern for evaluation
- `llm_judge.py` has `EvaluationBatch` that could be extended with new metrics
- `parallel_evaluator.py` already supports concurrent persona evaluation

---

## 5. Risks and Constraints

| Risk | Mitigation |
|------|-----------|
| Dynamic query generation costs LLM tokens | Cache generated queries; regenerate only when regulations change |
| Extended metrics need ground truth | Start with automated checks (citation verification); add manual ground truth incrementally |
| System health scans may be subjective | Use concrete metrics (TODO count, coverage %, etc.) |
| Adaptive difficulty could create flaky tests | Lock difficulty per run; increase between runs |
| Backward compatibility with existing evaluation | Keep 30-query baseline as "L1 tier"; add new tiers on top |

### Technical Constraints
- Must work with existing ChromaDB + OpenRouter LLM infrastructure
- Full evaluation run should complete within 30 minutes
- Results must be JSON for trend tracking
- Must not require human annotation for basic operation

---

## 6. Recommendations

**Primary Recommendation**: Build a "Perpetual Quality Discovery Engine" that:
1. Auto-generates test queries from regulation content (never runs out)
2. Measures 10+ quality dimensions (not just 4)
3. Scans system health beyond RAG quality
4. Adapts difficulty progressively (L1→L5)
5. Always shows the NEXT improvement frontier (never says "done")

**Secondary Recommendation**: Fix the evaluation infrastructure reliability:
- Detect and alert when LLM-as-Judge falls back to mock values
- Validate API connectivity before starting evaluation
- Add retry logic with exponential backoff for API calls

**SPEC ID**: SPEC-RAG-EVAL-002
**Priority**: High
**Estimated Complexity**: 5 requirement modules, medium-to-large scope
