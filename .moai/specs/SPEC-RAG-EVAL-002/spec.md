---
id: SPEC-RAG-EVAL-002
version: 1.0.0
status: complete
created: 2026-03-03
updated: 2026-03-04
author: MoAI
priority: high
title: "Perpetual Quality Discovery Engine"
tags: [evaluation, quality, dynamic-testing, system-health, adaptive-difficulty]
affected_files:
  - run_rag_quality_eval.py
  - src/rag/domain/evaluation/llm_judge.py
  - src/rag/domain/evaluation/parallel_evaluator.py
  - src/rag/domain/evaluation/query_synthesizer.py (new)
  - src/rag/domain/evaluation/extended_metrics.py (new)
  - src/rag/domain/evaluation/system_health.py (new)
  - src/rag/domain/evaluation/difficulty_manager.py (new)
  - src/rag/domain/evaluation/improvement_radar.py (new)
  - tests/rag/unit/evaluation/test_query_synthesizer.py (new)
  - tests/rag/unit/evaluation/test_extended_metrics.py (new)
  - tests/rag/unit/evaluation/test_system_health.py (new)
  - tests/rag/unit/evaluation/test_difficulty_manager.py (new)
  - tests/rag/unit/evaluation/test_improvement_radar.py (new)
---

# HISTORY

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-03-03 | MoAI | Initial SPEC creation |

---

# SPEC-RAG-EVAL-002: Perpetual Quality Discovery Engine

## Background

### Problem Statement

현재 `regulation-quality` 평가 스킬은 30개의 고정 쿼리와 4개의 메트릭으로 시스템 품질을 측정합니다.
14개의 SPEC(SPEC-RAG-QUALITY-001 ~ 014)을 통해 이 30개 쿼리 전부 통과에 성공했으나,
이로 인해 평가 시스템이 "개선할 것이 없다"고 선언하는 **구조적 한계(ceiling effect)**가 발생했습니다.

실제로는 다음과 같은 50+ 개선 기회가 존재하지만 현재 평가가 감지하지 못합니다:
- RAG 응답 품질의 10+ 미측정 차원 (응답 속도, 일관성, 가독성, 교차규정 합성 등)
- 코드 품질 이슈 15+ (silent exception, 중복 코드, hardcoded 값)
- 테스트 커버리지 갭 10+ (edge case, 동시성, 캐시 일관성)
- 보안/운영 이슈 4+ (rate limiting, 입력 검증, 로깅)
- 사용자 경험 이슈 5+ (CLI/Web 기능 불일치, 컨텍스트 지속성)

### Root Cause

평가 시스템의 "ceiling effect"는 3가지 원인에서 발생합니다:

1. **정적 쿼리 세트**: 30개 고정 쿼리가 전부 통과하면 더 이상 실패를 감지할 수 없음
2. **좁은 메트릭 범위**: RAG 답변 품질의 4개 메트릭만 측정하여 시스템 전체 건강 상태를 반영하지 못함
3. **난이도 고정**: 현재 테스트가 모두 통과해도 더 어려운 테스트를 자동 생성하는 메커니즘이 없음

### Target

평가 시스템이 **절대로 "개선할 것이 없다"고 말할 수 없도록** 다음을 구현:
- 30개 → 200+ 동적 쿼리 생성
- 4개 → 10+ 품질 메트릭
- RAG 답변 평가 → 시스템 전체 건강 평가
- 고정 난이도 → L1-L5 적응형 난이도 진행
- 단순 보고 → 자동 개선 로드맵 생성

---

## Requirements (EARS Format)

### Module 1: Dynamic Query Universe

#### EARS-U-001: Regulation-Based Query Synthesis

**When** the evaluation system runs with `--generate` flag or when the existing query pool falls below the configured minimum (default: 100),
**the system shall** automatically generate test queries by analyzing regulation JSON content in `data/output/`,
extracting key provisions (조), requirements (항), and exceptions (단서) to formulate natural language questions.

**Rationale:** Static 30 queries cannot cover the breadth of 1000+ regulation articles. Auto-generation from content ensures every regulation article has potential test coverage.

**Constraints:**
- Generated queries must be cached in `data/evaluations/generated_queries.json`
- Regeneration triggers: regulation data update or manual `--regenerate` flag
- Minimum 200 queries across all difficulty tiers
- Each query must include: question text, expected regulation source, difficulty tier (L1-L5), category

#### EARS-U-002: Cross-Regulation Query Generation

**When** generating queries at difficulty L3 or above,
**the system shall** create questions that require synthesizing information from 2+ regulation documents
(e.g., "학칙과 장학규정에서 성적 장학금의 자격 기준이 다른가요?").

**Rationale:** Real users frequently ask questions spanning multiple regulations. Current evaluation tests only single-regulation retrieval.

**Constraints:**
- At least 20% of L3+ queries must be cross-regulation type
- Cross-regulation pairs identified by shared keyword overlap analysis

#### EARS-U-003: Adversarial Query Generation

**When** generating queries at difficulty L4-L5,
**the system shall** include adversarial queries designed to expose system weaknesses:
- Out-of-domain questions (questions about regulations that don't exist)
- Hallucination triggers (questions with false premises)
- Ambiguous queries requiring disambiguation
- Extremely long queries (>200 characters)
- Queries with typos and colloquial expressions
- Questions about recently amended regulations

**Rationale:** The system should handle adversarial inputs gracefully. Current evaluation only tests "happy path" queries.

**Constraints:**
- At least 15% of generated queries must be adversarial
- Each adversarial query must have a labeled expected behavior (reject, clarify, partial answer)

---

### Module 2: Extended Quality Metrics

#### EARS-U-004: Response Latency Measurement

**When** evaluating any query,
**the system shall** measure and record wall-clock response time,
calculating p50, p95, and p99 percentiles per persona, per difficulty tier, and overall.

**Rationale:** A correct but slow answer degrades user experience. Current evaluation ignores performance entirely.

**Thresholds:**
- p50 < 3 seconds (simple queries)
- p95 < 8 seconds (complex queries)
- p99 < 15 seconds (all queries)

#### EARS-U-005: Answer Consistency Check

**When** the evaluation runs with `--consistency` flag,
**the system shall** execute each query 3 times and compute answer similarity (cosine similarity of response embeddings),
flagging any query where similarity < 0.85 as "inconsistent."

**Rationale:** Users expect stable answers for the same question. LLM non-determinism can cause confusing variations.

**Constraints:**
- Temperature must be set to 0 for consistency tests
- Report inconsistent queries with diff highlights

#### EARS-U-006: Citation Existence Verification

**When** a response includes regulation citations (e.g., "제3조 제2항"),
**the system shall** verify that the cited article actually exists in the vector database,
calculating a citation_existence_rate metric.

**Rationale:** The current "Citations" metric only checks if citations are present, not if they are real. Hallucinated citation numbers are a known LLM failure mode.

**Thresholds:**
- citation_existence_rate >= 0.95

#### EARS-U-007: Answer Readability Score

**When** evaluating responses,
**the system shall** compute a readability score based on:
- Structural organization (headers, bullet points, numbered lists presence)
- Response length appropriateness (not too short for complex queries, not too long for simple ones)
- Korean language quality (no mixed encoding, proper sentence endings)

**Rationale:** Well-structured answers are more useful even if content is identical.

**Thresholds:**
- readability_score >= 0.70

---

### Module 3: System Health Radar

#### EARS-U-008: Code Quality Scan

**When** the evaluation system runs with `--health` or `--full` flag,
**the system shall** scan the `src/` directory and report:
- Count of bare `except: pass` blocks (target: 0)
- Count of TODO/FIXME/HACK comments (track trend)
- Count of hardcoded magic numbers outside config.py (target: decreasing)
- Longest function by line count (target: < 50 lines)

**Rationale:** Code quality issues cause subtle bugs that RAG quality tests cannot detect. A correct answer today may break silently tomorrow due to swallowed exceptions.

**Constraints:**
- Use AST parsing or regex for Python-specific patterns
- Store results in `data/evaluations/health_scan.json` for trend tracking
- Display delta from previous scan

#### EARS-U-009: Test Coverage Delta

**When** the evaluation system runs with `--health` or `--full` flag,
**the system shall** read `coverage.json` (or run `pytest --cov` if stale),
calculate coverage delta from the previous evaluation, and flag modules with < 80% coverage.

**Rationale:** Test coverage regression indicates growing risk. The evaluation should track this independently of CI/CD.

**Thresholds:**
- Overall coverage >= 85%
- No module below 80%
- Coverage delta must be non-negative (no regression)

#### EARS-U-010: Configuration Drift Detection

**When** the evaluation system runs with `--health` or `--full` flag,
**the system shall** verify:
- All required environment variables are set (non-empty)
- No API keys hardcoded in source files
- Config values in `config.py` match expected ranges
- Dependencies have no known critical vulnerabilities (via `pip audit` or equivalent)

**Rationale:** Misconfiguration causes silent failures. Previous evaluation history shows 0.5 scores from API key issues that went undetected.

**Constraints:**
- Report as WARNING (not FAIL) to avoid blocking evaluation
- Store configuration snapshot for drift comparison

---

### Module 4: Adaptive Difficulty Progression

#### EARS-U-011: Difficulty Tier System

**The system shall** categorize all test queries into 5 difficulty tiers:

| Tier | Description | Example |
|------|-------------|---------|
| L1 | Single fact lookup | "휴학 신청 기간은?" |
| L2 | Multi-fact synthesis from single regulation | "장학금 자격요건과 신청 절차는?" |
| L3 | Cross-regulation synthesis | "학칙과 장학규정의 성적 기준이 다른가?" |
| L4 | Adversarial/edge case | "존재하지 않는 규정 조항 알려줘" |
| L5 | Multi-turn with context dependency | "앞서 말한 규정의 예외 조항은?" |

**Rationale:** Flat difficulty means once all tests pass, there's nowhere to go. Tiered difficulty ensures continuous improvement pressure.

#### EARS-U-012: Automatic Difficulty Escalation

**When** the pass rate for a given tier reaches 95% or above across two consecutive evaluation runs,
**the system shall** automatically:
1. Lock the current tier as "mastered"
2. Generate 50% more queries at the next difficulty tier
3. Report the new frontier tier as the current challenge

**When** no tier has been fully mastered,
**the system shall** focus evaluation resources on the lowest unmastered tier first.

**Rationale:** This ensures the evaluation system always has a "next frontier" and can never declare "nothing to improve."

**Constraints:**
- Mastery state persisted in `data/evaluations/difficulty_state.json`
- Manual override: `--tier L3` to force specific tier testing
- Display mastery progress: "L1 ✓ | L2 ✓ | L3 78% | L4 -- | L5 --"

---

### Module 5: Improvement Roadmap Engine

#### EARS-U-013: Failure Pattern Clustering

**When** evaluation completes with any failures,
**the system shall** cluster failures by root cause pattern:
- Retrieval failure (relevant regulation not found)
- Synthesis failure (information found but poorly combined)
- Citation failure (wrong or hallucinated citations)
- Classification failure (query misrouted)
- Latency failure (correct but too slow)
- Consistency failure (different answers for same query)
- System health failure (code quality, coverage regression)

**Rationale:** Clustered failures enable targeted fixes. Individual failure reports don't show systemic patterns.

**Constraints:**
- Minimum cluster size: 3 failures for pattern recognition
- Each cluster includes: root cause, affected queries, suggested fix

#### EARS-U-014: Prioritized Improvement Roadmap

**When** failure clustering completes,
**the system shall** generate a prioritized improvement roadmap with:
- Priority ranking based on: frequency × impact × fixability
- Specific code locations (file:line) where changes are needed
- Estimated complexity (small/medium/large)
- Suggested SPEC title for each improvement

**Rationale:** The evaluation should not just measure quality but actively guide next improvements. This closes the loop between evaluation and development.

**Output format:**
```markdown
## Improvement Roadmap (Generated: {date})

1. [HIGH] Fix Self-RAG keyword gap for 복무 terms
   - Impact: 5 queries affected
   - Location: src/rag/infrastructure/self_rag.py:36-70
   - Complexity: Small
   - Suggested SPEC: SPEC-RAG-QUALITY-015

2. [MEDIUM] Add response latency caching for common queries
   - Impact: p95 latency 12s → target 8s
   - Location: src/rag/infrastructure/cache.py
   - Complexity: Medium
   - Suggested SPEC: SPEC-RAG-PERF-001
```

#### EARS-U-015: Quality Trend Analysis

**When** the evaluation system runs with `--trend` flag or when 3+ historical evaluation results exist,
**the system shall** generate a trend analysis showing:
- Score trends per metric over time (improving/stable/regressing)
- Mastered tiers progression over time
- System health trend (code quality, coverage)
- Statistical significance of changes (reject noise, highlight real improvements)

**Rationale:** Single-point measurements can't distinguish improvement from noise. Trend analysis provides confidence in progress.

**Constraints:**
- Requires minimum 3 historical evaluation JSONs
- Use simple moving average for smoothing
- Flag statistically significant changes (>2 standard deviations)

---

### Non-Functional Requirements

#### EARS-E-001: Backward Compatibility

**The system shall** maintain full backward compatibility with the existing evaluation:
- `--quick` flag continues to run only original 30 queries with 4 metrics
- New features activated via explicit flags: `--generate`, `--health`, `--consistency`, `--trend`, `--full`
- `--full` activates all new features together

#### EARS-E-002: Performance Budget

**The system shall** complete a `--full` evaluation within 30 minutes,
and a `--quick` evaluation within 5 minutes.

#### EARS-E-003: Result Persistence

**The system shall** save all evaluation results in machine-readable JSON format at
`data/evaluations/eval_{timestamp}.json`, including all new metrics, health scan results, and difficulty state.

#### EARS-E-004: Never "Nothing To Improve"

**The system shall** always include at least one improvement recommendation in its output,
even when all quality metrics pass. If all queries pass at all tiers, the system shall recommend:
- New difficulty tier generation
- System health improvements
- Performance optimization opportunities
- Unexplored query categories

**Rationale:** This is the core requirement. The evaluation system must never declare work complete.
