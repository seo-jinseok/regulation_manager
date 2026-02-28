# SPEC-RAG-QUALITY-013: RAG Pipeline Quality Improvement — 16.7% → 80%+ Pass Rate

## 1. Overview

### Problem Statement

After initial bug fixes (SPEC-RAG-QUALITY-012: `self.config` AttributeError, FaithfulnessValidator over-filtering), the regulation Q&A system passes only **5/30 evaluation queries (16.7%)**. Target is **80%+ pass rate (24/30)**.

Deep analysis with Sequential Thinking identified **six root causes** across the pipeline:

| Root Cause | Affected Metric | Severity | Example |
|---|---|---|---|
| Search Irrelevance | context_relevance (gap -0.203) | Critical | "휴학 방법"에 비품/구매 규정 검색 |
| LLM Output Format | completeness (gap -0.238) | Critical | 영어 분석 형태 출력, 한국어 답변 미제공 |
| CoT Leakage | completeness, accuracy | High | Step 1-6, Role analysis가 최종 답변에 노출 |
| Citation Non-compliance | citations (gap -0.058) | Medium | 「규정명」 제X조 형식 미준수 |
| Completeness Gaps | completeness (gap -0.238) | High | 5개 검색 문서 중 1개만 사용 |
| Persona Inequity | overall (0.388~0.760) | Medium | staff-admin 최저, international 0% pass |

### Current Performance Baseline

```
Evaluation ID: rag_quality_20260228_120932
Total Queries: 30 | Passed: 5 | Failed: 25 | Pass Rate: 16.7%

Metric      | Average | Threshold | Gap
------------|---------|-----------|-------
Overall     | 0.625   | 0.800     | -0.175
Accuracy    | 0.742   | 0.850     | -0.108
Completeness| 0.512   | 0.750     | -0.238 ← WORST
Citations   | 0.642   | 0.700     | -0.058 ← CLOSEST
Ctx Relev   | 0.547   | 0.750     | -0.203

Persona      | Avg Score | Pass Rate
-------------|-----------|----------
parent       | 0.760     | 20.0%
student-ug   | 0.733     | 20.0%
professor    | 0.656     | 20.0%
student-grad | 0.637     | 20.0%
international| 0.576     | 0.0%
staff-admin  | 0.388     | 20.0%
```

### Target

- Pass Rate: **≥ 80%** (24/30 queries)
- All persona groups: **≥ 60%** pass rate
- Average context_relevance: **≥ 0.70**
- Average completeness: **≥ 0.70**
- Average citations: **≥ 0.70**

---

## 2. Root Cause Analysis

### RC-1: Search Irrelevance (Critical)

**Evidence**: "검색된 문서 관련성 낮음" — 가장 빈번한 실패 패턴 (5회 이상)

- "등록금 납부 기한" 쿼리 → 재수강규정, 수강신청규정 검색 (등록금 관련 0건)
- "비자 납부 절차" 쿼리 → 비품 지급기준 검색
- "장학금 신청" 쿼리 → 비품/구매 규정 검색

**Root Cause**: 
- `min_relevance_score = 0.25` — 임계값이 너무 낮아 무관한 문서 통과
- 한국어 행정 용어(신청, 규정, 제출, 기한)가 다양한 규정에 공통으로 등장 → 임베딩 유사도 오탐
- Corrective RAG 동적 임계값(simple:0.3, medium:0.4, complex:0.5)이 불충분

**Impact**: 검색 실패 시 후속 파이프라인 전체 실패 → 8-10개 쿼리에 영향

### RC-2: LLM Output Format (Critical)

**Evidence**: 
- 응답이 `* **Role:** University Regulation Expert`로 시작
- 영어 분석 형태: `* **Input Question:** "..."`, `* **Provided Context:** ...`
- 한국어 최종 답변 미제공 또는 분석 후 부분적 한국어

**Root Cause**:
- glm-4.7-flash 모델의 한국어 지시 따르기 능력 한계
- 현재 프롬프트(~3000 토큰)가 소형 모델에 과부하
- 모델이 시스템 프롬프트를 분석 과제로 해석

**Impact**: completeness 0.0인 쿼리 다수 발생 (답변 자체가 누락)

### RC-3: CoT Leakage (High)

**Evidence**: "내부 사고 과정(Step 1~6)이 최종 응답에 노출", "분석 로그 형태 응답"

**Root Cause**:
- `_strip_cot_from_answer()`의 패턴이 현재 모델 출력 형식과 불일치
- 현재 패턴: `^\d+\.\s+\*\*(?:Analyze|Check|...)`, `Step\s+\d+:`
- 실제 출력: `* **Role:**`, `* **Input Question:**`, `* **Provided Context:**`

**Impact**: 형식 오류로 인한 평가 점수 하락 (5-7개 쿼리)

### RC-4: Citation Non-compliance (Medium)

**Evidence**: "인용 형식 미준수 (규정명 + 제X조)" 6회, "규정 인용 누락" 4회

**Root Cause**:
- 모델이 `reg-0800` 형태의 코드만 출력, 「규정명」 변환 미수행
- 빈 괄호 `[]` 인용 또는 조항 번호 누락
- 기존 `_enhance_answer_citations()` 후처리가 불충분

### RC-5: Completeness Gaps (High)

**Evidence**: "5개 문서 중 4개 핵심 내용 무시", "답변 거부", "정보 부족"

**Root Cause**: 모델이 검색된 문서 전체를 종합하지 않고 첫 번째 문서만 참조

### RC-6: Persona Inequity (Medium)

**Evidence**: staff-admin 평균 0.388 (최저), student-international 0% 통과율

**Root Cause**: 모든 페르소나에 동일한 프롬프트 적용. 행정직의 절차 중심 질문, 유학생의 이중언어 필요에 대응 부족

---

## 3. Requirements (EARS Format)

### WS-1: Search Relevance Hardening

**REQ-001** (Ubiquitous): The system shall filter search results with relevance score below **0.40** before providing them as LLM context.
- Acceptance: No document with score < 0.40 appears in LLM context
- File: `src/rag/application/search_usecase.py` (~line 2730)

**REQ-002** (Event-Driven): When **all** search results have scores below 0.40, the system shall return a structured "information not found" response instead of generating an answer from low-quality context.
- Acceptance: Response includes "규정 정보를 찾을 수 없습니다" message
- Acceptance: No hallucinated answer generated from irrelevant context

**REQ-003** (Ubiquitous): The system shall perform a topic-mismatch detection check comparing query keywords against retrieved document titles/paths, and exclude documents with zero topical overlap.
- Acceptance: "비품 지급기준" excluded for "휴학 방법" queries
- Implementation: New `_filter_topic_relevance()` method

### WS-2: LLM Output Normalization

**REQ-004** (Event-Driven): When LLM output contains English analysis markers (`* **Role:**`, `* **Input Question:**`, `* **Provided Context:**`), the system shall extract only the Korean answer portion.
- Acceptance: Final user-facing answer contains no English analysis markers
- Implementation: New `OutputNormalizer` class

**REQ-005** (Ubiquitous): The system shall provide a compact prompt variant (≤ 800 tokens) that retains core instructions (context boundary, citation format, no-hallucination) for small language models.
- Acceptance: Compact prompt exists in `data/config/prompts.json`
- Acceptance: Compact prompt used when model is detected as < 10B parameters

**REQ-006** (Event-Driven): When the initial LLM response lacks a direct Korean answer (detected as analysis-only format), the system shall retry once with a focused re-prompt: "위 분석을 바탕으로 사용자에게 한국어로 직접 답변해주세요."
- Acceptance: Retry produces Korean answer in ≥ 80% of cases
- Acceptance: Maximum 1 retry per query (no infinite loops)

### WS-3: CoT Stripping Enhancement

**REQ-007** (Ubiquitous): The system shall strip bullet-point analysis format outputs including patterns: `* **Role:**`, `* **Input:**`, `* **Question:**`, `* **Provided Context:**`, `* **Constraint:**`.
- Acceptance: Zero analysis-format markers in final output across all 30 evaluation queries
- File: `src/rag/application/search_usecase.py`, `_COT_PATTERNS` list

**REQ-008** (Event-Driven): When the entire response is in analysis format with no Korean content paragraph, the system shall attempt to extract content from conclusion markers (결론, 답변, 요약, Final Answer).
- Acceptance: Extracted conclusion presented as final answer

### WS-4: Citation Quality Enhancement

**REQ-009** (Ubiquitous): The system shall auto-convert regulation codes (e.g., `reg-0800`) to 「규정명」 format using chunk metadata (`parent_path`, `title`).
- Acceptance: No `reg-XXXX` codes visible in final answer
- File: `src/rag/application/search_usecase.py`, `_enhance_answer_citations()`

**REQ-010** (Event-Driven): When answer references a regulation without article number, the system shall auto-append the article number (제X조) from the source chunk's metadata.
- Acceptance: ≥ 80% of citations include article numbers

**REQ-011** (Ubiquitous): Every answer that references regulation content shall include at least one properly formatted citation in 「규정명」 제X조 format.
- Acceptance: Zero answers with regulation content but no citation
- Implementation: Post-processing citation injection from source metadata

### WS-5: Completeness Improvement

**REQ-012** (Ubiquitous): The system shall synthesize information from **all** relevant retrieved documents, not just the first one. The prompt shall explicitly instruct: "검색된 모든 규정 문서를 종합하여 답변하세요."
- Acceptance: Answers reference content from ≥ 2 source documents when available

**REQ-013** (Event-Driven): When a query asks for procedures or steps (절차, 방법, 순서), the answer shall include **all** procedural steps found across retrieved documents.
- Acceptance: No missing procedural steps that exist in provided context

### WS-6: Per-Persona Optimization

**REQ-014** (State-Driven): While the detected audience is administrative staff, the system shall append persona-specific instructions emphasizing procedural compliance, form references, and approval processes.
- Acceptance: staff-admin avg score ≥ 0.60 (from 0.388)

**REQ-015** (State-Driven): While the detected audience is international student, the system shall include key regulatory terms in both Korean and English (e.g., "휴학 (Leave of Absence)").
- Acceptance: student-international pass rate ≥ 40% (from 0.0%)

---

## 4. Technical Approach

### Implementation Files

| Workstream | Primary Files | Change Type |
|---|---|---|
| WS-1 | `search_usecase.py` | Modify `_select_answer_sources()`, new `_filter_topic_relevance()` |
| WS-2 | New: `domain/llm/output_normalizer.py`, `prompts.json` | New class + prompt variant |
| WS-3 | `search_usecase.py` (`_COT_PATTERNS`) | Add regex patterns |
| WS-4 | `search_usecase.py` (`_enhance_answer_citations()`) | Enhance existing |
| WS-5 | `prompts.json` | Modify prompt text |
| WS-6 | `prompts.json`, `search_usecase.py` | Persona-specific prompt appendix |

### Implementation Priority

```
Phase 1 (Quick Wins — Expected +30%):
  ├── REQ-001: Raise min_relevance_score (0.25 → 0.40)
  ├── REQ-002: Aggressive no-info response for all-low-score results
  ├── REQ-003: Topic mismatch detection
  ├── REQ-007: CoT pattern additions for glm-4.7-flash
  └── REQ-009: Auto-convert reg-XXXX to 「규정명」

Phase 2 (Core Improvements — Expected +20%):
  ├── REQ-004: English analysis extraction
  ├── REQ-005: Compact prompt variant
  ├── REQ-006: Analysis-only retry mechanism
  ├── REQ-008: Conclusion extraction from analysis
  ├── REQ-010: Auto-append article numbers
  └── REQ-011: Mandatory citation injection

Phase 3 (Targeted Optimization — Expected +10%):
  ├── REQ-012: Multi-document synthesis instruction
  ├── REQ-013: Procedural completeness check
  ├── REQ-014: Staff-admin persona prompt
  └── REQ-015: International student bilingual support
```

### Risks

| Risk | Probability | Mitigation |
|---|---|---|
| glm-4.7-flash 모델 한계로 80% 미달 | Medium | Compact prompt + output normalization으로 최대한 보완. 달성 불가 시 모델 업그레이드 경로 권장 |
| min_relevance_score 상향으로 적중률 하락 | Low | Fallback: 0.40 → 0.30 단계적 하향 + max 3 결과 |
| CoT 패턴 추가로 정상 답변 부분 삭제 | Low | 테스트로 검증. 한국어 컨텐츠 우선 보존 |

---

## 5. Acceptance Criteria (Overall)

| Criteria | Current | Target | Method |
|---|---|---|---|
| Pass Rate | 16.7% (5/30) | ≥ 80% (24/30) | `run_rag_quality_eval.py --quick --summary` |
| Min Persona Pass Rate | 0.0% | ≥ 60% per group | Per-persona evaluation |
| Avg Context Relevance | 0.547 | ≥ 0.70 | LLM Judge evaluation |
| Avg Completeness | 0.512 | ≥ 0.70 | LLM Judge evaluation |
| Avg Citations | 0.642 | ≥ 0.70 | LLM Judge evaluation |
| No regression in tests | All pass | All pass | `uv run pytest` |

---

## 6. Dependencies

- SPEC-RAG-QUALITY-012: Bug fixes (self.config, FaithfulnessValidator) — **Completed**
- Evaluation framework: `run_rag_quality_eval.py` — **Available**
- LLM server: LM Studio at `game-mac-studio:1234` with `z-ai/glm-4.7-flash` — **Available**
- ChromaDB: 3,940 documents — **Available**

---

## 7. Out of Scope

- LLM model replacement or fine-tuning
- Embedding model change (jhgan/ko-sbert-sts → BGE-M3 migration)
- ChromaDB re-indexing or chunk restructuring
- Evaluation framework modification (thresholds, judge prompt)
- New persona types beyond existing 6

---

*Created: 2026-02-28*
*Status: Draft*
*Methodology: DDD (ANALYZE-PRESERVE-IMPROVE)*
