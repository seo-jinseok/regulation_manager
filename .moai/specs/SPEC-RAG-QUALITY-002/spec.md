# SPEC: RAG Quality Comprehensive Improvement

**SPEC ID:** SPEC-RAG-QUALITY-002
**Title:** RAG Quality Comprehensive Improvement (RAG 품질 종합 개선)
**Created:** 2026-02-15
**Status:** Planned
**Priority:** High
**Related SPECs:** SPEC-RAG-QUALITY-001 (Previous baseline improvement)
**Lifecycle Level:** spec-anchored (maintained alongside implementation)

---

## Problem Analysis

### Current State (2026-02-15 Evaluation)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Pass Rate | 83.3% | 75%+ | PASS |
| Avg Score | 0.863 | 0.80+ | PASS |
| Staff Pass Rate | 60% | 75%+ | FAIL |
| Citation Score | 0.850 | 0.90+ | BELOW TARGET |
| Completeness | 0.815 | 0.88+ | BELOW TARGET |
| Edge Case Tests | 0 | 10+ | MISSING |

### Baseline Comparison

- Previous (2026-02-13): 43.3% pass rate, 0.795 avg score
- Current (2026-02-15): 83.3% pass rate, 0.863 avg score
- Improvement: +40% pass rate, +0.068 score

### Critical Issues Identified

1. **ISSUE-001 (Medium):** Staff-related queries show lower completeness (0.760 vs target 0.85+)
2. **ISSUE-002 (Low):** Some responses lack specific article citations
3. **ISSUE-003 (Medium):** No edge case test coverage for typos, ambiguous terms

### Root Cause Analysis (Five Whys)

1. **Why Staff Pass Rate Low?** Staff regulation terms not properly indexed
2. **Why Not Indexed?** Synonym mapping lacks staff-specific terminology
3. **Why No Synonyms?** Dictionary manager focused on student/faculty terms
4. **Why Focus Limited?** Original persona testing didn't include staff scenarios
5. **Root Cause:** Dictionary configuration lacks comprehensive staff terminology coverage

---

## Environment

### Technology Stack

- Python 3.13+
- Dictionary Manager: `src/rag/infrastructure/dictionary_manager.py`
- Citation Validator: `src/rag/domain/citation/citation_validator.py`
- Multi-hop Handler: `src/rag/application/multi_hop_handler.py`
- Test Templates: `src/rag/automation/infrastructure/test_scenario_templates.py`
- Chat Logic: `src/rag/interface/chat_logic.py`

### Configuration Files

- `data/config/intents.json` - Intent definitions
- `data/config/synonyms.json` - Synonym mappings

### Dependencies

- SPEC-RAG-QUALITY-001 (completed baseline improvements)
- Existing RAG infrastructure components

---

## Assumptions

| Assumption | Confidence | Validation Method |
|------------|------------|-------------------|
| Staff regulations exist in database | High | Run `regulation sync` verification |
| Synonym expansion improves retrieval | High | A/B testing with/without synonyms |
| Citation format is parseable | High | Existing ArticleNumberExtractor |
| Multi-hop queries are decomposable | Medium | LLM decomposition capability |
| Edge cases are detectable | Medium | Pattern-based detection |

---

## Requirements (EARS Format)

### Phase 1 (P1 - Critical): Staff Regulation Indexing Enhancement

#### Event-Driven Requirements

- **WHEN** a user queries with keywords "직원", "교직원", "행정직", "일반직", "기술직", "행정", "사무"
- **THEN** the system **SHALL** return relevant staff regulations in top 5 search results

**Korean Translation:**
- 사용자가 "직원", "교직원", "행정직", "일반직", "기술직", "행정", "사무" 키워드로 질의하면
- 시스템은 상위 5개 검색 결과에 관련 직원 규정을 반환해야 한다

#### Ubiquitous Requirements

- The system **SHALL** maintain synonym mappings for staff-related terms in `synonyms.json`
- The system **SHALL** include metadata tagging for staff-related regulation chunks

**Korean Translation:**
- 시스템은 `synonyms.json`에 직원 관련 용어의 동의어 매핑을 유지해야 한다
- 시스템은 직원 관련 규정 청크에 메타데이터 태깅을 포함해야 한다

#### State-Driven Requirements

- **IF** staff regulation documents are not synchronized in the database
- **THEN** the system **SHALL** log a warning and suggest running `regulation sync`

**Korean Translation:**
- 직원 규정 문서가 데이터베이스에 동기화되지 않은 경우
- 시스템은 경고를 로깅하고 `regulation sync` 실행을 제안해야 한다

---

### Phase 2 (P2 - High): Citation Quality Enhancement

#### Ubiquitous Requirements

- The system **SHALL** include specific article numbers (e.g., "제15조 제2항") in responses that reference regulations

**Korean Translation:**
- 시스템은 규정을 참조하는 응답에 구체적인 조문 번호(예: "제15조 제2항")를 포함해야 한다

#### State-Driven Requirements

- **IF** a response references a regulation
- **THEN** the response **SHALL** pass citation validation in `CitationValidator`
- **AND** the citation confidence score **SHALL** be >= 0.9

**Korean Translation:**
- 응답이 규정을 참조하는 경우
- 응답은 `CitationValidator`에서 인용 검증을 통과해야 한다
- 인용 신뢰도 점수는 0.9 이상이어야 한다

#### Event-Driven Requirements

- **WHEN** the system generates a response
- **THEN** the LLM prompt **SHALL** include instructions encouraging citation inclusion

**Korean Translation:**
- 시스템이 응답을 생성할 때
- LLM 프롬프트는 인용 포함을 장려하는 지침을 포함해야 한다

---

### Phase 3 (P3 - Medium): Completeness Improvement

#### Event-Driven Requirements

- **WHEN** a multi-intent query is received (e.g., "휴직 신청 사유와 복직 절차")
- **THEN** the system **SHALL** address ALL intents mentioned in the query
- **AND** the completeness score **SHALL** be >= 0.88

**Korean Translation:**
- 다중 의도 쿼리(예: "휴직 신청 사유와 복직 절차")가 수신되면
- 시스템은 쿼리에 언급된 모든 의도를 다루어야 한다
- 완전성 점수는 0.88 이상이어야 한다

#### Ubiquitous Requirements

- The system **SHALL** implement query decomposition in `MultiHopHandler`
- The system **SHALL** validate response completeness before delivery

**Korean Translation:**
- 시스템은 `MultiHopHandler`에 쿼리 분해를 구현해야 한다
- 시스템은 전달 전 응답 완전성을 검증해야 한다

---

### Phase 4 (P4 - Low): Edge Case Test Coverage

#### Event-Driven Requirements

- **WHEN** a query contains typos, ambiguous terms, or non-existent regulation requests
- **THEN** the system **SHALL** provide appropriate guidance or correction suggestions

**Korean Translation:**
- 쿼리에 오타, 모호한 용어, 또는 존재하지 않는 규정 요청이 포함된 경우
- 시스템은 적절한 안내 또는 수정 제안을 제공해야 한다

#### Ubiquitous Requirements

- The system **SHALL** include 10+ edge case test scenarios in the evaluation framework
- Categories: typos (3), ambiguous (3), non-existent (2), out-of-scope (2)

**Korean Translation:**
- 시스템은 평가 프레임워크에 10개 이상의 엣지 케이스 테스트 시나리오를 포함해야 한다
- 카테고리: 오타(3), 모호한 용어(3), 존재하지 않는 규정(2), 범위 외(2)

#### Unwanted Behavior Requirements

- The system **SHALL NOT** crash or return empty responses for edge case queries
- The system **SHALL NOT** provide misleading information for ambiguous queries

**Korean Translation:**
- 시스템은 엣지 케이스 쿼리에 대해 충돌하거나 빈 응답을 반환해서는 안 된다
- 시스템은 모호한 쿼리에 대해 오해의 소지가 있는 정보를 제공해서는 안 된다

---

## Specifications

### Component Changes

#### 1. Dictionary Manager Enhancement

**File:** `src/rag/infrastructure/dictionary_manager.py`

**Changes:**
- Add staff-related synonym entries
- Implement automatic conflict detection for new terms
- Add metadata tagging for regulation chunks

**New Synonym Entries:**
```json
{
  "term": "직원",
  "synonyms": ["교직원", "행정직", "일반직", "기술직", "행정인", "사무직", "일반행정직"],
  "context": "staff_regulation"
}
```

#### 2. Citation Validator Enhancement

**File:** `src/rag/domain/citation/citation_validator.py`

**Changes:**
- Add citation confidence scoring
- Implement citation format validation
- Add specific article number validation

#### 3. Multi-hop Handler Enhancement

**File:** `src/rag/application/multi_hop_handler.py`

**Changes:**
- Implement query decomposition for multi-intent queries
- Add completeness validation
- Enhance retrieval for multiple sub-queries

#### 4. Test Scenario Templates Enhancement

**File:** `src/rag/automation/infrastructure/test_scenario_templates.py`

**Changes:**
- Add 10+ edge case test scenarios
- Categories: typos, ambiguous, non-existent, out-of-scope

#### 5. Chat Logic Prompt Enhancement

**File:** `src/rag/interface/chat_logic.py`

**Changes:**
- Update LLM prompt templates to encourage citation inclusion
- Add citation format instructions

---

## Constraints

### Technical Constraints

- No database schema changes
- No new LLM provider integration
- No UI/UX changes
- Existing API interfaces must remain compatible

### Performance Constraints

- Staff query response time: < 300ms
- Multi-hop query processing: < 2s
- Edge case handling: < 500ms

### Quality Constraints

- Test coverage: >= 85%
- Citation validation accuracy: >= 95%
- Completeness validation accuracy: >= 90%

---

## Out of Scope

1. New LLM provider integration
2. Database schema changes
3. UI/UX modifications
4. New API endpoints
5. Authentication/authorization changes
6. Performance optimization for large-scale deployment

---

## Risks and Mitigation

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Staff synonyms conflict with existing terms | Medium | Low | Use DictionaryManager conflict detection |
| Citation extraction accuracy drops | Medium | Medium | A/B testing before deployment |
| Multi-hop query decomposition fails | Low | Medium | Fallback to single-hop processing |
| Edge cases increase false positives | Low | Low | Comprehensive test coverage |

---

## Success Metrics

| Metric | Current | Target | Measurement Method |
|--------|---------|--------|-------------------|
| Staff Pass Rate | 60% | 85% | Persona-based evaluation |
| Citation Score | 0.850 | 0.920 | Citation validation |
| Completeness Score | 0.815 | 0.880 | Multi-intent query evaluation |
| Edge Case Tests | 0 | 10+ | Test scenario count |
| Overall Pass Rate | 83.3% | 90% | Full evaluation suite |

---

## Traceability

**TAG:** SPEC-RAG-QUALITY-002

**Related Files:**
- `src/rag/infrastructure/dictionary_manager.py` (modify)
- `src/rag/domain/citation/citation_validator.py` (modify)
- `src/rag/application/multi_hop_handler.py` (modify)
- `src/rag/automation/infrastructure/test_scenario_templates.py` (modify)
- `src/rag/interface/chat_logic.py` (modify)

**Configuration Files:**
- `data/config/synonyms.json` (update)
- `data/config/intents.json` (update)

---

**Document Status:** Complete
**Review Required:** Yes
**Implementation Ready:** Pending approval
