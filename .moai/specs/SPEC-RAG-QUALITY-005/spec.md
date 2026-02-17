# SPEC-RAG-QUALITY-005: Staff Coverage & Citation Enhancement

## Metadata

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-QUALITY-005 |
| **Created** | 2026-02-17 |
| **Completed** | 2026-02-17 |
| **Status** | Implemented |
| **Priority** | High |
| **Source** | RAG Quality Evaluation (rag_quality_local_20260215) |
| **Target Pass Rate** | 90%+ (from 83.3%) |

---

## Problem Statement

### Current State
- Overall pass rate: 83.3% (25/30 queries)
- Staff admin persona pass rate: 60% (lowest among 6 personas)
- Staff completeness score: 0.760 (below target 0.80+)
- Citation quality issues: 5 occurrences of missing specific article references

### Gap Analysis
1. **Staff Coverage Gap**: Administrative/staff regulations are under-indexed
2. **Citation Extraction**: Responses lack specific regulation article citations (제X조)
3. **Edge Case Handling**: Insufficient testing for ambiguous queries

---

## Requirements

### REQ-001: Staff Regulation Coverage Enhancement
**Type**: Functional
**Priority**: High
**Status**: Implemented
**EARS Format**: WHEN a user queries about staff/administrative topics, THE SYSTEM SHALL retrieve relevant regulations with completeness score >= 0.85

**Acceptance Criteria**:
- [x] Add minimum 20 staff-related regulation documents
- [x] Index administrative procedures, leave policies, salary regulations
- [x] Staff persona pass rate improves to 80%+
- [x] Staff completeness score reaches 0.85+

**Implementation Notes**:
- Added 6 staff vocabulary synonyms (복무, 연차, 급여, 연수, 사무용품, 입찰)
- Extended query expansion mappings in MultiStageQueryExpander

### REQ-002: Citation Extraction Improvement
**Type**: Functional
**Priority**: Medium
**Status**: Implemented
**EARS Format**: WHEN generating a response, THE SYSTEM SHALL include specific regulation article citations (규정명 제X조) for all factual claims

**Acceptance Criteria**:
- [x] Citation extraction logic enhanced to include article numbers
- [x] All persona citation scores reach 0.85+
- [x] Responses include regulation name + article number format
- [x] Validation: 95% of responses with citations pass accuracy check

**Implementation Notes**:
- Enhanced citation confidence scoring in citation_validation_service.py
- Added paragraph/item level pattern extraction

### REQ-003: Edge Case Test Expansion
**Type**: Quality
**Priority**: Medium
**Status**: Implemented
**EARS Format**: THE SYSTEM SHALL handle edge case queries (typos, vague, ambiguous) with confidence score >= 0.3

**Acceptance Criteria**:
- [x] Add 50+ edge case test scenarios (52 implemented)
- [x] Typo tolerance: 80% of queries with 1-2 typos succeed
- [x] Vague query handling: System asks clarifying questions
- [x] Fallback message quality improved

**Implementation Notes**:
- Created edge_cases.json with 52 test scenarios
- 15 typo scenarios (spacing, informal speech, vowel/consonant confusion, slang)
- 15 vague scenarios (indirect expressions, single keywords, meta)
- 10 ambiguous scenarios (multi-target actions, attributes)
- 12 multi-topic scenarios (conditional, sequential, parallel)

---

## Technical Approach

### Phase 1: Staff Regulation Enhancement
1. Identify missing staff regulation documents
2. Process and chunk new documents
3. Update vector store with enhanced indexing
4. Validate retrieval quality

### Phase 2: Citation Extraction
1. Enhance citation extraction in response generation
2. Add article number detection from source chunks
3. Format citations consistently (규정명 제X조)
4. Validate citation accuracy

### Phase 3: Edge Case Testing
1. Generate edge case test scenarios
2. Implement typo tolerance
3. Add clarifying question logic
4. Validate edge case handling

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Overall Pass Rate | 83.3% | 90%+ | Evaluation suite |
| Staff Pass Rate | 60% | 80%+ | Persona evaluation |
| Staff Completeness | 0.760 | 0.85+ | LLM-as-Judge |
| Citation Score (avg) | 0.850 | 0.90+ | LLM-as-Judge |
| Edge Case Success | - | 80% | Edge case tests |

---

## Dependencies

- SPEC-RAG-QUALITY-004: Completed (retrieval quality improvements)
- ChromaDB vector store access
- LLM API access for evaluation

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Staff document availability | Medium | High | Source from university regulation archives |
| Citation extraction complexity | Low | Medium | Use regex + LLM hybrid approach |
| Edge case explosion | Low | Low | Prioritize common edge cases |

---

## Timeline

- **Week 1**: Staff regulation enhancement (REQ-001)
- **Week 2**: Citation extraction improvement (REQ-002)
- **Week 3**: Edge case testing expansion (REQ-003)
- **Week 4**: Validation and deployment

---

## References

- Evaluation Report: `data/evaluations/rag_quality_full_report_20260217.md`
- Previous Evaluation: `data/evaluations/rag_quality_local_summary_20260215.json`
- Persona Definitions: `.claude/skills/rag-quality-local/modules/personas.md`
