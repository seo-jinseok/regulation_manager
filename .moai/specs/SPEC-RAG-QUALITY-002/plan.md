# Implementation Plan: RAG Quality Comprehensive Improvement

**SPEC ID:** SPEC-RAG-QUALITY-002
**Title:** Implementation Plan - RAG 품질 종합 개선
**Created:** 2026-02-15
**Status:** Planned
**Development Mode:** Hybrid (TDD for new code, DDD for modifications)

---

## Implementation Strategy

### Approach Overview

This implementation follows a phased approach with clear priority ordering:

1. **Phase 1 (P1 - Critical):** Staff Regulation Indexing Enhancement
2. **Phase 2 (P2 - High):** Citation Quality Enhancement
3. **Phase 3 (P3 - Medium):** Completeness Improvement
4. **Phase 4 (P4 - Low):** Edge Case Test Coverage

### Development Methodology

| Change Type | Methodology | Rationale |
|-------------|-------------|-----------|
| New synonym entries | Configuration update | No code changes required |
| Dictionary Manager modifications | DDD | Existing module modification |
| Citation Validator enhancements | DDD | Existing module modification |
| Multi-hop Handler improvements | DDD | Existing module modification |
| Edge case test templates | TDD | New test scenarios |
| Prompt template updates | Configuration update | No code changes required |

---

## Milestones

### Milestone 1: Staff Regulation Indexing (Primary Goal)

**Objective:** Improve Staff persona pass rate from 60% to 85%

**Tasks:**

1. **Dictionary Analysis**
   - Audit existing synonym mappings for staff-related terms
   - Identify missing staff terminology
   - Document conflict potential with existing terms

2. **Synonym Mapping Implementation**
   - Add staff-related synonym entries to `data/config/synonyms.json`
   - Implement conflict detection using `DictionaryManager`
   - Add metadata tagging for staff regulation chunks

3. **Verification**
   - Run `regulation sync` to verify document synchronization
   - Execute staff persona evaluation
   - Verify top-5 retrieval for staff queries

**Deliverables:**
- Updated `synonyms.json` with staff terminology
- Staff regulation metadata tags
- Verification report

**Success Criteria:**
- Staff persona pass rate >= 85%
- Staff queries appear in top 5 results

---

### Milestone 2: Citation Quality Enhancement (Secondary Goal)

**Objective:** Improve citation score from 0.850 to 0.920

**Tasks:**

1. **Citation Validator Enhancement**
   - Implement citation confidence scoring
   - Add specific article number validation (e.g., "제15조 제2항")
   - Enhance regulation name matching

2. **Prompt Template Updates**
   - Update LLM prompts to encourage citation inclusion
   - Add citation format instructions
   - Implement citation examples in prompts

3. **Validation Testing**
   - Create citation validation test cases
   - A/B test with/without enhanced prompts
   - Measure citation accuracy improvement

**Deliverables:**
- Enhanced `CitationValidator` with confidence scoring
- Updated prompt templates
- Citation test cases

**Success Criteria:**
- Citation score >= 0.920
- Citation validation accuracy >= 95%

---

### Milestone 3: Completeness Improvement (Secondary Goal)

**Objective:** Improve completeness score from 0.815 to 0.880

**Tasks:**

1. **Multi-hop Handler Enhancement**
   - Implement query decomposition for multi-intent queries
   - Add intent detection for combined queries
   - Enhance retrieval to support multiple sub-queries

2. **Completeness Validation**
   - Implement response completeness scoring
   - Add validation before response delivery
   - Create completeness test scenarios

3. **Integration Testing**
   - Test multi-intent query scenarios
   - Verify all intents are addressed
   - Measure completeness improvement

**Deliverables:**
- Enhanced `MultiHopHandler` with query decomposition
- Completeness validation module
- Multi-intent test scenarios

**Success Criteria:**
- Completeness score >= 0.880
- All multi-intent query intents addressed

---

### Milestone 4: Edge Case Test Coverage (Final Goal)

**Objective:** Add 10+ edge case test scenarios

**Tasks:**

1. **Test Template Creation**
   - Add typo test scenarios (3 cases)
   - Add ambiguous term test scenarios (3 cases)
   - Add non-existent regulation scenarios (2 cases)
   - Add out-of-scope scenarios (2 cases)

2. **Edge Case Handling**
   - Implement typo detection and correction suggestions
   - Add ambiguous query clarification prompts
   - Implement graceful handling for non-existent regulations

3. **Evaluation Integration**
   - Integrate edge case tests into evaluation framework
   - Add edge case scoring to overall metrics
   - Create edge case evaluation report

**Deliverables:**
- Edge case test templates in `test_scenario_templates.py`
- Edge case handling logic
- Edge case evaluation report

**Success Criteria:**
- 10+ edge case test scenarios passing
- Graceful handling of all edge case types

---

## Technical Approach

### File Modification Strategy

#### 1. Dictionary Manager (`dictionary_manager.py`)

**Current State:**
- Manages intents.json and synonyms.json
- Has LLM-based recommendation capability
- Includes conflict detection

**Required Changes:**
```python
# Add staff-specific synonym handling
STAFF_SYNONYMS = {
    "직원": ["교직원", "행정직", "일반직", "기술직"],
    "행정": ["사무", "행정업무", "행정사무"],
    # ... additional entries
}
```

**Approach:**
1. Add `add_staff_synonyms()` method
2. Implement metadata tagging for staff chunks
3. Add staff-specific validation

#### 2. Citation Validator (`citation_validator.py`)

**Current State:**
- Validates citations against source documents
- Detects hallucinated citations
- Has article number extraction

**Required Changes:**
```python
# Add confidence scoring
def validate_with_confidence(self, citation: str) -> tuple[ValidationResult, float]:
    # Enhanced validation with confidence score
    pass
```

**Approach:**
1. Add confidence scoring method
2. Enhance article number format validation
3. Add citation format templates

#### 3. Multi-hop Handler (`multi_hop_handler.py`)

**Current State:**
- Decomposes complex queries into sub-queries
- Has dependency cycle detection
- Supports up to 5 hops

**Required Changes:**
```python
# Add completeness validation
def validate_completeness(self, query: str, response: str) -> float:
    # Check if all intents are addressed
    pass
```

**Approach:**
1. Add intent detection for multi-intent queries
2. Implement completeness scoring
3. Add validation before response delivery

#### 4. Test Scenario Templates (`test_scenario_templates.py`)

**Current State:**
- Has ambiguous query templates
- Has multi-turn scenario templates
- Has edge case templates (emotional, deadline, etc.)

**Required Changes:**
```python
# Add new edge case categories
TYPO_EDGE_CASES = [
    {"query": "휴학 신정", "expected_correction": "휴학 신청"},
    # ... additional cases
]
```

**Approach:**
1. Add typo test scenarios
2. Add ambiguous term scenarios
3. Add non-existent regulation scenarios
4. Add out-of-scope scenarios

---

## Architecture Design

### Component Interaction

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Chat Logic                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Dictionary      │  │ Citation        │                   │
│  │ Manager         │  │ Validator       │                   │
│  │ (Enhanced)      │  │ (Enhanced)      │                   │
│  └─────────────────┘  └─────────────────┘                   │
│           │                    │                             │
│           ▼                    ▼                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Multi-hop Handler                        ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     ││
│  │  │ Query       │  │ Completeness│  │ Intent      │     ││
│  │  │ Decomposer  │  │ Validator   │  │ Detector    │     ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Response (with citations, complete)
```

### Data Flow

1. **Query Processing:**
   - Query received → Dictionary lookup → Synonym expansion
   - Staff terms → Enhanced retrieval

2. **Citation Generation:**
   - LLM response → Citation extraction → Validation
   - Confidence scoring → Response enhancement

3. **Completeness Validation:**
   - Multi-intent detection → Query decomposition
   - Sub-query processing → Completeness scoring

---

## Dependencies

### Internal Dependencies

| Component | Depends On | Type |
|-----------|------------|------|
| Dictionary Manager | synonyms.json, intents.json | Configuration |
| Citation Validator | ArticleNumberExtractor, VectorStore | Module |
| Multi-hop Handler | LLMClient, VectorStore | Service |
| Test Templates | None | Standalone |

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.13+ | Runtime |
| LLM Provider | Current | Query processing |
| Vector Database | Current | Retrieval |

---

## Testing Strategy

### Unit Tests

1. **Dictionary Manager Tests:**
   - Synonym addition
   - Conflict detection
   - Staff term recognition

2. **Citation Validator Tests:**
   - Citation format validation
   - Confidence scoring
   - Hallucination detection

3. **Multi-hop Handler Tests:**
   - Query decomposition
   - Completeness validation
   - Intent detection

### Integration Tests

1. **Staff Query Retrieval:**
   - Staff terms return relevant results
   - Top-5 ranking verification

2. **Citation Quality:**
   - Responses include citations
   - Citations are valid

3. **Completeness:**
   - Multi-intent queries addressed
   - All intents covered

### Evaluation Tests

1. **Persona-based Evaluation:**
   - Staff persona: target 85%+
   - Other personas: maintain 80%+

2. **Edge Case Evaluation:**
   - Typo handling
   - Ambiguous query handling
   - Non-existent regulation handling

---

## Rollback Plan

### Risk Mitigation

1. **Configuration Backup:**
   - Backup `synonyms.json` before modification
   - Backup `intents.json` before modification

2. **Feature Flags:**
   - Enable/disable enhanced retrieval
   - Enable/disable citation validation
   - Enable/disable completeness validation

3. **Rollback Triggers:**
   - Pass rate drops below 80%
   - Response time exceeds 500ms
   - Citation accuracy below 90%

### Rollback Procedure

1. Restore configuration backups
2. Disable enhanced features via flags
3. Verify system stability
4. Investigate root cause

---

## Performance Considerations

### Optimization Points

1. **Synonym Expansion:**
   - Cache expanded queries
   - Limit synonym count per term

2. **Citation Validation:**
   - Batch validation for multiple citations
   - Cache validation results

3. **Completeness Validation:**
   - Parallel sub-query processing
   - Early termination for complete responses

### Resource Usage

| Component | Memory | CPU | I/O |
|-----------|--------|-----|-----|
| Dictionary Manager | Low | Low | Medium |
| Citation Validator | Low | Medium | Low |
| Multi-hop Handler | Medium | High | Medium |
| Test Templates | Low | Low | Low |

---

## Documentation Requirements

### Code Documentation

- All new methods must have docstrings
- Complex logic must have inline comments
- Configuration changes must be documented

### User Documentation

- Update README with new features
- Document new configuration options
- Add edge case handling guide

---

## Next Steps

1. **Approval:** Get user approval for implementation plan
2. **Phase 1 Implementation:** Staff regulation indexing
3. **Evaluation:** Verify Phase 1 success criteria
4. **Phase 2-4 Implementation:** Sequential implementation
5. **Final Evaluation:** Full evaluation suite
6. **Documentation:** Update all documentation

---

**Plan Status:** Complete
**Ready for Implementation:** Pending approval
