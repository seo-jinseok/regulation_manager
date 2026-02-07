# Acceptance Criteria: SPEC-RAG-SEARCH-001

**Created**: 2026-02-07
**SPEC Version**: 1.0

## Overview

This document defines the acceptance criteria for the RAG search system Contextual Recall improvement implementation.

## Success Metrics

### Primary Metrics

| Metric | Current | Minimum Target | Expected Target | Stretch Goal |
|--------|---------|----------------|-----------------|--------------|
| Contextual Recall | 32% | 50% | 55-60% | 65% |
| Faithfulness | 51.7% | 51.7% (maintain) | 55%+ | 60%+ |
| Answer Relevancy | 73.3% | 73.3% (maintain) | 75%+ | 80%+ |
| Response Time | < 500ms | < 500ms | < 500ms | < 400ms |

### Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Test Coverage | 85%+ | All new code must be tested |
| Bug Count | 0 critical | No critical bugs in production |
| User Satisfaction | Maintain | No decrease in satisfaction |

## TAG-Level Acceptance Criteria

### TAG-001: Entity Recognition Enhancement

**Given** the EntityRecognizer is implemented
**When** a query contains regulation sections (조, 항, 호)
**Then** the system SHALL extract section numbers with 95%+ confidence

**Given** the EntityRecognizer is implemented
**When** a query contains procedure keywords (신청, 절차, 방법)
**Then** the system SHALL expand to related procedure terms

**Given** the EntityRecognizer is implemented
**When** a query contains requirement keywords (자격, 요건, 조건)
**Then** the system SHALL expand to related requirement terms

**Given** the EntityRecognizer is implemented
**When** a query contains benefit keywords (혜택, 지급, 지원)
**Then** the system SHALL expand to related benefit terms

**Given** the EntityRecognizer is implemented
**When** a query contains deadline keywords (기한, 마감, 날짜)
**Then** the system SHALL expand to related deadline terms

**Given** the EntityRecognizer is implemented
**When** a query contains hypernym mappings (등록금→학사→행정)
**Then** the system SHALL expand to hierarchical terms

**Acceptance Tests**:
```python
# tests/unit/test_entity_recognizer.py

def test_section_recognition():
    """Test section pattern recognition."""
    recognizer = RegulationEntityRecognizer()
    matches = recognizer.recognize("제15조의 휴학 절차")
    assert any(m.entity_type == EntityType.SECTION for m in matches)
    assert any(m.text == "제15조" for m in matches)

def test_procedure_expansion():
    """Test procedure keyword expansion."""
    recognizer = RegulationEntityRecognizer()
    matches = recognizer.recognize("장학금 신청 방법")
    procedure_matches = [m for m in matches if m.entity_type == EntityType.PROCEDURE]
    assert len(procedure_matches) > 0
    assert "신청" in procedure_matches[0].expanded_terms

def test_requirement_expansion():
    """Test requirement keyword expansion."""
    recognizer = RegulationEntityRecognizer()
    matches = recognizer.recognize("연구년 자격 요건")
    requirement_matches = [m for m in matches if m.entity_type == EntityType.REQUIREMENT]
    assert len(requirement_matches) > 0
    assert "자격" in requirement_matches[0].expanded_terms

def test_no_false_positives():
    """Test that entity recognition doesn't create false positives."""
    recognizer = RegulationEntityRecognizer()
    matches = recognizer.recognize("일반적인 문장입니다")
    assert len(matches) == 0
```

---

### TAG-002: Query Expansion Pipeline

**Given** the MultiStageQueryExpander is implemented
**When** a query is expanded through 3 stages
**Then** the system SHALL apply synonym, hypernym, and procedure expansions

**Given** the MultiStageQueryExpander is implemented
**When** Stage 1 synonym expansion is applied
**Then** the system SHALL add up to 3 relevant synonyms

**Given** the MultiStageQueryExpander is implemented
**When** Stage 2 hypernym expansion is applied
**Then** the system SHALL add up to 2 hierarchical terms

**Given** the MultiStageQueryExpander is implemented
**When** Stage 3 procedure expansion is applied
**Then** the system SHALL add up to 2 procedure terms

**Given** the MultiStageQueryExpander is implemented
**When** the total expanded terms exceed 10
**Then** the system SHALL limit to top 7 most relevant terms

**Given** the MultiStageQueryExpander is implemented
**When** the query is already formal regulation language
**Then** the system SHALL skip expansion

**Acceptance Tests**:
```python
# tests/unit/test_query_expander_v2.py

def test_three_stage_expansion():
    """Test all 3 expansion stages work together."""
    expander = MultiStageQueryExpander(entity_recognizer)
    result = expander.expand("장학금 신청 방법")
    assert len(result.stage1_synonyms) > 0
    assert len(result.stage2_hypernyms) >= 0
    assert len(result.stage3_procedures) > 0
    assert result.final_expanded != result.original_query

def test_expansion_limit():
    """Test expansion is limited to prevent noise."""
    expander = MultiStageQueryExpander(entity_recognizer)
    result = expander.expand("very long query with many keywords")
    term_count = len(result.final_expanded.split())
    assert term_count <= 10

def test_formal_query_skip():
    """Test formal queries skip expansion."""
    expander = MultiStageQueryExpander(entity_recognizer)
    result = expander.expand("교원인사규정 제15조")
    assert result.final_expanded == result.original_query
    assert result.confidence == 1.0

def test_synonym_expansion():
    """Test Stage 1 synonym expansion."""
    expander = MultiStageQueryExpander(entity_recognizer)
    result = expander.expand("장학금")
    assert "장학금 지급" in result.stage1_synonyms or "재정 지원" in result.stage1_synonyms
```

---

### TAG-003: Adaptive Top-K Implementation

**Given** the AdaptiveTopKSelector is implemented
**When** a simple query is classified (single keyword)
**Then** the system SHALL return Top-5

**Given** the AdaptiveTopKSelector is implemented
**When** a medium query is classified (natural question)
**Then** the system SHALL return Top-10

**Given** the AdaptiveTopKSelector is implemented
**When** a complex query is classified (multi-condition)
**Then** the system SHALL return Top-15

**Given** the AdaptiveTopKSelector is implemented
**When** a multi-part query is classified (multiple distinct queries)
**Then** the system SHALL return Top-20

**Given** the AdaptiveTopKSelector is implemented
**When** response time exceeds 500ms
**Then** the system SHALL reduce Top-K

**Acceptance Tests**:
```python
# tests/unit/test_adaptive_top_k.py

def test_simple_query_top5():
    """Test simple queries get Top-5."""
    selector = AdaptiveTopKSelector()
    top_k = selector.select_top_k("휴학")
    assert top_k == 5

def test_medium_query_top10():
    """Test medium queries get Top-10."""
    selector = AdaptiveTopKSelector()
    top_k = selector.select_top_k("휴학 어떻게 하나요")
    assert top_k == 10

def test_complex_query_top15():
    """Test complex queries get Top-15."""
    selector = AdaptiveTopKSelector()
    top_k = selector.select_top_k("장학금 신청 자격 요건")
    assert top_k == 15

def test_multi_part_query_top20():
    """Test multi-part queries get Top-20."""
    selector = AdaptiveTopKSelector()
    top_k = selector.select_top_k("휴학 방법 그리고 복학 절차")
    assert top_k == 20

def test_regulation_name_top5():
    """Test regulation names get Top-5."""
    selector = AdaptiveTopKSelector()
    top_k = selector.select_top_k("교원인사규정")
    assert top_k == 5
```

---

### TAG-004: Multi-Hop Retrieval

**Given** the MultiHopRetriever is implemented
**When** a document contains "제X조" references
**Then** the system SHALL follow the citation to the target regulation

**Given** the MultiHopRetriever is implemented
**When** a document contains "관련 규정" references
**Then** the system SHALL follow the reference to the related regulation

**Given** the MultiHopRetriever is implemented
**When** citation following reaches 2 hops
**Then** the system SHALL stop further traversal

**Given** the MultiHopRetriever is implemented
**When** a citation has low relevance (< 0.5)
**Then** the system SHALL skip that citation

**Given** the MultiHopRetriever is implemented
**When** a citation cycle is detected
**Then** the system SHALL break the cycle

**Acceptance Tests**:
```python
# tests/unit/test_multi_hop_retriever.py

def test_section_citation_following():
    """Test following '제X조' citations."""
    retriever = MultiHopRetriever(mock_vector_store, mock_chunk_store)
    initial_results = ["doc1"]  # doc1 contains "제15조를 참조"
    result = retriever.retrieve("query", initial_results)
    assert "doc15" in result.all_results  # 제15조 document

def test_two_hop_limit():
    """Test max 2 hops limit."""
    retriever = MultiHopRetriever(mock_vector_store, mock_chunk_store)
    initial_results = ["doc1"]  # doc1 -> doc2 -> doc3 -> doc4
    result = retriever.retrieve("query", initial_results)
    assert result.hops_performed <= 2
    assert "doc4" not in result.all_results  # Beyond 2 hops

def test_low_relevance_filtering():
    """Test low relevance citations are skipped."""
    retriever = MultiHopRetriever(mock_vector_store, mock_chunk_store)
    # Mock citation with low relevance
    result = retriever.retrieve("query", ["doc1"])
    assert len(result.hop_results) == 0 or all(
        r != "irrelevant_doc" for r in result.hop_results
    )

def test_cycle_detection():
    """Test citation cycles are broken."""
    retriever = MultiHopRetriever(mock_vector_store, mock_chunk_store)
    # doc1 -> doc2 -> doc1 (cycle)
    result = retriever.retrieve("query", ["doc1"])
    assert result.hops_performed <= 2
    assert len(result.all_results) <= 2  # doc1, doc2 only
```

---

### TAG-005: Integration & Testing

**Given** all components are implemented
**When** the enhanced search system is evaluated
**Then** Contextual Recall SHALL be at least 50%

**Given** all components are implemented
**When** the enhanced search system is evaluated
**Then** Faithfulness SHALL be maintained at 51.7% or improved

**Given** all components are implemented
**When** the enhanced search system is evaluated
**Then** Answer Relevancy SHALL be maintained at 73.3% or improved

**Given** all components are implemented
**When** a search query is processed
**Then** response time SHALL be under 500ms

**Given** all components are implemented
**When** search results are returned
**Then** there SHALL be no critical bugs or crashes

**Acceptance Tests**:
```python
# tests/integration/test_enhanced_search.py

@pytest.mark.asyncio
async def test_end_to_end_search():
    """Test complete enhanced search pipeline."""
    search_service = EnhancedSearchService()
    results = await search_service.search("장학금 신청 방법")

    # Verify results
    assert len(results) > 0
    assert any("신청" in r.text for r in results)
    assert any("절차" in r.text for r in results)

    # Verify no regressions
    assert results[0].score > 0.5  # Relevance threshold

@pytest.mark.asyncio
async def test_contextual_recall_improvement():
    """Test Contextual Recall improvement."""
    evaluator = RAGQualityEvaluator()

    # Test on 30 scenarios
    results = []
    for scenario in TEST_SCENARIOS:
        result = await evaluator.evaluate(
            query=scenario.query,
            answer=scenario.answer,
            contexts=scenario.contexts,
            ground_truth=scenario.ground_truth
        )
        results.append(result.contextual_recall)

    avg_recall = sum(results) / len(results)
    assert avg_recall >= 0.50  # Minimum 50%

@pytest.mark.asyncio
async def test_no_faithfulness_regression():
    """Test Faithfulness is maintained."""
    evaluator = RAGQualityEvaluator()

    results = []
    for scenario in TEST_SCENARIOS:
        result = await evaluator.evaluate(
            query=scenario.query,
            answer=scenario.answer,
            contexts=scenario.contexts
        )
        results.append(result.faithfulness)

    avg_faithfulness = sum(results) / len(results)
    assert avg_faithfulness >= 0.517  # Maintain or improve

@pytest.mark.asyncio
async def test_response_time():
    """Test response time is maintained."""
    search_service = EnhancedSearchService()

    import time
    start = time.time()
    results = await search_service.search("장학금 신청 방법")
    elapsed = (time.time() - start) * 1000

    assert elapsed < 500  # Under 500ms
```

---

## End-to-End Scenarios

### Scenario 1: Scholarship Application Query

**Given** a user searches for "장학금 신청 방법"
**When** the enhanced search processes the query
**Then** the system SHALL:
- Recognize "장학금" as benefit entity
- Recognize "신청" and "방법" as procedure entities
- Expand to related terms (장학금 지급, 절차, 서류)
- Use appropriate Top-K (15 for complex query)
- Return results covering application procedures

**Acceptance Criteria**:
- Results include both scholarship and procedure information
- No results about enrollment/registration only
- At least 3 relevant documents retrieved

### Scenario 2: Research Year Eligibility

**Given** a user searches for "연구년 자격 요건"
**When** the enhanced search processes the query
**Then** the system SHALL:
- Recognize "연구년" as entity
- Recognize "자격" and "요건" as requirement entities
- Expand to related terms (안식년, 교원연구년, 조건, 기준)
- Follow citations to related regulations if referenced
- Return results covering eligibility details

**Acceptance Criteria**:
- Results include eligibility criteria
- Results include research year duration
- At least 2 relevant documents retrieved

### Scenario 3: TA Working Hours

**Given** a user searches for "조교 근무 시간"
**When** the enhanced search processes the query
**Then** the system SHALL:
- Recognize "조교" as entity
- Expand to related terms (교육조교, 연구조교)
- Search for working hours information
- Return results covering benefits and working conditions

**Acceptance Criteria**:
- Results include working hours information
- Results include benefit information
- At least 2 relevant documents retrieved

---

## Performance Benchmarks

### Latency Requirements

| Operation | Target | Maximum |
|-----------|--------|---------|
| Entity Recognition | < 10ms | 20ms |
| Query Expansion | < 50ms | 100ms |
| Top-K Selection | < 5ms | 10ms |
| Multi-Hop Retrieval | < 100ms | 200ms |
| Total Search Time | < 500ms | 1000ms |

### Quality Requirements

| Component | Coverage Target |
|-----------|----------------|
| EntityRecognizer | 90% |
| MultiStageQueryExpander | 85% |
| AdaptiveTopKSelector | 85% |
| MultiHopRetriever | 80% |
| Integration Tests | 75% |

---

## Regression Testing

### Characterization Tests

**Given** existing search behavior is characterized
**When** new features are implemented
**Then** existing behavior SHALL be preserved

**Characterization Test Template**:
```python
def test_existing_behavior_preserved():
    """Ensure existing search behavior is not broken."""
    # Use existing test queries
    existing_queries = [
        "교원인사규정",
        "제15조",
        "휴학",
    ]

    search_service = EnhancedSearchService()

    for query in existing_queries:
        old_results = legacy_search(query)
        new_results = search_service.search(query)

        # Verify top results are similar
        assert new_results[0].id == old_results[0].id or \
               new_results[0].score >= old_results[0].score * 0.9
```

---

## Sign-Off Criteria

### For TAG Completion

- [ ] All unit tests pass (85%+ coverage)
- [ ] All integration tests pass
- [ ] No critical bugs
- [ ] Performance benchmarks met
- [ ] Code review approved
- [ ] Documentation updated

### For SPEC Completion

- [ ] All 5 TAGs completed
- [ ] Contextual Recall ≥ 50% (minimum) or 55%+ (expected)
- [ ] Faithfulness ≥ 51.7% (maintain or improve)
- [ ] Answer Relevancy ≥ 73.3% (maintain or improve)
- [ ] Response Time < 500ms
- [ ] User acceptance testing passed
- [ ] Production deployment successful

---

## Appendix

### Test Data

**Sample Queries for Testing**:
1. "장학금 신청 방법" - Procedure query with multiple entities
2. "연구년 자격 요건" - Requirement query with hypernym expansion
3. "조교 근무 시간" - Benefit query with multiple concepts
4. "교원인사규정 제15조" - Section reference query
5. "휴학 그리고 복학" - Multi-part query

### Test Environment

- Python 3.11+
- ChromaDB with test data
- Mock vector store for unit tests
- Real vector store for integration tests
- LLM-as-Judge evaluation framework

### References

- RAGAS Evaluation Metrics
- pytest Documentation
- Async Testing Best Practices
