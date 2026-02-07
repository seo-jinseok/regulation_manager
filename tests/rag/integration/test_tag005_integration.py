"""
Integration tests for SPEC-RAG-SEARCH-001 TAG-005.

Tests the complete integration of all TAG components:
- TAG-001: Enhanced Entity Recognition
- TAG-002: Multi-Stage Query Expansion
- TAG-003: Adaptive Top-K Selection
- TAG-004: Multi-Hop Retrieval

This test demonstrates the full RAG search pipeline with all enhancements.
"""

from unittest.mock import MagicMock

import pytest

from src.rag.domain.entities import SearchResult
from src.rag.domain.entity.entity_recognizer import RegulationEntityRecognizer
from src.rag.infrastructure.adaptive_top_k import (
    AdaptiveTopKSelector,
    ComplexityAnalysisResult,
    QueryComplexity,
)
from src.rag.infrastructure.multi_hop_retriever import (
    HopRetrievalResult,
    MultiHopRetriever,
)
from src.rag.infrastructure.query_expander_v2 import (
    MultiStageQueryExpander,
)


class TestTAGIntegration:
    """Test integration of all TAG components."""

    @pytest.fixture
    def entity_recognizer(self):
        """Create entity recognizer instance."""
        return RegulationEntityRecognizer()

    @pytest.fixture
    def query_expander(self, entity_recognizer):
        """Create query expander instance."""
        return MultiStageQueryExpander(entity_recognizer)

    @pytest.fixture
    def top_k_selector(self):
        """Create Top-K selector instance."""
        return AdaptiveTopKSelector()

    @pytest.fixture
    def multi_hop_retriever(self):
        """Create multi-hop retriever instance."""
        mock_store = MagicMock()
        return MultiHopRetriever(mock_store)

    def test_full_pipeline_simple_query(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test full pipeline with simple query (REQ-AT-002)."""
        query = "장학금"  # Single keyword

        # Step 1: Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Step 2: Query expansion
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Step 3: Adaptive Top-K
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 5  # SIMPLE query gets Top-5

    def test_full_pipeline_medium_query(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test full pipeline with medium query (REQ-AT-003)."""
        query = "장학금을 어떻게 신청하나요?"

        # Step 1: Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Step 2: Query expansion
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Step 3: Adaptive Top-K
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 10  # MEDIUM query gets Top-10

    def test_full_pipeline_complex_query(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test full pipeline with complex query (REQ-AT-004)."""
        query = "장학금 신청 절차와 구비 서류"

        # Step 1: Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Step 2: Query expansion
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Step 3: Adaptive Top-K
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 15  # COMPLEX query gets Top-15

    def test_full_pipeline_multi_part_query(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test full pipeline with multi-part query (REQ-AT-005)."""
        query = "장학금 신청 방법 그리고 자격 요건"

        # Step 1: Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Step 2: Query expansion
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Step 3: Adaptive Top-K
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 20  # MULTI_PART query gets Top-20

    def test_entity_recognition_to_expansion(self, entity_recognizer, query_expander):
        """Test integration between entity recognition and query expansion."""
        query = "장학금 신청 방법"

        # Recognize entities
        entity_result = entity_recognizer.recognize(query)

        # Get expanded query from entity result
        expanded_query = entity_result.get_expanded_query()

        # Should include original query plus expanded terms
        assert "장학금" in expanded_query
        assert len(expanded_query) >= len(query)

    def test_top_k_with_complexity_analysis(self, top_k_selector):
        """Test Top-K selection with detailed complexity analysis."""
        query = "장학금 신청 절차와 자격 요건"

        # Get full complexity analysis
        analysis = top_k_selector.analyze_complexity(query)

        # Should return detailed analysis
        assert isinstance(analysis, ComplexityAnalysisResult)
        assert analysis.complexity == QueryComplexity.COMPLEX
        assert analysis.top_k == 15
        assert 0 <= analysis.score <= 100
        assert "entity_count" in analysis.factors

    def test_multi_hop_with_mock_results(self, multi_hop_retriever):
        """Test multi-hop retrieval with mock results."""
        query = "제15조 장학금 지급"

        # Create mock initial results
        chunk1 = MagicMock(
            id="doc1", content="제15조에 따라 장학금을 지급한다. 제16조를 참조."
        )
        initial_results = [SearchResult(chunk=chunk1, score=0.9)]

        # Perform multi-hop retrieval
        hop_result = multi_hop_retriever.retrieve(query, initial_results, top_k=10)

        # Should return hop result structure
        assert isinstance(hop_result, HopRetrievalResult)
        assert len(hop_result.original_results) == 1
        assert hop_result.visited_docs is not None

    def test_latency_guardrail_integration(self, top_k_selector):
        """Test latency guardrail affects Top-K selection (REQ-AT-009)."""
        query = "장학금 신청 절차와 자격 요건"

        # Record high latencies
        for _ in range(5):
            top_k_selector.record_latency(600.0)

        # Get Top-K with guardrails enabled
        top_k_with_guardrails = top_k_selector.select_top_k(
            query, latency_guardrails=True
        )
        top_k_without_guardrails = top_k_selector.select_top_k(
            query, latency_guardrails=False
        )

        # With guardrails should be lower or equal
        assert top_k_with_guardrails <= top_k_without_guardrails

    def test_spec_scenario_1_complete_pipeline(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test SPEC scenario 1: 장학금 신청 방법."""
        query = "장학금 신청 방법"

        # Entity recognition (TAG-001)
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Query expansion (TAG-002)
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Adaptive Top-K (TAG-003)
        top_k = top_k_selector.select_top_k(query)
        # Could be MEDIUM (10) or COMPLEX (15) depending on classification
        assert top_k in [10, 15]

    def test_spec_scenario_2_complete_pipeline(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test SPEC scenario 2: 연구년 자격 요건."""
        query = "연구년 자격 요건"

        # Entity recognition (TAG-001)
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Query expansion (TAG-002)
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Adaptive Top-K (TAG-003)
        top_k = top_k_selector.select_top_k(query)
        assert top_k in [10, 15]  # MEDIUM or COMPLEX

    def test_spec_scenario_3_complete_pipeline(
        self, entity_recognizer, query_expander, top_k_selector
    ):
        """Test SPEC scenario 3: 조교 근무 시간."""
        query = "조교 근무 시간"

        # Entity recognition (TAG-001)
        entity_result = entity_recognizer.recognize(query)
        assert entity_result.has_entities

        # Query expansion (TAG-002)
        expansion_result = query_expander.expand(query)
        assert expansion_result.final_expanded

        # Adaptive Top-K (TAG-003)
        top_k = top_k_selector.select_top_k(query)
        assert top_k in [5, 10]  # SIMPLE or MEDIUM


class TestTAGComponentCompatibility:
    """Test compatibility between TAG components."""

    def test_entity_recognizer_output_compatible_with_expander(self):
        """Test that entity recognizer output works with query expander."""
        recognizer = RegulationEntityRecognizer()
        expander = MultiStageQueryExpander(recognizer)

        query = "장학금 신청 방법"

        # Entity recognition
        entity_result = recognizer.recognize(query)

        # Query expansion (should handle entity recognizer output)
        expansion_result = expander.expand(query)

        # Both should complete without errors
        assert entity_result is not None
        assert expansion_result is not None

    def test_top_k_selector_with_expanded_query(self):
        """Test Top-K selector works with expanded queries."""
        recognizer = RegulationEntityRecognizer()
        expander = MultiStageQueryExpander(recognizer)
        selector = AdaptiveTopKSelector()

        original_query = "장학금"
        expansion_result = expander.expand(original_query)

        # Top-K selection should work with both original and expanded
        top_k_original = selector.select_top_k(original_query)
        top_k_expanded = selector.select_top_k(expansion_result.final_expanded)

        # Both should return valid Top-K values
        assert isinstance(top_k_original, int)
        assert isinstance(top_k_expanded, int)

    def test_multi_hop_with_adaptive_top_k(self):
        """Test multi-hop retriever respects adaptive Top-K."""
        mock_store = MagicMock()
        retriever = MultiHopRetriever(mock_store)
        selector = AdaptiveTopKSelector()

        query = "제15조 장학금"
        top_k = selector.select_top_k(query)

        # Create mock results
        chunk = MagicMock(id="doc1", content="제15조 참조")
        initial_results = [SearchResult(chunk=chunk, score=0.9)]

        # Multi-hop should respect Top-K limit
        hop_result = retriever.retrieve(query, initial_results, top_k=top_k)

        # Results should be limited to top_k
        assert len(hop_result.all_results) <= top_k


class TestPerformanceRequirements:
    """Test performance requirements (REQ-PQ-001 ~ REQ-PQ-003)."""

    @pytest.fixture
    def entity_recognizer(self):
        """Create entity recognizer instance."""
        return RegulationEntityRecognizer()

    @pytest.fixture
    def top_k_selector(self):
        """Create Top-K selector instance."""
        return AdaptiveTopKSelector()

    def test_entity_recognition_performance(self, entity_recognizer):
        """Test entity recognition is fast (REQ-PQ-001)."""
        import time

        query = "장학금 신청 절차와 자격 요건 및 구비 서류"

        start = time.time()
        entity_result = entity_recognizer.recognize(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should be very fast (< 10ms for entity recognition)
        assert elapsed_ms < 10
        assert entity_result is not None

    def test_query_expansion_performance(self, entity_recognizer):
        """Test query expansion is fast (REQ-PQ-001)."""
        import time

        query = "장학금 신청 절차와 자격 요건"
        expander = MultiStageQueryExpander(entity_recognizer)

        start = time.time()
        expansion_result = expander.expand(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should be fast (< 50ms for query expansion)
        assert elapsed_ms < 50
        assert expansion_result is not None

    def test_top_k_selection_performance(self, top_k_selector):
        """Test Top-K selection is fast (REQ-PQ-001)."""
        import time

        query = "장학금 신청 절차와 자격 요건 및 구비 서류"

        start = time.time()
        top_k = top_k_selector.select_top_k(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should be very fast (< 5ms for Top-K selection)
        assert elapsed_ms < 5
        assert isinstance(top_k, int)

    def test_complexity_analysis_performance(self, top_k_selector):
        """Test complexity analysis is fast (REQ-PQ-001)."""
        import time

        query = "장학금 신청 절차와 자격 요건"

        start = time.time()
        analysis = top_k_selector.analyze_complexity(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should be fast (< 10ms for complexity analysis)
        assert elapsed_ms < 10
        assert analysis is not None

    def test_full_pipeline_latency_under_500ms(self, entity_recognizer, top_k_selector):
        """Test full pipeline latency under 500ms (REQ-PQ-001)."""
        import time

        query = "장학금 신청 절차와 자격 요건"
        expander = MultiStageQueryExpander(entity_recognizer)

        start = time.time()

        # Entity recognition
        entity_result = entity_recognizer.recognize(query)

        # Query expansion
        expansion_result = expander.expand(query)

        # Adaptive Top-K
        top_k = top_k_selector.select_top_k(query)

        elapsed_ms = (time.time() - start) * 1000

        # Full pipeline should be well under 500ms
        assert elapsed_ms < 100  # Should be much faster than 500ms
        assert entity_result is not None
        assert expansion_result is not None
        assert top_k is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def entity_recognizer(self):
        """Create entity recognizer instance."""
        return RegulationEntityRecognizer()

    @pytest.fixture
    def top_k_selector(self):
        """Create Top-K selector instance."""
        return AdaptiveTopKSelector()

    def test_empty_query_handling(self, entity_recognizer, top_k_selector):
        """Test handling of empty query."""
        query = ""
        expander = MultiStageQueryExpander(entity_recognizer)

        # Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert not entity_result.has_entities

        # Query expansion
        expansion_result = expander.expand(query)
        assert expansion_result.original_query == ""

        # Top-K selection
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 10  # Fallback to MEDIUM

    def test_whitespace_only_query(self, entity_recognizer, top_k_selector):
        """Test handling of whitespace-only query."""
        query = "   "
        expander = MultiStageQueryExpander(entity_recognizer)

        # Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert not entity_result.has_entities

        # Query expansion
        expansion_result = expander.expand(query)
        assert expansion_result.original_query == "   "

        # Top-K selection
        top_k = top_k_selector.select_top_k(query)
        assert top_k == 10  # Fallback to MEDIUM

    def test_very_long_query_handling(self, entity_recognizer, top_k_selector):
        """Test handling of very long query."""
        query = "장학금 " * 1000
        expander = MultiStageQueryExpander(entity_recognizer)

        # Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result is not None

        # Query expansion
        expansion_result = expander.expand(query)
        assert expansion_result is not None

        # Top-K selection
        top_k = top_k_selector.select_top_k(query)
        assert isinstance(top_k, int)
        assert top_k <= 25  # Max limit

    def test_unicode_query_handling(self, entity_recognizer, top_k_selector):
        """Test handling of unicode characters."""
        query = "장학금 ㅏㅏㅏ ㉿"
        expander = MultiStageQueryExpander(entity_recognizer)

        # Entity recognition
        entity_result = entity_recognizer.recognize(query)
        assert entity_result is not None

        # Query expansion
        expansion_result = expander.expand(query)
        assert expansion_result is not None

        # Top-K selection
        top_k = top_k_selector.select_top_k(query)
        assert isinstance(top_k, int)
