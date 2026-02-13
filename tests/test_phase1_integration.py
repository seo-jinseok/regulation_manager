"""
Phase 1 Integration Tests for RAG System Improvements.

Tests query expansion, citation enhancement, and evaluation prompts
without requiring MLX dependencies.

Run with: pytest tests/test_phase1_integration.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.rag.application.query_expansion import (
    QueryExpansionService,
    ExpandedQuery
)
from src.rag.domain.citation.citation_enhancer import (
    CitationEnhancer,
    EnhancedCitation
)
from src.rag.domain.evaluation.prompts import EvaluationPrompts
from src.rag.domain.entities import Chunk, SearchResult
from src.rag.domain.value_objects import Query


class TestQueryExpansion:
    """Test query expansion functionality."""

    @pytest.fixture
    def mock_store(self):
        """Create mock vector store."""
        store = Mock()
        store.search = Mock(return_value=[])
        return store

    @pytest.fixture
    def expansion_service(self, mock_store):
        """Create query expansion service."""
        return QueryExpansionService(store=mock_store)

    def test_expand_query_with_synonyms(self, expansion_service):
        """Test query expansion with academic synonyms."""
        query = "휴학 방법 알려줘"
        expanded = expansion_service.expand_query(
            query,
            max_variants=3,
            method="synonym"
        )

        # Should include original query
        assert any(e.expanded_text == query for e in expanded)

        # Should include synonym variants
        assert len(expanded) > 1
        assert all(isinstance(e, ExpandedQuery) for e in expanded)

        # Check that original is preserved
        original = [e for e in expanded if e.expansion_method == "original"]
        assert len(original) == 1
        assert original[0].confidence == 1.0

    def test_synonym_database_coverage(self, expansion_service):
        """Test that synonym database covers key academic terms."""
        key_terms = ["휴학", "복학", "등록금", "장학금", "성적", "수강신청", "졸업"]

        for term in key_terms:
            assert term in expansion_service.ACADEMIC_SYNONYMS, \
                f"Key term '{term}' not in synonym database"
            assert len(expansion_service.ACADEMIC_SYNONYMS[term]) > 0, \
                f"No synonyms provided for '{term}'"

    def test_english_korean_translation(self, expansion_service):
        """Test English-Korean translation mappings."""
        # Test Korean to English
        query_ko = "tuition 납부 방법"
        expanded_ko = expansion_service.expand_query(query_ko, method="translation")

        # Should include English translation
        assert any("등록금" in e.expanded_text for e in expanded_ko), \
            "Korean to English translation failed"

        # Test English to Korean
        query_en = "scholarship how to apply"
        expanded_en = expansion_service.expand_query(query_en, method="translation")

        # Should include Korean translation
        assert any("장학금" in e.expanded_text for e in expanded_en), \
            "English to Korean translation failed"

    def test_language_detection(self, expansion_service):
        """Test language detection for queries."""
        # Korean
        assert expansion_service._detect_language("휴학 방법") == "ko"

        # English
        assert expansion_service._detect_language("leave of absence") == "en"

        # Mixed
        assert expansion_service._detect_language("tuition 납부") == "mixed"

    def test_query_deduplication(self, expansion_service):
        """Test that duplicate queries are removed."""
        query = "휴학 방법"
        expanded = expansion_service.expand_query(
            query,
            max_variants=10,
            method="synonym"
        )

        # Check no duplicates
        texts = [e.expanded_text for e in expanded]
        assert len(texts) == len(set(texts)), "Duplicate queries found"

    def test_expansion_statistics(self, expansion_service):
        """Test expansion statistics generation."""
        test_queries = [
            "휴학 방법 알려줘",
            "장학금 신청",
            "성적 조회 방법",
            "수강신청 기간",
        ]

        stats = expansion_service.get_expansion_statistics(
            test_queries,
            method="synonym"
        )

        assert stats["total_queries"] == len(test_queries)
        assert stats["avg_variants_per_query"] >= 1.0
        assert "language_distribution" in stats
        assert "method_distribution" in stats


class TestCitationEnhancer:
    """Test citation enhancement functionality."""

    @pytest.fixture
    def citation_enhancer(self):
        """Create citation enhancer."""
        return CitationEnhancer()

    @pytest.fixture
    def sample_chunk(self):
        """Create sample chunk with article info."""
        chunk = Mock(spec=Chunk)
        chunk.id = "test_chunk_1"
        chunk.rule_code = "교원인사규정_제26조"
        chunk.article_number = "제26조"
        chunk.parent_path = ["교원인사규정", "제2장", "제26조"]
        chunk.title = "징계의 절차"
        chunk.text = "징계의결은 재적위원 과반수의 찬성으로 의결한다."
        return chunk

    @pytest.fixture
    def sample_attachment_chunk(self):
        """Create sample chunk for attachment (별표)."""
        chunk = Mock(spec=Chunk)
        chunk.id = "test_chunk_2"
        chunk.rule_code = "직원복무규정_별표1"
        chunk.article_number = "별표1"
        chunk.parent_path = ["직원복무규정"]
        chunk.title = "직원급별 봉급표"
        chunk.text = "|급별|봉급|...|"
        return chunk

    def test_enhance_citation_with_article(self, citation_enhancer, sample_chunk):
        """Test citation enhancement for article."""
        citation = citation_enhancer.enhance_citation(sample_chunk)

        assert citation is not None
        assert citation.regulation == "교원인사규정"
        assert citation.article_number == "제26조"
        assert citation.chunk_id == "test_chunk_1"
        assert citation.confidence == 1.0

    def test_enhance_citation_formatting(self, citation_enhancer, sample_chunk):
        """Test citation formatting."""
        citation = citation_enhancer.enhance_citation(sample_chunk)
        formatted = citation.format()

        assert "교원인사규정" in formatted
        assert "제26조" in formatted
        assert "「" in formatted and "」" in formatted

    def test_attachment_citation_formatting(
        self,
        citation_enhancer,
        sample_attachment_chunk
    ):
        """Test attachment citation formatting (별표)."""
        citation = citation_enhancer.enhance_citation(sample_attachment_chunk)
        formatted = citation.format()

        # Attachments don't use regulation quotes
        assert "별표1" in formatted
        assert "직원급별 봉급표" in formatted

    def test_citation_validation(self, citation_enhancer, sample_chunk):
        """Test rule code validation."""
        # Valid rule code
        assert citation_enhancer.validate_rule_code(sample_chunk) is True

        # Invalid rule code (missing underscore)
        invalid_chunk = Mock(spec=Chunk)
        invalid_chunk.rule_code = "교원인사규정제26조"
        assert citation_enhancer.validate_rule_code(invalid_chunk) is False

    def test_regulation_title_extraction(self, citation_enhancer, sample_chunk):
        """Test regulation title extraction."""
        title = citation_enhancer.extract_regulation_title(sample_chunk)

        assert title == "교원인사규정"

    def test_citation_deduplication(self, citation_enhancer, sample_chunk):
        """Test citation deduplication."""
        citations = [
            citation_enhancer.enhance_citation(sample_chunk),
            citation_enhancer.enhance_citation(sample_chunk),
        ]

        unique = citation_enhancer.deduplicate_citations(citations)

        assert len(unique) == 1  # Should remove duplicates

    def test_citation_sorting(self, citation_enhancer):
        """Test citation sorting by article number."""
        chunk1 = Mock(spec=Chunk)
        chunk1.rule_code = "규정_제10조"
        chunk1.article_number = "제10조"
        chunk1.parent_path = ["규정"]
        chunk1.id = "chunk1"
        chunk1.title = "규정1"
        chunk1.text = "text1"

        chunk2 = Mock(spec=Chunk)
        chunk2.rule_code = "규정_제2조"
        chunk2.article_number = "제2조"
        chunk2.parent_path = ["규정"]
        chunk2.id = "chunk2"
        chunk2.title = "규정2"
        chunk2.text = "text2"

        chunk3 = Mock(spec=Chunk)
        chunk3.rule_code = "규정_제15조"
        chunk3.article_number = "제15조"
        chunk3.parent_path = ["규정"]
        chunk3.id = "chunk3"
        chunk3.title = "규정3"
        chunk3.text = "text3"

        citations = [
            citation_enhancer.enhance_citation(chunk1),
            citation_enhancer.enhance_citation(chunk2),
            citation_enhancer.enhance_citation(chunk3),
        ]

        sorted_citations = citation_enhancer.sort_by_article_number(citations)

        # Should be sorted: 제2조, 제10조, 제15조
        assert "제2조" in sorted_citations[0].article_number
        assert "제10조" in sorted_citations[1].article_number
        assert "제15조" in sorted_citations[2].article_number


class TestEvaluationPrompts:
    """Test evaluation prompt functionality."""

    def test_accuracy_prompt_formatting(self):
        """Test accuracy evaluation prompt formatting."""
        query = "휴학 신청 방법 알려줘"
        answer = "휴학 신청은 학기 시작 전에 교육과정팀에 방문하여 신청서를 제출해야 합니다."
        context = [
            {
                "title": "학칙 제26조",
                "text": "휴학은 학기 개시 30일 전까지 신청하여야 한다.",
                "score": 0.95
            }
        ]

        system_prompt, user_prompt = EvaluationPrompts.format_accuracy_prompt(
            query=query,
            answer=answer,
            context=context
        )

        assert "정확성" in system_prompt
        assert query in user_prompt
        assert answer in user_prompt
        assert "학칙 제26조" in user_prompt

    def test_hallucination_detection_prompt(self):
        """Test hallucination detection prompt formatting."""
        text = "휴학 신청은 02-1234-5678로 전화하세요."
        context = [
            {
                "title": "학칙",
                "text": "휴학은 교육과정팀에 방문하여 신청한다.",
                "score": 0.9
            }
        ]

        system_prompt, user_prompt = EvaluationPrompts.format_hallucination_prompt(
            text=text,
            context=context
        )

        assert "환각" in system_prompt
        assert text in user_prompt
        assert "02-1234-5678" in user_prompt

    def test_factual_consistency_prompt(self):
        """Test factual consistency check prompt formatting."""
        answer = "휴학 기간은 1년 이내로 한다."
        context = [
            {
                "title": "학칙",
                "text": "휴학 기간은 1학기 또는 2학기로 한다.",
                "score": 0.85
            }
        ]

        system_prompt, user_prompt = EvaluationPrompts.format_factual_consistency_prompt(
            answer=answer,
            context=context
        )

        assert "사실적 일관성" in system_prompt
        assert answer in user_prompt
        assert "1년" in user_prompt

    def test_negative_examples(self):
        """Test negative examples for training."""
        examples = EvaluationPrompts.list_negative_examples()

        assert "hallucination" in examples
        assert "avoidance" in examples
        assert "insufficient_citation" in examples

        hallucination_example = EvaluationPrompts.get_negative_example("hallucination")

        assert "query" in hallucination_example
        assert "answer" in hallucination_example
        assert "issues" in hallucination_example
        assert len(hallucination_example["issues"]) > 0

    def test_context_formatting(self):
        """Test context formatting for prompts."""
        context = [
            {
                "title": "학칙 제26조",
                "text": "휴학은 학기 개시 30일 전까지 신청하여야 한다.",
                "score": 0.95
            },
            {
                "title": "학칙 제27조",
                "text": "휴학자는 복학 시 소정의 절차를 밟아야 한다.",
                "score": 0.87
            }
        ]

        formatted = EvaluationPrompts._format_context(context)

        assert "[문서 1]" in formatted
        assert "[문서 2]" in formatted
        assert "0.95" in formatted
        assert "0.87" in formatted


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def mock_store(self):
        """Create mock vector store with search results."""
        store = Mock()

        # Mock search results
        chunk1 = Mock(spec=Chunk)
        chunk1.id = "chunk1"
        chunk1.rule_code = "학칙_제26조"
        chunk1.article_number = "제26조"
        chunk1.parent_path = ["학칙"]
        chunk1.title = "휴학"
        chunk1.text = "휴학은 학기 개시 30일 전까지 신청하여야 한다."

        result1 = SearchResult(chunk=chunk1, score=0.95)

        store.search = Mock(return_value=[result1])
        return store

    @pytest.fixture
    def integrated_services(self, mock_store):
        """Create integrated services."""
        expansion_service = QueryExpansionService(store=mock_store)
        citation_enhancer = CitationEnhancer()

        return {
            "expansion": expansion_service,
            "citation": citation_enhancer
        }

    def test_query_expansion_to_citation_workflow(self, integrated_services, mock_store):
        """Test workflow from query expansion to citation enhancement."""
        # Step 1: Expand query
        query = "휴학 방법"
        expanded_queries = integrated_services["expansion"].expand_query(
            query,
            method="synonym"
        )

        assert len(expanded_queries) > 0

        # Step 2: Search with expanded query
        expanded_query = expanded_queries[0]
        search_results = mock_store.search(expanded_query.to_query())

        assert len(search_results) > 0

        # Step 3: Enhance citations
        chunk = search_results[0].chunk
        citation = integrated_services["citation"].enhance_citation(chunk)

        assert citation is not None
        assert citation.regulation == "학칙"
        assert citation.article_number == "제26조"

    def test_multilingual_query_handling(self, integrated_services):
        """Test handling of multilingual queries."""
        # Korean query
        ko_query = "휴학 방법"
        ko_expanded = integrated_services["expansion"].expand_query(
            ko_query,
            method="mixed"
        )
        assert any(e.language == "ko" for e in ko_expanded)

        # English query
        en_query = "leave of absence process"
        en_expanded = integrated_services["expansion"].expand_query(
            en_query,
            method="translation"
        )
        # Should include Korean translation
        assert any("휴학" in e.expanded_text for e in en_expanded)

    def test_evaluation_with_enhanced_citations(self):
        """Test evaluation process with enhanced citations."""
        query = "휴학 기간이 어떻게 되나요?"
        answer = "휴학 기간은 1학기 또는 2학기로 한다."

        # Create enhanced citation
        chunk = Mock(spec=Chunk)
        chunk.id = "test_chunk"
        chunk.rule_code = "학칙_제27조"
        chunk.article_number = "제27조"
        chunk.parent_path = ["학칙"]
        chunk.title = "휴학 기간"
        chunk.text = "휴학 기간은 1학기 또는 2학기로 한다."

        enhancer = CitationEnhancer()
        citation = enhancer.enhance_citation(chunk)

        # Format context with citation
        context = [{
            "title": f"{citation.regulation} {citation.article_number}",
            "text": citation.text,
            "score": 0.9
        }]

        # Create evaluation prompt
        system_prompt, user_prompt = EvaluationPrompts.format_accuracy_prompt(
            query=query,
            answer=answer,
            context=context
        )

        assert citation.regulation in user_prompt
        assert citation.article_number in user_prompt


def test_phase1_comprehensive_test():
    """Comprehensive test demonstrating all Phase 1 improvements."""
    # This test shows the complete integration without MLX

    # 1. Query Expansion
    store = Mock()
    expansion_service = QueryExpansionService(store=store)

    query = "휴학 방법 알려줘"
    expanded = expansion_service.expand_query(query, max_variants=3, method="synonym")

    assert len(expanded) >= 1, "Query expansion should return at least original query"
    assert expanded[0].expanded_text == query, "Original query should be preserved"

    # 2. Citation Enhancement
    enhancer = CitationEnhancer()

    chunk = Mock(spec=Chunk)
    chunk.id = "test"
    chunk.rule_code = "학칙_제26조"
    chunk.article_number = "제26조"
    chunk.parent_path = ["학칙"]
    chunk.title = "휴학"
    chunk.text = "휴학은 학기 개시 30일 전까지 신청하여야 한다."

    citation = enhancer.enhance_citation(chunk)
    assert citation is not None, "Citation enhancement should succeed"
    assert "학칙" in citation.format(), "Formatted citation should include regulation name"
    assert "제26조" in citation.format(), "Formatted citation should include article number"

    # 3. Evaluation Prompts
    context = [{
        "title": f"{citation.regulation} {citation.article_number}",
        "text": citation.text,
        "score": 0.95
    }]

    system_prompt, user_prompt = EvaluationPrompts.format_accuracy_prompt(
        query=query,
        answer="휴학은 학기 개시 30일 전까지 신청해야 합니다.",
        context=context
    )

    assert len(system_prompt) > 0, "System prompt should not be empty"
    assert query in user_prompt, "User prompt should include original query"
    assert citation.regulation in user_prompt, "User prompt should include citation info"

    # All Phase 1 components working together
    assert True, "Phase 1 integration test passed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
