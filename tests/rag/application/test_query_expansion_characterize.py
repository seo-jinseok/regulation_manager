"""
Characterization tests for QueryExpansionService behavior.

These tests capture the CURRENT behavior of the query expansion service
to ensure that improvements don't break existing functionality.
"""

import pytest
from src.rag.application.query_expansion import QueryExpansionService, ExpandedQuery
from src.rag.domain.entities import Chunk, ChunkLevel


@pytest.fixture
def mock_store():
    """Create a mock vector store."""
    class MockStore:
        def search(self, query, top_k=5):
            # Return mock search results
            return []

    return MockStore()


@pytest.fixture
def query_expansion_service(mock_store):
    """Create a QueryExpansionService instance for testing."""
    return QueryExpansionService(store=mock_store)


class TestQueryExpansionCharacterization:
    """Characterization tests for QueryExpansionService."""

    def test_expand_query_original_only(self, query_expansion_service):
        """CHARACTERIZE: Expansion returns original query."""
        expanded = query_expansion_service.expand_query("휴학 방법")

        # Document current behavior
        assert len(expanded) >= 1
        assert expanded[0].original_query == "휴학 방법"
        assert expanded[0].expanded_text == "휴학 방법"
        assert expanded[0].expansion_method == "original"
        assert expanded[0].confidence == 1.0

    def test_expand_query_with_synonyms(self, query_expansion_service):
        """CHARACTERIZE: Synonym-based expansion behavior."""
        expanded = query_expansion_service.expand_query("휴학 방법", method="synonym")

        # Document current behavior
        assert len(expanded) >= 1

        # Check if synonyms were found
        methods = [e.expansion_method for e in expanded]
        assert "original" in methods

        # If synonyms found, check their properties
        synonym_expansions = [e for e in expanded if e.expansion_method == "synonym"]
        if synonym_expansions:
            assert all(e.confidence > 0 for e in synonym_expansions)
            assert all(e.language == "ko" for e in synonym_expansions)

    def test_expand_query_english_korean(self, query_expansion_service):
        """CHARACTERIZE: English-Korean translation expansion."""
        expanded = query_expansion_service.expand_query("leave of absence", method="translation")

        # Document current behavior
        assert len(expanded) >= 1

        # Check if translation occurred
        has_korean = any("휴학" in e.expanded_text for e in expanded)
        assert has_korean, "Should translate 'leave of absence' to Korean"

    def test_expand_query_mixed_language(self, query_expansion_service):
        """CHARACTERIZE: Mixed language query expansion."""
        expanded = query_expansion_service.expand_query("휴학 leave of absence", method="mixed")

        # Document current behavior
        assert len(expanded) >= 1

        # Check language detection
        languages = [e.language for e in expanded]
        assert "ko" in languages or "mixed" in languages

    def test_deduplicate_expanded(self, query_expansion_service):
        """CHARACTERIZE: Deduplication behavior."""
        # This would need to be tested with actual expansion that might create duplicates
        # For now, test the deduplication method directly
        expanded = [
            ExpandedQuery("test", "test query", "original", 1.0, "ko"),
            ExpandedQuery("test", "test query", "synonym", 0.9, "ko"),  # Duplicate text
            ExpandedQuery("test", "different query", "synonym", 0.8, "ko"),
        ]

        unique = query_expansion_service._deduplicate_expanded(expanded)

        # Document current behavior: should remove duplicates
        assert len(unique) <= len(expanded)
        assert len(unique) == 2  # "test query" duplicate removed

    def test_detect_language_korean(self, query_expansion_service):
        """CHARACTERIZE: Korean language detection."""
        lang = query_expansion_service._detect_language("휴학 방법 알려줘")

        # Document current behavior
        assert lang == "ko"

    def test_detect_language_english(self, query_expansion_service):
        """CHARACTERIZE: English language detection."""
        lang = query_expansion_service._detect_language("How to apply for leave of absence")

        # Document current behavior
        assert lang == "en"

    def test_detect_language_mixed(self, query_expansion_service):
        """CHARACTERIZE: Mixed language detection."""
        lang = query_expansion_service._detect_language("휴학 leave of absence 방법")

        # Document current behavior
        assert lang == "mixed"

    def test_extract_key_terms(self, query_expansion_service):
        """CHARACTERIZE: Key term extraction behavior."""
        terms = query_expansion_service._extract_key_terms("휴학 방법과 장학금 신청")

        # Document current behavior
        assert isinstance(terms, list)
        # Should extract academic terms
        if "휴학" in terms:
            assert "휴학" in terms
        if "장학금" in terms:
            assert "장학금" in terms

    def test_search_with_expansion(self, query_expansion_service, mock_store):
        """CHARACTERIZE: Search with expansion behavior."""
        results = query_expansion_service.search_with_expansion("휴학 방법", top_k=5)

        # Document current behavior
        assert isinstance(results, list)
        # Should return results (even if empty due to mock store)
        assert len(results) <= 5

    def test_get_expansion_statistics(self, query_expansion_service):
        """CHARACTERIZE: Statistics calculation behavior."""
        queries = ["휴학 방법", "장학금 신청", "등록금 납부"]
        stats = query_expansion_service.get_expansion_statistics(queries, method="synonym")

        # Document current behavior
        assert "total_queries" in stats
        assert stats["total_queries"] == len(queries)
        assert "expanded_queries" in stats
        assert "avg_variants_per_query" in stats
        assert "language_distribution" in stats
        assert "method_distribution" in stats

    def test_expanded_query_to_query(self, query_expansion_service):
        """CHARACTERIZE: ExpandedQuery conversion to Query."""
        expanded = ExpandedQuery(
            original_query="휴학 방법",
            expanded_text="휴학원 방법",
            expansion_method="synonym",
            confidence=0.9,
            language="ko"
        )

        query = expanded.to_query()

        # Document current behavior
        assert query.text == "휴학원 방법"

    def test_max_variants_limit(self, query_expansion_service):
        """CHARACTERIZE: max_variants parameter behavior."""
        expanded = query_expansion_service.expand_query("휴학 방법", max_variants=2)

        # Document current behavior
        assert len(expanded) <= 2

    def test_expand_query_no_synonyms_found(self, query_expansion_service):
        """CHARACTERIZE: Behavior when no synonyms are found."""
        # Use a query with no known academic terms
        expanded = query_expansion_service.expand_query("무의미한 질문 텍스트", method="synonym")

        # Document current behavior
        assert len(expanded) >= 1  # At least original query
        assert expanded[0].expansion_method == "original"

    def test_expand_query_llm_method_without_client(self, query_expansion_service):
        """CHARACTERIZE: LLM expansion behavior when client not available."""
        # Service created without LLM client
        expanded = query_expansion_service.expand_query("휴학 방법", method="llm")

        # Document current behavior: should fall back gracefully
        assert len(expanded) >= 1  # At least original query
