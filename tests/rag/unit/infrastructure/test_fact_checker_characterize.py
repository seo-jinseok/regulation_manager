"""
Characterization tests for FactChecker source_chunks integration.

These tests capture the CURRENT BEHAVIOR before source_chunks enhancement.
They document what IS, not what SHOULD BE.

Purpose: Ensure backward compatibility when adding source_chunks parameter.
"""

from src.rag.infrastructure.fact_checker import (
    Citation,
    FactChecker,
    FactCheckResult,
    VerificationResult,
)


class FakeStore:
    """Fake store for testing."""

    def __init__(self, results=None):
        self._results = results or []

    def search(self, query, top_k=10):
        return self._results


class FakeChunk:
    """Fake chunk for testing."""

    def __init__(self, text, parent_path=None, title=""):
        self.text = text
        self.parent_path = parent_path or []
        self.title = title


class FakeSearchResult:
    """Fake search result for testing."""

    def __init__(self, chunk, score=0.8):
        self.chunk = chunk
        self.score = score


class TestVerifyCitationWithSourceChunks:
    """
    Tests for verify_citation with source_chunks parameter.

    These tests verify the new enhanced behavior using source chunks.
    """

    def test_verify_with_source_chunks_returns_verified(self):
        """
        Verify citation using source_chunks returns verified when matched.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            original_text="「학칙」 제25조",
        )

        source_chunks = [
            {
                "text": "제25조(학생의 의무) 학생은 학칙을 준수해야 한다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.verify_citation(citation, source_chunks=source_chunks)

        assert result.verified is True
        assert result.matched_content is not None
        assert result.confidence == 1.0  # High confidence for direct match

    def test_verify_with_source_chunks_returns_not_verified(self):
        """
        Verify citation using source_chunks returns not verified when no match.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="99",
            original_text="「학칙」 제99조",
        )

        source_chunks = [
            {
                "text": "제25조(학생의 의무) 학생은 학칙을 준수해야 한다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.verify_citation(citation, source_chunks=source_chunks)

        assert result.verified is False
        assert result.matched_content is None
        assert result.confidence == 0.0

    def test_verify_with_source_chunks_empty_list_returns_not_verified(self):
        """
        Verify citation with empty source_chunks returns not verified.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            original_text="「학칙」 제25조",
        )

        result = checker.verify_citation(citation, source_chunks=[])

        assert result.verified is False
        assert result.confidence == 0.0

    def test_verify_with_source_chunks_includes_content(self):
        """
        Verify citation with source_chunks includes matched content.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            original_text="「학칙」 제25조",
        )

        source_chunks = [
            {
                "text": "제25조(학생의 의무) 학생은 학칙을 준수해야 한다. 추가 내용이 있습니다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.verify_citation(citation, source_chunks=source_chunks)

        assert result.verified is True
        assert result.matched_content is not None
        assert len(result.matched_content) <= 200

    def test_verify_with_source_chunks_different_regulation(self):
        """
        Verify citation fails when regulation name differs.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        source_chunks = [
            {
                "text": "제25조(학생의 의무)",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.verify_citation(citation, source_chunks=source_chunks)

        assert result.verified is False


class TestCheckWithSourceChunks:
    """
    Tests for check() method with source_chunks parameter.
    """

    def test_check_with_source_chunks_all_verified(self):
        """
        check() with source_chunks verifies all citations.
        """
        store = FakeStore()
        checker = FactChecker(store)

        answer = "「학칙」 제25조에 따르면 학생은 의무가 있다."

        source_chunks = [
            {
                "text": "제25조(학생의 의무)",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.check(answer, source_chunks=source_chunks)

        assert result.all_verified is True
        assert result.verified_count == 1
        assert result.total_count == 1

    def test_check_with_source_chunks_partial_failure(self):
        """
        check() with source_chunks reports partial failures.
        """
        store = FakeStore()
        checker = FactChecker(store)

        answer = """
        「학칙」 제25조에 따르면 학생은 의무가 있다.
        「학칙」 제99조는 존재하지 않는다.
        """

        source_chunks = [
            {
                "text": "제25조(학생의 의무)",
                "metadata": {"regulation_name": "학칙", "article": 25},
            }
        ]

        result = checker.check(answer, source_chunks=source_chunks)

        assert result.all_verified is False
        assert result.verified_count == 1
        assert result.total_count == 2
        assert len(result.failed_citations) == 1

    def test_check_without_source_chunks_uses_store(self):
        """
        check() without source_chunks uses vector store (backward compat).
        """
        chunk = FakeChunk(
            text="제36조(휴직) 교원의 휴직에 관한 사항.",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        answer = "「교원인사규정」 제36조에 따르면..."

        # Call without source_chunks - should use store
        result = checker.check(answer)

        assert result.all_verified is True
        assert result.verified_count == 1


class TestCitationToExtractedConversion:
    """
    Tests for Citation to ExtractedCitation conversion helper.
    """

    def test_citation_to_extracted_basic(self):
        """Convert basic Citation to ExtractedCitation."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            original_text="「학칙」 제25조",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.regulation_name == "학칙"
        assert extracted.article == 25
        assert extracted.paragraph is None
        assert extracted.sub_article is None

    def test_citation_to_extracted_with_paragraph(self):
        """Convert Citation with paragraph to ExtractedCitation."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            paragraph="3",
            original_text="「학칙」 제25조 제3항",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.regulation_name == "학칙"
        assert extracted.article == 25
        assert extracted.paragraph == 3

    def test_citation_to_extracted_with_sub_article(self):
        """Convert Citation with sub_article to ExtractedCitation."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="10",
            article_sub="2",
            original_text="「학칙」 제10조의2",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.regulation_name == "학칙"
        assert extracted.article == 10
        assert extracted.sub_article == 2

    def test_citation_to_extracted_invalid_article(self):
        """Handle invalid article number gracefully."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="invalid",
            original_text="「학칙」 제invalid조",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.article == 0  # Fallback to 0

    def test_citation_to_extracted_invalid_paragraph(self):
        """Handle invalid paragraph gracefully."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="25",
            paragraph="invalid",
            original_text="「학칙」 제25조 제invalid항",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.paragraph is None  # Invalid paragraph is ignored

    def test_citation_to_extracted_invalid_sub_article(self):
        """Handle invalid sub_article gracefully."""
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="학칙",
            article="10",
            article_sub="invalid",
            original_text="「학칙」 제10조의invalid",
        )

        extracted = checker._citation_to_extracted(citation)

        assert extracted.sub_article is None  # Invalid sub_article is ignored


class TestGetVerificationService:
    """
    Tests for lazy initialization of CitationVerificationService.
    """

    def test_verification_service_lazy_init(self):
        """Verification service is created on first access."""
        store = FakeStore()
        checker = FactChecker(store)

        # Initially None
        assert checker._verification_service is None

        # Access triggers creation
        service = checker._get_verification_service()

        assert service is not None
        assert checker._verification_service is service

    def test_verification_service_singleton_per_checker(self):
        """Verification service is reused on subsequent calls."""
        store = FakeStore()
        checker = FactChecker(store)

        service1 = checker._get_verification_service()
        service2 = checker._get_verification_service()

        assert service1 is service2  # Same instance


class TestVerifyCitationCurrentBehavior:
    """
    Characterize current verify_citation() behavior.

    These tests document the CURRENT implementation that uses
    vector store search. This behavior MUST be preserved when
    source_chunks parameter is added.
    """

    def test_characterize_verify_uses_store_search(self):
        """
        CHARACTERIZE: verify_citation calls store.search().

        Current behavior: Uses vector store to find matching chunks.
        """
        chunk = FakeChunk(
            text="교원의 휴직에 관한 사항은...",
            parent_path=["교원인사규정", "제7장 신분보장"],
            title="제36조 휴직",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.85)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: uses store search for verification
        assert result.verified is True
        assert result.confidence == 0.85

    def test_characterize_verify_returns_verification_result(self):
        """
        CHARACTERIZE: verify_citation returns VerificationResult.

        Current behavior: Returns dataclass with citation, verified, matched_content, confidence.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="테스트규정",
            article="1",
            original_text="「테스트규정」 제1조",
        )

        result = checker.verify_citation(citation)

        # Current state: returns VerificationResult
        assert isinstance(result, VerificationResult)
        assert result.citation == citation
        assert result.verified is False
        assert result.matched_content is None
        assert result.confidence == 0.0

    def test_characterize_verify_builds_query_from_citation(self):
        """
        CHARACTERIZE: verify_citation builds search query from citation.

        Current behavior: Creates query like "{regulation} 제{article}조".
        """
        # Track what query was used
        queries = []

        class TrackingStore:
            def search(self, query, top_k=10):
                queries.append(query.text)
                return []

        checker = FactChecker(TrackingStore())

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        checker.verify_citation(citation)

        # Current state: query format is "{regulation} 제{article}조"
        assert len(queries) == 1
        assert "교원인사규정" in queries[0]
        assert "제36조" in queries[0]

    def test_characterize_verify_with_sub_article_query(self):
        """
        CHARACTERIZE: verify_citation handles sub-articles in query.

        Current behavior: Uses "{regulation} 제{article}조의{sub}" format.
        """
        queries = []

        class TrackingStore:
            def search(self, query, top_k=10):
                queries.append(query.text)
                return []

        checker = FactChecker(TrackingStore())

        citation = Citation(
            regulation="학칙",
            article="10",
            article_sub="2",
            original_text="「학칙」 제10조의2",
        )

        checker.verify_citation(citation)

        # Current state: includes sub-article in query
        assert len(queries) == 1
        assert "제10조의2" in queries[0]

    def test_characterize_verify_matches_regulation_in_parent_path(self):
        """
        CHARACTERIZE: verify_citation matches regulation in parent_path.

        Current behavior: Checks if normalized regulation name appears in parent_path.
        """
        chunk = FakeChunk(
            text="내용...",
            parent_path=["교원인사규정", "제7장"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: matches regulation from parent_path
        assert result.verified is True

    def test_characterize_verify_matches_article_in_title(self):
        """
        CHARACTERIZE: verify_citation matches article number in title.

        Current behavior: Checks "제N조" pattern in chunk title/text.
        """
        chunk = FakeChunk(
            text="내용...",
            parent_path=["교원인사규정"],
            title="제36조 휴직",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: matches article in title
        assert result.verified is True

    def test_characterize_verify_normalizes_text_for_matching(self):
        """
        CHARACTERIZE: verify_citation normalizes spaces and dots.

        Current behavior: Removes spaces and middle dots for comparison.
        """
        chunk = FakeChunk(
            text="내용",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: normalization works
        assert result.verified is True

    def test_characterize_verify_extracts_matched_content(self):
        """
        CHARACTERIZE: verify_citation extracts first 200 chars of matched content.

        Current behavior: Returns matched_content field with chunk text preview.
        """
        long_text = "이것은 매우 긴 텍스트입니다. " * 50
        chunk = FakeChunk(
            text=long_text,
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: extracts up to 200 characters
        assert result.verified is True
        assert result.matched_content is not None
        assert len(result.matched_content) <= 200

    def test_characterize_verify_returns_score_as_confidence(self):
        """
        CHARACTERIZE: verify_citation uses search score as confidence.

        Current behavior: confidence = result.score from search.
        """
        chunk = FakeChunk(
            text="내용",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.75)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="36",
            original_text="「교원인사규정」 제36조",
        )

        result = checker.verify_citation(citation)

        # Current state: confidence equals search score
        assert result.confidence == 0.75


class TestCheckMethodCurrentBehavior:
    """
    Characterize current check() method behavior.

    These tests document the CURRENT implementation flow.
    """

    def test_characterize_check_extracts_and_verifies(self):
        """
        CHARACTERIZE: check() extracts citations and verifies each.

        Current behavior: Calls extract_citations then verify_citation for each.
        """
        chunk = FakeChunk(
            text="내용",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        answer = "「교원인사규정」 제36조에 따르면..."

        result = checker.check(answer)

        # Current state: extracts, verifies, returns FactCheckResult
        assert isinstance(result, FactCheckResult)
        assert result.total_count == 1
        assert result.verified_count == 1
        assert result.all_verified is True

    def test_characterize_check_no_citations_returns_all_verified(self):
        """
        CHARACTERIZE: check() with no citations returns all_verified=True.

        Current behavior: Empty citations = nothing to fail.
        """
        store = FakeStore()
        checker = FactChecker(store)

        result = checker.check("규정 인용이 없는 텍스트입니다.")

        # Current state: no citations = all verified
        assert result.all_verified is True
        assert result.total_count == 0
        assert result.verified_count == 0

    def test_characterize_check_aggregates_results(self):
        """
        CHARACTERIZE: check() aggregates all verification results.

        Current behavior: Returns all VerificationResult objects in results list.
        """
        chunk1 = FakeChunk(
            text="내용1",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk1, score=0.9)])
        checker = FactChecker(store)

        answer = """
        「교원인사규정」 제36조에 따르면 휴직이 가능하며,
        「가상규정」 제999조에서는 추가 사항을 정합니다.
        """

        result = checker.check(answer)

        # Current state: aggregates results
        assert result.total_count == 2
        assert len(result.results) == 2
        assert result.verified_count == 1  # Only first citation verified

    def test_characterize_check_provides_failed_citations(self):
        """
        CHARACTERIZE: check() provides failed_citations property.

        Current behavior: Returns list of failed Citation objects.
        """
        store = FakeStore()
        checker = FactChecker(store)

        answer = "「가상규정」 제999조에 따르면..."

        result = checker.check(answer)

        # Current state: provides failed_citations
        assert len(result.failed_citations) == 1
        assert result.failed_citations[0].regulation == "가상규정"


class TestFactCheckerBackwardCompatibilityContract:
    """
    Document the public API contract for backward compatibility.

    These tests ensure that adding source_chunks parameter doesn't
    break existing usage patterns.
    """

    def test_characterize_verify_citation_single_parameter(self):
        """
        CHARACTERIZE: verify_citation accepts single Citation parameter.

        Current behavior: verify_citation(citation) - only one required parameter.
        This MUST remain callable with just citation after enhancement.
        """
        store = FakeStore()
        checker = FactChecker(store)

        citation = Citation(
            regulation="테스트",
            article="1",
            original_text="「테스트」 제1조",
        )

        # Current state: callable with single argument
        result = checker.verify_citation(citation)

        assert isinstance(result, VerificationResult)

    def test_characterize_check_single_parameter(self):
        """
        CHARACTERIZE: check() accepts single answer_text parameter.

        Current behavior: check(answer_text) - only one required parameter.
        This MUST remain callable with just text after enhancement.
        """
        store = FakeStore()
        checker = FactChecker(store)

        # Current state: callable with single argument
        result = checker.check("텍스트")

        assert isinstance(result, FactCheckResult)

    def test_characterize_fact_checker_requires_store(self):
        """
        CHARACTERIZE: FactChecker requires IVectorStore in constructor.

        Current behavior: __init__(self, store: IVectorStore).
        """
        store = FakeStore()
        checker = FactChecker(store)

        # Current state: store is required and stored
        assert checker.store == store
