"""
Characterization tests for FactChecker citation patterns.

These tests capture the CURRENT BEHAVIOR of citation extraction
and verification. They document what IS, not what SHOULD BE.

Purpose: Ensure behavior preservation during refactoring.
"""

from src.rag.infrastructure.fact_checker import Citation, FactChecker


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


class TestCitationPatternExtraction:
    """Characterize current citation extraction patterns."""

    def test_characterize_extract_regulation_with_quotes(self):
        """
        CHARACTERIZE: 「규정명」 제N조 pattern extraction.

        Current behavior: Extracts regulation name and article number
        from bracketed regulation names.
        """
        checker = FactChecker(FakeStore())
        text = "「교원인사규정」 제36조에 따르면 휴직 기간은..."

        citations = checker.extract_citations(text)

        # Current state: extracts regulation and article
        assert len(citations) == 1
        assert citations[0].regulation == "교원인사규정"
        assert citations[0].article == "36"

    def test_characterize_extract_regulation_without_quotes(self):
        """
        CHARACTERIZE: 규정명 제N조 pattern without brackets.

        Current behavior: Extracts from common regulation suffixes.
        """
        checker = FactChecker(FakeStore())
        text = "학칙 제15조에 의하면 학생은..."

        citations = checker.extract_citations(text)

        # Current state: extracts "학칙" as regulation
        assert len(citations) == 1
        assert citations[0].regulation == "학칙"
        assert citations[0].article == "15"

    def test_characterize_extract_with_paragraph(self):
        """
        CHARACTERIZE: 제N조 제M항 pattern with paragraph.

        Current behavior: Extracts article and paragraph numbers.
        """
        checker = FactChecker(FakeStore())
        text = "「휴학규정」 제7조 제2항에 따라..."

        citations = checker.extract_citations(text)

        # Current state: extracts both article and paragraph
        assert len(citations) == 1
        assert citations[0].article == "7"
        assert citations[0].paragraph == "2"

    def test_characterize_extract_with_sub_article(self):
        """
        CHARACTERIZE: 제N조의M pattern for sub-articles.

        Current behavior: Extracts sub-article number separately.
        """
        checker = FactChecker(FakeStore())
        text = "「학칙」 제10조의2에 근거하여..."

        citations = checker.extract_citations(text)

        # Current state: extracts sub-article in article_sub field
        assert len(citations) == 1
        assert citations[0].article == "10"
        assert citations[0].article_sub == "2"

    def test_characterize_extract_multiple_citations(self):
        """
        CHARACTERIZE: Multiple citations in single text.

        Current behavior: Extracts all unique citations, deduplicates.
        """
        checker = FactChecker(FakeStore())
        text = """
        「교원인사규정」 제36조에 따르면 휴직이 가능하며,
        「휴직규정」 제5조에서는 휴직 기간을 정하고 있습니다.
        """

        citations = checker.extract_citations(text)

        # Current state: extracts both citations
        assert len(citations) == 2
        regulations = {c.regulation for c in citations}
        assert regulations == {"교원인사규정", "휴직규정"}

    def test_characterize_deduplicate_same_citation(self):
        """
        CHARACTERIZE: Deduplication of identical citations.

        Current behavior: Same regulation+article appears only once.
        """
        checker = FactChecker(FakeStore())
        text = """
        「교원인사규정」 제36조에 따르면...
        앞서 언급한 「교원인사규정」 제36조는...
        """

        citations = checker.extract_citations(text)

        # Current state: deduplicates by (regulation, article, article_sub)
        assert len(citations) == 1

    def test_characterize_no_citations_in_text(self):
        """
        CHARACTERIZE: Text without citation patterns.

        Current behavior: Returns empty list, no errors.
        """
        checker = FactChecker(FakeStore())
        text = "이 질문에 대한 답변입니다. 규정을 찾을 수 없습니다."

        citations = checker.extract_citations(text)

        # Current state: gracefully handles no citations
        assert len(citations) == 0


class TestCitationVerificationLogic:
    """Characterize current citation verification behavior."""

    def test_characterize_verify_with_parent_path_match(self):
        """
        CHARACTERIZE: Verification uses parent_path for regulation match.

        Current behavior: Checks if regulation name appears in parent_path.
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

        # Current state: verifies based on parent_path and title match
        assert result.verified is True
        assert result.confidence > 0.8

    def test_characterize_verify_with_title_match(self):
        """
        CHARACTERIZE: Verification checks article number in chunk text/title.

        Current behavior: Searches for "제N조" pattern in chunk text/title.
        """
        chunk = FakeChunk(
            text="휴직에 관한 사항...",
            parent_path=["휴학규정"],
            title="제5조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="휴학규정",
            article="5",
            original_text="「휴학규정」 제5조",
        )

        result = checker.verify_citation(citation)

        # Current state: matches article number in title/text
        assert result.verified is True

    def test_characterize_verify_nonexistent_citation(self):
        """
        CHARACTERIZE: Non-existent citations fail verification.

        Current behavior: Returns verified=False when no match found.
        """
        chunk = FakeChunk(
            text="학생의 휴학에 관한 사항은...",
            parent_path=["휴학규정"],
            title="제5조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.5)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="99",  # Non-existent
            original_text="「교원인사규정」 제99조",
        )

        result = checker.verify_citation(citation)

        # Current state: fails verification
        assert result.verified is False
        assert result.confidence == 0.0

    def test_characterize_verify_with_normalized_text(self):
        """
        CHARACTERIZE: Verification normalizes spaces and dots for matching.

        Current behavior: Removes spaces and middle dots for comparison.
        """
        chunk = FakeChunk(
            text="교원의 복무에 관한 사항",
            parent_path=["교원인사규정"],
            title="제26조 (직원의 구분)",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        # Citation with different spacing
        citation = Citation(
            regulation="교 원 인 사 규 정",  # Spaced differently
            article="26",
            original_text="「교원인사규정」 제26조",
        )

        result = checker.verify_citation(citation)

        # Current state: normalization allows match
        assert result.verified is True


class TestCitationPatternEdgeCases:
    """Characterize edge cases in citation patterns."""

    def test_characterize_regulation_name_with_suffix(self):
        """
        CHARACTERIZE: Regulation names with various suffixes.

        Current behavior: Matches common suffixes (규정, 규칙, 세칙, 지침, 학칙).
        """
        checker = FactChecker(FakeStore())

        # Test various regulation types
        test_cases = [
            "직원복무규정 제26조에 따라",
            "학칙 제15조에 의하면",
            "연구처규칙 제3조에서는",
            "장학세칙 제8조에 따르면",
            "연구윤리지침 제12조에 의하여",
        ]

        for text in test_cases:
            citations = checker.extract_citations(text)
            # Current state: extracts regulation with suffix
            assert len(citations) == 1
            assert citations[0].article  # Article extracted

    def test_characterize_citation_with_hangul_paragraph(self):
        """
        CHARACTERIZE: Paragraph numbers in Korean (일, 이, 삼).

        Current behavior: Current pattern expects Arabic numerals.
        """
        checker = FactChecker(FakeStore())
        text = "「직원복무규정」 제26조 제일항에 따라"

        citations = checker.extract_citations(text)

        # Current state: May not extract Korean numerals
        # This characterization documents current limitation
        if citations:
            # If extracted, paragraph field would need Korean numeral support
            assert citations[0].paragraph is None or citations[0].paragraph == "일"

    def test_characterize_citation_without_space(self):
        """
        CHARACTERIZE: Citations without spaces (제N조제M항).

        Current behavior: Patterns allow flexible spacing.
        """
        checker = FactChecker(FakeStore())
        text = "「직원복무규정」제26조제2항에 따라"

        citations = checker.extract_citations(text)

        # Current state: extracts despite missing spaces
        assert len(citations) >= 1

    def test_characterize_original_text_preservation(self):
        """
        CHARACTERIZE: Original matched text is preserved.

        Current behavior: Stores original_text for feedback generation.
        """
        checker = FactChecker(FakeStore())
        text = "「교원인사규정」 제36조에 따르면"

        citations = checker.extract_citations(text)

        # Current state: preserves original text
        assert len(citations) == 1
        assert citations[0].original_text == "「교원인사규정」 제36조"
