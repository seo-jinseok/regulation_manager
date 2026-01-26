"""Unit tests for FactChecker."""


from src.rag.infrastructure.fact_checker import Citation, FactChecker, FactCheckResult


class FakeStore:
    """Fake store that returns predefined results."""

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


class TestCitationExtraction:
    """Tests for citation extraction from text."""

    def test_extract_citation_with_quotes(self):
        """「규정명」 제N조 형식 추출"""
        checker = FactChecker(FakeStore())
        text = "「교원인사규정」 제36조에 따르면 휴직 기간은..."

        citations = checker.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].regulation == "교원인사규정"
        assert citations[0].article == "36"

    def test_extract_citation_without_quotes(self):
        """따옴표 없는 규정명 제N조 형식 추출"""
        checker = FactChecker(FakeStore())
        text = "학칙 제15조에 의하면 학생은..."

        citations = checker.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].regulation == "학칙"
        assert citations[0].article == "15"

    def test_extract_citation_with_paragraph(self):
        """제N조 제M항 형식 추출"""
        checker = FactChecker(FakeStore())
        text = "「휴학규정」 제7조 제2항에 따라..."

        citations = checker.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].article == "7"
        assert citations[0].paragraph == "2"

    def test_extract_citation_with_sub_article(self):
        """제N조의M 형식 추출"""
        checker = FactChecker(FakeStore())
        text = "「학칙」 제10조의2에 근거하여..."

        citations = checker.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].article == "10"
        assert citations[0].article_sub == "2"

    def test_extract_multiple_citations(self):
        """여러 인용 추출"""
        checker = FactChecker(FakeStore())
        text = """
        「교원인사규정」 제36조에 따르면 휴직이 가능하며,
        「휴직규정」 제5조에서는 휴직 기간을 정하고 있습니다.
        """

        citations = checker.extract_citations(text)

        assert len(citations) == 2
        assert {c.regulation for c in citations} == {"교원인사규정", "휴직규정"}

    def test_deduplicate_citations(self):
        """중복 인용 제거"""
        checker = FactChecker(FakeStore())
        text = """
        「교원인사규정」 제36조에 따르면...
        앞서 언급한 「교원인사규정」 제36조는...
        """

        citations = checker.extract_citations(text)

        assert len(citations) == 1

    def test_no_citations(self):
        """인용 없는 텍스트"""
        checker = FactChecker(FakeStore())
        text = "이 질문에 대한 답변입니다. 규정을 찾을 수 없습니다."

        citations = checker.extract_citations(text)

        assert len(citations) == 0


class TestCitationVerification:
    """Tests for citation verification."""

    def test_verify_existing_citation(self):
        """존재하는 인용 검증 성공"""
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

        assert result.verified is True
        assert result.confidence > 0.8

    def test_verify_nonexistent_citation(self):
        """존재하지 않는 인용 검증 실패"""
        chunk = FakeChunk(
            text="학생의 휴학에 관한 사항은...",
            parent_path=["휴학규정"],
            title="제5조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.5)])
        checker = FactChecker(store)

        citation = Citation(
            regulation="교원인사규정",
            article="99",  # 존재하지 않는 조항
            original_text="「교원인사규정」 제99조",
        )

        result = checker.verify_citation(citation)

        assert result.verified is False


class TestFactCheck:
    """Tests for full fact check flow."""

    def test_fact_check_all_verified(self):
        """모든 인용 검증 성공"""
        chunk = FakeChunk(
            text="휴직에 관한 사항...",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        answer = "「교원인사규정」 제36조에 따르면..."

        result = checker.check(answer)

        assert result.all_verified is True
        assert result.verified_count == 1
        assert result.total_count == 1
        assert result.success_rate == 100.0

    def test_fact_check_partial_failure(self):
        """일부 인용 검증 실패"""
        # 첫 번째 인용만 검증 가능하게 설정
        chunk = FakeChunk(
            text="휴직에 관한 사항...",
            parent_path=["교원인사규정"],
            title="제36조",
        )
        store = FakeStore([FakeSearchResult(chunk, score=0.9)])
        checker = FactChecker(store)

        answer = """
        「교원인사규정」 제36조에 따르면 휴직이 가능하며,
        「가상규정」 제999조에서는 추가 사항을 정합니다.
        """

        result = checker.check(answer)

        assert result.all_verified is False
        assert result.verified_count == 1
        assert result.total_count == 2
        assert len(result.failed_citations) == 1
        assert result.failed_citations[0].regulation == "가상규정"

    def test_fact_check_no_citations(self):
        """인용 없는 답변은 검증 통과"""
        checker = FactChecker(FakeStore())

        answer = "이 질문에 대한 답변입니다."

        result = checker.check(answer)

        assert result.all_verified is True
        assert result.total_count == 0

    def test_build_correction_feedback(self):
        """수정 피드백 생성"""
        checker = FactChecker(FakeStore())

        failed_citation = Citation(
            regulation="가상규정",
            article="999",
            original_text="「가상규정」 제999조",
        )
        result = FactCheckResult(
            verified_count=1,
            total_count=2,
            results=[
                # One verified, one failed
                type(
                    "VerificationResult",
                    (),
                    {"citation": failed_citation, "verified": False},
                )()
            ],
            all_verified=False,
        )
        # Manually set failed citations for test
        result.results = [
            type(
                "VerificationResult",
                (),
                {"citation": failed_citation, "verified": False},
            )()
        ]

        feedback = checker.build_correction_feedback(result)

        assert "「가상규정」 제999조" in feedback
        assert "확인되지 않았습니다" in feedback
