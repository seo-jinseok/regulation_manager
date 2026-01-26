"""
Unit tests for HyDE integration in SearchUseCase - Cycle 8.
"""

from typing import List
from unittest.mock import MagicMock

from src.rag.config import reset_config


class FakeChunk:
    """Fake Chunk for testing."""

    def __init__(
        self,
        id: str,
        text: str,
        title: str = "",
        rule_code: str = "",
    ):
        self.id = id
        self.text = text
        self.title = title
        self.rule_code = rule_code
        self.keywords = []


class FakeSearchResult:
    """Fake SearchResult for testing."""

    def __init__(self, chunk: FakeChunk, score: float, rank: int = 1):
        self.chunk = chunk
        self.score = score
        self.rank = rank


def make_result(
    doc_id: str,
    text: str,
    score: float,
    title: str = "",
    rule_code: str = "",
    rank: int = 1,
) -> FakeSearchResult:
    """Helper to create fake search results."""
    chunk = FakeChunk(id=doc_id, text=text, title=title, rule_code=rule_code)
    return FakeSearchResult(chunk=chunk, score=score, rank=rank)


class FakeStore:
    """Fake Vector Store for testing."""

    def __init__(self, results: List[FakeSearchResult] = None):
        self._results = results or []
        self.search_calls = []

    def search(self, query, filter=None, top_k: int = 10):
        self.search_calls.append({"query": query, "filter": filter, "top_k": top_k})
        return self._results[:top_k]


class FakeLLMClient:
    """Fake LLM client for HyDE generation."""

    def __init__(self, response: str = "가상 문서 내용"):
        self._response = response

    def generate(self, **kwargs):
        return self._response


class TestHyDEIntegration:
    """Test HyDE integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_hyde_enabled_for_vague_queries(self):
        """모호한 쿼리에서 HyDE가 활성화되어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        vague_queries = [
            "학교 가기 싫어",
            "쉬고 싶어",
            "어떻게 해야 하나요",
            "가능한가요?",
        ]

        for query in vague_queries:
            assert generator.should_use_hyde(query, complexity="medium"), (
                f"'{query}'에서 HyDE가 활성화되어야 함"
            )

    def test_hyde_skipped_for_structural_queries(self):
        """구조적 쿼리(조문 번호 등)에서는 HyDE를 건너뛰어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        structural_queries = [
            "교원인사규정 제8조",
            "학칙",
            "3-1-24",
        ]

        for query in structural_queries:
            assert not generator.should_use_hyde(query, complexity="simple"), (
                f"'{query}'에서 HyDE가 비활성화되어야 함"
            )

    def test_hyde_skipped_for_very_short_queries(self):
        """매우 짧은 쿼리에서는 HyDE를 건너뛰어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        short_queries = ["휴직", "장학", "규정"]

        for query in short_queries:
            assert not generator.should_use_hyde(query, complexity="medium"), (
                f"'{query}'에서 HyDE가 비활성화되어야 함 (너무 짧음)"
            )

    def test_hyde_enabled_for_emotional_queries(self):
        """감정적 쿼리에서 HyDE가 활성화되어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        emotional_queries = [
            "학교생활이 너무 힘들어",
            "스트레스 받아",
            "걱정돼서",
        ]

        for query in emotional_queries:
            assert generator.should_use_hyde(query, complexity="medium"), (
                f"'{query}'에서 HyDE가 활성화되어야 함 (감정적 표현)"
            )

    def test_hyde_cache_uses_config_settings(self):
        """HyDE 캐시가 config 설정을 사용해야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(
            cache_dir="custom/cache/dir",
            enable_cache=True,
        )

        assert generator._enable_cache is True


class TestHyDESearchFlow:
    """Test the complete HyDE search flow."""

    def test_search_with_hyde_merges_results(self):
        """HyDE 검색이 원본 쿼리 결과와 병합되어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator, HyDEResult, HyDESearcher

        mock_generator = MagicMock(spec=HyDEGenerator)
        mock_generator.generate_hypothetical_doc.return_value = HyDEResult(
            original_query="휴직",
            hypothetical_doc="교원이 휴직을 신청하려면 인사규정에 따라",
            from_cache=False,
            quality_score=0.8,
        )

        mock_store = MagicMock()
        mock_store.search.return_value = [
            make_result("doc1", "휴직 규정", 0.9, rule_code="1-1-1"),
            make_result("doc2", "휴직 절차", 0.8, rule_code="1-1-2"),
        ]

        searcher = HyDESearcher(mock_generator, mock_store)

        merged_results = searcher.search_with_hyde("휴직", top_k=5)

        mock_generator.generate_hypothetical_doc.assert_called_once_with("휴직")
        assert mock_store.search.call_count == 2


class TestHyDEValidation:
    """Test HyDE input and result validation."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_generate_hypothetical_doc_with_empty_query(self):
        """Empty query should return empty hypothetical doc."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        result = generator.generate_hypothetical_doc("")
        assert result.original_query == ""
        assert result.hypothetical_doc == ""
        assert result.from_cache is False

    def test_generate_hypothetical_doc_with_invalid_llm_response(self):
        """Invalid LLM response should fallback to original query."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        class InvalidLLMClient:
            def generate(self, **kwargs):
                return "죄송합니다"

        generator = HyDEGenerator(llm_client=InvalidLLMClient(), enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        assert result.original_query == "test query"
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False

    def test_validate_hypothetical_doc_accepts_valid_response(self):
        """Validation should accept valid responses with quality score."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        doc = ("교직원의 휴직은 다음 각 호의 사유에 해당하는 경우 신청할 수 있다. "
               "휴직 기간은 1년 이내로 하며 이 경우 보수는 지급하지 않는다. "
               "휴직은 인사위원회의 심의를 거쳐 허가한다. "
               "교원은 휴직기간 중에도 직무를 수행할 수 있다. "
               "휴직은 연구년 휴직과 질병 휴직으로 구분한다.")

        is_valid, validated, quality = generator._validate_hypothetical_doc(doc, "test query")

        assert is_valid is True
        assert validated == doc
        assert quality >= 0.5


class TestHyDEMetrics:
    """Test HyDE performance metrics (Cycle 8)."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_metrics_initialization(self):
        """HyDEMetrics should initialize with zero values."""
        from src.rag.infrastructure.hyde import HyDEMetrics

        metrics = HyDEMetrics()

        assert metrics.total_generations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.validation_failures == 0
        assert metrics.total_quality_score == 0.0
        assert metrics.total_generation_time_ms == 0.0

    def test_cache_hit_rate_calculation(self):
        """Cache hit rate should be calculated correctly."""
        from src.rag.infrastructure.hyde import HyDEMetrics

        metrics = HyDEMetrics()
        metrics.total_generations = 10
        metrics.cache_hits = 7

        assert metrics.get_cache_hit_rate() == 0.7

    def test_average_quality_calculation(self):
        """Average quality score should be calculated correctly."""
        from src.rag.infrastructure.hyde import HyDEMetrics

        metrics = HyDEMetrics()
        metrics.total_generations = 5
        metrics.total_quality_score = 3.5

        assert metrics.get_average_quality() == 0.7

    def test_generator_tracks_metrics(self):
        """HyDEGenerator should track performance metrics."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        class MockLLMClient:
            def generate(self, **kwargs):
                return ("학생의 휴학은 질병 또는 가사사정으로 인하여 수학할 수 없는 "
                        "경우에 한하여 허가할 수 있다. 휴학 기간은 1회에 1학기를 "
                        "초과하지 못하며 통산 3학기를 초과할 수 없다. "
                        "휴학은 학기 개시일로부터 30일 이내에 신청하여야 한다. "
                        "휴학 신청은 소속 학장의 승인을 받아야 한다.")

        generator = HyDEGenerator(
            llm_client=MockLLMClient(),
            enable_cache=False
        )

        generator.generate_hypothetical_doc("학교 가기 싫어")
        generator.generate_hypothetical_doc("쉬고 싶어")

        metrics = generator.get_metrics()

        assert metrics.total_generations == 2
        assert metrics.validation_failures == 0
        assert metrics.total_quality_score > 0

    def test_generator_reset_metrics(self):
        """reset_metrics should clear all metrics."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        class MockLLMClient:
            def generate(self, **kwargs):
                return "테스트 응답 " * 20

        generator.set_llm_client(MockLLMClient())
        generator.generate_hypothetical_doc("test")

        generator.reset_metrics()

        metrics = generator.get_metrics()
        assert metrics.total_generations == 0


class TestHyDEQualityScoring:
    """Test HyDE quality scoring (Cycle 8)."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_quality_score_includes_length_bonus(self):
        """Quality score should include bonus for appropriate length."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        good_doc = ("학생의 휴학은 질병 또는 가사사정으로 인하여 수학할 수 없는 "
                    "경우에 한하여 허가할 수 있다. 휴학 기간은 1회에 1학기를 "
                    "초과하지 못하며 통산 3학기를 초과할 수 없다. "
                    "휴학은 학기 개시일로부터 30일 이내에 신청하여야 한다. "
                    "휴학 신청은 소속 학장의 승인을 받아야 한다. "
                    "휴학 중인 학생은 등록금을 납부하지 않아도 된다.")

        is_valid, _, quality = generator._validate_hypothetical_doc(good_doc, "query")

        assert is_valid is True
        assert quality > 0

    def test_quality_score_includes_regulatory_language(self):
        """Quality score should include bonus for regulatory language."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        regulatory_doc = ("학생은 휴학을 신청할 수 있다. 휴학은 허가되어야 한다. "
                          "장학금을 지급한다. 성적을 평가한다. "
                          "학점을 취득한다. 등록금을 납부한다. "
                          "졸업 요건을 충족한다. 교원을 임용한다. "
                          "휴직을 신청한다. 복직을 허가한다.")

        is_valid, _, quality = generator._validate_hypothetical_doc(regulatory_doc, "query")

        assert is_valid is True
        assert quality > 0.3

    def test_quality_score_includes_keywords(self):
        """Quality score should include bonus for education keywords."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        keyword_doc = ("학생의 휴학, 교원의 휴직, 장학금 지급, "
                       "등록금 납부, 성적 평가, 학점 취득, "
                       "졸업 요건, 복학 절차, 등록 신청, "
                       "수강 신청, 교과목 이수, 학위 수여. "
                       "휴학은 학기 개시일로부터 30일 이내에 신청하여야 한다. "
                       "장학금은 성적과 소득 수준을 고려하여 지급된다. "
                       "등록금은 분할 납부할 수 있다. "
                       "성적은 평점으로 평가한다. "
                       "학점은 3학기 이상 이수해야 한다.")

        is_valid, _, quality = generator._validate_hypothetical_doc(keyword_doc, "query")

        assert is_valid is True
        assert quality > 0.2
