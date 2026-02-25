"""
Unit tests for Self-RAG integration in SearchUseCase.

Tests cover:
- Self-RAG activation based on config
- Retrieval necessity check
- Relevance filtering
- Integration with search pipeline
- SPEC-RAG-QUALITY-011: Self-RAG Classification Fix
"""

import time
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


class TestSelfRAGIntegration:
    """Test Self-RAG integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_search_usecase_has_self_rag_attributes(self):
        """SearchUseCase가 Self-RAG 관련 속성을 가져야 함."""
        from src.rag.application.search_usecase import SearchUseCase

        store = FakeStore(
            [
                make_result("doc1", "휴직 신청", 0.8, rule_code="1-1-1"),
            ]
        )

        usecase = SearchUseCase(
            store,
            use_reranker=False,
            use_hybrid=False,
        )

        # Self-RAG 관련 속성들이 있어야 함
        assert hasattr(usecase, "_self_rag_pipeline")
        assert hasattr(usecase, "_enable_self_rag")

    def test_self_rag_enabled_by_default(self):
        """Self-RAG는 기본적으로 활성화되어야 함."""
        from src.rag.config import RAGConfig

        config = RAGConfig()
        assert config.enable_self_rag is True


class TestSelfRAGPipeline:
    """Test SelfRAGPipeline functionality."""

    def test_should_retrieve_returns_true_for_factual_queries(self):
        """사실적 질문에는 검색이 필요함."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline

        # LLM 없이도 기본적으로 True 반환
        pipeline = SelfRAGPipeline(
            enable_retrieval_check=False,
        )

        assert pipeline.should_retrieve("교원 휴직 신청 절차는?") is True

    def test_filter_relevant_results_returns_all_when_disabled(self):
        """enable_relevance_check=False면 모든 결과 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline(
            enable_relevance_check=False,
        )

        results = [
            make_result("doc1", "관련 내용", 0.9),
            make_result("doc2", "무관한 내용", 0.3),
        ]

        filtered = pipeline.filter_relevant_results("query", results)

        # 필터링 비활성화 시 모든 결과 반환
        assert len(filtered) == len(results)

    def test_evaluate_results_batch_skips_llm_for_high_scores(self):
        """점수가 높으면 LLM 호출 없이 결과 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline()

        results = [
            make_result("doc1", "관련 내용", 0.95),
            make_result("doc2", "또 다른 관련 내용", 0.85),
        ]

        is_relevant, filtered, confidence = pipeline.evaluate_results_batch(
            "query", results
        )

        # 높은 점수면 바로 관련성 있다고 판단
        assert is_relevant is True
        assert confidence > 0.8

    def test_evaluate_results_batch_handles_dict_representation(self):
        """dict 표현이 포함된 결과를 처리해야 함."""
        from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult
        from src.rag.infrastructure.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline()

        # Create a dict representation (as if deserialized from JSON)
        dict_result = {
            "chunk": {
                "id": "doc1",
                "rule_code": "1-1-1",
                "level": "text",
                "title": "휴직규정",
                "text": "휴직 신청 방법",
                "embedding_text": "휴직 신청 방법",
                "full_text": "휴직 신청 방법",
                "parent_path": [],
                "token_count": 10,
                "keywords": [],
                "is_searchable": True,
                "effective_date": None,
                "status": "active",
            },
            "score": 0.9,
            "rank": 1,
        }

        # Mix of SearchResult and dict
        chunk2 = Chunk(
            id="doc2",
            rule_code="1-1-2",
            level=ChunkLevel.TEXT,
            title="복지규정",
            text="복지 혜택",
            embedding_text="복지 혜택",
            full_text="복지 혜택",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
        )
        result2 = SearchResult(chunk=chunk2, score=0.85, rank=2)

        results = [dict_result, result2]

        # Should handle both dict and SearchResult objects
        is_relevant, filtered, confidence = pipeline.evaluate_results_batch(
            "query", results
        )

        # High score should be detected from converted dict
        assert is_relevant is True
        assert confidence > 0.8
        assert len(filtered) > 0


class TestSelfRAGEvaluator:
    """Test SelfRAGEvaluator functionality."""

    def test_evaluator_needs_llm_for_retrieval_check(self):
        """검색 필요성 판단은 LLM이 필요함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        # LLM 없으면 기본적으로 검색 필요하다고 판단
        result = evaluator.needs_retrieval("오늘 날씨 어때?")
        assert result is True  # Fallback to True when no LLM

    def test_evaluator_evaluate_relevance_without_llm(self):
        """LLM 없이도 관련성 평가 가능해야 함 (fallback)."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        results = [
            make_result("doc1", "휴직 신청 방법", 0.8),
        ]

        # LLM 없으면 모든 결과를 관련있다고 판단 (fallback)
        is_relevant, filtered = evaluator.evaluate_relevance("휴직", results)

        assert is_relevant is True
        assert len(filtered) == len(results)


class TestSearchUseCaseSelfRAGIntegration:
    """Test complete Self-RAG integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_ask_method_can_use_self_rag(self):
        """ask 메서드가 Self-RAG를 사용할 수 있어야 함."""
        from src.rag.application.search_usecase import SearchUseCase

        store = FakeStore(
            [
                make_result(
                    "doc1",
                    "휴직 신청은 인사과에 서류를 제출하면 됩니다.",
                    0.9,
                    rule_code="1-1-1",
                    title="휴직규정",
                ),
            ]
        )

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "휴직 신청은 인사과에 서류를 제출하면 됩니다."

        usecase = SearchUseCase(
            store,
            llm_client=mock_llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._corrective_rag_enabled = False

        # Self-RAG pipeline이 초기화되어 있어야 함
        assert hasattr(usecase, "_self_rag_pipeline")


# ============================================================================
# SPEC-RAG-QUALITY-011: Self-RAG Classification Fix Tests
# ============================================================================


class TestSPEC_RAG_QUALITY_011_Req001:
    """REQ-001: Self-RAG Prompt Improvement Tests."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_prompt_includes_regulation_domain_context(self):
        """AC-001.1: 프롬프트에 규정 도메인 컨텍스트가 포함되어야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        prompt = SelfRAGEvaluator.RETRIEVAL_NEEDED_PROMPT

        # 프롬프트에 규정 관련 키워드가 포함되어야 함
        assert "규정" in prompt or "학칙" in prompt or "대학" in prompt

    def test_prompt_defaults_to_retrieval_for_uncertain(self):
        """AC-001.2: 불확실한 경우 검색을 기본값으로 설정해야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        prompt = SelfRAGEvaluator.RETRIEVAL_NEEDED_PROMPT

        # 불확실한 경우 검색하라는 지시가 있어야 함
        assert "불확실" in prompt or "RETRIEVE_YES" in prompt or "기본" in prompt

    def test_prompt_includes_examples(self):
        """AC-001.3: 검색이 필요한 경우 예시가 포함되어야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        prompt = SelfRAGEvaluator.RETRIEVAL_NEEDED_PROMPT

        # 예시가 포함되어야 함 (질문어 또는 구체적인 예시)
        assert "어떻게" in prompt or "언제" in prompt or "예시" in prompt or "경우" in prompt


class TestSPEC_RAG_QUALITY_011_Req002:
    """REQ-002: Keyword-Based Pre-Filtering Tests."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_has_regulation_keywords_method_exists(self):
        """AC-002.1: _has_regulation_keywords 메서드가 존재해야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()
        assert hasattr(evaluator, "_has_regulation_keywords")

    def test_keyword_detection_succeeds(self):
        """AC-002.1: 키워드가 포함된 쿼리를 감지해야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()

        # 규정 관련 키워드가 포함된 쿼리
        assert evaluator._has_regulation_keywords("이 규정에 대해 알려주세요") is True
        assert evaluator._has_regulation_keywords("학칙 제5조가 뭐야?") is True
        assert evaluator._has_regulation_keywords("휴학 신청 방법") is True

    def test_no_keyword_detection(self):
        """AC-002.2: 키워드가 없는 쿼리는 False 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()

        # 키워드가 없는 일반적인 쿼리
        assert evaluator._has_regulation_keywords("오늘 날씨 어때?") is False
        assert evaluator._has_regulation_keywords("안녕하세요") is False

    def test_keyword_matching_completes_quickly(self):
        """AC-002.3: 키워드 매칭이 1ms 이내에 완료되어야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()

        # 여러 쿼리에 대한 매칭 시간 측정
        queries = [
            "이 규정에 대해 알려주세요",
            "학칙 제5조가 뭐야?",
            "휴학 신청 방법",
            "장학금 신청 자격이 뭐야?",
            "졸업 요건이 어떻게 돼?",
        ]

        start_time = time.time()
        for query in queries:
            evaluator._has_regulation_keywords(query)
        elapsed_ms = (time.time() - start_time) * 1000

        # 5개 쿼리 합쳐서 5ms 이내 (각 1ms 이내)
        assert elapsed_ms < 5.0


class TestSPEC_RAG_QUALITY_011_Req003:
    """REQ-003: Fallback Retrieval Mechanism Tests."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_keywords_bypass_llm(self):
        """AC-002.3: 키워드가 있으면 LLM 호출 없이 True 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        # LLM이 [RETRIEVE_NO]를 반환하도록 설정해도
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"

        evaluator = SelfRAGEvaluator(llm_client=mock_llm)

        # 키워드가 포함된 쿼리는 LLM 호출 없이 True 반환해야 함
        result = evaluator.needs_retrieval("휴학 규정이 뭐야?")
        assert result is True

    def test_override_activates_when_keywords_exist(self):
        """AC-003.1: LLM이 NO를 반환해도 키워드가 있으면 override."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"

        evaluator = SelfRAGEvaluator(llm_client=mock_llm)

        # 키워드가 포함된 쿼리 (override 발생)
        result = evaluator.needs_retrieval("장학금 신청 자격이 어떻게 돼?")
        assert result is True

    def test_greeting_returns_false(self):
        """AC-001.2: 단순 인사말은 False 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"

        evaluator = SelfRAGEvaluator(llm_client=mock_llm)

        # 단순 인사말 (키워드 없음, LLM도 NO 반환)
        result = evaluator.needs_retrieval("안녕하세요")
        assert result is False


class TestSPEC_RAG_QUALITY_011_Req006:
    """REQ-006: Classification Metrics Tests."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_metrics_method_exists(self):
        """AC-006.1: get_metrics 메서드가 존재해야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()
        assert hasattr(evaluator, "get_metrics")

    def test_metrics_track_retrieval_yes_count(self):
        """AC-006.1: retrieval_yes_count 추적."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_YES]"

        evaluator = SelfRAGEvaluator(llm_client=mock_llm)

        # 키워드 없는 쿼리로 LLM 호출 유도
        evaluator.needs_retrieval("이거 뭐야?")

        metrics = evaluator.get_metrics()
        assert "retrieval_yes_count" in metrics
        assert metrics["retrieval_yes_count"] >= 1

    def test_metrics_track_bypass_count(self):
        """AC-006.1: bypass_count 추적."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        # 키워드가 포함된 쿼리 (bypass 발생)
        evaluator.needs_retrieval("휴학 규정 알려줘")

        metrics = evaluator.get_metrics()
        assert "bypass_count" in metrics

    def test_metrics_track_override_count(self):
        """AC-003.3: override_count 추적."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"

        evaluator = SelfRAGEvaluator(llm_client=mock_llm)

        # 키워드 포함 + LLM NO = override 발생
        evaluator.needs_retrieval("장학금 규정이 뭐야?")

        metrics = evaluator.get_metrics()
        assert "override_count" in metrics


class TestSPEC_RAG_QUALITY_011_Accuracy:
    """Classification Accuracy Tests (AC-001.4)."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_regulation_queries_return_true(self):
        """규정 관련 쿼리는 True를 반환해야 함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()

        regulation_queries = [
            "휴학 신청 방법이 뭐야?",
            "졸업 요건이 어떻게 돼?",
            "장학금 신청 자격이 뭐야?",
            "학칙 제5조가 뭐야?",
            "등록금 납부 기간이 언제야?",
            "교원 임용 규정이 어떻게 돼?",
            "복수전공 신청은 어떻게 해?",
            "학점 인정 기준이 뭐야?",
            "성적 이의신청 방법 알려줘",
            "휴직 규정이 어떻게 돼?",
        ]

        correct = 0
        for query in regulation_queries:
            if evaluator.needs_retrieval(query):
                correct += 1

        # 95% 이상 정확도 (10개 중 10개)
        accuracy = correct / len(regulation_queries)
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} < 95%"


class TestSPEC_RAG_QUALITY_011_Req008:
    """REQ-008: Configuration for Self-RAG Behavior Tests."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_self_rag_config_fields_exist(self):
        """AC-008.1, AC-008.2, AC-008.3: Self-RAG 설정 필드가 존재해야 함."""
        from src.rag.config import RAGConfig

        config = RAGConfig()

        # 기본 설정
        assert hasattr(config, "enable_self_rag")
        # SPEC-RAG-QUALITY-011 추가 설정
        assert hasattr(config, "self_rag_keywords_path")
        assert hasattr(config, "self_rag_override_on_keywords")
        assert hasattr(config, "self_rag_log_overrides")

    def test_self_rag_defaults(self):
        """Self-RAG 기본값 확인."""
        from src.rag.config import RAGConfig

        config = RAGConfig()

        # 기본적으로 활성화
        assert config.enable_self_rag is True
        assert config.self_rag_override_on_keywords is True
        assert config.self_rag_log_overrides is True

    def test_self_rag_disabled_via_environment(self):
        """AC-008.1: 환경 변수로 Self-RAG 비활성화."""
        import os

        from src.rag.config import RAGConfig, reset_config

        # 환경 변수 설정
        os.environ["ENABLE_SELF_RAG"] = "false"
        reset_config()
        config = RAGConfig()

        assert config.enable_self_rag is False

        # 정리
        del os.environ["ENABLE_SELF_RAG"]
        reset_config()
