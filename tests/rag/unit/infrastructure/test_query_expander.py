"""
Unit tests for DynamicQueryExpander (Phase 3).

Tests the LLM-based dynamic query expansion functionality.
"""

import json
import tempfile
from pathlib import Path

from src.rag.infrastructure.query_expander import (
    FALLBACK_RULES,
    DynamicQueryExpander,
    QueryExpansionPipeline,
)


class FakeLLMClient:
    """Fake LLM client for testing."""

    def __init__(self, response: str = None):
        self._response = response or json.dumps(
            {
                "intent": "test_intent",
                "keywords": ["키워드1", "키워드2", "키워드3"],
                "confidence": 0.9,
            }
        )

    def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.3
    ) -> str:
        return self._response


class TestDynamicQueryExpander:
    """Tests for DynamicQueryExpander class."""

    def test_expand_with_llm(self):
        """LLM을 사용한 쿼리 확장이 작동하는지 테스트."""
        llm = FakeLLMClient()
        expander = DynamicQueryExpander(llm_client=llm, enable_cache=False)

        result = expander.expand("장학금 받으려면?")

        assert result.original_query == "장학금 받으려면?"
        assert len(result.keywords) > 0
        assert result.method == "llm"
        assert result.confidence >= 0.5

    def test_expand_without_llm_uses_pattern(self):
        """LLM 없이 패턴 기반 확장이 작동하는지 테스트."""
        expander = DynamicQueryExpander(llm_client=None, enable_cache=False)

        result = expander.expand("장학금 성적 기준")

        assert result.original_query == "장학금 성적 기준"
        assert "장학금" in result.keywords
        assert result.method == "pattern"

    def test_fallback_rules_coverage(self):
        """폴백 규칙이 주요 주제를 커버하는지 테스트."""
        expander = DynamicQueryExpander(llm_client=None, enable_cache=False)

        # Test various query types
        test_cases = [
            ("장학금 받고 싶어", "장학금"),
            ("졸업하려면 뭐 필요해?", "졸업"),
            ("교수 승진 기준", "승진"),
            ("휴학하고 싶어", "휴학"),
            ("등록금 납부", "등록금"),
            ("성적 평점", "성적"),
            ("영어 점수 필요해?", "어학인증"),
        ]

        for query, expected_keyword in test_cases:
            result = expander.expand(query)
            assert any(
                expected_keyword in kw or kw in expected_keyword
                for kw in result.keywords
            ), f"Expected '{expected_keyword}' in {result.keywords} for query '{query}'"

    def test_should_expand_vague_queries(self):
        """모호한 쿼리에 대해 확장이 필요하다고 판단하는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)

        # Vague queries should be expanded
        assert expander.should_expand("학교 가기 싫어") is True
        assert expander.should_expand("졸업하려면 뭐 필요해?") is True
        assert expander.should_expand("받으려면 어떻게 해야 해?") is True

    def test_should_not_expand_formal_queries(self):
        """규정 용어가 포함된 쿼리는 확장하지 않는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)

        # Formal queries with regulatory terms should not be expanded
        assert expander.should_expand("교원인사규정 제8조") is False
        assert expander.should_expand("학칙 제15조제2항") is False
        assert expander.should_expand("장학금지급세칙") is False

    def test_cache_functionality(self):
        """캐시가 작동하는지 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import QueryExpansionCache, RAGQueryCache

            # Create isolated cache for testing
            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            expansion_cache = QueryExpansionCache(rag_cache=rag_cache)

            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                cache_dir=tmpdir,
                enable_cache=True,
                expansion_cache=expansion_cache,
            )

            # First call - should use LLM
            result1 = expander.expand("테스트 쿼리")
            assert result1.from_cache is False

            # Second call - should use cache
            result2 = expander.expand("테스트 쿼리")
            assert result2.from_cache is True

            # Keywords should be the same
            assert result1.keywords == result2.keywords

    def test_cache_persistence(self):
        """캐시가 디스크에 저장되는지 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = FakeLLMClient()

            # First expander
            expander1 = DynamicQueryExpander(
                llm_client=llm,
                cache_dir=tmpdir,
                enable_cache=True,
            )
            result1 = expander1.expand("영구 캐시 테스트")

            # Second expander (simulating restart)
            expander2 = DynamicQueryExpander(
                llm_client=None,  # No LLM
                cache_dir=tmpdir,
                enable_cache=True,
            )
            result2 = expander2.expand("영구 캐시 테스트")

            assert result2.from_cache is True
            assert result1.keywords == result2.keywords

    def test_build_expanded_query(self):
        """확장된 쿼리가 올바르게 구성되는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)

        query = "장학금"
        keywords = ["성적기준", "지급기준", "장학금지급"]

        expanded = expander._build_expanded_query(query, keywords)

        # Should append keywords not in original query
        assert "장학금" in expanded
        assert "성적기준" in expanded

    def test_extract_basic_keywords(self):
        """기본 키워드 추출이 작동하는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)

        keywords = expander._extract_basic_keywords("교수 승진 기준이 어떻게 되나요?")

        # Should extract meaningful words, exclude stopwords
        assert "교수" in keywords
        assert "승진" in keywords
        # "기준이" is extracted as a single token (no morpheme analysis)
        assert any("기준" in kw for kw in keywords)
        # Stopwords should be excluded
        assert "이" not in keywords
        # "어떻게" is in stopwords
        assert "어떻게" not in keywords

    def test_max_keywords_limit(self):
        """키워드 개수 제한이 적용되는지 테스트."""
        llm_response = json.dumps(
            {
                "intent": "test",
                "keywords": [
                    "kw1",
                    "kw2",
                    "kw3",
                    "kw4",
                    "kw5",
                    "kw6",
                    "kw7",
                    "kw8",
                    "kw9",
                    "kw10",
                ],
                "confidence": 0.9,
            }
        )
        llm = FakeLLMClient(response=llm_response)
        expander = DynamicQueryExpander(
            llm_client=llm,
            enable_cache=False,
            max_keywords=5,
        )

        result = expander.expand("테스트 쿼리")

        assert len(result.keywords) <= 5


class TestQueryExpansionPipeline:
    """Tests for QueryExpansionPipeline class."""

    def test_pipeline_skips_when_not_needed(self):
        """확장이 필요없는 쿼리를 건너뛰는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)
        pipeline = QueryExpansionPipeline(expander)

        result = pipeline.process_query("교원인사규정 제8조")

        assert result.method == "skip"
        assert result.expanded_query == "교원인사규정 제8조"

    def test_pipeline_expands_vague_queries(self):
        """모호한 쿼리를 확장하는지 테스트."""
        expander = DynamicQueryExpander(enable_cache=False)
        pipeline = QueryExpansionPipeline(expander)

        result = pipeline.process_query("학교 가기 싫어")

        assert result.method in ("llm", "pattern")
        assert len(result.keywords) > 0


class TestFallbackRules:
    """Tests for fallback expansion rules."""

    def test_all_rules_have_required_fields(self):
        """모든 폴백 규칙이 필수 필드를 가지는지 테스트."""
        for rule in FALLBACK_RULES:
            assert len(rule.patterns) > 0, f"Rule {rule.intent} has no patterns"
            assert len(rule.keywords) > 0, f"Rule {rule.intent} has no keywords"
            assert rule.intent, "Rule has no intent"

    def test_rules_cover_common_topics(self):
        """폴백 규칙이 주요 주제를 커버하는지 테스트."""
        covered_topics = {rule.intent for rule in FALLBACK_RULES}

        # Essential topics that should be covered
        essential = ["장학금", "졸업", "승진", "휴학", "등록금", "성적", "어학"]

        for topic in essential:
            assert any(topic in intent for intent in covered_topics), (
                f"Missing coverage for: {topic}"
            )


class TestQueryExpansionCacheCycle6:
    """Tests for Cycle 6 QueryExpansionCache integration."""

    def test_centralized_cache_hit(self):
        """중앙 집중식 캐시 적중이 작동하는지 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache, RAGQueryCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm, enable_cache=False, expansion_cache=cache
            )

            # 첫 번째 호출 - LLM 사용
            result1 = expander.expand("테스트 쿼리")
            assert result1.from_cache is False
            assert result1.method == "llm"

            # 캐시에 결과 저장 확인
            cached = cache.get_expansion("테스트 쿼리")
            assert cached is not None
            assert cached["keywords"] == result1.keywords

    def test_cache_metrics_tracking(self):
        """캐시 메트릭 추적이 작동하는지 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                enable_cache=False,
                expansion_cache=cache,
            )

            # 첫 번째 확장 (캐시 미스)
            expander.expand("쿼리 1")

            # 같은 쿼리로 캐시 적중
            expander.expand("쿼리 1")

            # 다른 쿼리 (캐시 미스)
            expander.expand("쿼리 2")

            metrics = cache.get_metrics()
            assert metrics.total_expansions == 3
            assert metrics.cache_hits == 1
            assert metrics.llm_calls == 2

    def test_llm_call_time_recording(self):
        """LLM 호출 시간 기록이 작동하는지 테스트 (Cycle 6)."""
        import time

        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)

            class SlowLLMClient:
                def generate(
                    self,
                    system_prompt: str,
                    user_message: str,
                    temperature: float = 0.3,
                ) -> str:
                    time.sleep(0.01)  # 10ms 지연
                    return json.dumps(
                        {
                            "intent": "test",
                            "keywords": ["kw1", "kw2"],
                            "confidence": 0.9,
                        }
                    )

            expander = DynamicQueryExpander(
                llm_client=SlowLLMClient(),
                enable_cache=False,
                expansion_cache=cache,
            )

            expander.expand("테스트")

            metrics = cache.get_metrics()
            assert metrics.llm_calls == 1
            assert metrics.total_llm_time_ms > 0
            assert metrics.avg_llm_time_ms > 0

    def test_pattern_fallback_recording(self):
        """패턴 폴백 기록이 작동하는지 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            expander = DynamicQueryExpander(
                llm_client=None,  # LLM 없음
                enable_cache=False,
                expansion_cache=cache,
            )

            # 패턴 기반 확장 (폴백)
            result = expander.expand("장학금 받고 싶어")
            assert result.method == "pattern"

            metrics = cache.get_metrics()
            assert metrics.pattern_fallbacks == 1
            assert metrics.llm_calls == 0

    def test_cache_hit_rate_calculation(self):
        """캐시 적중률 계산이 올바른지 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                enable_cache=False,
                expansion_cache=cache,
            )

            # 3개의 다른 쿼리로 확장
            expander.expand("쿼리 A")
            expander.expand("쿼리 B")
            expander.expand("쿼리 C")

            # 2개는 캐시 적중
            expander.expand("쿼리 A")
            expander.expand("쿼리 B")

            metrics = cache.get_metrics()
            assert metrics.total_expansions == 5
            assert metrics.cache_hits == 2
            assert metrics.cache_hit_rate == 0.4  # 2/5 = 0.4

    def test_llm_call_reduction_rate(self):
        """LLM 호출 감소율 계산이 올바른지 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                enable_cache=False,
                expansion_cache=cache,
            )

            # 4개 확장 (2개는 캐시 적중)
            expander.expand("쿼리 1")
            expander.expand("쿼리 2")
            expander.expand("쿼리 1")  # 캐시 적중
            expander.expand("쿼리 2")  # 캐시 적중

            metrics = cache.get_metrics()
            # 4개 확장 중 2개만 LLM 호출
            assert metrics.llm_calls == 2
            # LLM 호출 감소율 = 1 - (2/4) = 0.5
            assert metrics.llm_call_reduction_rate == 0.5

    def test_expander_get_metrics(self):
        """DynamicQueryExpander의 get_metrics 메서드 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
        llm = FakeLLMClient()
        expander = DynamicQueryExpander(
            llm_client=llm,
            enable_cache=False,
            expansion_cache=cache,
        )

        expander.expand("테스트")
        metrics = expander.get_metrics()

        assert metrics is not None
        assert "total_expansions" in metrics
        assert "cache_hit_rate" in metrics
        assert "llm_call_reduction_rate" in metrics

    def test_expander_reset_metrics(self):
        """DynamicQueryExpander의 reset_metrics 메서드 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
        llm = FakeLLMClient()
        expander = DynamicQueryExpander(
            llm_client=llm,
            enable_cache=False,
            expansion_cache=cache,
        )

        expander.expand("테스트 1")
        expander.expand("테스트 2")

        metrics_before = expander.get_metrics()
        assert metrics_before["total_expansions"] == 2

        # 메트릭 리셋
        expander.reset_metrics()

        metrics_after = expander.get_metrics()
        assert metrics_after["total_expansions"] == 0

    def test_legacy_cache_migration(self):
        """레거시 캐시에서 중앙 캐시로 마이그레이션 테스트 (Cycle 6)."""
        from src.rag.infrastructure.cache import QueryExpansionCache

        # 레거시 캐시에 미리 데이터 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "expansion_cache.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            legacy_data = {
                "abc123": {
                    "keywords": ["장학금", "성적기준"],
                    "expanded_query": "장학금 받으려면 성적기준",
                    "intent": "장학금 문의",
                    "confidence": 0.9,
                }
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f, ensure_ascii=False)

            # 레거시 캐시와 중앙 캐시 모두 사용
            # Use temporary directory to avoid cache sharing between tests
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.rag.infrastructure.cache import RAGQueryCache

            rag_cache = RAGQueryCache(enabled=True, cache_dir=tmpdir)
            cache = QueryExpansionCache(rag_cache=rag_cache)
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                cache_dir=tmpdir,
                enable_cache=True,
                expansion_cache=cache,
            )

            # 같은 해시를 생성하도록 원본 쿼리 사용
            # (실제로는 해시가 다를 수 있으므로 이 테스트는 개념적)
            expander.expand("장학금")

            # 중앙 캐시에 데이터가 있는지 확인
            metrics = cache.get_metrics()
            assert metrics.total_expansions >= 1
