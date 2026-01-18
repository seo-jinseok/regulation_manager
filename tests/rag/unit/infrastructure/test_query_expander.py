"""
Unit tests for DynamicQueryExpander (Phase 3).

Tests the LLM-based dynamic query expansion functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json

from src.rag.infrastructure.query_expander import (
    DynamicQueryExpander,
    QueryExpansionResult,
    QueryExpansionPipeline,
    FALLBACK_RULES,
)


class FakeLLMClient:
    """Fake LLM client for testing."""

    def __init__(self, response: str = None):
        self._response = response or json.dumps({
            "intent": "test_intent",
            "keywords": ["키워드1", "키워드2", "키워드3"],
            "confidence": 0.9,
        })

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
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
            assert any(expected_keyword in kw or kw in expected_keyword 
                      for kw in result.keywords), f"Expected '{expected_keyword}' in {result.keywords} for query '{query}'"

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
            llm = FakeLLMClient()
            expander = DynamicQueryExpander(
                llm_client=llm,
                cache_dir=tmpdir,
                enable_cache=True,
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
        llm_response = json.dumps({
            "intent": "test",
            "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6", "kw7", "kw8", "kw9", "kw10"],
            "confidence": 0.9,
        })
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
            assert rule.intent, f"Rule has no intent"

    def test_rules_cover_common_topics(self):
        """폴백 규칙이 주요 주제를 커버하는지 테스트."""
        covered_topics = {rule.intent for rule in FALLBACK_RULES}

        # Essential topics that should be covered
        essential = ["장학금", "졸업", "승진", "휴학", "등록금", "성적", "어학"]

        for topic in essential:
            assert any(topic in intent for intent in covered_topics), f"Missing coverage for: {topic}"
