"""
Tests for UnstructuredRegulationAnalyzer (SPEC-HWXP-002, TASK-005).

TDD Approach: RED Phase
- Write failing tests first
- Define interface through tests
- Test LLM integration, prompt engineering, JSON parsing
- Test confidence scoring, timeout handling, fallback behavior
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.parsing.analyzers.unstructured_regulation_analyzer import (
    UnstructuredRegulationAnalyzer,
    LLMResponse,
    StructureInferenceResult
)
from src.parsing.format.format_type import FormatType
import json


class TestUnstructuredRegulationAnalyzer:
    """Test suite for UnstructuredRegulationAnalyzer initialization."""

    def test_init_with_default_parameters(self):
        """Test analyzer initialization with default parameters."""
        analyzer = UnstructuredRegulationAnalyzer()
        assert analyzer.timeout == 30
        assert analyzer.confidence_threshold == 0.7
        assert analyzer.model == "gpt-4o-mini"

    def test_init_with_custom_parameters(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = UnstructuredRegulationAnalyzer(
            timeout=60,
            confidence_threshold=0.8,
            model="gpt-4o"
        )
        assert analyzer.timeout == 60
        assert analyzer.confidence_threshold == 0.8
        assert analyzer.model == "gpt-4o"

    def test_init_with_llm_client(self):
        """Test analyzer initialization with custom LLM client."""
        mock_llm_client = Mock()
        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm_client)
        assert analyzer.llm_client == mock_llm_client

    def test_init_without_llm_client_raises_error(self):
        """Test that missing LLM client raises appropriate error."""
        with patch('src.parsing.analyzers.unstructured_regulation_analyzer.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="openai is required"):
                UnstructuredRegulationAnalyzer(llm_client=None)


class TestPromptBuilding:
    """Test suite for LLM prompt engineering."""

    def test_build_system_prompt_korean(self):
        """Test system prompt is built in Korean for legal text."""
        analyzer = UnstructuredRegulationAnalyzer()
        prompt = analyzer._build_system_prompt()

        # Should contain Korean legal structure keywords
        assert "규정" in prompt or "structure" in prompt
        assert "조문" in prompt or "article" in prompt
        assert "항" in prompt or "item" in prompt

    def test_build_system_prompt_contains_instructions(self):
        """Test system prompt contains JSON output instructions."""
        analyzer = UnstructuredRegulationAnalyzer()
        prompt = analyzer._build_system_prompt()

        # Should specify JSON output format
        assert "JSON" in prompt or "json" in prompt

    def test_build_user_prompt_with_simple_text(self):
        """Test user prompt construction with simple text."""
        analyzer = UnstructuredRegulationAnalyzer()
        title = "테스트 규정"
        content = "이것은 테스트 내용입니다."

        prompt = analyzer._build_user_prompt(title, content)

        assert title in prompt
        assert content in prompt

    def test_build_user_prompt_with_complex_text(self):
        """Test user prompt with complex Korean legal text."""
        analyzer = UnstructuredRegulationAnalyzer()
        title = "대학 규정"
        content = "제1조 목적\n이 규정은 대학의 운영에 관한 사항을 규정함을 목적으로 한다."

        prompt = analyzer._build_user_prompt(title, content)

        assert title in prompt
        assert content in prompt

    def test_build_prompt_with_empty_content(self):
        """Test prompt building with empty content."""
        analyzer = UnstructuredRegulationAnalyzer()
        prompt = analyzer._build_user_prompt("제목", "")

        # Should handle empty content gracefully
        assert "제목" in prompt


class TestJSONResponseParsing:
    """Test suite for LLM JSON response parsing."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response from LLM."""
        analyzer = UnstructuredRegulationAnalyzer()
        json_response = '''
        {
            "structure_type": "article",
            "confidence": 0.9,
            "provisions": [
                {"number": "1", "content": "목적"},
                {"number": "2", "content": "정의"}
            ]
        }
        '''

        result = analyzer._parse_json_response(json_response)

        assert result.structure_type == "article"
        assert result.confidence == 0.9
        assert len(result.provisions) == 2
        assert result.provisions[0]["content"] == "목적"

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON surrounded by extra text (common LLM behavior)."""
        analyzer = UnstructuredRegulationAnalyzer()
        json_with_text = '''
        다음은 분석 결과입니다:

        ```json
        {
            "structure_type": "list",
            "confidence": 0.85,
            "provisions": [{"number": "1", "content": "항목 1"}]
        }
        ```

        분석이 완료되었습니다.
        '''

        result = analyzer._parse_json_response(json_with_text)

        assert result.structure_type == "list"
        assert result.confidence == 0.85

    def test_parse_json_with_korean_fields(self):
        """Test parsing JSON with Korean field names."""
        analyzer = UnstructuredRegulationAnalyzer()
        korean_json = '''
        {
            "구조_유형": "article",
            "신뢰도": 0.88,
            "조문": [
                {"번호": "1", "내용": "목적"}
            ]
        }
        '''

        result = analyzer._parse_json_response(korean_json)

        # Should handle Korean field names or normalized to English
        assert result.confidence == 0.88

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises parsing error."""
        analyzer = UnstructuredRegulationAnalyzer()
        invalid_json = "This is not valid JSON at all"

        with pytest.raises(json.JSONDecodeError):
            analyzer._parse_json_response(invalid_json)

    def test_parse_json_with_missing_required_fields(self):
        """Test JSON with missing required fields."""
        analyzer = UnstructuredRegulationAnalyzer()
        incomplete_json = '''
        {
            "structure_type": "article"
        }
        '''

        # Should handle missing fields gracefully
        result = analyzer._parse_json_response(incomplete_json)
        assert result.structure_type == "article"

    def test_parse_json_with_invalid_confidence_range(self):
        """Test JSON with confidence outside 0-1 range."""
        analyzer = UnstructuredRegulationAnalyzer()
        invalid_confidence = '''
        {
            "structure_type": "article",
            "confidence": 1.5
        }
        '''

        result = analyzer._parse_json_response(invalid_confidence)
        # Should normalize or handle invalid confidence
        assert 0 <= result.confidence <= 1


class TestConfidenceScoring:
    """Test suite for confidence scoring algorithm."""

    def test_calculate_confidence_from_structure_clarity(self):
        """Test confidence based on structure clarity indicators."""
        analyzer = UnstructuredRegulationAnalyzer()

        # Clear article markers → high confidence
        text_with_articles = "제1조 목적\n제2조 정의"
        score1 = analyzer._calculate_structure_confidence(text_with_articles)
        assert score1 > 0.7

        # No clear structure → lower confidence
        text_without_structure = "이것은 그냥 일반 텍스트입니다."
        score2 = analyzer._calculate_structure_confidence(text_without_structure)
        assert score2 < score1

    def test_confidence_with_numbered_list(self):
        """Test confidence calculation for numbered lists."""
        analyzer = UnstructuredRegulationAnalyzer()

        numbered_list = "1. 첫 번째 항목\n2. 두 번째 항목\n3. 세 번째 항목"
        score = analyzer._calculate_structure_confidence(numbered_list)
        assert score > 0.6

    def test_confidence_with_korean_alphabet_list(self):
        """Test confidence for Korean alphabet lists."""
        analyzer = UnstructuredRegulationAnalyzer()

        korean_list = "가. 첫 번째\n나. 두 번째\n다. 세 번째"
        score = analyzer._calculate_structure_confidence(korean_list)
        assert score > 0.6

    def test_confidence_combines_multiple_factors(self):
        """Test that confidence combines multiple structural indicators."""
        analyzer = UnstructuredRegulationAnalyzer()

        # Multiple indicators → higher confidence
        multi_indicator = "제1조 목적\n1. 세부 사항\n가. 상세 내용"
        score = analyzer._calculate_structure_confidence(multi_indicator)
        assert score > 0.8

    def test_confidence_normalization(self):
        """Test confidence is always between 0 and 1."""
        analyzer = UnstructuredRegulationAnalyzer()

        for test_text in [
            "",  # Empty
            "단어",  # Very short
            "제1조" * 100,  # Very long
            "제1조\n제2조\n제3조",  # Multiple articles
        ]:
            score = analyzer._calculate_structure_confidence(test_text)
            assert 0 <= score <= 1


class TestTimeoutHandling:
    """Test suite for timeout handling."""

    def test_timeout_default_30_seconds(self):
        """Test default timeout is 30 seconds."""
        analyzer = UnstructuredRegulationAnalyzer()
        assert analyzer.timeout == 30

    @pytest.mark.timeout(35)  # Allow slightly more than timeout
    def test_llm_call_respects_timeout(self):
        """Test that LLM call respects timeout setting."""
        mock_llm = Mock()
        # Simulate slow LLM that would exceed timeout
        import time

        def slow_generate(*args, **kwargs):
            time.sleep(35)  # Exceeds default 30s timeout
            return "Response"

        mock_llm.generate = slow_generate

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm, timeout=30)

        # Should handle timeout gracefully
        result = analyzer.analyze("제목", "내용")
        assert result is not None

    def test_timeout_falls_back_to_raw_text(self):
        """Test that timeout triggers fallback to raw text."""
        analyzer = UnstructuredRegulationAnalyzer()
        analyzer.llm_client = Mock()

        # Simulate timeout
        import asyncio

        def timeout_generate(*args, **kwargs):
            raise TimeoutError("LLM request timed out")

        analyzer.llm_client.generate = timeout_generate

        result = analyzer.analyze("제목", "내용")

        # Should fallback to raw text analysis
        assert result is not None
        assert "metadata" in result

    def test_custom_timeout_setting(self):
        """Test custom timeout configuration."""
        analyzer = UnstructuredRegulationAnalyzer(timeout=60)
        assert analyzer.timeout == 60


class TestFallbackBehavior:
    """Test suite for fallback to raw text on LLM failure."""

    def test_fallback_on_llm_error(self):
        """Test fallback when LLM raises exception."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM API error")

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("제목", "내용")

        # Should still return result with raw text
        assert result is not None
        assert "metadata" in result
        assert result["metadata"]["format_type"] == FormatType.UNSTRUCTURED.value

    def test_fallback_on_invalid_json(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "This is not valid JSON"

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("제목", "내용")

        # Should fallback gracefully
        assert result is not None

    def test_fallback_on_low_confidence(self):
        """Test fallback when confidence is below threshold."""
        mock_llm = Mock()
        # Return low confidence response
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "unstructured",
            "confidence": 0.5,  # Below default 0.7 threshold
            "provisions": []
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm, confidence_threshold=0.7)
        result = analyzer.analyze("제목", "내용")

        # Should still return result
        assert result is not None

    def test_fallback_preserves_content(self):
        """Test that fallback preserves original content."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        title = "테스트 규정"
        content = "이것은 테스트 내용입니다."

        result = analyzer.analyze(title, content)

        # Should preserve original content
        assert result["metadata"]["title"] == title


class TestAnalyzeMethod:
    """Test suite for the main analyze method."""

    def test_analyze_returns_valid_structure(self):
        """Test analyze returns valid result structure."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "article",
            "confidence": 0.9,
            "provisions": [
                {"number": "1", "content": "목적"}
            ]
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("제목", "내용")

        # Should have required fields
        assert "provisions" in result
        assert "articles" in result
        assert "metadata" in result
        assert result["metadata"]["format_type"] == FormatType.UNSTRUCTURED.value

    def test_analyze_with_high_confidence(self):
        """Test analyze with high confidence LLM response."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "article",
            "confidence": 0.95,
            "provisions": [
                {"number": "1", "content": "제1조 목적"},
                {"number": "2", "content": "제2조 정의"}
            ]
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("규정", "제1조 목적\n제2조 정의")

        # Should use LLM structure
        assert len(result["provisions"]) >= 2
        assert result["metadata"]["confidence"] >= 0.95

    def test_analyze_with_empty_content(self):
        """Test analyze with empty content."""
        mock_llm = Mock()
        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("제목", "")

        # Should handle empty content
        assert result is not None
        assert result["provisions"] == []

    def test_analyze_integration_with_llm_client(self):
        """Test analyze method integrates with LLM client correctly."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "list",
            "confidence": 0.85,
            "provisions": [
                {"number": "1", "content": "항목 1"}
            ]
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze("리스트 규정", "1. 항목 1\n2. 항목 2")

        # Verify LLM was called
        assert mock_llm.generate.called
        # Should return structured result
        assert result["metadata"]["format_type"] == FormatType.UNSTRUCTURED.value


class TestDataModels:
    """Test suite for data models."""

    def test_llm_response_model(self):
        """Test LLMResponse data model."""
        response = LLMResponse(
            structure_type="article",
            confidence=0.9,
            provisions=[{"number": "1", "content": "내용"}]
        )

        assert response.structure_type == "article"
        assert response.confidence == 0.9
        assert len(response.provisions) == 1

    def test_structure_inference_result_model(self):
        """Test StructureInferenceResult data model."""
        result = StructureInferenceResult(
            inferred_type="article",
            confidence=0.85,
            provisions=["제1조", "제2조"]
        )

        assert result.inferred_type == "article"
        assert result.confidence == 0.85
        assert len(result.provisions) == 2


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_very_long_content_handling(self):
        """Test handling of very long content."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "unstructured",
            "confidence": 0.6,
            "provisions": []
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        long_content = "내용" * 10000  # Very long content

        # Should handle without crashing
        result = analyzer.analyze("긴 규정", long_content)
        assert result is not None

    def test_special_characters_in_content(self):
        """Test handling of special Korean characters."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "article",
            "confidence": 0.8,
            "provisions": [{"number": "1", "content": "내용"}]
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        special_content = "① ② ③ Ⅳ Ⅴ ← → ↑ ↓"

        result = analyzer.analyze("특수 문자", special_content)
        assert result is not None

    def test_mixed_structure_content(self):
        """Test content with mixed structure indicators."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "structure_type": "mixed",
            "confidence": 0.75,
            "provisions": [
                {"number": "1", "content": "항목 1"},
                {"number": "가", "content": "세부 항목"}
            ]
        })

        analyzer = UnstructuredRegulationAnalyzer(llm_client=mock_llm)
        mixed_content = "제1조 목적\n1. 세부 사항\n가. 상세 내용"

        result = analyzer.analyze("혼합 구조", mixed_content)
        assert result is not None
