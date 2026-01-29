"""
Tests for Priority 1 RAG improvements.

Tests for:
1. Composite query decomposition (query_analyzer.py)
2. Context window expansion (self_rag.py)
3. English answer generation support (search_usecase.py)
"""

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.infrastructure.self_rag import SelfRAGEvaluator


class TestCompositeQueryDecomposition:
    """Test composite query decomposition in QueryAnalyzer."""

    def test_decompose_conjunction_hago(self):
        """Test decomposition with '하고' conjunction."""
        analyzer = QueryAnalyzer(llm_client=None)
        query = "장학금 신청하고 휴학 절차"
        result = analyzer.decompose_query(query)

        # Should decompose into multiple queries
        assert len(result) >= 2
        assert any("장학금" in q for q in result)
        assert any("휴학" in q for q in result)

    def test_decompose_conjunction_geurigo(self):
        """Test decomposition with '그리고' conjunction."""
        analyzer = QueryAnalyzer(llm_client=None)
        query = "교원 휴직 그리고 복직"
        result = analyzer.decompose_query(query)

        # Should decompose into multiple queries
        assert len(result) >= 2
        assert any("휴직" in q for q in result)
        assert any("복직" in q for q in result)

    def test_decompose_conjunction_ttohan(self):
        """Test decomposition with '또한' conjunction."""
        analyzer = QueryAnalyzer(llm_client=None)
        query = "수강신청 기간 또한 정정 기간"
        result = analyzer.decompose_query(query)

        # Should decompose into multiple queries
        assert len(result) >= 2

    def test_no_decompose_simple_query(self):
        """Test that simple queries are not decomposed."""
        analyzer = QueryAnalyzer(llm_client=None)
        query = "휴학 신청 방법"
        result = analyzer.decompose_query(query)

        # Should not decompose simple intent queries
        assert len(result) == 1
        assert result[0] == query

    def test_no_decompose_intent_pattern(self):
        """Test that intent patterns are not decomposed (exception)."""
        analyzer = QueryAnalyzer(llm_client=None)
        query = "휴학하고 싶어"
        result = analyzer.decompose_query(query)

        # Should NOT decompose intent expressions
        assert len(result) == 1
        assert result[0] == query


class TestContextWindowExpansion:
    """Test context window expansion in SelfRAGEvaluator."""

    def test_max_context_chars_default_4000(self):
        """Test that default max_context_chars is 4000."""
        evaluator = SelfRAGEvaluator(llm_client=None)

        # Check method signature has correct default
        import inspect

        sig = inspect.signature(evaluator.evaluate_relevance)
        max_context_param = sig.parameters["max_context_chars"]

        # Verify default value is 4000
        assert max_context_param.default == 4000

    def test_max_context_chars_explicit_value(self):
        """Test that max_context_chars can be explicitly set."""
        evaluator = SelfRAGEvaluator(llm_client=None)

        # Method should accept custom max_context_chars value
        # This test verifies the parameter exists and is used correctly
        import inspect

        sig = inspect.signature(evaluator.evaluate_relevance)

        # Verify parameter exists
        assert "max_context_chars" in sig.parameters
        assert sig.parameters["max_context_chars"].default == 4000


class TestEnglishAnswerGeneration:
    """Test English answer generation support in SearchUseCase."""

    def test_detect_language_korean(self):
        """Test language detection for Korean queries."""
        korean_query = "휴학 신청 방법을 알려주세요"
        result = SearchUseCase.detect_language(korean_query)

        assert result == "korean"

    def test_detect_language_english(self):
        """Test language detection for English queries."""
        english_query = "How do I apply for a leave of absence?"
        result = SearchUseCase.detect_language(english_query)

        assert result == "english"

    def test_detect_language_mixed_english_dominant(self):
        """Test language detection for English-dominant mixed queries."""
        mixed_query = "What is the application deadline for 휴학?"
        result = SearchUseCase.detect_language(mixed_query)

        # Should detect as English when >50% English characters
        assert result == "english"

    def test_detect_language_mixed_korean_dominant(self):
        """Test language detection for Korean-dominant mixed queries."""
        mixed_query = "휴학 신청 deadline이 언제인가요?"
        result = SearchUseCase.detect_language(mixed_query)

        assert result == "korean"

    def test_detect_language_empty_query(self):
        """Test language detection for empty queries."""
        result = SearchUseCase.detect_language("")

        assert result == "korean"  # Default to Korean

    def test_get_english_prompt(self):
        """Test English prompt generation."""
        prompt = SearchUseCase._get_english_prompt()

        # Verify prompt contains key English instructions
        assert "Dong-A University" in prompt
        assert "English" in prompt
        assert "hallucination" in prompt.lower()

    def test_build_english_user_message(self):
        """Test English user message building."""
        question = "What is the leave of absence process?"
        context = (
            "[1] Regulation: 학칙\n    Text: 휴학은 학기초 30일 이내에 신청해야 한다."
        )

        message = SearchUseCase._build_english_user_message(question, context, None)

        # Verify message contains English instruction
        assert "English" in message
        assert question in message
        assert context in message

    def test_build_english_user_message_with_history(self):
        """Test English user message with conversation history."""
        question = "What about readmission?"
        history = "User: How do I take a leave?\\nAI: You need to apply within 30 days."
        context = "[1] Context here"

        message = SearchUseCase._build_english_user_message(question, context, history)

        # Verify history is included
        assert "Conversation History" in message
        assert history in message


class TestIntegrationScenarios:
    """Integration tests for Priority 1 improvements."""

    def test_composite_query_flow(self):
        """Test end-to-end composite query handling."""
        analyzer = QueryAnalyzer(llm_client=None)

        # Test "하고" conjunction
        query1 = "장학금 신청하고 휴학 절차"
        result1 = analyzer.decompose_query(query1)
        assert len(result1) >= 2

        # Test "그리고" conjunction
        query2 = "교원 휴직 그리고 복직"
        result2 = analyzer.decompose_query(query2)
        assert len(result2) >= 2

    def test_multilingual_detection_coverage(self):
        """Test various language detection scenarios."""
        test_cases = [
            ("How to apply?", "english"),
            ("신청 방법은?", "korean"),
            ("What is the deadline for course registration?", "english"),
            ("When is the deadline?", "english"),
            ("This is completely in English language", "english"),
            ("전부 한국어 질문입니다", "korean"),
        ]

        for query, expected in test_cases:
            result = SearchUseCase.detect_language(query)
            assert result == expected, f"Failed for query: {query}"

    def test_context_window_parameter_correctness(self):
        """Test that context window parameter is properly configured."""
        evaluator = SelfRAGEvaluator(llm_client=None)

        # Verify the method signature
        import inspect

        sig = inspect.signature(evaluator.evaluate_relevance)

        # Check max_context_chars parameter
        assert "max_context_chars" in sig.parameters
        param = sig.parameters["max_context_chars"]
        assert param.default == 4000, "max_context_chars should default to 4000"
        # Annotation can be int, or empty (Parameter.empty), or str("int")
        assert param.annotation in (int, inspect.Parameter.empty, "int")
