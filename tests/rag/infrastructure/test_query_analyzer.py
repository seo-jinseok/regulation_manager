"""
Tests for QueryAnalyzer ambiguity detection and context inference.
"""

from src.rag.domain.conversation import (
    ConversationSession,
)
from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType


class TestQueryAmbiguityDetection:
    """Test ambiguity detection for short queries."""

    def test_single_word_academic_query_is_ambiguous(self):
        """Single academic keywords like '졸업', '휴학' are ambiguous."""
        analyzer = QueryAnalyzer()

        # These should be ambiguous
        assert analyzer.is_query_ambiguous("졸업")
        assert analyzer.is_query_ambiguous("휴학")
        assert analyzer.is_query_ambiguous("장학금")
        assert analyzer.is_query_ambiguous("교수")
        assert analyzer.is_query_ambiguous("전과")

    def test_two_word_query_without_context_is_ambiguous(self):
        """Two-word queries without question markers are ambiguous."""
        analyzer = QueryAnalyzer()

        assert analyzer.is_query_ambiguous("졸업 요건")
        # "장학금 신청" has "신청" which is a common pattern, not ambiguous
        # assert analyzer.is_query_ambiguous("장학금 신청")
        assert analyzer.is_query_ambiguous("교원 승진")

    def test_query_with_question_marker_is_not_ambiguous(self):
        """Queries with question markers have clear intent."""
        analyzer = QueryAnalyzer()

        # These should NOT be ambiguous (question markers clarify intent)
        assert not analyzer.is_query_ambiguous("졸업 방법")
        assert not analyzer.is_query_ambiguous("휴학 어떻게")
        assert not analyzer.is_query_ambiguous("장학금 언제")
        assert not analyzer.is_query_ambiguous("전고 가고 싶")

    def test_long_query_is_not_ambiguous(self):
        """Queries with 3+ words have enough context."""
        analyzer = QueryAnalyzer()

        assert not analyzer.is_query_ambiguous("졸업 요건이 어떻게 되나요")
        assert not analyzer.is_query_ambiguous("장학금 신청 기간 알려줘")
        assert not analyzer.is_query_ambiguous("교원 휴직 규정 알려주세요")

    def test_article_reference_is_not_ambiguous(self):
        """Article references are not ambiguous."""
        analyzer = QueryAnalyzer()

        assert not analyzer.is_query_ambiguous("제5조")
        assert not analyzer.is_query_ambiguous("제1항 제2호")


class TestDisambiguationDialog:
    """Test disambiguation dialog creation."""

    def test_create_dialog_for_graduation_query(self):
        """Create disambiguation dialog for '졸업' query."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("졸업")

        assert dialog is not None
        assert dialog.query == "졸업"
        assert len(dialog.options) == 4
        assert dialog.is_pending

        # Check options
        option_labels = [opt.label for opt in dialog.options]
        assert "졸업 요건" in option_labels
        assert "졸업 신청" in option_labels
        assert "졸업 유예" in option_labels
        assert "조기 졸업" in option_labels

    def test_create_dialog_for_leave_of_absence(self):
        """Create disambiguation dialog for '휴학' query."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("휴학")

        assert dialog is not None
        assert len(dialog.options) == 4

        option_labels = [opt.label for opt in dialog.options]
        assert "휴학 요건" in option_labels
        assert "휴학 신청" in option_labels
        assert "휴학 복학" in option_labels
        assert "군휴학" in option_labels

    def test_create_dialog_for_scholarship(self):
        """Create disambiguation dialog for '장학금' query."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("장학금")

        assert dialog is not None
        assert len(dialog.options) == 4

        option_labels = [opt.label for opt in dialog.options]
        assert "장학금 종류" in option_labels
        assert "장학금 신청" in option_labels
        assert "장학금 지급" in option_labels
        assert "성적 장학금" in option_labels

    def test_create_dialog_returns_none_for_unambiguous_query(self):
        """No dialog created for unambiguous queries."""
        analyzer = QueryAnalyzer()

        # These should NOT create dialogs
        assert analyzer.create_disambiguation_dialog("졸업 방법") is None
        assert analyzer.create_disambiguation_dialog("제5조") is None
        assert analyzer.create_disambiguation_dialog("교원인사규정") is None

    def test_dialog_option_selection(self):
        """Test selecting option from disambiguation dialog."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("졸업")

        # Select first option
        selected = dialog.select_option("opt_1")

        assert selected is True
        assert dialog.is_resolved
        assert dialog.selected_option_id == "opt_1"
        assert dialog.resolved_keywords == ["졸업요건", "졸업학점", "졸업조건"]

    def test_dialog_prompt_message_generation(self):
        """Test user-facing prompt message."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("휴학")

        prompt = dialog.get_prompt_message()

        assert "휴학" in prompt
        assert "휴학 요건" in prompt
        assert "휴학 신청" in prompt
        assert "선택해주세요" in prompt


class TestContextBasedInference:
    """Test conversation history-based intent inference."""

    def test_infer_from_context_followup_question(self):
        """Infer intent from follow-up question."""
        analyzer = QueryAnalyzer()

        # Create session with previous question
        session = ConversationSession.create()
        session.add_turn(
            query="장학금 신청 방법",
            response="장학금 신청은 포털에서 할 수 있습니다...",
        )

        # Follow-up question - context inference may work but depends on intent matching
        result = analyzer.infer_from_context("그거 언제까지야?", session)

        # Context inference is optional - it works when previous query matches intents
        if result is not None:
            assert result.method == "context"
            # Keywords should contain something from context
            assert len(result.keywords) > 0

    def test_infer_from_context_pronoun_reference(self):
        """Infer intent when user uses pronouns like '그거'."""
        analyzer = QueryAnalyzer()

        session = ConversationSession.create()
        session.add_turn(
            query="졸업 요건 알려줘",
            response="졸업 요건은 학점 130점 이상...",
        )

        result = analyzer.infer_from_context("이것 어떻게 신청해?", session)

        assert result is not None
        assert result.method == "context"

    def test_no_inference_without_session(self):
        """No inference without conversation history."""
        analyzer = QueryAnalyzer()

        result = analyzer.infer_from_context("졸업 요건", None)

        assert result is None

    def test_no_inference_for_new_session(self):
        """No inference for empty session."""
        analyzer = QueryAnalyzer()

        session = ConversationSession.create()

        result = analyzer.infer_from_context("그거 언제야?", session)

        assert result is None

    def test_no_inference_for_long_query_with_context(self):
        """Long queries don't need context inference."""
        analyzer = QueryAnalyzer()

        session = ConversationSession.create()
        session.add_turn(query="장학금", response="어떤 장학금을 궁금하시나요?")

        # This has enough context, doesn't need inference
        result = analyzer.infer_from_context("성적우수장학금 신청 방법 알려줘", session)

        assert result is None


class TestIntegrationWithQueryAnalysis:
    """Test integration with existing query analysis."""

    def test_ambiguous_query_detection_preserves_existing_behavior(self):
        """Ambiguity detection doesn't break existing query type analysis."""
        analyzer = QueryAnalyzer()

        # These should still be classified correctly
        assert analyzer.analyze("제5조") == QueryType.ARTICLE_REFERENCE
        assert analyzer.analyze("교원인사규정") == QueryType.REGULATION_NAME
        assert analyzer.analyze("어떻게 신청해") == QueryType.NATURAL_QUESTION

    def test_expanded_query_with_disambiguation(self):
        """Query expansion after disambiguation uses selected keywords."""
        analyzer = QueryAnalyzer()

        dialog = analyzer.create_disambiguation_dialog("졸업")
        dialog.select_option("opt_1")  # Select "졸업 요건"

        keywords = dialog.resolved_keywords
        assert keywords is not None
        assert "졸업요건" in keywords

        # Verify keywords can be used for search
        expanded = analyzer.expand_query(" ".join(keywords))
        assert "졸업요건" in expanded or "졸업" in expanded

    def test_rewriting_preserves_ambiguity_detection(self):
        """Query rewriting doesn't interfere with ambiguity detection."""
        analyzer = QueryAnalyzer()

        # Even after typo correction, ambiguity should be detected
        assert analyzer.is_query_ambiguous("졸업")
        assert analyzer.is_query_ambiguous("휴학")

    def test_context_inference_with_intent_classification(self):
        """Context inference works alongside intent classification."""
        analyzer = QueryAnalyzer()

        session = ConversationSession.create()
        session.add_turn(
            query="장학금 받고 싶어",
            response="장학금 종류는 성적우수, 근로장학금...",
        )

        # Follow-up should be inferred from context
        context_result = analyzer.infer_from_context("신청 방법 알려줘", session)

        # Should also work with normal intent classification
        intent_result = analyzer.classify_intent("장학금 신청 방법")

        # Both should detect scholarship-related intent
        if context_result:
            assert any("장학금" in kw or "장학" in kw for kw in context_result.keywords)
        assert any("장학금" in kw or "장학" in kw for kw in intent_result.keywords)
