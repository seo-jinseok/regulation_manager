"""
Characterization tests for emotional query detection.

These tests capture the CURRENT behavior of emotion-related query processing
before implementing Emotional Query Support (Component 4).

All tests document WHAT IS, not what SHOULD BE.
"""

from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType


class TestEmotionalIntentPatternsCharacterization:
    """Characterize current emotional intent pattern matching behavior."""

    def test_distress_pattern_himdul_sipseo(self):
        """Characterize: Intent pattern matches '힘들어' with '싶어' combinations."""
        analyzer = QueryAnalyzer()

        # Current behavior: matches intent pattern for "학업하기 싫" -> 휴학, 자퇴
        result = analyzer.rewrite_query_with_info("공부하기 힘들어서 싫어")

        # Characterize current behavior
        # Note: '싶' appears in label text "싫어/안 하고 싶은 표현 이의"
        assert result.used_intent is True
        # 휴학 is extracted as intent keyword
        assert "휴학" in result.rewritten

    def test_distress_pattern_otokhey(self):
        """Characterize: Intent pattern matching for '어떡해요' expressions."""
        analyzer = QueryAnalyzer()

        # "어떡해요" doesn't have specific pattern yet
        result = analyzer.rewrite_query_with_info("장학금 어떡해요")

        # Characterize: no specific intent match, uses general expansion
        assert result.used_intent is False

    def test_frustration_pattern_andwaeyo(self):
        """Characterize: '안돼요' frustration expressions."""
        analyzer = QueryAnalyzer()

        # "안돼요" has intent pattern for 거절/반대
        result = analyzer.rewrite_query_with_info("안하고 싶어")

        # Characterize: matches rejection/intent pattern
        assert result.used_intent is True
        # Matched intents include the full label text from intent patterns
        assert any(
            "거절" in intent or "반대" in intent or "거부" in intent
            for intent in result.matched_intents
        )

    def test_seeking_help_pattern_mall(self):
        """Characterize: '몰라요' seeking help expressions."""
        analyzer = QueryAnalyzer()

        result = analyzer.rewrite_query_with_info("휴학 어떻게 해요 몰라요")

        # Characterize: NO specific HOW-TO patterns, just synonym expansion
        # Current behavior: does NOT extract 방법/절차 keywords
        assert "방법" not in result.rewritten
        assert "절차" not in result.rewritten
        # But does expand with synonyms
        assert result.used_synonyms is True

    def test_urgency_pattern_geuphey(self):
        """Characterize: Urgency indicator '급해요' patterns."""
        # Characterize: urgency is not a separate classification
        # Just treated as regular query
        # EMOTIONAL QueryType does not exist yet
        assert "EMOTIONAL" not in [t.name for t in QueryType]
        assert "URGENT" not in [t.name for t in QueryType]


class TestEmotionalQueryTypeClassificationCharacterization:
    """Characterize current QueryType classification for emotional queries."""

    def test_emotional_intent_classified_as_intent_type(self):
        """Characterize: Queries with emotion markers classified as INTENT type."""
        analyzer = QueryAnalyzer()

        # "싫어" triggers intent marker
        query_type = analyzer.analyze("학교 가기 싫어")

        # Current behavior: Intent markers trigger INTENT type
        assert query_type == QueryType.INTENT
        # EMOTIONAL QueryType does not exist yet
        assert "EMOTIONAL" not in [t.name for t in QueryType]

    def test_distress_with_academic_keyword(self):
        """Characterize: Distress + academic keyword classified as REGULATION_NAME."""
        analyzer = QueryAnalyzer()

        # "힘들어" + "휴학" - academic keyword takes precedence
        query_type = analyzer.analyze("휴학 힘들어요")

        # Current behavior: Academic keywords trigger REGULATION_NAME
        assert query_type == QueryType.REGULATION_NAME

    def test_seeking_help_question_classified_as_natural_question(self):
        """Characterize: '어떻게 해요' classification."""
        analyzer = QueryAnalyzer()

        query_type = analyzer.analyze("휴학 어떻게 해요")

        # Current behavior: Academic keywords take precedence over question markers
        assert query_type == QueryType.REGULATION_NAME


class TestEmotionalKeywordExtractionCharacterization:
    """Characterize current keyword extraction for emotional queries."""

    def test_distress_keywords_extracted_from_patterns(self):
        """Characterize: Distress-related intent keywords extracted."""
        analyzer = QueryAnalyzer()

        result = analyzer.rewrite_query_with_info("공부하기 힘들어서 싫어")

        # Current behavior: Intent pattern extracts 휴학 keywords
        extracted_keywords = result.rewritten.split()
        assert any("휴학" in k for k in extracted_keywords)
        # Note: 자퇴 is NOT extracted (only 휴학 is in the intent pattern)

    def test_frustration_keywords_extracted(self):
        """Characterize: Frustration patterns extract related keywords."""
        analyzer = QueryAnalyzer()

        result = analyzer.rewrite_query_with_info("안하고 싶어")

        # Current behavior: '안하고 싶' pattern extracts 거절, 반대 keywords
        assert result.used_intent is True
        # Check that frustration-related keywords appear in rewritten query
        assert any(
            kw in result.rewritten for kw in ["거절", "반대", "거부", "불만", "민원"]
        )


class TestEmotionalAudienceDetectionCharacterization:
    """Characterize audience detection with emotional context."""

    def test_student_distress_detected(self):
        """Characterize: Student context detected in distress queries."""
        analyzer = QueryAnalyzer()

        # "공부" context + "힘들어" distress
        audience = analyzer.detect_audience("공부하기 힘들어요")

        # Current behavior: student context keywords detected
        assert audience.name in [
            "STUDENT",
            "ALL",
        ]  # May be ALL without explicit student keyword

    def test_faculty_frustration_detected(self):
        """Characterize: Faculty context detected in frustration queries."""
        analyzer = QueryAnalyzer()

        audience = analyzer.detect_audience("교수 연구 힘들어요")

        # Current behavior: faculty keyword detected
        assert audience.name == "FACULTY"


class TestNoEmotionalTypeExistsCharacterization:
    """Characterize that EMOTIONAL QueryType does not exist yet."""

    def test_emotional_query_type_not_exists(self):
        """Characterize: QueryType.EMOTIONAL does not exist in current code."""
        # Verify EMOTIONAL is not in QueryType enum
        assert "EMOTIONAL" not in [t.name for t in QueryType]

    def test_no_emotional_classifier_exists(self):
        """Characterize: EmotionalClassifier component does not exist."""
        # Characterize: EmotionalClassifier file does not exist
        import pathlib

        classifier_path = (
            pathlib.Path(__file__).parent.parent.parent.parent
            / "src"
            / "rag"
            / "domain"
            / "llm"
            / "emotional_classifier.py"
        )
        assert not classifier_path.exists()


class TestBackwardsCompatibilityCharacterization:
    """Characterize that non-emotional queries work unchanged."""

    def test_factual_query_unchanged(self):
        """Characterize: Factual queries work as before."""
        analyzer = QueryAnalyzer()

        query_type = analyzer.analyze("교원인사규정 제8조")
        result = analyzer.rewrite_query_with_info("교원인사규정 제8조")

        # Should work exactly as before
        assert query_type == QueryType.ARTICLE_REFERENCE
        assert "교원인사규정" in result.rewritten
        assert "제8조" in result.rewritten

    def test_academic_query_unchanged(self):
        """Characterize: Academic queries work as before."""
        analyzer = QueryAnalyzer()

        query_type = analyzer.analyze("휴학 절차")
        result = analyzer.rewrite_query_with_info("휴학 절차")

        # Should work exactly as before
        assert query_type == QueryType.REGULATION_NAME
        assert "휴학" in result.rewritten
