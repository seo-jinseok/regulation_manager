"""
Unit tests for Emotional Classifier.

Tests emotional state detection, confidence scoring, urgency detection,
and prompt adaptation per REQ-EMO-001 through REQ-EMO-015.
"""

from src.rag.domain.llm.emotional_classifier import (
    EmotionalClassificationResult,
    EmotionalClassifier,
    EmotionalClassifierConfig,
    EmotionalState,
)


class TestEmotionalClassifierInit:
    """Test emotional classifier initialization."""

    def test_init_with_defaults(self):
        """Characterize: Default configuration values."""
        classifier = EmotionalClassifier()

        assert isinstance(classifier.config, EmotionalClassifierConfig)
        assert classifier.config.neutral_threshold == 0.3
        assert classifier.config.seeking_help_threshold == 0.5
        assert classifier.config.frustrated_threshold == 0.6
        assert classifier.config.distressed_threshold == 0.7

    def test_init_with_custom_config(self):
        """Characterize: Custom configuration values."""
        config = EmotionalClassifierConfig(
            neutral_threshold=0.4,
            seeking_help_threshold=0.6,
            frustrated_threshold=0.7,
            distressed_threshold=0.8,
        )
        classifier = EmotionalClassifier(config)

        assert classifier.config.neutral_threshold == 0.4
        assert classifier.config.seeking_help_threshold == 0.6
        assert classifier.config.frustrated_threshold == 0.7
        assert classifier.config.distressed_threshold == 0.8


class TestNeutralClassification:
    """Test NEUTRAL emotional state classification."""

    def test_factual_query_classified_as_neutral(self):
        """Characterize: Factual queries without emotion markers classified as NEUTRAL."""
        classifier = EmotionalClassifier()

        result = classifier.classify("교원인사규정 제8조")

        assert result.state == EmotionalState.NEUTRAL
        assert result.confidence == 1.0
        assert len(result.detected_keywords) == 0

    def test_academic_query_classified_as_neutral(self):
        """Characterize: Academic queries without emotion markers classified as NEUTRAL."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 규정")

        assert result.state == EmotionalState.NEUTRAL
        assert result.confidence == 1.0
        assert len(result.detected_keywords) == 0

    def test_empty_query_returns_neutral(self):
        """Characterize: Empty query returns NEUTRAL with high confidence."""
        classifier = EmotionalClassifier()

        result = classifier.classify("")

        assert result.state == EmotionalState.NEUTRAL
        assert result.confidence == 1.0
        assert len(result.detected_keywords) == 0


class TestSeekingHelpClassification:
    """Test SEEKING_HELP emotional state classification (REQ-EMO-006)."""

    def test_explicit_help_request_classified_seeking_help(self):
        """Characterize: Explicit help requests classified as SEEKING_HELP."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 절차 알려주세요")

        assert result.state == EmotionalState.SEEKING_HELP
        assert result.confidence >= 0.5
        # Either "알려주세요" or "절차" should be detected
        assert any(kw in result.detected_keywords for kw in ["알려주세요", "절차"])

    def test_how_to_question_classified_seeking_help(self):
        """Characterize: '어떻게 해요' questions classified as SEEKING_HELP."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 방법 알려줘")

        assert result.state == EmotionalState.SEEKING_HELP
        # Should detect seeking help keywords (e.g., "방법 알", "알려줘")
        assert len(result.detected_keywords) > 0

    def test_procedure_request_classified_seeking_help(self):
        """Characterize: Procedure requests classified as SEEKING_HELP."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 방법 좀 알려줘")

        assert result.state == EmotionalState.SEEKING_HELP
        # "방법" or "알려줘" should be detected
        assert any(kw in result.detected_keywords for kw in ["방법", "알려줘"])

    def test_teach_me_request_classified_seeking_help(self):
        """Characterize: '가르쳐줘' requests classified as SEEKING_HELP."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 가르쳐주세요")

        assert result.state == EmotionalState.SEEKING_HELP
        assert "가르쳐주세요" in result.detected_keywords


class TestDistressedClassification:
    """Test DISTRESSED emotional state classification (REQ-EMO-004, REQ-EMO-008)."""

    def test_hardship_expression_classified_distressed(self):
        """Characterize: '힘들어요' expressions classified as DISTRESSED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("공부하기 힘들어요 휴학")

        assert result.state == EmotionalState.DISTRESSED
        assert result.confidence >= 0.7
        assert "힘들어" in result.detected_keywords

    def test_despair_expression_classified_distressed(self):
        """Characterize: '어떡해요' expressions classified as DISTRESSED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("장학금이 어떡해요")

        assert result.state == EmotionalState.DISTRESSED
        assert "어떡해요" in result.detected_keywords

    def test_frustration_expression_classified_distressed(self):
        """Characterize: '답답해요' expressions classified as DISTRESSED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("규정이 너무 답답해요")

        assert result.state == EmotionalState.DISTRESSED
        # "답답해요" should be in detected keywords
        assert "답답해요" in result.detected_keywords

    def test_giving_up_expression_classified_distressed(self):
        """Characterize: '포기' expressions classified as DISTRESSED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("학교 포기하고 싶어")

        assert result.state == EmotionalState.DISTRESSED
        # Should detect "포기하고" or similar variation
        assert any("포기" in kw for kw in result.detected_keywords)

    def test_suffering_expression_classified_distressed(self):
        """Characterize: '괴로워요' expressions classified as DISTRESSED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("학사경고 때문에 괴로워요")

        assert result.state == EmotionalState.DISTRESSED
        assert "괴로워요" in result.detected_keywords


class TestFrustratedClassification:
    """Test FRUSTRATED emotional state classification (REQ-EMO-005)."""

    def test_rejection_expression_classified_frustrated(self):
        """Characterize: '안돼요' expressions classified as FRUSTRATED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("수강신청 안돼요 왜 안돼요")

        assert result.state == EmotionalState.FRUSTRATED
        # Should detect "안돼요" or similar variation
        assert any("안돼" in kw for kw in result.detected_keywords)

    def test_confusion_expression_classified_frustrated(self):
        """Characterize: '이해 안돼요' expressions classified as FRUSTRATED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("규정이 이해 안돼요")

        assert result.state == EmotionalState.FRUSTRATED
        assert "이해 안" in result.detected_keywords

    def test_complexity_expression_classified_frustrated(self):
        """Characterize: '너무 복잡해요' expressions classified as FRUSTRATED."""
        classifier = EmotionalClassifier()

        result = classifier.classify("절차가 너무 복잡해요")

        assert result.state == EmotionalState.FRUSTRATED
        # Should detect "너무 복잡" or similar variation
        assert any("복잡" in kw for kw in result.detected_keywords)


class TestUrgencyDetection:
    """Test urgency indicator detection (REQ-EMO-009)."""

    def test_urgency_geuphaeyo_detected(self):
        """Characterize: '급해요' triggers urgency detection."""
        classifier = EmotionalClassifier()

        result = classifier.classify("장학금 급해요")

        assert result.has_urgency is True

    def test_urgency_ppalli_detected(self):
        """Characterize: '빨리' triggers urgency detection."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 빨리 알려줘")

        assert result.has_urgency is True

    def test_urgency_jigeum_detected(self):
        """Characterize: '지금' triggers urgency detection."""
        classifier = EmotionalClassifier()

        result = classifier.classify("지금 당장 필요해")

        assert result.has_urgency is True

    def test_urgency_dangjang_detected(self):
        """Characterize: '당장' triggers urgency detection."""
        classifier = EmotionalClassifier()

        result = classifier.classify("지금 당장 필요해")

        assert result.has_urgency is True

    def test_no_urgency_in_factual_query(self):
        """Characterize: Factual queries don't trigger urgency."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 규정")

        assert result.has_urgency is False


class TestConflictResolution:
    """Test conflict resolution when multiple emotional signals present (REQ-EMO-011)."""

    def test_distressed_takes_precedence_over_frustrated(self):
        """Characterize: DISTRESSED takes precedence over FRUSTRATED."""
        classifier = EmotionalClassifier()

        # Both distress and frustration present
        result = classifier.classify("너무 힘들어요 안돼요")

        # Distressed has higher priority
        assert result.state == EmotionalState.DISTRESSED

    def test_distressed_takes_precedence_over_seeking_help(self):
        """Characterize: DISTRESSED takes precedence over SEEKING_HELP."""
        classifier = EmotionalClassifier()

        # Both distress and seeking help present
        result = classifier.classify("힘들어요 방법 알려주세요")

        # Distressed has higher priority
        assert result.state == EmotionalState.DISTRESSED

    def test_frustrated_takes_precedence_over_seeking_help(self):
        """Characterize: FRUSTRATED takes precedence over SEEKING_HELP."""
        classifier = EmotionalClassifier()

        # Both frustration and seeking help present
        result = classifier.classify("이해 안돼요 방법 알려줘")

        # Frustrated has higher priority
        assert result.state == EmotionalState.FRUSTRATED


class TestConfidenceScoring:
    """Test confidence score calculation (REQ-EMO-003)."""

    def test_single_emotional_keywordModerate_confidence(self):
        """Characterize: Single emotional keyword yields moderate confidence."""
        classifier = EmotionalClassifier()

        result = classifier.classify("힘들어요")

        # Single match: score 2.0, normalized to 1.0
        assert result.confidence >= 0.7  # At or above distressed threshold

    def test_multiple_emotional_keywords_higher_confidence(self):
        """Characterize: Multiple emotional keywords increase confidence."""
        classifier = EmotionalClassifier()

        result = classifier.classify("너무 힘들어요 정말 힘들어요 어떡해요")

        # Multiple matches increase confidence
        assert result.confidence > 0.7

    def test_confidence_capped_at_1_0(self):
        """Characterize: Confidence is capped at 1.0."""
        classifier = EmotionalClassifier()

        # Many emotional keywords
        result = classifier.classify("힘들어요 힘들어요 힘들어요 힘들어요 힘들어요")

        assert result.confidence <= 1.0

    def test_neutral_always_high_confidence(self):
        """Characterize: NEUTRAL classification always has high confidence."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 규정")

        assert result.confidence == 1.0


class TestPromptAdaptation:
    """Test empathy-aware prompt adaptation (REQ-EMO-004 through REQ-EMO-007)."""

    def test_distressed_prompt_adds_empathy(self):
        """Characterize: DISTRESSED queries add empathetic acknowledgment (REQ-EMO-004)."""
        classifier = EmotionalClassifier()
        classification = EmotionalClassificationResult(
            state=EmotionalState.DISTRESSED,
            confidence=0.8,
            detected_keywords=["힘들어"],
            has_urgency=False,
        )

        adapted = classifier.generate_empathy_prompt(
            classification, "규정에 따라 답변하세요."
        )

        # Should prepend empathetic acknowledgment
        assert "공감" in adapted or "위로" in adapted
        assert "규정에 따라 답변하세요" in adapted

    def test_frustrated_prompt_adds_clarity(self):
        """Characterize: FRUSTRATED queries add calming guidance (REQ-EMO-005)."""
        classifier = EmotionalClassifier()
        classification = EmotionalClassificationResult(
            state=EmotionalState.FRUSTRATED,
            confidence=0.7,
            detected_keywords=["안돼"],
            has_urgency=False,
        )

        adapted = classifier.generate_empathy_prompt(
            classification, "규정에 따라 답변하세요."
        )

        # Should add calming + step-by-step instruction
        assert "단계별" in adapted or "차분하게" in adapted
        assert "규정에 따라 답변하세요" in adapted

    def test_seeking_help_prompt_adds_clarity(self):
        """Characterize: SEEKING_HELP queries prioritize clarity (REQ-EMO-006)."""
        classifier = EmotionalClassifier()
        classification = EmotionalClassificationResult(
            state=EmotionalState.SEEKING_HELP,
            confidence=0.6,
            detected_keywords=["방법"],
            has_urgency=False,
        )

        adapted = classifier.generate_empathy_prompt(
            classification, "규정에 따라 답변하세요."
        )

        # Should add clarity instruction
        assert "명확" in adapted or "자세" in adapted
        assert "규정에 따라 답변하세요" in adapted

    def test_neutral_prompt_unchanged(self):
        """Characterize: NEUTRAL queries leave prompt unchanged (REQ-EMO-010)."""
        classifier = EmotionalClassifier()
        classification = EmotionalClassificationResult(
            state=EmotionalState.NEUTRAL,
            confidence=1.0,
            detected_keywords=[],
            has_urgency=False,
        )

        base_prompt = "규정에 따라 답변하세요."
        adapted = classifier.generate_empathy_prompt(classification, base_prompt)

        # Should not modify base prompt
        assert adapted == base_prompt


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_whitespace_only_query(self):
        """Characterize: Whitespace-only query returns NEUTRAL."""
        classifier = EmotionalClassifier()

        result = classifier.classify("   ")

        assert result.state == EmotionalState.NEUTRAL
        assert result.confidence == 1.0

    def test_mixed_emotion_with_academic_keyword(self):
        """Characterize: Emotion + academic keyword detects emotion."""
        classifier = EmotionalClassifier()

        result = classifier.classify("휴학 힘들어요")

        # Should detect distress despite academic keyword
        assert result.state == EmotionalState.DISTRESSED
        assert "힘들어" in result.detected_keywords

    def test_partial_keyword_match(self):
        """Characterize: Partial keyword matches are detected."""
        classifier = EmotionalClassifier()

        # "어떡해요" should match
        result = classifier.classify("장학금 어떡해요")

        assert result.state == EmotionalState.DISTRESSED

    def test_longest_keyword_match_preferred(self):
        """Characterize: Longer keywords match before shorter ones."""
        classifier = EmotionalClassifier()

        # "너무 힘들어" contains both "힘들어" and "너무 힘"
        result = classifier.classify("너무 힘들어요")

        # Should match and detect distress
        assert result.state == EmotionalState.DISTRESSED

    def test_low_confidence_downgrades_to_neutral(self):
        """Characterize: Low confidence emotional detection downgrades to NEUTRAL (REQ-EMO-010)."""
        # Use custom config with high thresholds to trigger downgrade
        config = EmotionalClassifierConfig(
            neutral_threshold=0.1,
            seeking_help_threshold=2.0,  # Very high - single match won't reach
            frustrated_threshold=2.0,
            distressed_threshold=2.0,
        )
        classifier = EmotionalClassifier(config)

        # Single emotional keyword match (score = 1.0 * 2.0 = 2.0, normalized to 1.0)
        # But threshold is 2.0, so confidence < threshold
        result = classifier.classify("알려줘")

        # Should downgrade to NEUTRAL due to low confidence
        assert result.state == EmotionalState.NEUTRAL
        assert result.confidence == 1.0


class TestMetrics:
    """Test emotional state metrics (REQ-EMO-003)."""

    def test_get_emotional_metrics(self):
        """Characterize: Emotional metrics returns keyword counts."""
        classifier = EmotionalClassifier()

        metrics = classifier.get_emotional_metrics()

        assert isinstance(metrics, dict)
        assert "distressed_keywords_count" in metrics
        assert "frustrated_keywords_count" in metrics
        assert "seeking_help_keywords_count" in metrics
        assert "urgency_keywords_count" in metrics

        # Verify counts are positive
        assert metrics["distressed_keywords_count"] > 0
        assert metrics["frustrated_keywords_count"] > 0
        assert metrics["seeking_help_keywords_count"] > 0
        assert metrics["urgency_keywords_count"] > 0
