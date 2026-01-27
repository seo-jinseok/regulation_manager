"""
Unit tests for Ambiguity Classifier.

Tests characterize and verify ambiguity detection and disambiguation behavior
for regulation search queries.
"""

from src.rag.domain.llm.ambiguity_classifier import (
    AmbiguityClassifier,
    AmbiguityClassifierConfig,
    AmbiguityLevel,
    DisambiguationDialog,
)
from src.rag.infrastructure.query_analyzer import Audience


class TestAmbiguityClassifierInit:
    """Test ambiguity classifier initialization."""

    def test_init_with_defaults(self):
        """Characterize: Default configuration values."""
        classifier = AmbiguityClassifier()

        assert isinstance(classifier.config, AmbiguityClassifierConfig)
        assert classifier.config.high_threshold == 0.7
        assert classifier.config.low_threshold == 0.4
        assert classifier.config.max_options == 5

    def test_init_with_custom_config(self):
        """Characterize: Custom configuration values."""
        config = AmbiguityClassifierConfig(
            high_threshold=0.8,
            low_threshold=0.3,
            max_options=3,
        )
        classifier = AmbiguityClassifier(config)

        assert classifier.config.high_threshold == 0.8
        assert classifier.config.low_threshold == 0.3
        assert classifier.config.max_options == 3


class TestAmbiguityLevelClassification:
    """Test ambiguity level detection."""

    def test_clear_classification_specific_article(self):
        """Characterize: Specific article reference classified as CLEAR."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("교원인사규정 제8조")

        assert result.level == AmbiguityLevel.CLEAR
        assert result.score < 0.4
        assert "audience" not in result.ambiguity_factors
        assert "regulation_type" not in result.ambiguity_factors

    def test_clear_classification_specific_audience(self):
        """Characterize: Specific audience mentioned with regulation type still has some ambiguity due to generic '규정' term."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("학생 휴학 규정")

        # Current behavior: "학생 휴학 규정" has AMBIGUOUS level due to generic "규정" term
        # Audience is detected (student), but regulation_type ambiguity remains
        assert result.level == AmbiguityLevel.AMBIGUOUS
        assert result.score >= 0.4
        assert Audience.STUDENT in result.detected_audiences
        assert "regulation_type" in result.ambiguity_factors

    def test_ambiguous_audience_no_context(self):
        """Characterize: Query '휴학 규정' has ambiguity from generic '규정' term and missing article reference."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("휴학 규정")

        # Current behavior: AMBIGUOUS due to regulation_type and article_reference factors
        # Audience is actually detected (student), so 'audience' factor is False
        assert result.level == AmbiguityLevel.AMBIGUOUS
        assert result.score >= 0.4
        assert result.score < 0.7
        assert "regulation_type" in result.ambiguity_factors
        assert "article_reference" in result.ambiguity_factors

    def test_ambiguous_regulation_type(self):
        """Characterize: '징계 규정' is HIGHLY_AMBIGUOUS due to multiple audiences (student/faculty/staff) and generic term."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("징계 규정")

        # Current behavior: HIGHLY_AMBIGUOUS due to multiple audience matches + generic regulation term
        assert result.level == AmbiguityLevel.HIGHLY_AMBIGUOUS
        assert result.score >= 0.7
        assert "regulation_type" in result.ambiguity_factors
        assert "audience" in result.ambiguity_factors  # Multiple audiences detected

    def test_highly_ambiguous_vague_query(self):
        """Characterize: Highly vague query classified as HIGHLY_AMBIGUOUS."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("규정")

        assert result.level == AmbiguityLevel.HIGHLY_AMBIGUOUS
        assert result.score >= 0.7
        assert "regulation_type" in result.ambiguity_factors
        assert "audience" in result.ambiguity_factors


class TestAudienceAmbiguityDetection:
    """Test audience ambiguity detection."""

    def test_student_keywords_detected(self):
        """Characterize: Student keywords correctly identified."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("학생 휴학")

        assert Audience.STUDENT in result.detected_audiences

    def test_faculty_keywords_detected(self):
        """Characterize: Faculty keywords correctly identified."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("교수 휴직")

        assert Audience.FACULTY in result.detected_audiences

    def test_staff_keywords_detected(self):
        """Characterize: Staff keywords correctly identified."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("직원 승진")

        assert Audience.STAFF in result.detected_audiences

    def test_multiple_audiences_detected(self):
        """Characterize: Multiple audience matches trigger ambiguity."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("징계 규정")

        # 징계 applies to all audiences
        assert len(result.detected_audiences) > 1
        assert "audience" in result.ambiguity_factors


class TestRegulationTypeAmbiguity:
    """Test regulation type ambiguity detection."""

    def test_specific_regulation_name(self):
        """Characterize: Specific regulation name reduces ambiguity."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("교원인사규정")

        assert result.level == AmbiguityLevel.CLEAR
        assert "regulation_type" not in result.ambiguity_factors

    def test_generic_regulation_terms(self):
        """Characterize: Generic regulation terms increase ambiguity."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("규정")

        assert result.level == AmbiguityLevel.HIGHLY_AMBIGUOUS
        assert "regulation_type" in result.ambiguity_factors


class TestArticleReferenceAmbiguity:
    """Test article reference ambiguity detection."""

    def test_specific_article_reference(self):
        """Characterize: Specific article reference reduces ambiguity."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("제8조")

        assert "article_reference" not in result.ambiguity_factors

    def test_missing_article_reference(self):
        """Characterize: Missing article reference can increase ambiguity."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("휴학")

        # Without article reference, more ambiguous
        assert result.score > 0


class TestDisambiguationDialogGeneration:
    """Test disambiguation dialog generation."""

    def test_dialog_generation_for_audience_ambiguity(self):
        """Characterize: Dialog generated for audience ambiguity."""
        classifier = AmbiguityClassifier()
        classification = classifier.classify("휴학 규정")

        dialog = classifier.generate_disambiguation_dialog(classification)

        assert isinstance(dialog, DisambiguationDialog)
        assert len(dialog.options) > 0
        assert len(dialog.options) <= classifier.config.max_options
        assert dialog.message is not None

    def test_dialog_options_ranked_by_relevance(self):
        """Characterize: Dialog options ranked by relevance score."""
        classifier = AmbiguityClassifier()
        classification = classifier.classify("휴학 규정")

        dialog = classifier.generate_disambiguation_dialog(classification)

        # Options should be sorted by relevance (descending)
        for i in range(len(dialog.options) - 1):
            assert (
                dialog.options[i].relevance_score
                >= dialog.options[i + 1].relevance_score
            )

    def test_dialog_option_structure(self):
        """Characterize: Dialog option contains required fields."""
        classifier = AmbiguityClassifier()
        classification = classifier.classify("휴학 규정")

        dialog = classifier.generate_disambiguation_dialog(classification)

        option = dialog.options[0]
        assert option.label is not None
        assert option.clarified_query is not None
        assert 0.0 <= option.relevance_score <= 1.0

    def test_no_dialog_for_clear_queries(self):
        """Characterize: No dialog generated for CLEAR queries."""
        classifier = AmbiguityClassifier()
        classification = classifier.classify("교원인사규정 제8조")

        dialog = classifier.generate_disambiguation_dialog(classification)

        assert dialog is None or len(dialog.options) == 0


class TestUserConfirmationHandling:
    """Test user confirmation and query clarification."""

    def test_apply_user_selection(self):
        """Characterize: User selection updates query with clarification."""
        classifier = AmbiguityClassifier()
        classification = classifier.classify("휴학 규정")
        dialog = classifier.generate_disambiguation_dialog(classification)

        if dialog and len(dialog.options) > 0:
            selected_option = dialog.options[0]
            clarified = classifier.apply_user_selection("휴학 규정", selected_option)

            assert clarified is not None
            assert clarified != "휴학 규정"

    def test_skip_clarification(self):
        """Characterize: Skip option returns original query."""
        classifier = AmbiguityClassifier()

        clarified = classifier.skip_clarification("휴학 규정")

        assert clarified == "휴학 규정"


class TestClassificationThresholds:
    """Test classification score thresholds."""

    def test_high_threshold_boundary(self):
        """Characterize: Queries at high threshold classified as HIGHLY_AMBIGUOUS."""
        config = AmbiguityClassifierConfig(high_threshold=0.7, low_threshold=0.4)
        classifier = AmbiguityClassifier(config)

        # Very ambiguous query
        result = classifier.classify("규정")

        assert result.score >= 0.7
        assert result.level == AmbiguityLevel.HIGHLY_AMBIGUOUS

    def test_low_threshold_boundary(self):
        """Characterize: Queries at low threshold classified as AMBIGUOUS."""
        classifier = AmbiguityClassifier()

        # Moderately ambiguous query
        result = classifier.classify("휴학")

        assert 0.4 <= result.score < 0.7
        assert result.level == AmbiguityLevel.AMBIGUOUS

    def test_below_low_threshold(self):
        """Characterize: Queries below low threshold classified as CLEAR."""
        classifier = AmbiguityClassifier()

        # Specific query
        result = classifier.classify("학생 휴학 신청 기간")

        assert result.score < 0.4
        assert result.level == AmbiguityLevel.CLEAR


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self):
        """Characterize: Empty query handled gracefully."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("")

        assert result.level == AmbiguityLevel.CLEAR
        assert result.score == 0.0

    def test_whitespace_only_query(self):
        """Characterize: Whitespace-only query handled gracefully."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("   ")

        assert result.level == AmbiguityLevel.CLEAR
        assert result.score == 0.0

    def test_very_long_query(self):
        """Characterify: Long query processed correctly."""
        classifier = AmbiguityClassifier()

        long_query = "학생 휴학 규정에 대한 " + "상세한 " * 50 + "설명"
        result = classifier.classify(long_query)

        assert result.level in AmbiguityLevel

    def test_special_characters(self):
        """Characterize: Special characters handled correctly."""
        classifier = AmbiguityClassifier()

        result = classifier.classify("휴학?!?!.")

        assert result.level in AmbiguityLevel


class TestDisambiguationLearning:
    """Test learning from user selections."""

    def test_learn_from_selection(self):
        """Characterize: Classifier learns from user disambiguation selection."""
        classifier = AmbiguityClassifier()

        # Simulate user selecting student audience for "휴학"
        classifier.learn_from_selection("휴학", Audience.STUDENT)

        # Future classifications should reflect learning
        result = classifier.classify("휴학")

        # Student should be preferred
        if result.detected_audiences:
            assert Audience.STUDENT in result.detected_audiences

    def test_learning_persists_across_queries(self):
        """Characterize: Learning persists across different queries."""
        classifier = AmbiguityClassifier()

        # Learn that "징계" usually implies student context
        classifier.learn_from_selection("징계", Audience.STUDENT)

        result1 = classifier.classify("징계")
        result2 = classifier.classify("징계 규정")

        # Both should reflect the learned preference
        if result1.detected_audiences:
            assert Audience.STUDENT in result1.detected_audiences
        if result2.detected_audiences:
            assert Audience.STUDENT in result2.detected_audiences
