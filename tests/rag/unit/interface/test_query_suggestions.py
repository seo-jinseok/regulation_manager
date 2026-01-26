"""
Unit tests for query_suggestions module.
"""


from src.rag.interface.query_suggestions import (
    INITIAL_QUERY_EXAMPLES,
    format_examples_for_cli,
    format_suggestions_for_cli,
    get_followup_suggestions,
    get_initial_examples,
)


class TestGetInitialExamples:
    """Tests for get_initial_examples function."""

    def test_returns_list(self):
        """Should return a list of examples."""
        examples = get_initial_examples()
        assert isinstance(examples, list)

    def test_returns_five_examples(self):
        """Should return exactly 5 examples."""
        examples = get_initial_examples()
        assert len(examples) == 5

    def test_returns_copy(self):
        """Should return a copy, not the original list."""
        examples = get_initial_examples()
        examples.append("테스트")
        assert len(INITIAL_QUERY_EXAMPLES) == 5  # Original unchanged

    def test_examples_are_strings(self):
        """All examples should be strings."""
        examples = get_initial_examples()
        assert all(isinstance(e, str) for e in examples)

    def test_examples_not_empty(self):
        """All examples should be non-empty strings."""
        examples = get_initial_examples()
        assert all(len(e) > 0 for e in examples)


class TestGetFollowupSuggestions:
    """Tests for get_followup_suggestions function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_followup_suggestions("휴학")
        assert isinstance(result, list)

    def test_max_three_suggestions(self):
        """Should return at most 3 suggestions."""
        result = get_followup_suggestions("휴학 등록금 장학금")
        assert len(result) <= 3

    def test_keyword_matching_휴학(self):
        """Should suggest followups for 휴학 keyword."""
        result = get_followup_suggestions("휴학")
        assert len(result) > 0
        assert any("복학" in s or "휴학" in s for s in result)

    def test_keyword_matching_등록금(self):
        """Should suggest followups for 등록금 keyword."""
        result = get_followup_suggestions("등록금")
        assert len(result) > 0
        assert any("납부" in s or "장학" in s or "감면" in s for s in result)

    def test_regulation_title_fallback(self):
        """Should use regulation title for default suggestions."""
        result = get_followup_suggestions(
            "알 수 없는 쿼리", regulation_title="학칙"
        )
        assert len(result) > 0
        assert any("학칙" in s for s in result)

    def test_no_duplicates(self):
        """Should not return duplicate suggestions."""
        result = get_followup_suggestions("휴학", regulation_title="학칙")
        assert len(result) == len(set(result))

    def test_empty_query(self):
        """Should handle empty query gracefully."""
        result = get_followup_suggestions("")
        assert isinstance(result, list)

    def test_answer_text_keyword_extraction(self):
        """Should extract keywords from answer text."""
        result = get_followup_suggestions(
            "질문", answer_text="휴학 절차에 대해 안내드립니다."
        )
        assert len(result) > 0


class TestFormatExamplesForCli:
    """Tests for format_examples_for_cli function."""

    def test_numbering(self):
        """Should add numbering to examples."""
        examples = ["예시1", "예시2"]
        result = format_examples_for_cli(examples)
        assert "[1]" in result
        assert "[2]" in result

    def test_includes_all_examples(self):
        """Should include all example texts."""
        examples = ["휴학 신청", "연구년"]
        result = format_examples_for_cli(examples)
        assert "휴학 신청" in result
        assert "연구년" in result


class TestFormatSuggestionsForCli:
    """Tests for format_suggestions_for_cli function."""

    def test_empty_suggestions(self):
        """Should return empty string for empty list."""
        result = format_suggestions_for_cli([])
        assert result == ""

    def test_header_text(self):
        """Should include header text."""
        result = format_suggestions_for_cli(["질문1"])
        assert "연관 질문" in result

    def test_numbering(self):
        """Should add numbering to suggestions."""
        result = format_suggestions_for_cli(["질문1", "질문2"])
        assert "[1]" in result
        assert "[2]" in result
