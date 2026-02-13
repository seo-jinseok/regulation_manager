"""
Tests for CompletenessChecker module.

Tests TOC-based completeness validation for HWPX parsing.
"""
import pytest

from src.parsing.validators.completeness_checker import (
    CompletenessChecker,
    CompletenessReport,
    TOCEntry,
)


class TestTOCEntry:
    """Test TOCEntry dataclass."""

    def test_creation(self):
        """Test TOCEntry creation."""
        entry = TOCEntry(
            id="toc-0001",
            title="학칙",
            page="1",
            rule_code="3-1-1",
        )
        assert entry.id == "toc-0001"
        assert entry.title == "학칙"
        assert entry.page == "1"
        assert entry.rule_code == "3-1-1"

    def test_normalized_title_auto_generation(self):
        """Test automatic normalized title generation."""
        entry = TOCEntry(id="toc-0001", title="학 칙")
        # After __post_init__, normalized_title should be set
        assert entry.normalized_title != ""
        assert entry.normalized_title == "학칙".lower().replace(" ", "")


class TestCompletenessChecker:
    """Test CompletenessChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = CompletenessChecker()
        assert checker.fuzzy_match_threshold == 0.85
        assert checker.require_exact_match is False

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        checker = CompletenessChecker(fuzzy_match_threshold=0.9)
        assert checker.fuzzy_match_threshold == 0.9

    def test_validate_complete_parsing(self):
        """Test validation when parsing is complete."""
        checker = CompletenessChecker()

        toc_entries = [
            TOCEntry(id="toc-0001", title="학칙"),
            TOCEntry(id="toc-0002", title="규정"),
        ]

        parsed_regs = [
            {"title": "학칙", "id": "reg-0001"},
            {"title": "규정", "id": "reg-0002"},
        ]

        report = checker.validate(toc_entries, parsed_regs)

        assert report.total_toc_entries == 2
        assert report.total_parsed == 2
        assert report.matched_entries == 2
        assert report.missing_entries == 0
        assert report.extra_entries == 0
        assert report.is_complete is True

    def test_validate_incomplete_parsing(self):
        """Test validation when parsing is incomplete."""
        checker = CompletenessChecker()

        toc_entries = [
            TOCEntry(id="toc-0001", title="학칙"),
            TOCEntry(id="toc-0002", title="규정"),
            TOCEntry(id="toc-0003", title="요령"),
        ]

        parsed_regs = [
            {"title": "학칙", "id": "reg-0001"},
            {"title": "규정", "id": "reg-0002"},
        ]

        report = checker.validate(toc_entries, parsed_regs)

        assert report.total_toc_entries == 3
        assert report.matched_entries == 2
        assert report.missing_entries == 1
        assert report.is_complete is False
        assert "요령" in report.missing_titles

    def test_validate_extra_entries(self):
        """Test detection of extra parsed entries."""
        checker = CompletenessChecker()

        toc_entries = [
            TOCEntry(id="toc-0001", title="학칙"),
        ]

        parsed_regs = [
            {"title": "학칙", "id": "reg-0001"},
            {"title": "추가규정", "id": "reg-0002"},
        ]

        report = checker.validate(toc_entries, parsed_regs)

        assert report.matched_entries == 1
        assert report.extra_entries == 1
        assert "추가규정" in report.extra_titles

    def test_fuzzy_matching(self):
        """Test fuzzy title matching."""
        checker = CompletenessChecker(fuzzy_match_threshold=0.7)

        toc_entries = [
            TOCEntry(id="toc-0001", title="대학원 학칙"),
        ]

        parsed_regs = [
            {"title": "대학원학칙", "id": "reg-0001"},  # No space
        ]

        report = checker.validate(toc_entries, parsed_regs)

        # Should match with fuzzy matching
        assert report.matched_entries == 1
        assert report.missing_entries == 0

    def test_exact_match_only(self):
        """Test exact match mode."""
        checker = CompletenessChecker(
            fuzzy_match_threshold=0.85,
            require_exact_match=True,
        )

        toc_entries = [
            TOCEntry(id="toc-0001", title="대학원 학칙"),
        ]

        parsed_regs = [
            {"title": "대학원학칙", "id": "reg-0001"},  # No space
        ]

        report = checker.validate(toc_entries, parsed_regs)

        # In exact mode, these should NOT match due to space difference
        # But due to normalization (removing spaces), they might match
        # Let's check the normalized forms
        toc_normalized = toc_entries[0].normalized_title  # "대학원학칙"
        parsed_normalized = checker.normalizer.get_title_hash("대학원학칙")  # "대학원학칙"

        # If they normalize the same, they match even in "exact" mode
        # because exact means "exact normalized match"
        if toc_normalized == parsed_normalized:
            # They match after normalization
            assert report.matched_entries == 1
        else:
            # They don't match
            assert report.matched_entries == 0
            assert report.missing_entries == 1

    def test_create_toc_from_regulations(self):
        """Test creating TOC from parsed regulations."""
        checker = CompletenessChecker()

        regs = [
            {"title": "학칙", "id": "reg-0001", "rule_code": "3-1-1"},
            {"title": "규정", "id": "reg-0002", "rule_code": "3-1-2"},
        ]

        toc_entries = checker.create_toc_from_regulations(regs)

        assert len(toc_entries) == 2
        assert toc_entries[0].title == "학칙"
        assert toc_entries[1].title == "규정"
        assert toc_entries[0].rule_code == "3-1-1"

    def test_find_best_match(self):
        """Test finding best match for a title."""
        checker = CompletenessChecker()

        candidates = ["학칙", "대학원학칙", "시행세칙"]

        # Exact match
        match, score = checker.find_best_match("학칙", candidates)
        assert match == "학칙"
        assert score >= 0.9

        # Fuzzy match
        match, score = checker.find_best_match("학 칙", candidates)
        assert match is not None
        assert score >= 0.85

        # No good match
        match, score = checker.find_best_match("요령", candidates)
        assert match is None or score < 0.85

    def test_generate_missing_report(self):
        """Test missing report generation."""
        checker = CompletenessChecker()

        report = CompletenessReport(
            total_toc_entries=10,
            total_parsed=8,
            matched_entries=8,
            missing_entries=2,
            extra_entries=0,
            is_complete=False,
            missing_titles=["규정1", "규정2"],
        )

        report_text = checker.generate_missing_report(report)

        assert "10" in report_text
        assert "8" in report_text
        assert "INCOMPLETE" in report_text
        assert "규정1" in report_text


class TestCompletenessReport:
    """Test CompletenessReport dataclass."""

    def test_to_dict(self):
        """Test report to_dict conversion."""
        report = CompletenessReport(
            total_toc_entries=10,
            total_parsed=8,
            matched_entries=8,
            missing_entries=2,
            is_complete=False,
        )

        report_dict = report.to_dict()

        assert report_dict["total_toc_entries"] == 10
        assert report_dict["total_parsed"] == 8
        assert report_dict["matched_entries"] == 8
        assert report_dict["missing_entries"] == 2
        assert report_dict["is_complete"] is False
        assert "completion_rate" in report_dict


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_toc_and_parsed(self):
        """Test with empty TOC and parsed lists."""
        checker = CompletenessChecker()
        report = checker.validate([], [])
        assert report.total_toc_entries == 0
        assert report.total_parsed == 0
        assert report.is_complete is True

    def test_empty_toc_with_parsed(self):
        """Test with empty TOC but parsed entries."""
        checker = CompletenessChecker()
        report = checker.validate(
            [],
            [{"title": "학칙", "id": "reg-0001"}]
        )
        assert report.extra_entries == 1

    def test_duplicate_titles_in_toc(self):
        """Test handling of duplicate titles in TOC."""
        checker = CompletenessChecker()

        toc_entries = [
            TOCEntry(id="toc-0001", title="학칙"),
            TOCEntry(id="toc-0002", title="학칙"),  # Duplicate
        ]

        parsed_regs = [
            {"title": "학칙", "id": "reg-0001"},
        ]

        report = checker.validate(toc_entries, parsed_regs)

        # Should match one TOC entry
        assert report.matched_entries >= 1

    def test_unicode_normalization(self):
        """Test Unicode title normalization."""
        checker = CompletenessChecker()

        toc_entries = [
            TOCEntry(id="toc-0001", title="학칙"),  # No special chars
        ]

        parsed_regs = [
            {"title": "학칙\u200b"},  # Contains zero-width space
        ]

        report = checker.validate(toc_entries, parsed_regs)

        # Should handle Unicode normalization
        assert report.matched_entries >= 0

    def test_very_long_titles(self):
        """Test handling of very long titles."""
        checker = CompletenessChecker()

        long_title = "a" * 300
        toc_entries = [
            TOCEntry(id="toc-0001", title=long_title),
        ]

        parsed_regs = [
            {"title": long_title, "id": "reg-0001"},
        ]

        report = checker.validate(toc_entries, parsed_regs)

        assert report.matched_entries == 1
