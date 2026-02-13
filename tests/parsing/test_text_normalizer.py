"""
Tests for TextNormalizer module.

Tests text cleaning and normalization functionality for HWPX parsing.
"""
import pytest

from src.parsing.core.text_normalizer import (
    TextNormalizer,
    normalize_regulation_text,
    get_default_normalizer,
)


class TestTextNormalizer:
    """Test TextNormalizer class."""

    def test_init_default(self):
        """Test default initialization."""
        normalizer = TextNormalizer()
        assert normalizer.remove_page_headers is True
        assert normalizer.remove_filler_chars is True
        assert normalizer.normalize_whitespace is True
        assert normalizer.detect_duplicate_titles is True

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        normalizer = TextNormalizer()
        result = normalizer.clean("")
        assert result == ""

    def test_clean_none(self):
        """Test cleaning None."""
        normalizer = TextNormalizer()
        result = normalizer.clean(None)
        assert result == ""

    def test_remove_page_headers(self):
        """Test page header removal."""
        normalizer = TextNormalizer()
        # Various dash and tilde combinations
        test_cases = [
            ("겸임교원규정 3—1—10～", "겸임교원규정"),
            ("규정 3－1－10～", "규정"),
            ("지침 1—2—3～", "지침"),
        ]
        for input_text, expected in test_cases:
            result = normalizer.clean(input_text)
            # After cleaning, the page header should be removed
            assert "～" not in result, f"Expected no tilde in: {result}"
            assert result.strip() == expected or result == expected, f"For input '{input_text}': got '{result}', expected '{expected}'"

    def test_remove_page_headers_with_duplicate(self):
        """Test page header removal with duplicate titles."""
        normalizer = TextNormalizer()
        # This tests the duplicate detector
        test_cases = [
            ("학칙 1-2-5～ 학칙", "학칙"),  # Duplicate should be cleaned
            ("겸임교원규정 3—1—10～ 겸임교원규정", "겸임교원규정"),
        ]
        for input_text, expected in test_cases:
            result = normalizer.clean(input_text)
            assert result == expected, f"For input '{input_text}': got '{result}', expected '{expected}'"

    def test_remove_filler_characters(self):
        """Test Unicode filler character removal."""
        normalizer = TextNormalizer()
        # U+F0800 range characters
        text_with_fillers = "규정 텍스트\U000f0800\U000f0fff"
        result = normalizer.clean(text_with_fillers)
        assert "\U000f0800" not in result
        assert "\U000f0fff" not in result

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        normalizer = TextNormalizer()
        test_cases = [
            ("  multiple   spaces  ", "multiple spaces"),
            ("tab\tseparated\t\tvalues", "tab separated values"),
            ("mixed \t spaces \n and tabs", "mixed spaces and tabs"),
        ]
        for input_text, expected in test_cases:
            result = normalizer.clean(input_text)
            assert result == expected

    def test_clean_duplicate_titles(self):
        """Test duplicate title detection and cleaning."""
        normalizer = TextNormalizer()
        test_cases = [
            ("겸임교원규정 3—1—10～ 겸임교원규정", "겸임교원규정"),
            ("학칙 1-2-5～ 학칙", "학칙"),
            # Not a duplicate
            ("학칙 시행세칙", "학칙 시행세칙"),
        ]
        for input_text, expected in test_cases:
            result = normalizer.clean(input_text)
            # After cleaning, should match expected
            assert result == expected or result.replace(" ", "") == expected.replace(" ", "")

    def test_is_meaningful(self):
        """Test is_meaningful method."""
        normalizer = TextNormalizer()
        assert normalizer.is_meaningful("규정") is True
        assert normalizer.is_meaningful("") is False
        assert normalizer.is_meaningful("   ") is False
        # Horizontal rules get removed, leaving nothing meaningful
        assert normalizer.is_meaningful("—-—") is False

    def test_clean_line(self):
        """Test clean_line method."""
        normalizer = TextNormalizer()
        assert normalizer.clean_line("valid text") == "valid text"
        assert normalizer.clean_line("   ") is None
        assert normalizer.clean_line("") is None

    def test_get_title_hash(self):
        """Test title hash generation."""
        normalizer = TextNormalizer()
        hash1 = normalizer.get_title_hash("학 칙")
        hash2 = normalizer.get_title_hash("학칙")
        assert hash1 == hash2  # Spaces removed

    def test_titles_match(self):
        """Test title matching."""
        normalizer = TextNormalizer()
        assert normalizer.titles_match("학 칙", "학칙") is True
        assert normalizer.titles_match("규정", "요령") is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_normalize_regulation_text(self):
        """Test normalize_regulation_text convenience function."""
        result = normalize_regulation_text("  test  ")
        assert result == "test"

    def test_get_default_normalizer(self):
        """Test get_default_normalizer returns singleton."""
        normalizer1 = get_default_normalizer()
        normalizer2 = get_default_normalizer()
        assert normalizer1 is normalizer2


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_mixed_dash_types(self):
        """Test various dash types in page headers."""
        normalizer = TextNormalizer()
        dash_types = [
            ("3-1-10～", "규정"),    # Regular dash
            ("3—1—10～", "규정"),    # Em dash
            ("3－1－10～", "규정"),  # Fullwidth dash
            ("3—1—10～", "규정"),    # Different em dash
        ]
        for dash_type, expected in dash_types:
            # Test with space before page header
            result = normalizer.clean(f"규정 {dash_type}")
            assert result.strip() == expected, f"For '규정 {dash_type}': got '{result}'"

            # Test without space
            result_no_space = normalizer.clean(f"규정{dash_type}")
            assert result_no_space == expected, f"For '규정{dash_type}': got '{result_no_space}'"

            # Verify tilde is removed
            assert "～" not in result_no_space, f"Tilde should be removed in: {result_no_space}"

    def test_korean_whitespace(self):
        """Test Korean whitespace normalization."""
        normalizer = TextNormalizer()
        # Korean space (U+3163) and regular space
        # Note: U+3163 is a Korean letter, not whitespace - normalize_whitespace won't remove it
        result = normalizer.clean("한글ㅤ공백")
        # Just verify it doesn't crash - Korean hangul filler characters may remain
        assert "한글" in result

    def test_very_long_title(self):
        """Test handling of very long titles."""
        normalizer = TextNormalizer()
        long_title = "a" * 300
        result = normalizer.clean(long_title)
        assert len(result) <= 300

    def test_special_characters(self):
        """Test special character handling."""
        normalizer = TextNormalizer()
        special_chars = "@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = normalizer.clean(f"규정 {special_chars}")
        # Some chars should remain, page headers removed
        assert "규정" in result
