"""
Unit tests for _extract_source_metadata in main.py
"""


from src.main import _extract_source_metadata


class TestExtractSourceMetadata:
    """Tests for _extract_source_metadata function."""

    def test_standard_filename(self):
        """Test extraction from standard filename format."""
        result = _extract_source_metadata("규정집9-343(20250909).hwp")
        assert result["source_serial"] == "9-343"
        assert result["source_date"] == "2025-09-09"

    def test_alternative_serial_format(self):
        """Test extraction with different serial format."""
        result = _extract_source_metadata("규정집10-500(20231215).hwp")
        assert result["source_serial"] == "10-500"
        assert result["source_date"] == "2023-12-15"

    def test_missing_serial(self):
        """Test when serial is not in filename."""
        result = _extract_source_metadata("문서(20240101).hwp")
        assert result["source_serial"] is None
        assert result["source_date"] == "2024-01-01"

    def test_missing_date(self):
        """Test when date is not in filename."""
        result = _extract_source_metadata("규정집8-200.hwp")
        assert result["source_serial"] == "8-200"
        assert result["source_date"] is None

    def test_no_matching_pattern(self):
        """Test when no patterns match."""
        result = _extract_source_metadata("random_file.hwp")
        assert result["source_serial"] is None
        assert result["source_date"] is None

    def test_json_extension(self):
        """Test with JSON extension."""
        result = _extract_source_metadata("규정집9-343(20250909).json")
        assert result["source_serial"] == "9-343"
        assert result["source_date"] == "2025-09-09"
