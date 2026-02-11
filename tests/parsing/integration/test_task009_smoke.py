"""
TASK-009: Smoke Tests for Quick Validation.

Fast-running tests for quick validation of core functionality.
These tests can be run frequently during development.

Version: 1.0.0
Reference: SPEC-HWXP-002, TASK-009
"""
from pathlib import Path
import pytest
import zipfile


@pytest.fixture
def hwpx_file():
    """Path to the HWPX file."""
    return Path("data/input/규정집9-343(20250909).hwpx")


class TestSmokeFileStructure:
    """Quick file structure validation."""

    def test_file_exists(self, hwpx_file):
        """Verify HWPX file exists."""
        assert hwpx_file.exists()
        assert hwpx_file.is_file()

    def test_file_size(self, hwpx_file):
        """Verify file size is reasonable (~4MB)."""
        size = hwpx_file.stat().st_size
        assert 3_000_000 < size < 10_000_000

    def test_is_zip_archive(self, hwpx_file):
        """Verify file is a valid ZIP archive."""
        try:
            with zipfile.ZipFile(hwpx_file, 'r') as zf:
                namelist = zf.namelist()
                assert len(namelist) > 0
        except Exception as e:
            pytest.fail(f"Not a valid ZIP: {e}")

    def test_has_required_sections(self, hwpx_file):
        """Verify required sections exist."""
        with zipfile.ZipFile(hwpx_file, 'r') as zf:
            namelist = zf.namelist()
            assert "Contents/section0.xml" in namelist
            assert "Contents/section1.xml" in namelist


class TestSmokeParserImport:
    """Verify parser modules can be imported."""

    def test_import_standard_parser(self):
        """Verify HWPXMultiFormatParser can be imported."""
        from src.parsing.multi_format_parser import HWPXMultiFormatParser
        assert HWPXMultiFormatParser is not None

    def test_import_optimized_parser(self):
        """Verify OptimizedHWPXMultiFormatParser can be imported."""
        from src.parsing.optimized_multi_format_parser import OptimizedHWPXMultiFormatParser
        assert OptimizedHWPXMultiFormatParser is not None

    def test_import_format_types(self):
        """Verify FormatType enum can be imported."""
        from src.parsing.format.format_type import FormatType
        assert FormatType.ARTICLE is not None
        assert FormatType.LIST is not None
        assert FormatType.GUIDELINE is not None
        assert FormatType.UNSTRUCTURED is not None


class TestSmokeParserInstantiation:
    """Verify parsers can be instantiated."""

    def test_instantiate_standard_parser(self):
        """Verify standard parser can be created."""
        from src.parsing.multi_format_parser import HWPXMultiFormatParser
        parser = HWPXMultiFormatParser()
        assert parser is not None

    def test_instantiate_optimized_parser(self):
        """Verify optimized parser can be created."""
        from src.parsing.optimized_multi_format_parser import OptimizedHWPXMultiFormatParser
        parser = OptimizedHWPXMultiFormatParser()
        assert parser is not None


@pytest.mark.smoke
def test_smoke_validation(hwpx_file):
    """
    Quick smoke test for core functionality.

    This test:
    1. Verifies file exists
    2. Imports parser
    3. Instantiates parser
    4. Does NOT parse (to keep it fast)
    """
    assert hwpx_file.exists()

    from src.parsing.multi_format_parser import HWPXMultiFormatParser
    parser = HWPXMultiFormatParser()
    assert parser is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
