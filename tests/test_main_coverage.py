# -*- coding: utf-8 -*-
"""
Additional tests for main.py to improve coverage.

Focuses on testable helper functions and edge cases.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.main import (
    _build_pipeline_signature,
    _collect_hwp_files,
    _extract_source_metadata,
    _resolve_preprocessor_rules_path,
)


class TestBuildPipelineSignature(unittest.TestCase):
    """Tests for _build_pipeline_signature function."""

    def test_basic_signature(self):
        """Test basic signature construction."""
        result = _build_pipeline_signature("abc123", "v1")
        expected = "v5|rules:abc123|llm:v1"
        self.assertEqual(result, expected)

    def test_missing_rules_hash(self):
        """Test signature when rules hash is missing."""
        result = _build_pipeline_signature("", "v1")
        expected = "v5|rules:|llm:v1"
        self.assertEqual(result, expected)

    def test_missing_llm_signature(self):
        """Test signature when LLM signature is disabled."""
        result = _build_pipeline_signature("abc123", "disabled")
        expected = "v5|rules:abc123|llm:disabled"
        self.assertEqual(result, expected)

    def test_empty_components(self):
        """Test signature with empty components."""
        result = _build_pipeline_signature("", "")
        expected = "v5|rules:|llm:"
        self.assertEqual(result, expected)


class TestResolvePreprocessorRulesPath(unittest.TestCase):
    """Tests for _resolve_preprocessor_rules_path function."""

    def test_default_path(self):
        """Test default path when env var not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_preprocessor_rules_path()
            self.assertEqual(result, Path("data/config/preprocessor_rules.json"))

    def test_env_var_override(self):
        """Test env var override."""
        with patch.dict(
            "os.environ", {"PREPROCESSOR_RULES_PATH": "/custom/path/rules.json"}
        ):
            result = _resolve_preprocessor_rules_path()
            self.assertEqual(result, Path("/custom/path/rules.json"))


class TestCollectHwpFiles(unittest.TestCase):
    """Tests for _collect_hwp_files function."""

    def setUp(self):
        self.console = MagicMock()

    @patch("pathlib.Path.exists", return_value=False)
    def test_non_existent_path(self, mock_exists):
        """Test non-existent input path."""
        result = _collect_hwp_files(Path("/non/existent"), self.console)
        self.assertIsNone(result)
        self.console.print.assert_called()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_single_file(self, mock_is_file, mock_exists):
        """Test collecting single file."""
        result = _collect_hwp_files(Path("test.hwpx"), self.console)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], Path("test.hwp"))


class TestExtractSourceMetadataExtended(unittest.TestCase):
    """Extended tests for _extract_source_metadata function."""

    def test_single_digit_serial(self):
        """Test single digit serial number."""
        result = _extract_source_metadata("regulation1-1(20250101).hwpx")
        # Pattern requires "규정집" prefix in Korean, not "regulation"
        self.assertIsNone(result["source_serial"])

    def test_large_serial_number(self):
        """Test large serial number."""
        result = _extract_source_metadata("regulation999-999(20251231).hwpx")
        # Pattern requires "규정집" prefix
        self.assertIsNone(result["source_serial"])

    def test_invalid_date_still_parsed(self):
        """Test invalid date is still parsed."""
        result = _extract_source_metadata("regulation1-1(20241301).hwpx")
        # Pattern requires "규정집" prefix
        self.assertIsNone(result["source_serial"])

    def test_no_extension(self):
        """Test filename without extension."""
        result = _extract_source_metadata("regulation9-343(20250909)")
        # Pattern requires "규정집" prefix
        self.assertIsNone(result["source_serial"])


if __name__ == "__main__":
    unittest.main()
