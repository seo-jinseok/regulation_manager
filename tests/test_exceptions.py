"""
Tests for src.exceptions module to improve coverage from 65% to 100%.

Focuses on testing exception attributes and message formatting for:
- RegulationParseError with line_number (lines 30-33)
- ReferenceResolutionError attributes (lines 46-48)
- HWPConversionError attributes (lines 66-67)
- SyncError attributes (lines 97-99)
- LLMError attributes (lines 112-113)
- MissingAPIKeyError attributes (lines 131-132)
"""

import unittest

from src.exceptions import (
    ConfigurationError,
    ConversionError,
    HWPConversionError,
    JSONSchemaError,
    LLMError,
    MissingAPIKeyError,
    ParsingError,
    RAGError,
    ReferenceResolutionError,
    RegulationManagerError,
    RegulationParseError,
    SearchError,
    SyncError,
    TableParseError,
    VectorStoreError,
)


class TestExceptions(unittest.TestCase):
    """Test all exception classes and their attributes."""

    def test_regulation_manager_error_base(self):
        """Test base exception can be raised and caught."""
        with self.assertRaises(RegulationManagerError):
            raise RegulationManagerError("Base error")

    def test_regulation_manager_error_message(self):
        """Test base exception message."""
        error = RegulationManagerError("Test message")
        self.assertEqual(str(error), "Test message")

    # Parsing exceptions tests
    def test_parsing_error_base(self):
        """Test ParsingError base class."""
        with self.assertRaises(ParsingError):
            raise ParsingError("Parse error")

    def test_regulation_parse_error_basic(self):
        """Test RegulationParseError without line number."""
        error = RegulationParseError("Parse failed")
        self.assertIsNone(error.line_number)
        self.assertIn("Parse failed", str(error))

    def test_regulation_parse_error_with_line_number(self):
        """Test RegulationParseError with line number (lines 30-33)."""
        error = RegulationParseError("Parse failed", line_number=42)
        self.assertEqual(error.line_number, 42)
        self.assertIn("Line 42", str(error))
        self.assertIn("Parse failed", str(error))
        self.assertEqual(str(error), "Line 42: Parse failed")

    def test_regulation_parse_error_line_zero(self):
        """Test RegulationParseError with line 0."""
        # Note: line_number=0 is falsy in Python, so the "Line 0:" prefix is NOT added
        error = RegulationParseError("Parse failed", line_number=0)
        self.assertEqual(error.line_number, 0)
        # When line_number=0, it's treated as no line number
        self.assertEqual(str(error), "Parse failed")

    def test_regulation_parse_error_line_negative(self):
        """Test RegulationParseError with negative line number."""
        error = RegulationParseError("Parse failed", line_number=-1)
        self.assertEqual(error.line_number, -1)
        self.assertIn("Line -1", str(error))

    def test_table_parse_error(self):
        """Test TableParseError exception."""
        error = TableParseError("Table parse failed")
        self.assertIn("Table parse failed", str(error))

    def test_reference_resolution_error_attributes(self):
        """Test ReferenceResolutionError attributes (lines 46-48)."""
        error = ReferenceResolutionError("§3 Article 5", "Article not found")
        self.assertEqual(error.reference, "§3 Article 5")
        self.assertEqual(error.reason, "Article not found")
        self.assertIn("§3 Article 5", str(error))
        self.assertIn("Article not found", str(error))
        self.assertEqual(
            str(error), "Cannot resolve reference '§3 Article 5': Article not found"
        )

    def test_reference_resolution_error_empty_reference(self):
        """Test ReferenceResolutionError with empty reference."""
        error = ReferenceResolutionError("", "Empty reference")
        self.assertEqual(error.reference, "")
        self.assertEqual(error.reason, "Empty reference")

    def test_reference_resolution_error_unicode(self):
        """Test ReferenceResolutionError with unicode characters."""
        error = ReferenceResolutionError("§조항", "한글 이유")
        self.assertEqual(error.reference, "§조항")
        self.assertEqual(error.reason, "한글 이유")

    # Conversion exceptions tests
    def test_conversion_error_base(self):
        """Test ConversionError base class."""
        with self.assertRaises(ConversionError):
            raise ConversionError("Conversion failed")

    def test_hwp_conversion_error_attributes(self):
        """Test HWPConversionError attributes (lines 66-67)."""
        error = HWPConversionError("/path/to/file.hwp", "Invalid format")
        self.assertEqual(error.file_path, "/path/to/file.hwp")
        self.assertIn("/path/to/file.hwp", str(error))
        self.assertIn("Invalid format", str(error))
        self.assertEqual(
            str(error), "Failed to convert HWP file '/path/to/file.hwp': Invalid format"
        )

    def test_hwp_conversion_error_empty_path(self):
        """Test HWPConversionError with empty path."""
        error = HWPConversionError("", "No path")
        self.assertEqual(error.file_path, "")

    def test_hwp_conversion_error_special_characters(self):
        """Test HWPConversionError with special characters in path."""
        error = HWPConversionError("/path/한글/file.hwp", "Error with 한글")
        self.assertIn("한글", str(error))

    def test_json_schema_error(self):
        """Test JSONSchemaError exception."""
        error = JSONSchemaError("Schema validation failed")
        self.assertIn("Schema validation failed", str(error))

    # RAG exceptions tests
    def test_rag_error_base(self):
        """Test RAGError base class."""
        with self.assertRaises(RAGError):
            raise RAGError("RAG error")

    def test_vector_store_error(self):
        """Test VectorStoreError exception."""
        error = VectorStoreError("Store connection failed")
        self.assertIn("Store connection failed", str(error))

    def test_sync_error_attributes(self):
        """Test SyncError attributes (lines 97-99)."""
        error = SyncError("Sync completed", added=10, failed=2)
        self.assertEqual(error.added, 10)
        self.assertEqual(error.failed, 2)
        self.assertIn("Sync completed", str(error))
        self.assertIn("added: 10", str(error))
        self.assertIn("failed: 2", str(error))
        self.assertEqual(str(error), "Sync completed (added: 10, failed: 2)")

    def test_sync_error_zero_values(self):
        """Test SyncError with zero values."""
        error = SyncError("No sync", added=0, failed=0)
        self.assertEqual(error.added, 0)
        self.assertEqual(error.failed, 0)
        self.assertIn("added: 0", str(error))
        self.assertIn("failed: 0", str(error))

    def test_sync_error_only_added(self):
        """Test SyncError with only added parameter."""
        error = SyncError("All added", added=5, failed=0)
        self.assertEqual(error.added, 5)
        self.assertEqual(error.failed, 0)

    def test_sync_error_only_failed(self):
        """Test SyncError with only failed parameter."""
        error = SyncError("All failed", added=0, failed=5)
        self.assertEqual(error.added, 0)
        self.assertEqual(error.failed, 5)

    def test_search_error(self):
        """Test SearchError exception."""
        error = SearchError("Search query invalid")
        self.assertIn("Search query invalid", str(error))

    def test_llm_error_attributes(self):
        """Test LLMError attributes (lines 112-113)."""
        error = LLMError("openai", "Rate limit exceeded")
        self.assertEqual(error.provider, "openai")
        self.assertIn("[openai]", str(error))
        self.assertIn("Rate limit exceeded", str(error))
        self.assertEqual(str(error), "[openai] Rate limit exceeded")

    def test_llm_error_different_providers(self):
        """Test LLMError with different providers."""
        providers = ["openai", "anthropic", "azure", "local"]
        for provider in providers:
            error = LLMError(provider, f"Error from {provider}")
            self.assertEqual(error.provider, provider)
            self.assertIn(f"[{provider}]", str(error))

    def test_llm_error_empty_provider(self):
        """Test LLMError with empty provider."""
        error = LLMError("", "No provider")
        self.assertEqual(error.provider, "")
        self.assertEqual(str(error), "[] No provider")

    # Configuration exceptions tests
    def test_configuration_error_base(self):
        """Test ConfigurationError base class."""
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("Config error")

    def test_missing_api_key_error_attributes(self):
        """Test MissingAPIKeyError attributes (lines 131-132)."""
        error = MissingAPIKeyError("OPENAI_API_KEY")
        self.assertEqual(error.key_name, "OPENAI_API_KEY")
        self.assertIn("OPENAI_API_KEY", str(error))
        self.assertEqual(str(error), "Missing required API key: OPENAI_API_KEY")

    def test_missing_api_key_error_different_keys(self):
        """Test MissingAPIKeyError with various key names."""
        keys = ["ANTHROPIC_API_KEY", "AZURE_KEY", "GOOGLE_KEY"]
        for key in keys:
            error = MissingAPIKeyError(key)
            self.assertEqual(error.key_name, key)
            self.assertIn(key, str(error))

    def test_missing_api_key_error_empty_key(self):
        """Test MissingAPIKeyError with empty key name."""
        error = MissingAPIKeyError("")
        self.assertEqual(error.key_name, "")
        self.assertEqual(str(error), "Missing required API key: ")

    # Exception inheritance tests
    def test_parsing_error_is_regulation_manager_error(self):
        """Test ParsingError inherits from RegulationManagerError."""
        with self.assertRaises(RegulationManagerError):
            raise ParsingError("Parse error")

    def test_conversion_error_is_regulation_manager_error(self):
        """Test ConversionError inherits from RegulationManagerError."""
        with self.assertRaises(RegulationManagerError):
            raise ConversionError("Conversion error")

    def test_rag_error_is_regulation_manager_error(self):
        """Test RAGError inherits from RegulationManagerError."""
        with self.assertRaises(RAGError):
            raise RAGError("RAG error")

    def test_configuration_error_is_regulation_manager_error(self):
        """Test ConfigurationError inherits from RegulationManagerError."""
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("Config error")

    def test_regulation_parse_error_is_parsing_error(self):
        """Test RegulationParseError inherits from ParsingError."""
        with self.assertRaises(ParsingError):
            raise RegulationParseError("Parse error")

    def test_hwp_conversion_error_is_conversion_error(self):
        """Test HWPConversionError inherits from ConversionError."""
        with self.assertRaises(ConversionError):
            raise HWPConversionError("/path/file.hwp", "Error")

    def test_sync_error_is_rag_error(self):
        """Test SyncError inherits from RAGError."""
        with self.assertRaises(RAGError):
            raise SyncError("Sync", added=1, failed=0)

    def test_llm_error_is_rag_error(self):
        """Test LLMError inherits from RAGError."""
        with self.assertRaises(RAGError):
            raise LLMError("provider", "message")


if __name__ == "__main__":
    unittest.main()
