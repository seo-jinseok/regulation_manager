"""
Extended tests for analyze_json.py to reach 85% coverage.

Covers missing lines:
- 83-85: File exists but error loading JSON
- 89: Print source file from data
- 107-111: Sort no validation (not a dict)
- 116: children not a list error
- 121-141: Main function with various edge cases
- 145: Invalid input path
"""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyze_json import (
    _validate_nodes,
    analyze_json,
    check_doc,
    check_structure,
    main,
)


class TestAnalyzeJsonExtendedCoverage(unittest.TestCase):
    """Extended tests for analyze_json coverage gaps."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.valid_json_path = os.path.join(self.test_dir, "valid.json")
        self.invalid_json_path = os.path.join(self.test_dir, "invalid.json")

        # Create valid JSON file
        valid_data = {
            "file_name": "test_regulation.json",
            "docs": [
                {
                    "title": "Test Regulation",
                    "content": [
                        {
                            "type": "chapter",
                            "title": "Chapter 1",
                            "children": [{"type": "article", "text": "Article text"}],
                            "sort_no": {"value": 1},
                        }
                    ],
                    "addenda": [{"type": "addendum", "text": "Addendum text"}],
                    "attached_files": [
                        {"title": "Attachment1.pdf", "file_ref": "ref123"}
                    ],
                }
            ],
        }
        with open(self.valid_json_path, "w", encoding="utf-8") as f:
            json.dump(valid_data, f)

        # Create invalid JSON file
        with open(self.invalid_json_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_analyze_json_error_loading(self):
        """Test error handling when JSON file exists but fails to load (lines 83-85)."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            analyze_json(self.invalid_json_path)
            output = mock_stdout.getvalue()
            self.assertIn("Error loading JSON", output)

    def test_analyze_json_print_file_name(self):
        """Test printing source file name from data (line 89)."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            analyze_json(self.valid_json_path)
            output = mock_stdout.getvalue()
            self.assertIn("Source File:", output)
            self.assertIn("test_regulation.json", output)

    def test_validate_sort_no_not_dict(self):
        """Test sort_no validation when not a dict (lines 107-111)."""
        # Node with sort_no that's not a dict
        node = {
            "type": "article",
            "title": "Test",
            "children": [],
            "sort_no": "not_a_dict",  # Invalid: should be dict
        }
        errors = _validate_nodes([node], "Test content")
        self.assertTrue(any("sort_no" in e and "not a dict" in e for e in errors))

    def test_validate_children_not_list(self):
        """Test children validation when not a list (line 116)."""
        # Node with children that's not a list
        node = {"type": "article", "title": "Test", "children": "not_a_list"}
        errors = _validate_nodes([node], "Test content")
        self.assertTrue(any("children" in e and "not a list" in e for e in errors))

    def test_validate_missing_type(self):
        """Test validation when type is missing."""
        node = {"title": "Test", "children": []}  # Missing 'type'
        errors = _validate_nodes([node], "Test content")
        self.assertTrue(any("missing 'type'" in e for e in errors))

    def test_validate_node_not_dict(self):
        """Test validation when node is not a dict."""
        errors = _validate_nodes(["not_a_dict"], "Test content")
        self.assertTrue(any("invalid node" in e and "not a dict" in e for e in errors))

    def test_check_structure_missing_keys(self):
        """Test check_structure with missing root keys."""
        # Missing 'docs' key
        data = {"file_name": "test.json"}
        errors = check_structure(data)
        self.assertIn("Missing root key: docs", errors)

        # Missing 'file_name' key
        data = {"docs": []}
        errors = check_structure(data)
        self.assertIn("Missing root key: file_name", errors)

    def test_check_doc_title_validation(self):
        """Test check_doc title validation."""
        # Missing title
        doc = {"content": []}
        errors = check_doc(doc, 0)
        self.assertTrue(any("Missing 'title'" in e for e in errors))

        # Empty title
        doc = {"title": "   ", "content": []}
        errors = check_doc(doc, 0)
        self.assertTrue(any("Empty 'title'" in e for e in errors))

    def test_check_doc_content_not_list(self):
        """Test check_doc when content is not a list."""
        doc = {"title": "Test", "content": "not_a_list"}
        errors = check_doc(doc, 0)
        self.assertTrue(any("'content' is not a list" in e for e in errors))

    def test_check_doc_addenda_not_list(self):
        """Test check_doc when addenda is not a list."""
        doc = {"title": "Test", "content": [], "addenda": "not_a_list"}
        errors = check_doc(doc, 0)
        self.assertTrue(any("'addenda' is not a list" in e for e in errors))

    def test_check_doc_attached_files_not_list(self):
        """Test check_doc when attached_files is not a list."""
        doc = {"title": "Test", "content": [], "attached_files": "not_a_list"}
        errors = check_doc(doc, 0)
        self.assertTrue(any("'attached_files' is not a list" in e for e in errors))

    def test_check_doc_attached_file_item_not_dict(self):
        """Test check_doc when attached file item is not a dict."""
        doc = {
            "title": "Test",
            "content": [],
            "attached_files": ["not_a_dict"],
        }
        errors = check_doc(doc, 0)
        self.assertTrue(any("invalid type" in e for e in errors))

    def test_check_doc_attached_file_missing_title(self):
        """Test check_doc when attached file item missing title."""
        doc = {
            "title": "Test",
            "content": [],
            "attached_files": [{"file_ref": "ref123"}],  # Missing title
        }
        errors = check_doc(doc, 0)
        self.assertTrue(any("missing title" in e for e in errors))

    def test_main_with_file(self):
        """Test main function with file argument (lines 120-141)."""
        with patch("sys.argv", ["analyze_json.py", self.valid_json_path]):
            with patch("sys.stdout", new_callable=io.StringIO):
                # Should not raise exception
                main()

    def test_main_with_directory(self):
        """Test main function with directory argument."""
        with patch("sys.argv", ["analyze_json.py", self.test_dir]):
            with patch("sys.stdout", new_callable=io.StringIO):
                # Should not raise exception
                main()

    def test_main_no_arguments(self):
        """Test main function with no arguments (line 122)."""
        with patch("sys.argv", ["analyze_json.py"]):
            with self.assertRaises(SystemExit) as context:
                main()
            self.assertEqual(context.exception.code, 1)

    def test_main_directory_no_json_files(self):
        """Test main function with empty directory (lines 135-136)."""
        # Create empty directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)

        with patch("sys.argv", ["analyze_json.py", empty_dir]):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                self.assertIn("No JSON files found", output)

    def test_main_invalid_path(self):
        """Test main function with invalid path (line 140, 145)."""
        with patch("sys.argv", ["analyze_json.py", "/nonexistent/path"]):
            with self.assertRaises(SystemExit) as context:
                main()
            self.assertEqual(context.exception.code, 1)

    def test_validate_nodes_deep_recursion(self):
        """Test _validate_nodes with deep nested structure."""
        deep_structure = {
            "type": "root",
            "children": [
                {
                    "type": "level1",
                    "children": [
                        {
                            "type": "level2",
                            "children": [{"type": "level3", "children": []}],
                        }
                    ],
                }
            ],
        }
        errors = _validate_nodes([deep_structure], "Deep structure")
        # Should not crash and should handle recursion
        self.assertEqual(errors, [])  # Valid structure, no errors

    def test_check_doc_multiple_errors(self):
        """Test check_doc with multiple validation errors."""
        doc = {
            # Missing title
            "content": "not_a_list",  # Invalid content
            "addenda": "not_a_list",  # Invalid addenda
            "attached_files": ["invalid"],  # Invalid attached_files
        }
        errors = check_doc(doc, 0)
        # Should have multiple errors
        self.assertGreater(len(errors), 1)
        self.assertTrue(any("title" in e for e in errors))
        self.assertTrue(any("content" in e for e in errors))

    def test_analyze_json_success_case(self):
        """Test analyze_json with valid data showing success message."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            analyze_json(self.valid_json_path)
            output = mock_stdout.getvalue()
            self.assertIn("SUCCESS", output)
            self.assertIn("All checks passed", output)

    def test_analyze_json_with_errors(self):
        """Test analyze_json with validation errors."""
        # Create JSON with errors
        error_json_path = os.path.join(self.test_dir, "errors.json")
        error_data = {
            "file_name": "errors.json",
            "docs": [
                {
                    # Missing title
                    "content": [{"type": "article"}]  # Missing children
                }
            ],
        }
        with open(error_json_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            analyze_json(error_json_path)
            output = mock_stdout.getvalue()
            self.assertIn("FAIL", output)
            self.assertIn("validation issues", output)


if __name__ == "__main__":
    unittest.main()
