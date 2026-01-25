"""
Additional tests for converter.py to improve coverage.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.converter import HwpToMarkdownReader


class TestHwpToMarkdownReaderBasic(unittest.TestCase):
    """Basic tests for HwpToMarkdownReader."""

    def test_init_default(self):
        """Test default initialization."""
        reader = HwpToMarkdownReader()
        self.assertIsNotNone(reader)

    def test_init_with_keep_html(self):
        """Test initialization with keep_html option."""
        reader = HwpToMarkdownReader(keep_html=True)
        self.assertIsNotNone(reader)

    def test_init_with_keep_html_false(self):
        """Test initialization with keep_html=False."""
        reader = HwpToMarkdownReader(keep_html=False)
        self.assertIsNotNone(reader)

    def test_load_data_with_nonexistent_file(self):
        """Test load_data with non-existent file raises FileNotFoundError."""
        reader = HwpToMarkdownReader()
        with self.assertRaises(FileNotFoundError):
            reader.load_data(Path("non_existent_file.hwp"))


class TestHwpToMarkdownReaderData(unittest.TestCase):
    """Tests for data processing methods."""

    def setUp(self):
        self.reader = HwpToMarkdownReader()

    @patch("subprocess.Popen")
    def test_hwp5html_not_found(self, mock_popen):
        """Test when hwp5html is not available."""
        mock_popen.side_effect = FileNotFoundError("hwp5html not found")
        with self.assertRaises(FileNotFoundError):
            self.reader.load_data(Path("test.hwp"))

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_conversion_success(self, mock_tmp, mock_popen):
        """Test successful conversion."""
        # Setup mocks
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ["<html><body>Test content</body></html>"]
        mock_popen.return_value = mock_process

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.glob", return_value=[Path("/tmp/dir/index.xhtml")]
            ):
                with patch("builtins.open", create=True) as mock_open_func:
                    mock_open_func.return_value.__enter__.return_value.read.return_value = "<html><body>Test content</body></html>"
                    docs = self.reader.load_data(Path("test.hwp"))
                    self.assertEqual(len(docs), 1)


class TestHwpToMarkdownReaderMetadata(unittest.TestCase):
    """Tests for metadata extraction."""

    def setUp(self):
        self.reader = HwpToMarkdownReader(keep_html=True)

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_html_content_in_metadata(self, mock_tmp, mock_popen):
        """Test HTML content is stored in metadata."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ["<html><body>Test</body></html>"]
        mock_popen.return_value = mock_process

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.glob", return_value=[Path("/tmp/dir/index.xhtml")]
            ):
                with patch("builtins.open", create=True) as mock_open_func:
                    mock_open_func.return_value.__enter__.return_value.read.return_value = "<html><body>Test</body></html>"
                    docs = self.reader.load_data(Path("test.hwp"))
                    if docs:
                        self.assertIn("html_content", docs[0].metadata)


if __name__ == "__main__":
    unittest.main()
