import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.converter import HwpToMarkdownReader


class TestConverter(unittest.TestCase):
    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="<html><body>Test HTML</body></html>",
    )
    def test_load_data_success(self, mock_open, mock_tmp, mock_popen):
        # Setup mocks
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ["line1", "line2"]
        mock_popen.return_value = mock_process

        # Mock Path.exists and glob
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.glob", return_value=[Path("/tmp/dir/index.xhtml")]
            ):
                reader = HwpToMarkdownReader()
                docs = reader.load_data(Path("test.hwpx"))

                self.assertEqual(len(docs), 1)
                self.assertIn("Test HTML", docs[0].text)

    def test_load_data_file_not_found(self):
        reader = HwpToMarkdownReader()
        with self.assertRaises(FileNotFoundError):
            reader.load_data(Path("non_existent.hwpx"))

    def test_load_data_invalid_extension_hwp(self):
        """Test that .hwp files (legacy binary format) are rejected."""
        reader = HwpToMarkdownReader()
        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaises(ValueError) as context:
                reader.load_data(Path("test.hwp"))

            self.assertIn("Unsupported file format", str(context.exception))
            self.assertIn(".hwpx", str(context.exception))
            self.assertIn("convert them to .hwpx", str(context.exception))


if __name__ == "__main__":
    unittest.main()
