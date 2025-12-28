import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.main import main


class TestMainFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tmp_integration_test")
        self.test_dir.mkdir(exist_ok=True)
        self.input_file = self.test_dir / "test.hwp"
        self.input_file.write_text("dummy content")
        self.output_dir = self.test_dir / "output"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("src.main.Preprocessor")
    @patch("src.main.RegulationFormatter")
    @patch("src.main.LLMClient")
    @patch("src.main.HwpToMarkdownReader", create=True)
    def test_main_full_flow(self, MockReaderClass, MockLLM, MockFormatter, MockPre):
        # Setup mocks
        mock_reader = MockReaderClass.return_value
        mock_reader.load_data.return_value = [
            MagicMock(
                text="# Title\nContent", metadata={"html_content": "<html>...</html>"}
            )
        ]
        MockPre.return_value.clean.return_value = "# Cleaned Title\nContent"
        MockPre.return_value.clean_pua.return_value = ("<html>...</html>", 0)
        MockFormatter.return_value.parse.return_value = [
            {"title": "Test Doc", "content": [], "metadata": {}}
        ]

        # Patch sys.argv and call main
        with patch(
            "sys.argv",
            [
                "main.py",
                str(self.input_file),
                "--output_dir",
                str(self.output_dir),
                "--force",
            ],
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify JSON was created
        json_files = [
            p
            for p in self.output_dir.glob("*.json")
            if not p.name.endswith("_metadata.json")
        ]
        self.assertGreater(len(json_files), 0)

    @patch("sys.argv", ["main.py", "non_existent_path_xyz.hwp"])
    def test_main_error_handling(self):
        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_with(1)


if __name__ == "__main__":
    unittest.main()
