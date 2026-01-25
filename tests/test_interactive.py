import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.getcwd())


class TestInteractive(unittest.TestCase):
    @patch("questionary.path")
    @patch("questionary.select")
    @patch("questionary.text")
    @patch("questionary.confirm")
    def test_run_interactive(self, mock_confirm, mock_text, mock_select, mock_path):
        from src.interactive import run_interactive

        # Reset all mocks to ensure clean state
        mock_path.reset_mock()
        mock_select.reset_mock()
        mock_text.reset_mock()
        mock_confirm.reset_mock()

        # Setup mocks for sequential answers
        # questionary.path() is used for directory path selection (returns Path)
        mock_path.return_value.ask.return_value = Path("data/input")

        # Create separate mock instances for different select calls
        # File selection (line 207) returns Path object
        file_select_mock = MagicMock()
        file_select_mock.ask.return_value = Path("input.hwp")

        # Provider selection (line 263) returns string
        provider_select_mock = MagicMock()
        provider_select_mock.ask.return_value = "openai"

        # questionary.text() is used for model name (line 278)
        mock_text.return_value.ask.return_value = "gpt-4o"

        # questionary.confirm() is used for use_llm, verbose, force (lines 236, 246, 288)
        # First confirm (use_llm) should be True, others can be False
        mock_confirm.return_value.ask.side_effect = [True, False, False, False]

        # Patch questionary.select to return different mocks based on call
        # We need to ensure proper isolation between calls

        # We need to mock existence of HWP files for the wizard to find them
        with patch("pathlib.Path.glob", return_value=[Path("input.hwp")]):
            with patch("pathlib.Path.exists", return_value=True):
                # Patch questionary.select with proper side_effect
                with patch("questionary.select") as patched_select:
                    # Create a mock that tracks which call it's on
                    call_count = [0]

                    def select_mock(*args, **kwargs):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            # File selection - return mock that returns Path
                            m = MagicMock()
                            m.ask.return_value = Path("input.hwp")
                            return m
                        else:
                            # Provider selection - return mock that returns string
                            m = MagicMock()
                            m.ask.return_value = "openai"
                            return m

                    patched_select.side_effect = select_mock
                    args = run_interactive()

        self.assertEqual(str(args.input_path), "input.hwp")
        self.assertEqual(args.provider, "openai")
        self.assertTrue(args.use_llm)


if __name__ == "__main__":
    unittest.main()
