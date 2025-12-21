import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

class TestInteractive(unittest.TestCase):
    @patch('questionary.path')
    @patch('questionary.select')
    @patch('questionary.text')
    @patch('questionary.confirm')
    def test_run_interactive(self, mock_confirm, mock_text, mock_select, mock_path):
        from src.interactive import run_interactive
        
        # Setup mocks for sequential answers
        mock_path.return_value.ask.return_value = "input.hwp"
        # Select returns a Path object in real app
        mock_select.return_value.ask.side_effect = [Path("input.hwp"), "openai", "gpt-4o", "Yes"]
        mock_confirm.return_value.ask.return_value = True
        mock_text.return_value.ask.return_value = "data/output"
        
        # We need to mock existence of HWP files for the wizard to find them
        with patch('pathlib.Path.glob', return_value=[Path("input.hwp")]):
            with patch('pathlib.Path.exists', return_value=True):
                args = run_interactive()
        
        self.assertEqual(str(args.input_path), "input.hwp")
        self.assertEqual(args.provider, "openai")
        self.assertTrue(args.use_llm)

if __name__ == "__main__":
    unittest.main()