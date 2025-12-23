import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from src.main import main

class TestMainFailureModes(unittest.TestCase):
    @patch('src.main.Path.rglob', return_value=[])
    @patch('sys.argv', ['main.py', 'missing_dir'])
    def test_main_no_files_found(self, mock_rglob):
        with self.assertRaises(SystemExit):
            main()

    @patch('src.main.CacheManager')
    def test_main_reader_import_error(self, MockCache):
        # Simulate reader being None (ImportError fallback)
        with patch('src.main.Path.rglob', return_value=[Path("test.hwp")]):
            with patch('src.main.Path.is_file', return_value=True):
                with patch('src.main.Path.exists', return_value=False):
                    with patch('src.main.Path.mkdir'):
                        # Use side_effect to mock the reader being None inside main()
                        with patch('src.main.reader', None, create=True):
                            with patch('sys.argv', ['main.py', 'test.hwp']):
                                with patch('builtins.open', MagicMock()):
                                    with self.assertRaises(SystemExit):
                                        main()

    @patch('src.main.LLMClient', side_effect=Exception("LLM init failed"))
    @patch('src.main.Path.mkdir')
    @patch('src.main.Path.is_file', return_value=True)
    @patch('src.main.Path.exists', return_value=True)
    def test_main_llm_required_init_failure(self, mock_exists, mock_is_file, mock_mkdir, mock_llm):
        with patch('sys.argv', ['main.py', 'test.hwp', '--use_llm']):
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_with(1)

if __name__ == "__main__":
    unittest.main()
