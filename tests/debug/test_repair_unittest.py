import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class TestRepair(unittest.TestCase):
    def test_repair_text_call(self):
        try:
            from src.repair import RegulationRepair
        except ImportError:
            self.fail("src.repair module not found")

        with patch('src.repair.LLMClient') as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.complete.return_value = "Repaired Text"
            
            repair = RegulationRepair()
            result = repair.repair_text("Broken Text")
            
            self.assertEqual(result, "Repaired Text")
            mock_instance.complete.assert_called_once()

if __name__ == "__main__":
    unittest.main()