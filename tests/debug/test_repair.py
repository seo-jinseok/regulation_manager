from unittest.mock import MagicMock, patch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

def test_repair_broken_lines_call():
    with patch('src.repair.LLMClient') as MockClient:
        try:
            from src.repair import RegulationRepair
        except ImportError:
            pytest.fail("src.repair module not found")

        mock_instance = MockClient.return_value
        # Use return_value for complete method
        mock_instance.complete.return_value = "Repaired Text"
        
        repair = RegulationRepair()
        # Input must be multi-line to bypass optimization
        result = repair.repair_broken_lines("Broken Text\nMore Text")
        
        assert result == "Repaired Text"
        mock_instance.complete.assert_called()

if __name__ == "__main__":
    test_repair_broken_lines_call()