from unittest.mock import MagicMock
from src.preprocessor import Preprocessor
from src.llm_client import LLMClient

def test_preprocessor_uses_repair():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = "Repaired"
    
    preprocessor = Preprocessor(llm_client=mock_llm)
    
    # Must be multi-line AFTER regex joining to trigger repair
    # Regex joins lines not ending in punctuation.
    # We use a header pattern to prevent joining.
    text = "제1조(목적)\n이 규정은 목적을 정의한다.\n① 이 항은 계층 구조를 가진다."
    
    result = preprocessor.clean(text)
    
    # Check if repair was called.
    # repair_broken_lines calls client.complete
    mock_llm.complete.assert_called()
    
    print("PASS: Preprocessor integrated with Repair.")

if __name__ == "__main__":
    test_preprocessor_uses_repair()
