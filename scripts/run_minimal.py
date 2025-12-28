#!/usr/bin/env python3
import importlib.util
import os
import sys
import warnings
from unittest.mock import MagicMock

# Add the project root directory to sys.path to ensure src module is found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Suppress annoying warnings from transformers/accelerate if PyTorch/TF not found
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="hwp5.importhelper")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# 1. Mock src.llm_client to avoid loading llama_index
mock_llm_client = MagicMock()


# Ensure LLMClient class exists in the mock
class MockLLMClient:
    def __init__(self, *args, **kwargs):
        pass


mock_llm_client.LLMClient = MockLLMClient
sys.modules["src.llm_client"] = mock_llm_client

# 2. Mock llama_index (if needed by other modules unrelated to LLMClient)
sys.modules["llama_index"] = MagicMock()
sys.modules["llama_index.llms"] = MagicMock()
sys.modules["llama_index.llms.openai"] = MagicMock()
sys.modules["llama_index.core"] = MagicMock()

# 3. Mock rich if missing
if importlib.util.find_spec("rich") is None:
    # Create valid class for inheritance without 'rich' installed
    class MockProgressColumn:
        pass

    class MockConsole:
        def __init__(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            # Strip tags maybe? or just print
            print(*args)

        def rule(self, *args, **kwargs):
            print("-" * 30)

    class MockProgress:
        def __init__(self, *args, **kwargs):
            self.console = MockConsole()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            pass

    # Mocking modules
    sys.modules["rich"] = MagicMock()

    m_console = MagicMock()
    m_console.Console = MockConsole
    sys.modules["rich.console"] = m_console

    m_progress = MagicMock()
    m_progress.Progress = MockProgress
    m_progress.ProgressColumn = MockProgressColumn  # Needed for inheritance
    # Columns
    m_progress.SpinnerColumn = MagicMock
    m_progress.TimeElapsedColumn = MagicMock
    m_progress.BarColumn = MagicMock
    m_progress.TextColumn = MagicMock
    m_progress.TaskProgressColumn = MagicMock
    sys.modules["rich.progress"] = m_progress

    sys.modules["rich.text"] = MagicMock()
    sys.modules["rich.table"] = MagicMock()


def _run():
    from src.main import main

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    _run()
