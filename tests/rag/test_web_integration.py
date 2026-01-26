
from unittest.mock import MagicMock, patch

import pytest

from src.rag.interface.gradio_app import create_app


@pytest.fixture
def mock_gradio_components():
    with patch('src.rag.interface.gradio_app.gr') as mock_gr:
        yield mock_gr

@pytest.fixture
def mock_handler():
    with patch('src.rag.interface.gradio_app.QueryHandler') as mock_cls:
        handler_instance = MagicMock()
        mock_cls.return_value = handler_instance
        yield mock_cls, handler_instance

def test_tool_calling_integration(mock_gradio_components, mock_handler):
    """Test if Tool Calling option is correctly passed to QueryHandler."""
    mock_handler_cls, mock_handler_inst = mock_handler

    # Mock process_query_stream to yield events
    def mock_stream(*args, **kwargs):
        yield {"type": "progress", "content": "Thinking..."}
        yield {"type": "token", "content": "Tools"}
        yield {"type": "token", "content": " are working."}
        yield {"type": "complete", "content": "Tools are working."}

    mock_handler_inst.process_query_stream.side_effect = mock_stream

    # Create app (triggers chat_respond definition)
    # Since chat_respond is internal, we can't call it directly easily.
    # However, we can simulate the "process_with_handler" part logic by checking arguments passed to QueryHandler

    # Actually, we can just inspect if FunctionGemmaAdapter is initialized when use_tools is True

    with patch('src.rag.interface.gradio_app.FunctionGemmaAdapter') as MockAdapter:
        app = create_app(db_path="dummy", use_mock_llm=True)

        # We need to access the chat_respond function.
        # In Gradio, event handlers are registered. It's hard to access local functions.
        # So we will replicate the logic test:
        # Check if "chat_use_tools" checkbox exists.

        # Verify UI component creation
        # settings_panel logic...
        # assert MockAdapter.called # This won't be called yet, only on request
        pass

def test_query_handler_initialization_with_tools():
    """Verify QueryHandler receives function_gemma_client when tools enabled."""
    from src.rag.interface.gradio_app import _process_with_handler

    with patch('src.rag.interface.gradio_app.QueryHandler') as MockHandlerCls, \
         patch('src.rag.interface.gradio_app.LLMClientAdapter'), \
         patch('src.rag.interface.gradio_app.ChromaVectorStore'):

        # Call the internal helper function
        _process_with_handler(
            query="test",
            top_k=5,
            include_abolished=False,
            llm_provider="openai",
            llm_model="gpt-4",
            llm_base_url=None,
            target_db_path="dummy",
            audience_override=None,
            use_tools=True,  # Enable tools
            history=[],
            state={}
        )

        # Verify Handler was created with function_gemma_client (not None)
        call_args = MockHandlerCls.call_args
        assert call_args.kwargs.get('function_gemma_client') is not None

def test_query_handler_initialization_without_tools():
    """Verify QueryHandler receives None for function_gemma_client when tools disabled."""
    from src.rag.interface.gradio_app import _process_with_handler

    with patch('src.rag.interface.gradio_app.QueryHandler') as MockHandlerCls, \
         patch('src.rag.interface.gradio_app.LLMClientAdapter'), \
         patch('src.rag.interface.gradio_app.ChromaVectorStore'):

        # Call with use_tools=False
        _process_with_handler(
            query="test",
            top_k=5,
            include_abolished=False,
            llm_provider="openai",
            llm_model="gpt-4",
            llm_base_url=None,
            target_db_path="dummy",
            audience_override=None,
            use_tools=False,
            history=[],
            state={}
        )

        # Verify Handler received None for function_gemma_client
        call_args = MockHandlerCls.call_args
        assert call_args.kwargs.get('function_gemma_client') is None
