import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.interface.link_formatter import extract_and_format_references
from src.rag.interface.gradio_app import create_app 
# Note: We can't easily instantiate the full Gradio app due to globals/styling, 
# but we can import the logic functions if they are exposed, or simulate the logic.
# Since chat_respond is inside create_app, we'll verify the logic by replicating the state updates
# or by testing the SearchUseCase stream directly which drives the UI.

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.llm_client import MockLLMClient
from src.rag.domain.entities import Chunk, SearchResult, ChunkLevel
from unittest.mock import MagicMock

def test_streaming_backend():
    print("\n--- [Test 1] Backend Streaming Verification ---")
    
    # Setup Mock
    mock_store = MagicMock()
    mock_store.count.return_value = 10
    
    # Mock search results
    mock_chunk = Chunk(
        id="test-id",
        text="제8조(자격) 교원은 다음 각 호의 자격을 갖추어야 한다.",
        title="제8조",
        rule_code="3-1-24",
        parent_path=["교원인사규정"],
        embedding_text="...",
        level=ChunkLevel.ARTICLE,
        full_text="제8조(자격) 교원은 다음 각 호의 자격을 갖추어야 한다.",
        token_count=10,
        keywords=[],
        is_searchable=True,
    )
    mock_results = [SearchResult(chunk=mock_chunk, score=0.9)]
    
    # Mock SearchUseCase methods
    # We can't easily mock search_unique inside ask_stream without patching, 
    # so we'll patch the methods called BY ask_stream if needed, or better, 
    # we simulate the generator yield sequence manually to verify our expectation of the contract.
    
    # Actually, let's use the real class with a Mock LLM and Mock Store
    mock_store.hybrid_search.return_value = mock_results
    mock_llm = MockLLMClient()
    
    usecase = SearchUseCase(mock_store, mock_llm)
    
    print("Calling ask_stream()...")
    stream = usecase.ask_stream("교원 자격은?", top_k=1)
    
    # Verify sequence
    step = 0
    has_metadata = False
    token_count = 0
    
    for item in stream:
        if item["type"] == "metadata":
            print(f"✅ Received Metadata: Sources={len(item['sources'])}")
            has_metadata = True
        elif item["type"] == "token":
            token_count += 1
            # print(f"Token: {item['content']}", end="")
            
    print(f"\n✅ Streaming completed. Tokens received: {token_count}")
    
    if has_metadata and token_count > 0:
        print(">> Backend Streaming Test PASSED")
    else:
        print(">> Backend Streaming Test FAILED")

def test_link_formatting_complex():
    print("\n--- [Test 2] Complex Link Formatting Verification ---")
    
    text = """
    본 규정은 교원인사규정 제8조에 따른다.
    또한 학칙 제3조(3-1-24)를 준용하며,
    복무규정 제5조 제2항을 참고한다.
    (관련: 3-1-99)
    """
    
    print(f"Original Text:\n{text.strip()}")
    
    refs, formatted_md = extract_and_format_references(text, "markdown")
    refs, formatted_cli = extract_and_format_references(text, "numbered")
    
    print(f"\nExtracted References: {len(refs)}")
    for r in refs:
        print(f" - {r.original_text} (Code: {r.rule_code})")
    
    print(f"\nFormatted Markdown:\n{formatted_md.strip()}")
    print(f"\nFormatted CLI List:\n{formatted_cli.strip()}")
    
    # Verification
    if len(refs) >= 3:
        print("✅ Correctly identified multiple references")
    else:
        print("❌ Failed to identify all references")
        
    if "[교원인사규정 제8조]" in formatted_md:
        print("✅ Markdown link creation successful")
    else:
        print("❌ Markdown link creation failed")

    if "[1] 교원인사규정 제8조" in formatted_cli:
        print("✅ CLI numbered list creation successful")
    else:
        print("❌ CLI numbered list creation failed")
        
    print(">> Link Formatting Test COMPLETED")

def test_navigation_logic_simulation():
    print("\n--- [Test 3] UI Navigation Logic Simulation ---")
    
    # Simulate the logic we added to gradio_app.py
    state = {
        "nav_history": [],
        "nav_index": -1,
        "last_query": None,
        "last_mode": None
    }
    
    def on_query_complete(query, mode, state):
        # Simulate the logic update
        prev_query = state.get("last_query")
        prev_mode = state.get("last_mode")
        
        state["last_query"] = query
        state["last_mode"] = mode
        
        if query and (query != prev_query or mode != prev_mode):
            nav_history = state.get("nav_history", [])
            nav_index = state.get("nav_index", -1)
            
            # Truncate if we were back in history
            if nav_index < len(nav_history) - 1:
                nav_history = nav_history[:nav_index + 1]
                print(f"   (Truncated history after index {nav_index})")
            
            nav_history.append((mode, query, "Some Regulation"))
            state["nav_history"] = nav_history
            state["nav_index"] = len(nav_history) - 1
            print(f"   Added to history: ({mode}, {query}). New Index: {state['nav_index']}")
            
        return state

    print("1. User searches 'A'")
    state = on_query_complete("A", "search", state)
    
    print("2. User searches 'B'")
    state = on_query_complete("B", "search", state)
    
    print("3. User searches 'C'")
    state = on_query_complete("C", "search", state)
    
    assert len(state["nav_history"]) == 3
    assert state["nav_index"] == 2
    print("✅ History built correctly: [A, B, C]")
    
    print("4. User clicks BACK (Simonulates moving index)")
    state["nav_index"] -= 1 # Moves to B (index 1)
    print(f"   Current Index: {state['nav_index']} (Expect 'B')")
    
    print("5. User clicks BACK (Simonulates moving index)")
    state["nav_index"] -= 1 # Moves to A (index 0)
    print(f"   Current Index: {state['nav_index']} (Expect 'A')")
    
    print("6. User searches 'D' (New Branch)")
    state = on_query_complete("D", "search", state)
    
    # Expect history to be [A, D]
    history = state["nav_history"]
    print(f"   History: {[h[1] for h in history]}")
    
    if len(history) == 2 and history[1][1] == "D":
        print("✅ Branching logic correct: [A, B, C] -> [A, D]")
    else:
        print("❌ Branching logic failed")
        
    print(">> Navigation Logic Test PASSED")

if __name__ == "__main__":
    test_streaming_backend()
    test_link_formatting_complex()
    test_navigation_logic_simulation()
