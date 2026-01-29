#!/usr/bin/env python3
"""
Quick test script to verify RAG system functionality before running full RAGAS evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.rag.config import get_config
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.interface.query_handler import QueryHandler, QueryOptions

print("🧪 Quick RAG System Test")
print("=" * 80)

# Initialize RAG components
print("🔧 Initializing RAG system...")
config = get_config()
store = ChromaVectorStore(persist_directory="data/chroma_db")
llm_client = LLMClientAdapter(
    provider=config.llm_provider,
    model=config.llm_model,
    base_url=config.llm_base_url,
)
query_handler = QueryHandler(
    store=store,
    llm_client=llm_client,
    use_reranker=True,
)
print("✅ RAG system initialized\n")

# Test query
test_query = "휴학 절차가 어떻게 되나요?"
print(f"📝 Test Query: {test_query}\n")

try:
    # Process query
    options = QueryOptions(
        top_k=5,
        use_rerank=True,
        force_mode="ask",
    )

    result = query_handler.process_query(
        query=test_query,
        options=options,
    )

    print("✅ Query processed successfully")
    print(f"Answer preview: {result.content[:200]}...")
    print(f"Success: {result.success}")

    # Extract contexts
    if result.data:
        print(f"\n📊 Data keys: {list(result.data.keys())}")

        if "tool_results" in result.data:
            print(f"Tool results found: {len(result.data['tool_results'])}")
            for tr in result.data["tool_results"][:2]:
                print(f"  - {tr.get('tool_name')}: {list(tr.get('result', {}).keys())}")

    print("\n✅ RAG system is working correctly!")
    print("Ready to run RAGAS evaluation.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
