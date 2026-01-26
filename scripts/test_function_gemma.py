#!/usr/bin/env python3
"""
FunctionGemma Test Script.

Tests the FunctionGemma integration with the regulation RAG system.
Requires: ollama with functiongemma model installed.

Usage:
    uv run scripts/test_function_gemma.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_ollama_available():
    """Check if Ollama is available and functiongemma is installed."""
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, 'models') else []

        print("✅ Ollama package available")
        print(f"📦 Installed models: {model_names}")

        has_functiongemma = any("functiongemma" in m.lower() for m in model_names)
        if has_functiongemma:
            print("✅ FunctionGemma model found")
        else:
            print("❌ FunctionGemma model not found")
            print("   Install with: ollama pull functiongemma")

        return has_functiongemma
    except ImportError:
        print("❌ Ollama package not installed")
        print("   Install with: uv add ollama")
        return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        print("   Make sure Ollama is running (open Ollama app)")
        return False


def test_tool_definitions():
    """Test tool definitions loading."""
    from rag.infrastructure.tool_definitions import TOOL_DEFINITIONS, get_tool_names

    print("\n📋 Testing Tool Definitions...")
    print(f"   Total tools: {len(TOOL_DEFINITIONS)}")
    print(f"   Tool names: {get_tool_names()}")
    return True


def test_tool_executor():
    """Test tool executor initialization."""
    from rag.infrastructure.tool_executor import ToolExecutor

    print("\n🔧 Testing Tool Executor...")
    executor = ToolExecutor()

    # Test a simple analysis tool without dependencies
    result = executor.execute("analyze_query", {"query": "교원 연구년 신청"})
    print(f"   analyze_query result: {result.success}")
    if result.success:
        print(f"   Query type: {result.result.get('query_type', 'unknown')}")

    return True


def test_function_gemma_adapter():
    """Test FunctionGemma adapter initialization."""
    from rag.infrastructure.function_gemma_adapter import (
        OLLAMA_AVAILABLE,
        FunctionGemmaAdapter,
    )

    print("\n🤖 Testing FunctionGemma Adapter...")
    print(f"   Ollama available: {OLLAMA_AVAILABLE}")

    adapter = FunctionGemmaAdapter(
        model="functiongemma",
        use_native_api=True,
    )

    print(f"   Use native API: {adapter._use_native_api}")
    return True


def test_full_pipeline():
    """Test full FunctionGemma query pipeline (requires Ollama)."""
    print("\n🚀 Testing Full Pipeline...")

    try:

        from rag.infrastructure.function_gemma_adapter import FunctionGemmaAdapter
        from rag.infrastructure.query_analyzer import QueryAnalyzer
        from rag.infrastructure.tool_executor import ToolExecutor

        # Initialize components
        query_analyzer = QueryAnalyzer()
        tool_executor = ToolExecutor(query_analyzer=query_analyzer)

        adapter = FunctionGemmaAdapter(
            tool_executor=tool_executor,
            model="functiongemma",
            use_native_api=True,
        )

        # Test query
        test_query = "교원 연구년 신청 자격은?"
        print(f"   Query: {test_query}")

        answer, tool_results = adapter.process_query(test_query)

        print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
        print(f"   Tools called: {[r.tool_name for r in tool_results]}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("FunctionGemma Integration Test")
    print("=" * 60)

    results = []

    # Check Ollama availability
    ollama_ok = check_ollama_available()

    # Test components
    results.append(("Tool Definitions", test_tool_definitions()))
    results.append(("Tool Executor", test_tool_executor()))
    results.append(("Adapter Init", test_function_gemma_adapter()))

    if ollama_ok:
        results.append(("Full Pipeline", test_full_pipeline()))
    else:
        print("\n⚠️ Skipping full pipeline test (Ollama/FunctionGemma not available)")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
