#!/usr/bin/env python3
"""
Professor Persona Search-Only Test
Tests document retrieval without LLM component.
"""
import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/Users/truestone/Dropbox/repo/University/regulation_manager')

try:
    from src.rag.infrastructure.hybrid_search import HybridSearcher
except Exception as e:
    print(f"Error importing HybridSearcher: {e}")
    sys.exit(1)


def test_search_only(query: str, category: str, difficulty: str):
    """Test document retrieval for a professor query."""
    start_time = time.time()

    try:
        # Initialize retriever
        retriever = HybridSearcher()

        # Retrieve relevant documents
        docs = retriever.retrieve(query, top_k=5)

        execution_time = (time.time() - start_time) * 1000

        # Extract content and metadata
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": {
                    "file_name": doc.metadata.get('file_name', 'Unknown'),
                    "regulation_title": doc.metadata.get('regulation_title', 'Unknown'),
                    "article": doc.metadata.get('article', 'Unknown'),
                    "section": doc.metadata.get('section', 'Unknown'),
                }
            })

        return {
            "persona": "professor",
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "result": {
                "documents_found": len(results),
                "documents": results,
                "execution_time_ms": round(execution_time, 2)
            },
            "evaluation": {
                "documents_found_score": min(1.0, len(results) / 5),
                "has_content": len(results) > 0
            }
        }

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        import traceback
        return {
            "persona": "professor",
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "error": str(e),
            "traceback": traceback.format_exc()[:500],
            "execution_time_ms": round(execution_time, 2)
        }


def main():
    """Run all professor queries with search only."""
    queries = [
        {
            "query": "학사위원회의 심의 권한과 절차에 대해 설명하시오",
            "category": "academic_governance",
            "difficulty": "medium"
        },
        {
            "query": "교원 인사 평가 기준과 승진 요건은 무엇인가",
            "category": "faculty_affairs",
            "difficulty": "medium"
        },
        {
            "query": "대학원 입학 전형 방법과 평가 비율은?",
            "category": "graduate_admissions",
            "difficulty": "medium"
        },
        {
            "query": "연구 윤리 규정과 연구 부정행위의 정의는?",
            "category": "research_administration",
            "difficulty": "medium"
        },
        {
            "query": "교과 개설 및 변경 절차에 대해 알려줘",
            "category": "academic_governance",
            "difficulty": "medium"
        }
    ]

    results = []

    print("=" * 80)
    print("Professor Persona RAG System Test - Search Only (No LLM)")
    print("=" * 80)
    print()

    for i, test_case in enumerate(queries, 1):
        print(f"Test {i}/{len(queries)}: {test_case['query'][:50]}...")
        result = test_search_only(
            test_case['query'],
            test_case['category'],
            test_case['difficulty']
        )
        results.append(result)

        if 'error' in result:
            print(f"  ✗ Error: {result['error'][:100]}...")
        else:
            print(f"  ✓ Found {result['result']['documents_found']} documents in {result['result']['execution_time_ms']}ms")
        print()

    # Print JSON output
    print("=" * 80)
    print("JSON Output")
    print("=" * 80)
    print()

    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
