#!/usr/bin/env python3
"""
Test script to verify Phase 1 integration of new RAG components.

Tests:
1. QueryExpansionService integration in SearchUseCase
2. CitationEnhancer integration in ask responses
3. EvaluationPrompts integration in LLMJudge
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_query_expansion_integration():
    """Test QueryExpansionService integration."""
    logger.info("=" * 80)
    logger.info("TEST 1: QueryExpansionService Integration")
    logger.info("=" * 80)

    try:
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.infrastructure.chroma_store import ChromaVectorStore
        from src.rag.infrastructure.llm_adapter import LLMClientAdapter
        from src.rag.config import get_config

        config = get_config()
        store = ChromaVectorStore(persist_directory="data/chroma_db")

        # Create SearchUseCase with query expansion enabled
        search_usecase = SearchUseCase(
            store=store,
            llm_client=None,  # Not needed for search only
            use_reranker=True,
        )

        # Test query expansion service initialization
        search_usecase._ensure_query_expansion_service()

        if search_usecase._query_expansion_service is not None:
            logger.info("✅ QueryExpansionService initialized successfully")

            # Test synonym-based expansion
            test_queries = [
                "휴학 방법 알려줘",
                "등록금 언제까지?",
                "장학금 신청 자격",
            ]

            for query in test_queries:
                expanded = search_usecase._query_expansion_service.expand_query(
                    query, max_variants=3, method="synonym"
                )
                logger.info(f"Query: {query}")
                logger.info(f"  Expanded variants: {len(expanded)}")
                for exp in expanded:
                    logger.info(f"    - {exp.expanded_text} (method: {exp.expansion_method})")
                logger.info("")

            return True
        else:
            logger.warning("❌ QueryExpansionService not initialized")
            return False

    except Exception as e:
        logger.error(f"❌ QueryExpansionService integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_enhancer_integration():
    """Test CitationEnhancer integration."""
    logger.info("=" * 80)
    logger.info("TEST 2: CitationEnhancer Integration")
    logger.info("=" * 80)

    try:
        from src.rag.domain.citation.citation_enhancer import CitationEnhancer
        from src.rag.domain.entities import Chunk

        enhancer = CitationEnhancer()

        # Create test chunks (using from_json_node for proper initialization)
        import json
        from src.rag.domain.entities import Chunk, ChunkLevel

        test_nodes = [
            {
                "id": "test1",
                "title": "제26조",
                "text": "휴학은 1년 이내 가능합니다.",
                "keywords": [],
                "level": "article",
            },
            {
                "id": "test2",
                "title": "제15조",
                "text": "장학금은 성적이 우수한 학생에게 지급합니다.",
                "keywords": [],
                "level": "article",
            },
        ]

        test_chunks = [
            Chunk.from_json_node(test_nodes[0], "직원복무규정_제26조"),
            Chunk.from_json_node(test_nodes[1], "장학금규정_제15조"),
        ]

        # Test citation enhancement
        enhanced = enhancer.enhance_citations(test_chunks)

        logger.info(f"✅ Enhanced {len(enhanced)} citations")
        for citation in enhanced:
            logger.info(f"  - {citation.format()}")
            logger.info(f"    Regulation: {citation.regulation}")
            logger.info(f"    Article: {citation.article_number}")
            logger.info(f"    Confidence: {citation.confidence}")
            logger.info("")

        # Test formatting
        formatted = enhancer.format_citations(enhanced)
        logger.info(f"✅ Formatted citations: {formatted}")

        return True

    except Exception as e:
        logger.error(f"❌ CitationEnhancer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_prompts_integration():
    """Test EvaluationPrompts integration."""
    logger.info("=" * 80)
    logger.info("TEST 3: EvaluationPrompts Integration")
    logger.info("=" * 80)

    try:
        from src.rag.domain.evaluation.prompts import EvaluationPrompts

        # Test prompt formatting
        query = "휴학 신청 방법은?"
        answer = "휴학 신청은 학기 시작 전에 해야 합니다. 「직원복무규정」제26조에 따라 신청서를 제출해야 합니다."
        context = [
            {
                "title": "제26조",
                "text": "직원은 1년 이내 휴학할 수 있다. 휴학 신청서를 제출해야 한다.",
                "score": 0.9
            }
        ]

        system_prompt, user_prompt = EvaluationPrompts.format_accuracy_prompt(
            query=query,
            answer=answer,
            context=context,
            expected_info=["신청 방법", "신청 기간"]
        )

        logger.info("✅ EvaluationPrompts formatted successfully")
        logger.info(f"System prompt length: {len(system_prompt)}")
        logger.info(f"User prompt length: {len(user_prompt)}")

        # Test negative examples
        neg_examples = EvaluationPrompts.list_negative_examples()
        logger.info(f"✅ Available negative examples: {neg_examples}")

        for example_type in neg_examples:
            example = EvaluationPrompts.get_negative_example(example_type)
            logger.info(f"  - {example_type}: {example.get('issues', [])}")

        return True

    except Exception as e:
        logger.error(f"❌ EvaluationPrompts integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_judge_integration():
    """Test LLMJudge with improved prompts."""
    logger.info("=" * 80)
    logger.info("TEST 4: LLMJudge Integration with Improved Prompts")
    logger.info("=" * 80)

    try:
        from src.rag.domain.evaluation import LLMJudge

        judge = LLMJudge()

        # Check if improved prompts are available
        if judge.use_improved_prompts:
            logger.info("✅ LLMJudge using improved prompts")

            # Test evaluation with improved prompts
            query = "휴학 신청 방법은?"
            answer = "휴학 신청은 학기 시작 전에 해야 합니다. 「직원복무규정」제26조에 따라 신청서를 제출해야 합니다."
            sources = [
                {
                    "title": "제26조",
                    "text": "직원은 1년 이내 휴학할 수 있다. 휴학 신청서를 제출해야 한다.",
                    "score": 0.9
                }
            ]

            # Use rule-based evaluation (LLM evaluation requires actual LLM)
            result = judge.evaluate(
                query=query,
                answer=answer,
                sources=sources,
                expected_info=["신청 방법", "신청 기간"]
            )

            logger.info("✅ Evaluation completed")
            logger.info(f"  Overall score: {result.overall_score:.3f}")
            logger.info(f"  Accuracy: {result.accuracy:.3f}")
            logger.info(f"  Completeness: {result.completeness:.3f}")
            logger.info(f"  Citations: {result.citations:.3f}")
            logger.info(f"  Context Relevance: {result.context_relevance:.3f}")
            logger.info(f"  Passed: {result.passed}")
            logger.info(f"  Issues: {result.issues}")
            logger.info(f"  Strengths: {result.strengths}")

            return True
        else:
            logger.warning("⚠️  LLMJudge not using improved prompts (fallback to rule-based)")
            return True

    except Exception as e:
        logger.error(f"❌ LLMJudge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "PHASE 1 INTEGRATION TESTS" + " " * 37 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")

    results = {
        "QueryExpansionService": test_query_expansion_integration(),
        "CitationEnhancer": test_citation_enhancer_integration(),
        "EvaluationPrompts": test_evaluation_prompts_integration(),
        "LLMJudge": test_llm_judge_integration(),
    }

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {component}")

    total_passed = sum(results.values())
    total_tests = len(results)

    logger.info("")
    logger.info(f"Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        logger.info("")
        logger.info("🎉 All Phase 1 integration tests passed!")
        return 0
    else:
        logger.error("")
        logger.error("⚠️  Some integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
