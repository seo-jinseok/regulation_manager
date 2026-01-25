"""
Focused tests for self_rag module to improve coverage from 68% toward 85%.
"""

from unittest.mock import MagicMock

from src.rag.infrastructure.self_rag import SelfRAGEvaluator, SelfRAGPipeline


class FakeChunk:
    def __init__(self, id, text, title="", rule_code=""):
        self.id = id
        self.text = text
        self.title = title
        self.rule_code = rule_code
        self.keywords = []


class FakeSearchResult:
    def __init__(self, chunk, score, rank=1):
        self.chunk = chunk
        self.score = score
        self.rank = rank


def make_result(id, text, score):
    return FakeSearchResult(FakeChunk(id, text), score)


# Test set_llm_client (line 87)
def test_evaluator_set_llm_client():
    evaluator = SelfRAGEvaluator(llm_client=None)
    mock_llm = MagicMock()
    evaluator.set_llm_client(mock_llm)
    assert evaluator._llm_client is mock_llm


# Test short query returns True (line 108)
def test_needs_retrieval_short_query():
    evaluator = SelfRAGEvaluator(llm_client=None)
    result = evaluator.needs_retrieval("안녕")
    assert result is True


# Test LLM returning [RETRIEVE_NO] (lines 120-123)
def test_needs_retrieval_llm_says_no():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[RETRIEVE_NO]"
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    result = evaluator.needs_retrieval("query")
    assert result is False


# Test LLM exception defaults to True (line 122-123)
def test_needs_retrieval_llm_exception():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("error")
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    result = evaluator.needs_retrieval("query")
    assert result is True


# Test evaluate_relevance with empty results (lines 153)
def test_evaluate_relevance_empty():
    evaluator = SelfRAGEvaluator(llm_client=None)
    is_relevant, filtered = evaluator.evaluate_relevance("query", [])
    assert is_relevant is False
    assert filtered == []


# Test evaluate_relevance with IRRELEVANT (lines 168-169)
def test_evaluate_relevance_irrelevant():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[IRRELEVANT]"
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    results = [make_result("doc1", "content", 0.8)]
    is_relevant, filtered = evaluator.evaluate_relevance("query", results)
    assert is_relevant is False
    assert filtered == []


# Test evaluate_relevance exception (line 168-169)
def test_evaluate_relevance_exception():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("error")
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    results = [make_result("doc1", "content", 0.8)]
    is_relevant, filtered = evaluator.evaluate_relevance("query", results)
    assert is_relevant is True  # Default on error


# Test evaluate_support without LLM (line 184)
def test_evaluate_support_no_llm():
    evaluator = SelfRAGEvaluator(llm_client=None)
    result = evaluator.evaluate_support("query", "context", "answer")
    assert result == "SUPPORTED"


# Test evaluate_support PARTIALLY_SUPPORTED (line 201)
def test_evaluate_support_partial():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[PARTIALLY_SUPPORTED]"
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    result = evaluator.evaluate_support("query", "context", "answer")
    assert result == "PARTIALLY_SUPPORTED"


# Test evaluate_support NOT_SUPPORTED (line 204-205)
def test_evaluate_support_not_supported():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[NOT_SUPPORTED]"
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    result = evaluator.evaluate_support("query", "context", "answer")
    assert result == "NOT_SUPPORTED"


# Test evaluate_support exception (line 204-205)
def test_evaluate_support_exception():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("error")
    evaluator = SelfRAGEvaluator(llm_client=mock_llm)
    result = evaluator.evaluate_support("query", "context", "answer")
    assert result == "SUPPORTED"  # Default on error


# Test SelfRAGPipeline set_llm_client (lines 252-253)
def test_pipeline_set_llm_client():
    pipeline = SelfRAGPipeline(llm_client=None)
    mock_llm = MagicMock()
    pipeline.set_llm_client(mock_llm)
    assert pipeline._llm_client is mock_llm
    assert pipeline._evaluator._llm_client is mock_llm


# Test filter_relevant_results (lines 267-268)
def test_filter_relevant_results_with_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[RELEVANT]"
    pipeline = SelfRAGPipeline(llm_client=mock_llm, enable_relevance_check=True)
    results = [make_result("doc1", "content", 0.8)]
    filtered = pipeline.filter_relevant_results("query", results)
    assert len(filtered) == 1


# Test get_support_level (lines 272-274)
def test_get_support_level_disabled():
    pipeline = SelfRAGPipeline(enable_support_check=False)
    result = pipeline.get_support_level("query", "context", "answer")
    assert result == "SUPPORTED"


def test_get_support_level_enabled():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[SUPPORTED]"
    pipeline = SelfRAGPipeline(llm_client=mock_llm, enable_support_check=True)
    result = pipeline.get_support_level("query", "context", "answer")
    assert result == "SUPPORTED"


# Test start_async_support_check (lines 294, 297)
def test_start_async_disabled():
    pipeline = SelfRAGPipeline(enable_support_check=False)
    future = pipeline.start_async_support_check("query", "context", "answer")
    assert future is None


def test_start_async_sync_mode():
    pipeline = SelfRAGPipeline(enable_support_check=True, async_support_check=False)
    future = pipeline.start_async_support_check("query", "context", "answer")
    assert future is None


# Test get_async_support_result (line 306, 322-334)
def test_get_async_none():
    pipeline = SelfRAGPipeline()
    result = pipeline.get_async_support_result()
    assert result is None


def test_get_async_timeout():
    mock_llm = MagicMock()
    import time

    mock_llm.generate.side_effect = lambda **kw: time.sleep(10)
    pipeline = SelfRAGPipeline(
        llm_client=mock_llm, enable_support_check=True, async_support_check=True
    )
    pipeline.start_async_support_check("query", "context", "answer")
    result = pipeline.get_async_support_result(timeout=0.1)
    assert result is None


# Test evaluate_results_batch (lines 346, 361)
def test_evaluate_results_batch_empty():
    pipeline = SelfRAGPipeline()
    is_relevant, filtered, confidence = pipeline.evaluate_results_batch("query", [])
    assert is_relevant is False
    assert filtered == []
    assert confidence == 0.0


def test_evaluate_results_batch_high_score():
    mock_llm = MagicMock()
    pipeline = SelfRAGPipeline(llm_client=mock_llm, enable_relevance_check=True)
    results = [make_result("doc1", "content", 0.9)]
    is_relevant, filtered, confidence = pipeline.evaluate_results_batch(
        "query", results
    )
    assert is_relevant is True
    assert confidence == 0.9
