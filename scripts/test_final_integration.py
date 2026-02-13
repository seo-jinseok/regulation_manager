"""Final integration test with actual RAG components."""
from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import Chunk, SearchResult
from unittest.mock import Mock

print("=== Final Integration Test ===\n")

# Create mock store
mock_store = Mock()
mock_store.count.return_value = 1

# Create mock chunk
mock_chunk = Mock(spec=Chunk)
mock_chunk.id = "test-1"
mock_chunk.text = "휴학은 학기 시작 전에 신청해야 합니다."
mock_chunk.rule_code = "학칙-001"
mock_chunk.title = "학칙"
mock_chunk.parent_path = ["학칙"]

mock_result = SearchResult(chunk=mock_chunk, score=0.9, rank=1)
mock_store.search.return_value = [mock_result]

# Create mock LLM
mock_llm = Mock()
mock_llm.stream_generate.return_value = ["휴학", "신청은", "학기", "시작", "전에", "해야", "합니다."]
mock_llm.generate.return_value = "휴학 신청은 학기 시작 전에 해야 합니다."

# Create SearchUseCase
usecase = SearchUseCase(mock_store, llm_client=mock_llm)

# Test 1: Query expansion integration
print("Test 1: Query Expansion Integration")
usecase._apply_dynamic_expansion = Mock(return_value=("휴학 방법", ["휴학원", "학업", "중단"]))

events = list(usecase.ask_stream(question="휴학 방법", top_k=3))

# Verify query expansion was called
assert usecase._apply_dynamic_expansion.call_count > 0, "Query expansion should be called"
print("✅ Query expansion is called in ask_stream")

# Test 2: Citation enhancement integration
print("\nTest 2: Citation Enhancement Integration")
usecase._enhance_answer_citations = Mock(return_value="답변 [학칙 제1조]")

events = list(usecase.ask_stream(question="휴학 방법", top_k=3))

# Verify citation enhancement was called
assert usecase._enhance_answer_citations.call_count > 0, "Citation enhancement should be called"
print("✅ Citation enhancement is called in ask_stream")

# Test 3: Non-streaming ask
print("\nTest 3: Non-Streaming ask Integration")
usecase._apply_dynamic_expansion = Mock(return_value=("휴학 방법", ["휴학원"]))
usecase._enhance_answer_citations = Mock(return_value="답변 [학칙 제1조]")

answer = usecase.ask(question="휴학 방법", top_k=3)

# Verify both methods were called
assert usecase._apply_dynamic_expansion.call_count > 0, "Query expansion should be called in ask"
assert usecase._enhance_answer_citations.call_count > 0, "Citation enhancement should be called in ask"
print("✅ Both query expansion and citation enhancement are called in ask")

print("\n=== All Integration Tests Passed! ===")
print("✅ Phase 1 components are properly integrated")
print("✅ Query expansion is active")
print("✅ Citation enhancement is active")
