import gc
import json
import sys
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Track created vector stores for cleanup
_vector_store_refs: list[weakref.ref] = []


def _cleanup_bm25_cache():
    """Clean up BM25 index cache file to prevent test isolation issues."""
    cache_path = ROOT / "data" / "cache" / "bm25_index.pkl"
    if cache_path.exists():
        cache_path.unlink()


def _cleanup_vector_stores():
    """Clean up any vector stores that weren't properly closed."""
    global _vector_store_refs

    # Clean up dead references
    _vector_store_refs = [ref for ref in _vector_store_refs if ref() is not None]

    # Close all remaining vector stores
    for ref in _vector_store_refs:
        store = ref()
        if store is not None and hasattr(store, "close"):
            try:
                store.close()
            except Exception:
                pass  # Silently fail during cleanup

    _vector_store_refs.clear()


def _register_vector_store(store: Any) -> None:
    """Register a vector store for cleanup."""
    _vector_store_refs.append(weakref.ref(store))


@pytest.fixture(autouse=True)
def reset_rag_config():
    """Reset RAG config before and after each test."""
    from src.rag.config import reset_config

    reset_config()
    _cleanup_bm25_cache()
    _cleanup_vector_stores()
    gc.collect()  # Force garbage collection to prevent memory leaks
    yield
    reset_config()
    _cleanup_bm25_cache()
    _cleanup_vector_stores()
    gc.collect()  # Force garbage collection after test


@pytest.fixture
def vector_store():
    """
    Create a test vector store with automatic cleanup.

    Uses context manager pattern to ensure resources are released.
    """
    # Use temporary directory for test isolation
    import tempfile
    import uuid

    from src.rag.infrastructure.chroma_store import ChromaVectorStore

    temp_dir = tempfile.mkdtemp(prefix=f"chroma_test_{uuid.uuid4().hex[:8]}_")
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"

    store = ChromaVectorStore(
        persist_directory=temp_dir,
        collection_name=collection_name,
    )

    _register_vector_store(store)

    yield store

    # Cleanup
    store.close()
    _cleanup_vector_stores()


@pytest.fixture
def clean_mock():
    """
    Fixture that provides a mock with automatic cleanup.

    Helps prevent mock object memory leaks by ensuring patches are undone.
    """
    from unittest.mock import patch

    class CleanMock:
        def __init__(self):
            self._patches = []

        def patch(self, target, **kwargs):
            """Create a patch that will be automatically cleaned up."""
            p = patch(target, **kwargs)
            self._patches.append(p)
            return p

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Stop all patches
            for p in self._patches:
                try:
                    p.stop()
                except Exception:
                    pass
            self._patches.clear()
            return False

    mock = CleanMock()
    yield mock
    # Cleanup on fixture teardown
    for p in mock._patches:
        try:
            p.stop()
        except Exception:
            pass


def pytest_collection_modifyitems(config, items):
    debug_root = ROOT / "tests" / "debug"
    for item in items:
        try:
            path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        if debug_root in path.parents:
            item.add_marker(pytest.mark.debug)


# =============================================================================
# Memory Management Hooks
# =============================================================================


def pytest_configure(config):
    """Configure pytest with memory-saving settings."""
    # Set garbage collection thresholds more aggressively for testing
    # Default is (700, 10, 10) - we use (100, 5, 5) for more frequent GC
    gc.set_threshold(100, 5, 5)

    # Enable auto-gc for each test
    config.option.usepdb = False  # Disable pdb to prevent hanging


def pytest_sessionfinish(session, exitstatus):
    """
    Run after the entire test session finishes.

    Performs final cleanup and reports memory usage.
    """
    # Final garbage collection
    gc.collect()

    # Clean up any remaining vector stores
    _cleanup_vector_stores()

    # Clean up BM25 cache
    _cleanup_bm25_cache()

    # Report if there are remaining references (potential leaks)
    if _vector_store_refs:
        print(
            f"\nWarning: {_vector_store_refs} vector store references remain after session",
            file=sys.stderr,
        )

    # Memory usage report
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        print(
            f"\nMemory usage: {mem_info.rss / (1024**3):.2f}GB RSS, "
            f"{mem_info.vms / (1024**3):.2f}GB VMS",
            file=sys.stderr,
        )

        # Warn if memory usage is excessive (>8GB)
        if mem_info.rss > 8 * 1024**3:
            print(
                "WARNING: High memory usage detected! Consider running tests in smaller batches.",
                file=sys.stderr,
            )
    except Exception:
        pass


def pytest_runtest_teardown(item, nextitem):
    """
    Hook called after each test execution.

    Performs per-test memory monitoring and cleanup.
    """
    # Force garbage collection after each test
    gc.collect()

    # Check memory usage every 10 tests
    if hasattr(item, "execution_count"):
        item.execution_count += 1
    else:
        item.execution_count = 1

    if item.execution_count % 10 == 0:
        try:
            import psutil

            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024**3)

            # Warn if memory usage exceeds 4GB
            if mem_gb > 4:
                print(
                    f"\n[WARNING] High memory usage after {item.execution_count} tests: {mem_gb:.2f}GB",
                    file=sys.stderr,
                )

            # Abort if memory usage exceeds 12GB to prevent system freeze
            if mem_gb > 12:
                pytest.fail(
                    f"Memory usage exceeded safety limit ({mem_gb:.2f}GB > 12GB). "
                    "Run tests in smaller batches using specific test paths."
                )
        except Exception:
            pass


# =============================================================================
# Coverage exclusion hooks (when not using --no-cov)
# =============================================================================


def pytest_report_header(config):
    """Add custom header to pytest output."""
    parts = []

    # Memory info if available
    try:
        import psutil

        mem = psutil.virtual_memory()
        parts.append(f"Memory: {mem.available / (1024**3):.1f}GB available")
    except Exception:
        pass

    # GC settings
    parts.append(f"GC thresholds: {gc.get_threshold()}")

    if parts:
        return "\n".join(parts)
    return None


# =============================================================================
# Mock Fixtures for SPEC-TEST-COV-001
# =============================================================================


@pytest.fixture
def llm_client_mock():
    """
    Mock LLM client for deterministic testing.

    Supports:
    - generate() method with configurable responses
    - Error simulation via side_effect
    - No actual network calls

    Usage:
        mock = llm_client_mock()
        mock.generate.return_value = '{"normalized": "query", "keywords": ["kw1"]}'
        # Or simulate error:
        mock.generate.side_effect = Exception("LLM error")
    """

    class MockLLMClient:
        """Synchronous mock LLM client for testing."""

        def __init__(self):
            self._default_response = json.dumps(
                {"normalized": "normalized query", "keywords": ["keyword1", "keyword2"]}
            )
            self._response: Optional[str] = None
            self._side_effect: Optional[Exception] = None
            self._call_count = 0
            self._call_history: List[Dict[str, Any]] = []

        def generate(
            self,
            system_prompt: str,
            user_message: str,
            temperature: float = 0.0,
        ) -> str:
            """Generate a response - mock implementation."""
            self._call_count += 1
            self._call_history.append(
                {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "temperature": temperature,
                }
            )

            if self._side_effect:
                raise self._side_effect

            return self._response or self._default_response

        def set_response(self, response: str) -> None:
            """Set the response to return."""
            self._response = response

        def set_json_response(self, data: Dict[str, Any]) -> None:
            """Set response as JSON string."""
            self._response = json.dumps(data, ensure_ascii=False)

        def set_error(self, error: Exception) -> None:
            """Set an error to raise on generate()."""
            self._side_effect = error

        def reset(self) -> None:
            """Reset mock state."""
            self._response = None
            self._side_effect = None
            self._call_count = 0
            self._call_history = []

        @property
        def call_count(self) -> int:
            """Number of times generate() was called."""
            return self._call_count

        @property
        def last_call(self) -> Optional[Dict[str, Any]]:
            """Get the last call arguments."""
            return self._call_history[-1] if self._call_history else None

    return MockLLMClient()


@pytest.fixture
def chroma_collection_mock():
    """
    Mock ChromaDB collection for deterministic testing.

    Supports:
    - query() method with configurable results
    - add() method for document insertion
    - delete() method for document deletion
    - Edge cases: empty results, large result sets

    Usage:
        mock = chroma_collection_mock()
        mock.set_query_results({
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]]
        })
    """

    class MockChromaCollection:
        """Mock ChromaDB collection for testing."""

        def __init__(self):
            self._query_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            self._documents: Dict[str, str] = {}
            self._metadatas: Dict[str, Dict] = {}
            self._call_count = {"query": 0, "add": 0, "delete": 0}

        def query(
            self,
            query_texts: Optional[List[str]] = None,
            query_embeddings: Optional[List[List[float]]] = None,
            n_results: int = 10,
            where: Optional[Dict] = None,
            **kwargs,
        ) -> Dict[str, Any]:
            """Mock query method."""
            self._call_count["query"] += 1
            # Return limited results based on n_results
            results = {
                "ids": [self._query_results["ids"][0][:n_results]],
                "documents": [self._query_results["documents"][0][:n_results]],
                "metadatas": [self._query_results["metadatas"][0][:n_results]],
                "distances": [self._query_results["distances"][0][:n_results]],
            }
            return results

        def add(
            self,
            ids: List[str],
            documents: List[str],
            metadatas: Optional[List[Dict]] = None,
            embeddings: Optional[List[List[float]]] = None,
            **kwargs,
        ) -> None:
            """Mock add method."""
            self._call_count["add"] += 1
            for i, doc_id in enumerate(ids):
                self._documents[doc_id] = documents[i]
                if metadatas:
                    self._metadatas[doc_id] = metadatas[i]

        def delete(
            self,
            ids: Optional[List[str]] = None,
            where: Optional[Dict] = None,
            **kwargs,
        ) -> None:
            """Mock delete method."""
            self._call_count["delete"] += 1
            if ids:
                for doc_id in ids:
                    self._documents.pop(doc_id, None)
                    self._metadatas.pop(doc_id, None)

        def set_query_results(self, results: Dict[str, Any]) -> None:
            """Set the results to return from query()."""
            self._query_results = results

        def set_empty_results(self) -> None:
            """Set empty query results."""
            self._query_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        def set_large_results(self, count: int = 100) -> None:
            """Set large result set for performance testing."""
            self._query_results = {
                "ids": [[f"id_{i}" for i in range(count)]],
                "documents": [[f"document {i}" for i in range(count)]],
                "metadatas": [[{"index": i} for i in range(count)]],
                "distances": [[0.1 * i for i in range(count)]],
            }

        @property
        def call_count(self) -> Dict[str, int]:
            """Get call counts for each method."""
            return self._call_count

        def reset(self) -> None:
            """Reset mock state."""
            self._query_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            self._documents.clear()
            self._metadatas.clear()
            self._call_count = {"query": 0, "add": 0, "delete": 0}

    return MockChromaCollection()


@pytest.fixture
def embedding_function_mock():
    """
    Mock embedding function for deterministic testing.

    Supports:
    - Single and batch embedding calls
    - Configurable dimension (default 1024)
    - Deterministic vectors based on input text

    Usage:
        mock = embedding_function_mock()
        mock.set_dimension(768)
        vectors = mock(["query1", "query2"])
    """

    class MockEmbeddingFunction:
        """Mock embedding function for testing."""

        def __init__(self, dimension: int = 1024):
            self._dimension = dimension
            self._call_count = 0
            self._call_history: List[List[str]] = []

        def __call__(self, texts: List[str]) -> List[List[float]]:
            """Generate deterministic embeddings."""
            self._call_count += 1
            self._call_history.append(texts)
            return [self._generate_vector(text) for text in texts]

        def _generate_vector(self, text: str) -> List[float]:
            """Generate a deterministic vector based on text hash."""
            # Use hash of text to generate deterministic values
            text_hash = hash(text)
            base_value = (text_hash % 1000) / 10000.0
            return [base_value + (i * 0.0001) for i in range(self._dimension)]

        def set_dimension(self, dimension: int) -> None:
            """Set the embedding dimension."""
            self._dimension = dimension

        @property
        def dimension(self) -> int:
            """Get the current dimension."""
            return self._dimension

        @property
        def call_count(self) -> int:
            """Get number of times called."""
            return self._call_count

        @property
        def last_texts(self) -> Optional[List[str]]:
            """Get the last texts embedded."""
            return self._call_history[-1] if self._call_history else None

        def reset(self) -> None:
            """Reset mock state."""
            self._call_count = 0
            self._call_history = []

    return MockEmbeddingFunction()


@pytest.fixture
def typo_corrector_mock():
    """
    Mock TypoCorrector for deterministic testing.

    Supports:
    - Configurable correction pairs
    - "No correction" scenario
    - "Multiple corrections" scenario
    - Method tracking (rule, symspell, edit_distance, llm)

    Usage:
        mock = typo_corrector_mock()
        mock.add_correction("시퍼", "싶어")
        mock.set_method("rule")
        result = mock.correct("받고시퍼")
    """

    class MockTypoCorrector:
        """Mock TypoCorrector for testing."""

        def __init__(self):
            self._corrections: Dict[str, str] = {}
            self._method = "rule"
            self._confidence = 0.95
            self._call_count = 0
            self._call_history: List[str] = []
            self._regulation_names: List[str] = []

        def correct(
            self, query: str, use_llm_fallback: bool = True
        ) -> Any:  # Returns TypoCorrectionResult
            """Mock correct method."""
            from dataclasses import dataclass
            from typing import List, Tuple

            @dataclass(frozen=True)
            class TypoCorrectionResult:
                original: str
                corrected: str
                method: str
                corrections: List[Tuple[str, str]]
                confidence: float

            self._call_count += 1
            self._call_history.append(query)

            corrected = query
            corrections = []

            # Apply configured corrections
            for original, replacement in self._corrections.items():
                if original in corrected:
                    corrected = corrected.replace(original, replacement)
                    corrections.append((original, replacement))

            return TypoCorrectionResult(
                original=query,
                corrected=corrected,
                method=self._method if corrections else "none",
                corrections=corrections,
                confidence=self._confidence if corrections else 1.0,
            )

        def add_correction(self, original: str, corrected: str) -> None:
            """Add a correction pair."""
            self._corrections[original] = corrected

        def set_corrections(self, corrections: Dict[str, str]) -> None:
            """Set all correction pairs."""
            self._corrections = corrections

        def set_method(self, method: str) -> None:
            """Set the method to report (rule, symspell, edit_distance, llm, none)."""
            self._method = method

        def set_confidence(self, confidence: float) -> None:
            """Set the confidence level."""
            self._confidence = confidence

        def set_regulation_names(self, regulation_names: List[str]) -> None:
            """Set regulation names for edit distance matching."""
            self._regulation_names = regulation_names

        def clear_cache(self) -> None:
            """Mock clear cache method."""
            pass

        @property
        def call_count(self) -> int:
            """Get number of times correct() was called."""
            return self._call_count

        @property
        def last_query(self) -> Optional[str]:
            """Get the last query corrected."""
            return self._call_history[-1] if self._call_history else None

        def reset(self) -> None:
            """Reset mock state."""
            self._corrections.clear()
            self._method = "rule"
            self._confidence = 0.95
            self._call_count = 0
            self._call_history = []

    return MockTypoCorrector()
