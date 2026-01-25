import gc
import sys
import weakref
from pathlib import Path
from typing import Any

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
