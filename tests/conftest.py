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
