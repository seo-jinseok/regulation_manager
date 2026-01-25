import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cleanup_bm25_cache():
    """Clean up BM25 index cache file to prevent test isolation issues."""
    cache_path = ROOT / "data" / "cache" / "bm25_index.pkl"
    if cache_path.exists():
        cache_path.unlink()


@pytest.fixture(autouse=True)
def reset_rag_config():
    """Reset RAG config before and after each test."""
    from src.rag.config import reset_config

    reset_config()
    _cleanup_bm25_cache()
    yield
    reset_config()
    _cleanup_bm25_cache()


def pytest_collection_modifyitems(config, items):
    debug_root = ROOT / "tests" / "debug"
    for item in items:
        try:
            path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        if debug_root in path.parents:
            item.add_marker(pytest.mark.debug)
