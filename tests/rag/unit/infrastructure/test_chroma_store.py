import pytest

from src.rag.domain.value_objects import Query, SearchFilter
from src.rag.domain.entities import RegulationStatus
from src.rag.infrastructure.chroma_store import ChromaVectorStore


def test_build_where_defaults_to_active():
    query = Query(text="규정", include_abolished=False)
    where = ChromaVectorStore._build_where(query, None)
    assert where == {"status": "active"}


def test_build_where_no_filter_include_abolished():
    query = Query(text="규정", include_abolished=True)
    where = ChromaVectorStore._build_where(query, None)
    assert where is None


def test_build_where_respects_explicit_status_filter():
    query = Query(text="규정", include_abolished=False)
    filter = SearchFilter(status=RegulationStatus.ABOLISHED)
    where = ChromaVectorStore._build_where(query, filter)
    assert where == {"status": "abolished"}


def test_build_where_adds_active_when_status_missing():
    query = Query(text="규정", include_abolished=False)
    filter = SearchFilter(rule_codes=["A-1-1"])
    where = ChromaVectorStore._build_where(query, filter)
    assert where and "$and" in where
    clauses = where["$and"]
    assert {"rule_code": {"$in": ["A-1-1"]}} in clauses
    assert {"status": "active"} in clauses


def test_distance_to_score_clamps():
    assert ChromaVectorStore._distance_to_score(0.3) == pytest.approx(0.7)
    assert ChromaVectorStore._distance_to_score(1.2) == 0.0


def test_clear_all_reuses_embedding_function():
    class DummyCollection:
        def count(self):
            return 0

    class DummyClient:
        def __init__(self):
            self.created = None

        def delete_collection(self, name):
            self.deleted = name

        def create_collection(self, name, metadata=None, embedding_function=None):
            self.created = {
                "name": name,
                "metadata": metadata,
                "embedding_function": embedding_function,
            }
            return DummyCollection()

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection_name = "regulations"
    store._embedding_function = object()
    store._client = DummyClient()
    store._collection = DummyCollection()

    store.clear_all()

    assert store._client.created["embedding_function"] is store._embedding_function
