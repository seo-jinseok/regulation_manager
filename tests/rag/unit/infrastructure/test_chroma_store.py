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
