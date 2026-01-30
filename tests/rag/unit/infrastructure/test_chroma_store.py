from unittest.mock import MagicMock

import pytest

from src.rag.domain.entities import RegulationStatus
from src.rag.domain.value_objects import Query, SearchFilter
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


def test_search_coerces_non_string_query_text():
    class DummyCollection:
        def __init__(self):
            self.last_query_texts = None

        def query(self, query_texts, n_results, where, include):
            self.last_query_texts = query_texts
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    class DummyQuery:
        def __init__(self, text):
            self.text = text
            self.include_abolished = False

    def dummy_embedding(texts):
        return [[0.0] * 768 for _ in texts]

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._embedding_function = dummy_embedding
    store._collection = DummyCollection()

    query = DummyQuery(["휴학", "관련", "규정"])
    results = store.search(query, None, 5)

    assert results == []
    assert store._collection.last_query_texts == ["휴학 관련 규정"]


# Additional tests for edge cases and error handling (missing lines coverage)


def test_init_without_chromadb():
    """Test that init raises ImportError when chromadb is not available."""
    from unittest.mock import patch

    # Create a mock module that simulates chromadb not being available
    with patch("src.rag.infrastructure.chroma_store.CHROMADB_AVAILABLE", False):
        # Need to bypass the early return in __init__
        with pytest.raises(ImportError, match="chromadb is required"):
            ChromaVectorStore.__new__(ChromaVectorStore).__init__(
                ChromaVectorStore.__new__(ChromaVectorStore)
            )


def test_distance_to_score_negative():
    """Negative distance results in score > 1.0 which gets clamped to 1.0."""
    # 1 - (-0.5) = 1.5, clamped to 1.0
    assert ChromaVectorStore._distance_to_score(-0.5) == 1.0


def test_distance_to_score_none_input():
    assert ChromaVectorStore._distance_to_score(None) == 0.0


def test_distance_to_score_above_one():
    assert ChromaVectorStore._distance_to_score(1.5) == 0.0


def test_add_chunks_empty_list():
    """Test that add_chunks returns 0 for empty list."""
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = MagicMock()

    result = store.add_chunks([])
    assert result == 0


def test_add_chunks_deduplicates_by_id():
    """Test that add_chunks removes duplicates by ID."""

    class DummyCollection:
        def __init__(self):
            self.added_ids = []

        def add(self, ids, documents, metadatas):
            self.added_ids = ids

    from src.rag.domain.entities import Chunk, ChunkLevel

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._embedding_function = object()
    store._collection = DummyCollection()

    chunk1 = Chunk(
        id="id1",
        rule_code="1-1-1",
        level=ChunkLevel.TEXT,
        title="T1",
        text="text1",
        embedding_text="text1",
        full_text="text1",
        parent_path=[],
        token_count=1,
        keywords=[],
        is_searchable=True,
    )
    chunk2 = Chunk(
        id="id1",  # Duplicate ID
        rule_code="1-1-2",
        level=ChunkLevel.TEXT,
        title="T2",
        text="text2",
        embedding_text="text2",
        full_text="text2",
        parent_path=[],
        token_count=1,
        keywords=[],
        is_searchable=True,
    )
    chunk3 = Chunk(
        id="id3",
        rule_code="1-1-3",
        level=ChunkLevel.TEXT,
        title="T3",
        text="text3",
        embedding_text="text3",
        full_text="text3",
        parent_path=[],
        token_count=1,
        keywords=[],
        is_searchable=True,
    )

    count = store.add_chunks([chunk1, chunk2, chunk3])

    assert count == 2  # Only id1 and id3
    assert len(store._collection.added_ids) == 2
    assert "id1" in store._collection.added_ids
    assert "id3" in store._collection.added_ids


def test_add_chunks_batch_processing():
    """Test that add_chunks processes in batches."""

    class DummyCollection:
        def __init__(self):
            self.add_calls = []

        def add(self, ids, documents, metadatas):
            self.add_calls.append(len(ids))

    from src.rag.domain.entities import Chunk, ChunkLevel

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._embedding_function = object()
    store._collection = DummyCollection()

    # Create a smaller batch for testing (BATCH_SIZE is 5000)
    chunks = [
        Chunk(
            id=f"id{i}",
            rule_code="1-1-1",
            level=ChunkLevel.TEXT,
            title=f"T{i}",
            text=f"text{i}",
            embedding_text=f"text{i}",
            full_text=f"text{i}",
            parent_path=[],
            token_count=1,
            keywords=[],
            is_searchable=True,
        )
        for i in range(100)  # Smaller batch for faster test
    ]

    count = store.add_chunks(chunks)

    assert count == 100
    # With 100 chunks and BATCH_SIZE=5000, should be called once
    assert len(store._collection.add_calls) == 1
    assert sum(store._collection.add_calls) == 100


def test_delete_by_rule_codes_empty_list():
    """Test that delete_by_rule_codes returns 0 for empty list."""
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = MagicMock()

    result = store.delete_by_rule_codes([])
    assert result == 0
    store._collection.get.assert_not_called()


def test_delete_by_rule_codes_no_matches():
    """Test delete when no chunks match the rule codes."""

    class DummyCollection:
        def get(self, where):
            return {"ids": []}

        def delete(self, ids):
            pass

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = DummyCollection()

    result = store.delete_by_rule_codes(["1-1-1", "1-1-2"])
    assert result == 0


def test_delete_by_rule_codes_with_matches():
    """Test delete with matching chunks."""

    class DummyCollection:
        def get(self, where):
            return {"ids": ["id1", "id2", "id3"]}

        def delete(self, ids):
            pass

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = DummyCollection()

    result = store.delete_by_rule_codes(["1-1-1"])
    assert result == 3


def test_search_empty_query_text():
    """Test that search returns empty results for empty query."""

    class DummyCollection:
        def query(self, query_texts, n_results, where, include):
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    class DummyQuery:
        text = "   "  # Whitespace only
        include_abolished = False

    def dummy_embedding(texts):
        return [[0.0] * 768 for _ in texts]

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._embedding_function = dummy_embedding
    store._collection = DummyCollection()

    results = store.search(DummyQuery(), None, 5)
    assert results == []


def test_search_with_valid_results():
    """Test search converts results to SearchResult correctly."""

    class DummyCollection:
        def query(self, query_texts, n_results, where, include):
            return {
                "ids": [["id1", "id2"]],
                "documents": [["doc1", "doc2"]],
                "metadatas": [[{"rule_code": "1-1-1"}, {"rule_code": "1-1-2"}]],
                "distances": [[0.3, 0.5]],
            }

    class DummyQuery:
        text = "test query"
        include_abolished = False

    def dummy_embedding(texts):
        return [[0.0] * 768 for _ in texts]

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._embedding_function = dummy_embedding
    store._collection = DummyCollection()

    results = store.search(DummyQuery(), None, 5)

    assert len(results) == 2
    assert results[0].chunk.id == "id1"
    assert results[0].score == 0.7  # 1 - 0.3
    assert results[1].score == 0.5  # 1 - 0.5


def test_get_all_rule_codes():
    """Test get_all_rule_codes extracts unique rule codes."""

    class DummyCollection:
        def get(self, include):
            return {
                "metadatas": [
                    {"rule_code": "1-1-1"},
                    {"rule_code": "1-1-2"},
                    {"rule_code": "1-1-1"},  # Duplicate
                    None,  # Missing metadata
                    {},  # Empty metadata
                ]
            }

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = DummyCollection()

    codes = store.get_all_rule_codes()

    assert codes == {"1-1-1", "1-1-2"}
    assert len(codes) == 2


def test_count():
    """Test count returns collection count."""

    class DummyCollection:
        def count(self):
            return 12345

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = DummyCollection()

    assert store.count() == 12345


def test_get_all_documents():
    """Test get_all_documents returns document tuples."""

    class DummyCollection:
        def get(self, include):
            return {
                "ids": ["id1", "id2", "id3"],
                "documents": ["text1", "", "text3"],  # One empty
                "metadatas": [{"meta": 1}, {"meta": 2}, {"meta": 3}],
            }

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._collection = DummyCollection()

    docs = store.get_all_documents()

    assert len(docs) == 2  # Empty document skipped
    assert docs[0] == ("id1", "text1", {"meta": 1})
    assert docs[1] == ("id3", "text3", {"meta": 3})


def test_clear_all():
    """Test clear_all deletes and recreates collection."""

    class DummyCollection:
        def count(self):
            return 100

    class DummyClient:
        def __init__(self):
            self.deleted = None
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
    store.collection_name = "test_collection"
    store._embedding_function = object()
    store._client = DummyClient()
    store._collection = DummyCollection()

    count = store.clear_all()

    assert count == 100
    assert store._client.deleted == "test_collection"
    assert store._client.created["name"] == "test_collection"
    assert store._client.created["embedding_function"] is store._embedding_function


def test_metadata_to_chunk():
    """Test _metadata_to_chunk creates Chunk from metadata."""

    store = ChromaVectorStore.__new__(ChromaVectorStore)

    chunk = store._metadata_to_chunk(
        "test_id",
        "test document",
        {
            "rule_code": "1-1-1",
            "title": "Test Title",
            "level": "regulation",
        },
    )

    assert chunk.id == "test_id"
    assert chunk.embedding_text == "test document"
    assert chunk.rule_code == "1-1-1"
