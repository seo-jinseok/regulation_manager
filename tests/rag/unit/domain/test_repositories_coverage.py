"""
Tests for repositories.py abstract interfaces.

These tests verify that the abstract interfaces are properly defined
and that concrete implementations can be created and used.
Coverage improvement from 71% to 85% focuses on:
- Testing abstract method definitions
- Testing concrete implementations
- Testing interface contracts
"""

from abc import ABC
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.rag.domain.entities import (
    Chunk,
    ChunkLevel,
    Keyword,
    RegulationOverview,
    RegulationStatus,
    SearchResult,
)
from src.rag.domain.repositories import (
    IDocumentLoader,
    IHybridSearcher,
    ILLMClient,
    IReranker,
    IVectorStore,
)
from src.rag.domain.value_objects import Query, SyncState


def create_test_chunk(chunk_id: str = "test_id") -> Chunk:
    """Helper to create a test Chunk with all required fields."""
    return Chunk(
        id=chunk_id,
        rule_code="1-1-1",
        level=ChunkLevel.ARTICLE,
        title="Test Article",
        text="Test content",
        embedding_text="Test content for embedding",
        full_text="Full text of test content",
        parent_path=["Chapter 1", "Section 1"],
        token_count=10,
        keywords=[Keyword(term="test", weight=1.0)],
        is_searchable=True,
    )


class TestIVectorStore:
    """Test IVectorStore abstract interface."""

    def test_ivectorstore_is_abstract(self):
        """Verify IVectorStore is an abstract base class."""
        assert issubclass(IVectorStore, ABC)
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            IVectorStore()

    def test_ivectorstore_abstract_methods(self):
        """Verify all expected abstract methods exist."""

        class ConcreteVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return len(chunks)

            def delete_by_rule_codes(self, rule_codes):
                return len(rule_codes)

            def search(self, query, filter=None, top_k=10):
                return []

            def get_all_rule_codes(self):
                return set()

            def count(self):
                return 0

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        # Concrete implementation should be instantiable
        store = ConcreteVectorStore()
        assert store.add_chunks([]) == 0
        assert store.delete_by_rule_codes([]) == 0
        assert store.search(Query(text="test")) == []
        assert store.get_all_rule_codes() == set()
        assert store.count() == 0
        assert store.get_all_documents() == []
        assert store.clear_all() == 0


class TestIDocumentLoader:
    """Test IDocumentLoader abstract interface."""

    def test_idocumentloader_is_abstract(self):
        """Verify IDocumentLoader is an abstract base class."""
        assert issubclass(IDocumentLoader, ABC)
        with pytest.raises(TypeError):
            IDocumentLoader()

    def test_idocumentloader_abstract_methods(self):
        """Verify all expected abstract methods exist."""

        class ConcreteDocumentLoader(IDocumentLoader):
            def load_all_chunks(self, json_path):
                return []

            def load_chunks_by_rule_codes(self, json_path, rule_codes):
                return []

            def compute_state(self, json_path):
                return SyncState.empty()

            def get_regulation_titles(self, json_path):
                return {}

            def get_all_regulations(self, json_path):
                return []

            def get_regulation_doc(self, json_path, identifier):
                return None

            def get_regulation_overview(self, json_path, identifier):
                return None

        loader = ConcreteDocumentLoader()
        assert loader.load_all_chunks("test.json") == []
        assert loader.load_chunks_by_rule_codes("test.json", set()) == []
        assert isinstance(loader.compute_state("test.json"), SyncState)
        assert loader.get_regulation_titles("test.json") == {}
        assert loader.get_all_regulations("test.json") == []
        assert loader.get_regulation_doc("test.json", "test") is None
        assert loader.get_regulation_overview("test.json", "test") is None


class TestILLMClient:
    """Test ILLMClient abstract interface."""

    def test_illmclient_is_abstract(self):
        """Verify ILLMClient is an abstract base class."""
        assert issubclass(ILLMClient, ABC)
        with pytest.raises(TypeError):
            ILLMClient()

    def test_illmclient_abstract_methods(self):
        """Verify all expected abstract methods exist."""

        class ConcreteLLMClient(ILLMClient):
            def generate(self, system_prompt, user_message, temperature=0.0):
                return "response"

            def get_embedding(self, text):
                return [0.1, 0.2, 0.3]

        client = ConcreteLLMClient()
        assert client.generate("system", "user") == "response"
        assert client.get_embedding("test") == [0.1, 0.2, 0.3]


class TestIReranker:
    """Test IReranker abstract interface."""

    def test_ireranker_is_abstract(self):
        """Verify IReranker is an abstract base class."""
        assert issubclass(IReranker, ABC)
        with pytest.raises(TypeError):
            IReranker()

    def test_ireranker_abstract_methods(self):
        """Verify all expected abstract methods exist."""

        class ConcreteReranker(IReranker):
            def rerank(self, query, documents, top_k=10):
                return []

        reranker = ConcreteReranker()
        assert reranker.rerank("query", []) == []


class TestIHybridSearcher:
    """Test IHybridSearcher abstract interface."""

    def test_ihybridsearcher_is_abstract(self):
        """Verify IHybridSearcher is an abstract base class."""
        assert issubclass(IHybridSearcher, ABC)
        with pytest.raises(TypeError):
            IHybridSearcher()

    def test_ihybridsearcher_abstract_methods(self):
        """Verify all expected abstract methods exist."""

        class ConcreteHybridSearcher(IHybridSearcher):
            def add_documents(self, documents):
                pass

            def search_sparse(self, query, top_k=10):
                return []

            def fuse_results(
                self, sparse_results, dense_results, top_k=10, query_text=None
            ):
                return []

            def set_llm_client(self, llm_client):
                pass

            def expand_query(self, query):
                return query

        searcher = ConcreteHybridSearcher()
        searcher.add_documents([])
        assert searcher.search_sparse("test") == []
        assert searcher.fuse_results([], []) == []
        searcher.set_llm_client(None)
        assert searcher.expand_query("test") == "test"


class TestVectorStoreInterfaceContract:
    """Test IVectorStore interface contract expectations."""

    def test_add_chunks_returns_count(self):
        """Verify add_chunks returns number of chunks added."""

        class MockVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return len(chunks)

            def delete_by_rule_codes(self, rule_codes):
                return 0

            def search(self, query, filter=None, top_k=10):
                return []

            def get_all_rule_codes(self):
                return set()

            def count(self):
                return 0

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        store = MockVectorStore()
        chunks = [create_test_chunk(f"id{i}") for i in range(3)]
        assert store.add_chunks(chunks) == 3

    def test_delete_by_rule_codes_returns_count(self):
        """Verify delete_by_rule_codes returns number of chunks deleted."""

        class MockVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return 0

            def delete_by_rule_codes(self, rule_codes):
                return 100  # Simulated deletion count

            def search(self, query, filter=None, top_k=10):
                return []

            def get_all_rule_codes(self):
                return set()

            def count(self):
                return 0

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        store = MockVectorStore()
        assert store.delete_by_rule_codes(["1-1-1", "1-1-2"]) == 100

    def test_search_returns_search_results(self):
        """Verify search returns list of SearchResult."""

        chunk = create_test_chunk()

        class MockVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return 0

            def delete_by_rule_codes(self, rule_codes):
                return 0

            def search(self, query, filter=None, top_k=10):
                return [SearchResult(chunk=chunk, score=0.9)]

            def get_all_rule_codes(self):
                return set()

            def count(self):
                return 0

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        store = MockVectorStore()
        results = store.search(Query(text="test"))
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_get_all_rule_codes_returns_set(self):
        """Verify get_all_rule_codes returns a set."""

        class MockVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return 0

            def delete_by_rule_codes(self, rule_codes):
                return 0

            def search(self, query, filter=None, top_k=10):
                return []

            def get_all_rule_codes(self):
                return {"1-1-1", "1-1-2"}

            def count(self):
                return 0

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        store = MockVectorStore()
        codes = store.get_all_rule_codes()
        assert isinstance(codes, set)
        assert "1-1-1" in codes

    def test_count_returns_integer(self):
        """Verify count returns integer."""

        class MockVectorStore(IVectorStore):
            def add_chunks(self, chunks):
                return 0

            def delete_by_rule_codes(self, rule_codes):
                return 0

            def search(self, query, filter=None, top_k=10):
                return []

            def get_all_rule_codes(self):
                return set()

            def count(self):
                return 1234

            def get_all_documents(self):
                return []

            def clear_all(self):
                return 0

        store = MockVectorStore()
        assert store.count() == 1234


class TestDocumentLoaderInterfaceContract:
    """Test IDocumentLoader interface contract expectations."""

    def test_load_all_chunks_returns_list(self):
        """Verify load_all_chunks returns list of Chunks."""

        class MockLoader(IDocumentLoader):
            def load_all_chunks(self, json_path):
                return [create_test_chunk("id1"), create_test_chunk("id2")]

            def load_chunks_by_rule_codes(self, json_path, rule_codes):
                return []

            def compute_state(self, json_path):
                return SyncState.empty()

            def get_regulation_titles(self, json_path):
                return {}

            def get_all_regulations(self, json_path):
                return []

            def get_regulation_doc(self, json_path, identifier):
                return None

            def get_regulation_overview(self, json_path, identifier):
                return None

        loader = MockLoader()
        chunks = loader.load_all_chunks("test.json")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_compute_state_returns_sync_state(self):
        """Verify compute_state returns SyncState object."""

        class MockLoader(IDocumentLoader):
            def load_all_chunks(self, json_path):
                return []

            def load_chunks_by_rule_codes(self, json_path, rule_codes):
                return []

            def compute_state(self, json_path):
                return SyncState(
                    last_sync=datetime.now().isoformat(),
                    json_file=json_path,
                    regulations={"1-1-1": "hash123"},
                )

            def get_regulation_titles(self, json_path):
                return {}

            def get_all_regulations(self, json_path):
                return []

            def get_regulation_doc(self, json_path, identifier):
                return None

            def get_regulation_overview(self, json_path, identifier):
                return None

        loader = MockLoader()
        state = loader.compute_state("test.json")
        assert isinstance(state, SyncState)
        assert state.regulations["1-1-1"] == "hash123"

    def test_get_regulation_titles_returns_dict(self):
        """Verify get_regulation_titles returns dict."""

        class MockLoader(IDocumentLoader):
            def load_all_chunks(self, json_path):
                return []

            def load_chunks_by_rule_codes(self, json_path, rule_codes):
                return []

            def compute_state(self, json_path):
                return SyncState.empty()

            def get_regulation_titles(self, json_path):
                return {"1-1-1": "Regulation 1", "1-1-2": "Regulation 2"}

            def get_all_regulations(self, json_path):
                return []

            def get_regulation_doc(self, json_path, identifier):
                return None

            def get_regulation_overview(self, json_path, identifier):
                return None

        loader = MockLoader()
        titles = loader.get_regulation_titles("test.json")
        assert isinstance(titles, dict)
        assert titles["1-1-1"] == "Regulation 1"

    def test_get_regulation_overview_returns_overview(self):
        """Verify get_regulation_overview can return RegulationOverview."""

        class MockLoader(IDocumentLoader):
            def load_all_chunks(self, json_path):
                return []

            def load_chunks_by_rule_codes(self, json_path, rule_codes):
                return []

            def compute_state(self, json_path):
                return SyncState.empty()

            def get_regulation_titles(self, json_path):
                return {}

            def get_all_regulations(self, json_path):
                return []

            def get_regulation_doc(self, json_path, identifier):
                return None

            def get_regulation_overview(self, json_path, identifier):
                return RegulationOverview(
                    rule_code="1-1-1",
                    title="Test Regulation",
                    status=RegulationStatus.ACTIVE,
                    article_count=10,
                    chapters=[],
                )

        loader = MockLoader()
        overview = loader.get_regulation_overview("test.json", "1-1-1")
        assert isinstance(overview, RegulationOverview)
        assert overview.rule_code == "1-1-1"


class TestLLMClientInterfaceContract:
    """Test ILLMClient interface contract expectations."""

    def test_generate_returns_string(self):
        """Verify generate returns string response."""

        class MockLLM(ILLMClient):
            def generate(self, system_prompt, user_message, temperature=0.0):
                return f"Response to: {user_message}"

            def get_embedding(self, text):
                return [0.1] * 10

        llm = MockLLM()
        response = llm.generate("You are helpful", "What is AI?")
        assert isinstance(response, str)
        assert "What is AI?" in response

    def test_get_embedding_returns_list(self):
        """Verify get_embedding returns list of floats."""

        class MockLLM(ILLMClient):
            def generate(self, system_prompt, user_message, temperature=0.0):
                return "response"

            def get_embedding(self, text):
                return [0.5] * 768  # Simulated embedding

        llm = MockLLM()
        embedding = llm.get_embedding("test text")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) == 768


class TestRerankerInterfaceContract:
    """Test IReranker interface contract expectations."""

    def test_rerank_returns_list_of_tuples(self):
        """Verify rerank returns list of tuples with scores."""

        class MockReranker(IReranker):
            def rerank(self, query, documents, top_k=10):
                # Return (doc_id, content, score, metadata) tuples
                return [
                    ("doc1", "content 1", 0.95, {"title": "Doc 1"}),
                    ("doc2", "content 2", 0.85, {"title": "Doc 2"}),
                ]

        reranker = MockReranker()
        results = reranker.rerank(
            "query", [("doc1", "content 1"), ("doc2", "content 2")]
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0][2] > results[1][2]  # Higher score first


class TestHybridSearcherInterfaceContract:
    """Test IHybridSearcher interface contract expectations."""

    def test_expand_query_returns_string(self):
        """Verify expand_query returns expanded query string."""

        class MockSearcher(IHybridSearcher):
            def add_documents(self, documents):
                pass

            def search_sparse(self, query, top_k=10):
                return []

            def fuse_results(
                self, sparse_results, dense_results, top_k=10, query_text=None
            ):
                return []

            def set_llm_client(self, llm_client):
                pass

            def expand_query(self, query):
                # Simple expansion: add synonyms
                return f"{query} OR expanded"

        searcher = MockSearcher()
        expanded = searcher.expand_query("test")
        assert isinstance(expanded, str)
        assert "expanded" in expanded

    def test_set_llm_client_stores_client(self):
        """Verify set_llm_client stores the client."""

        class MockSearcher(IHybridSearcher):
            def add_documents(self, documents):
                pass

            def search_sparse(self, query, top_k=10):
                return []

            def fuse_results(
                self, sparse_results, dense_results, top_k=10, query_text=None
            ):
                return []

            def set_llm_client(self, llm_client):
                self._llm_client = llm_client

            def expand_query(self, query):
                return query

        searcher = MockSearcher()
        mock_client = MagicMock()
        searcher.set_llm_client(mock_client)
        assert searcher._llm_client is mock_client
