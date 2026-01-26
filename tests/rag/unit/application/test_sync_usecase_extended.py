"""
Extended unit tests for SyncUseCase.

Tests clean architecture compliance with mocked dependencies.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.rag.application.sync_usecase import SyncUseCase
from src.rag.domain.entities import Chunk, ChunkLevel, RegulationOverview
from src.rag.domain.repositories import IDocumentLoader, IVectorStore
from src.rag.domain.value_objects import SyncState

# ====================
# Helper factories
# ====================

def make_chunk(
    chunk_id: str = "test-chunk-1",
    rule_code: str = "1-1-1",
    text: str = "테스트 내용",
) -> Chunk:
    """Factory for creating test chunks."""
    return Chunk(
        id=chunk_id,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
        title="제1조",
        text=text,
        embedding_text=text,
        full_text=text,
        parent_path=["테스트규정"],
        token_count=len(text),
        keywords=[],
        is_searchable=True,
    )


def make_state(
    regulations: Optional[Dict[str, str]] = None,
    last_sync: str = "2025-01-01T00:00:00Z",
) -> SyncState:
    """Factory for creating test sync states."""
    return SyncState(
        last_sync=last_sync,
        json_file="test.json",
        regulations=regulations or {},
    )


# ====================
# Mock implementations
# ====================

@dataclass
class MockDocumentLoader(IDocumentLoader):
    """Mock document loader for testing."""

    state: SyncState = field(default_factory=SyncState.empty)
    chunks_by_code: Dict[str, List[Chunk]] = field(default_factory=dict)
    titles: Dict[str, str] = field(default_factory=dict)
    load_all_called: bool = False
    load_chunks_called_with: Optional[Set[str]] = None

    def load_all_chunks(self, json_path: str) -> List[Chunk]:
        self.load_all_called = True
        chunks: List[Chunk] = []
        for items in self.chunks_by_code.values():
            chunks.extend(items)
        return chunks

    def load_chunks_by_rule_codes(
        self, json_path: str, rule_codes: Set[str]
    ) -> List[Chunk]:
        self.load_chunks_called_with = set(rule_codes)
        chunks: List[Chunk] = []
        for code in rule_codes:
            chunks.extend(self.chunks_by_code.get(code, []))
        return chunks

    def compute_state(self, json_path: str) -> SyncState:
        return self.state

    def get_regulation_titles(self, json_path: str) -> dict:
        return self.titles

    def get_all_regulations(self, json_path: str) -> List[tuple]:
        return [(code, self.titles.get(code, f"규정 {code}"))
                for code in self.state.regulations.keys()]

    def get_regulation_doc(self, json_path: str, identifier: str) -> Optional[dict]:
        return None

    def get_regulation_overview(
        self, json_path: str, identifier: str
    ) -> Optional[RegulationOverview]:
        return None


@dataclass
class MockVectorStore(IVectorStore):
    """Mock vector store for testing."""

    codes: Set[str] = field(default_factory=set)
    count_value: int = 0
    cleared: bool = False
    deleted_codes: List[str] = field(default_factory=list)
    added_chunks: List[Chunk] = field(default_factory=list)

    def add_chunks(self, chunks: List[Chunk]) -> int:
        self.added_chunks.extend(chunks)
        for chunk in chunks:
            self.codes.add(chunk.rule_code)
        self.count_value += len(chunks)
        return len(chunks)

    def delete_by_rule_codes(self, rule_codes: List[str]) -> int:
        self.deleted_codes.extend(rule_codes)
        self.codes -= set(rule_codes)
        return len(rule_codes)

    def search(self, query, filter=None, top_k: int = 10):
        return []

    def get_all_rule_codes(self) -> Set[str]:
        return set(self.codes)

    def count(self) -> int:
        return self.count_value

    def get_all_documents(self) -> list:
        return []

    def clear_all(self) -> int:
        self.cleared = True
        old_count = self.count_value
        self.codes = set()
        self.count_value = 0
        return old_count


# ====================
# Unit tests for SyncUseCase
# ====================

class TestSyncUseCaseInit:
    """Tests for SyncUseCase initialization."""

    def test_init_with_dependencies(self, tmp_path):
        """Create use case with dependencies."""
        loader = MockDocumentLoader()
        store = MockVectorStore()
        state_path = tmp_path / "sync_state.json"

        usecase = SyncUseCase(loader, store, state_path=str(state_path))

        assert usecase.loader is loader
        assert usecase.store is store
        assert usecase.state_path == state_path


class TestSyncUseCaseFullSync:
    """Tests for SyncUseCase.full_sync() method."""

    def test_full_sync_clears_store(self, tmp_path):
        """Full sync clears existing store data."""
        state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=state,
            chunks_by_code={"A-1-1": [make_chunk(rule_code="A-1-1")]},
        )
        store = MockVectorStore(codes={"B-1-1"}, count_value=5)
        state_path = tmp_path / "sync_state.json"

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.full_sync("test.json")

        assert store.cleared is True
        assert loader.load_all_called is True

    def test_full_sync_adds_all_chunks(self, tmp_path):
        """Full sync adds all chunks to store."""
        chunks = [
            make_chunk(chunk_id="c1", rule_code="A-1-1"),
            make_chunk(chunk_id="c2", rule_code="B-1-1"),
        ]
        state = make_state(regulations={"A-1-1": "hash-a", "B-1-1": "hash-b"})
        loader = MockDocumentLoader(
            state=state,
            chunks_by_code={
                "A-1-1": [chunks[0]],
                "B-1-1": [chunks[1]],
            },
        )
        store = MockVectorStore()
        state_path = tmp_path / "sync_state.json"

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.full_sync("test.json")

        assert len(store.added_chunks) == 2
        assert result.added == 2

    def test_full_sync_saves_state(self, tmp_path):
        """Full sync saves state to file."""
        state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=state,
            chunks_by_code={"A-1-1": [make_chunk()]},
        )
        store = MockVectorStore()
        state_path = tmp_path / "sync_state.json"

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        usecase.full_sync("test.json")

        assert state_path.exists()
        saved_data = json.loads(state_path.read_text())
        assert "A-1-1" in saved_data["regulations"]

    def test_full_sync_returns_error_on_exception(self, tmp_path):
        """Full sync returns error in result on exception."""
        loader = MockDocumentLoader()
        store = MockVectorStore()
        state_path = tmp_path / "sync_state.json"

        # Make loader raise exception
        def raise_error(_):
            raise ValueError("Test error")
        loader.load_all_chunks = raise_error

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.full_sync("test.json")

        assert result.has_errors is True
        assert "Test error" in result.errors[0]


class TestSyncUseCaseIncrementalSync:
    """Tests for SyncUseCase.incremental_sync() method."""

    def test_incremental_sync_full_sync_when_no_state(self, tmp_path):
        """Incremental sync does full sync when no previous state."""
        state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=state,
            chunks_by_code={"A-1-1": [make_chunk()]},
        )
        store = MockVectorStore(codes=set(), count_value=0)
        state_path = tmp_path / "sync_state.json"
        # No existing state file

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        # Should do full sync
        assert store.cleared is True

    def test_incremental_sync_full_sync_when_store_empty(self, tmp_path):
        """Incremental sync does full sync when store is empty."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={"A-1-1": "hash-a"})
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={"A-1-1": [make_chunk()]},
        )
        store = MockVectorStore(codes=set(), count_value=0)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert store.cleared is True
        assert result.added == 1

    def test_incremental_sync_detects_added(self, tmp_path):
        """Incremental sync detects added regulations."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={"A-1-1": "hash-a"})
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",  # New
        })
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={
                "A-1-1": [make_chunk(rule_code="A-1-1")],
                "B-1-1": [make_chunk(rule_code="B-1-1")],
            },
        )
        store = MockVectorStore(codes={"A-1-1"}, count_value=1)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.added == 1
        assert "B-1-1" in store.codes

    def test_incremental_sync_detects_removed(self, tmp_path):
        """Incremental sync detects removed regulations."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",
        })
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={"A-1-1": "hash-a"})  # B removed
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={"A-1-1": [make_chunk(rule_code="A-1-1")]},
        )
        store = MockVectorStore(codes={"A-1-1", "B-1-1"}, count_value=2)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.removed == 1
        assert "B-1-1" in store.deleted_codes

    def test_incremental_sync_detects_modified(self, tmp_path):
        """Incremental sync detects modified regulations."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={
            "A-1-1": "hash-a-old",
        })
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={"A-1-1": "hash-a-new"})  # Hash changed
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={"A-1-1": [make_chunk(rule_code="A-1-1")]},
        )
        store = MockVectorStore(codes={"A-1-1"}, count_value=1)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.modified == 1
        assert "A-1-1" in store.deleted_codes  # Old deleted
        assert "A-1-1" in store.codes  # New added

    def test_incremental_sync_detects_unchanged(self, tmp_path):
        """Incremental sync counts unchanged regulations."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",
        })
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={
            "A-1-1": "hash-a",  # Unchanged
            "B-1-1": "hash-b",  # Unchanged
        })
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={
                "A-1-1": [make_chunk(rule_code="A-1-1")],
                "B-1-1": [make_chunk(rule_code="B-1-1")],
            },
        )
        store = MockVectorStore(codes={"A-1-1", "B-1-1"}, count_value=2)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.unchanged == 2
        assert result.added == 0
        assert result.modified == 0

    def test_incremental_sync_repairs_missing_in_store(self, tmp_path):
        """Incremental sync repairs codes missing from store."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",
        })
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",
        })
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={
                "A-1-1": [make_chunk(rule_code="A-1-1")],
                "B-1-1": [make_chunk(rule_code="B-1-1")],
            },
        )
        # B-1-1 is missing from store (state says it exists)
        store = MockVectorStore(codes={"A-1-1"}, count_value=1)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert loader.load_chunks_called_with == {"B-1-1"}
        assert "B-1-1" in store.codes

    def test_incremental_sync_removes_extra_in_store(self, tmp_path):
        """Incremental sync removes codes not in JSON."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={"A-1-1": "hash-a"})
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={"A-1-1": [make_chunk(rule_code="A-1-1")]},
        )
        # Store has extra code not in JSON
        store = MockVectorStore(codes={"A-1-1", "ORPHAN"}, count_value=2)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert "ORPHAN" in store.deleted_codes


class TestSyncUseCaseStatus:
    """Tests for SyncUseCase.get_sync_status() method."""

    def test_get_sync_status_returns_info(self, tmp_path):
        """Returns sync status information."""
        state_path = tmp_path / "sync_state.json"
        state = make_state(
            regulations={"A-1-1": "hash-a", "B-1-1": "hash-b"},
            last_sync="2025-01-02T10:00:00Z",
        )
        state_path.write_text(json.dumps(state.to_dict()))

        loader = MockDocumentLoader(state=state)
        store = MockVectorStore(codes={"A-1-1", "B-1-1"}, count_value=5)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        status = usecase.get_sync_status()

        assert status["last_sync"] == "2025-01-02T10:00:00Z"
        assert status["state_regulations"] == 2
        assert status["store_regulations"] == 2
        assert status["store_chunks"] == 5

    def test_get_sync_status_handles_no_state(self, tmp_path):
        """Returns empty status when no state file."""
        state_path = tmp_path / "sync_state.json"
        # No state file

        loader = MockDocumentLoader()
        store = MockVectorStore()

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        status = usecase.get_sync_status()

        assert status["state_regulations"] == 0


class TestSyncUseCaseResetState:
    """Tests for SyncUseCase.reset_state() method."""

    def test_reset_state_deletes_file(self, tmp_path):
        """Reset deletes the state file."""
        state_path = tmp_path / "sync_state.json"
        state_path.write_text('{"test": "data"}')

        loader = MockDocumentLoader()
        store = MockVectorStore()

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        usecase.reset_state()

        assert not state_path.exists()

    def test_reset_state_handles_missing_file(self, tmp_path):
        """Reset handles missing state file gracefully."""
        state_path = tmp_path / "sync_state.json"
        # No state file

        loader = MockDocumentLoader()
        store = MockVectorStore()

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        # Should not raise
        usecase.reset_state()

        assert not state_path.exists()


class TestSyncResultIntegration:
    """Tests for SyncResult behavior."""

    def test_sync_result_has_changes(self, tmp_path):
        """SyncResult.has_changes() works correctly."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={"A-1-1": "hash-a"})
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={
            "A-1-1": "hash-a",
            "B-1-1": "hash-b",  # New
        })
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={
                "A-1-1": [make_chunk(rule_code="A-1-1")],
                "B-1-1": [make_chunk(rule_code="B-1-1")],
            },
        )
        store = MockVectorStore(codes={"A-1-1"}, count_value=1)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.has_changes is True

    def test_sync_result_no_changes(self, tmp_path):
        """SyncResult.has_changes() returns False when nothing changed."""
        state_path = tmp_path / "sync_state.json"
        old_state = make_state(regulations={"A-1-1": "hash-a"})
        state_path.write_text(json.dumps(old_state.to_dict()))

        new_state = make_state(regulations={"A-1-1": "hash-a"})
        loader = MockDocumentLoader(
            state=new_state,
            chunks_by_code={"A-1-1": [make_chunk(rule_code="A-1-1")]},
        )
        store = MockVectorStore(codes={"A-1-1"}, count_value=1)

        usecase = SyncUseCase(loader, store, state_path=str(state_path))
        result = usecase.incremental_sync("test.json")

        assert result.has_changes is False
