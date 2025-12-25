import json
from dataclasses import dataclass
from typing import List, Optional, Set

from src.rag.application.sync_usecase import SyncUseCase
from src.rag.domain.entities import Chunk, ChunkLevel
from src.rag.domain.value_objects import SyncState


def make_chunk(rule_code: str, chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
        title="",
        text="text",
        embedding_text="text",
        full_text="",
        parent_path=[],
        token_count=1,
        keywords=[],
        is_searchable=True,
    )


@dataclass
class FakeLoader:
    state: SyncState
    chunks_by_code: dict
    load_all_called: bool = False
    load_chunks_called_with: Optional[Set[str]] = None

    def load_all_chunks(self, json_path: str) -> List[Chunk]:
        self.load_all_called = True
        chunks: List[Chunk] = []
        for items in self.chunks_by_code.values():
            chunks.extend(items)
        return chunks

    def load_chunks_by_rule_codes(self, json_path: str, rule_codes: Set[str]) -> List[Chunk]:
        self.load_chunks_called_with = set(rule_codes)
        chunks: List[Chunk] = []
        for code in rule_codes:
            chunks.extend(self.chunks_by_code.get(code, []))
        return chunks

    def compute_state(self, json_path: str) -> SyncState:
        return self.state


@dataclass
class FakeStore:
    codes: Set[str]
    count_value: int
    cleared: bool = False
    deleted_codes: List[str] = None
    added_chunks: List[Chunk] = None

    def __post_init__(self) -> None:
        if self.deleted_codes is None:
            self.deleted_codes = []
        if self.added_chunks is None:
            self.added_chunks = []

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

    def clear_all(self) -> int:
        self.cleared = True
        self.codes = set()
        self.count_value = 0
        return 0


def write_state(path, state: SyncState) -> None:
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")


def test_incremental_sync_full_sync_when_store_empty(tmp_path):
    state_path = tmp_path / "sync_state.json"
    old_state = SyncState(
        last_sync="2025-01-01T00:00:00Z",
        json_file="test.json",
        regulations={"A": "hash-a"},
    )
    write_state(state_path, old_state)

    new_state = SyncState(
        last_sync="2025-01-02T00:00:00Z",
        json_file="test.json",
        regulations={"A": "hash-a"},
    )

    loader = FakeLoader(state=new_state, chunks_by_code={"A": [make_chunk("A", "1")]})
    store = FakeStore(codes=set(), count_value=0)
    usecase = SyncUseCase(loader, store, state_path=str(state_path))

    result = usecase.incremental_sync("dummy.json")

    assert store.cleared is True
    assert loader.load_all_called is True
    assert result.added == 1


def test_incremental_sync_repairs_missing_codes(tmp_path):
    state_path = tmp_path / "sync_state.json"
    old_state = SyncState(
        last_sync="2025-01-01T00:00:00Z",
        json_file="test.json",
        regulations={"A": "hash-a", "B": "hash-b"},
    )
    write_state(state_path, old_state)

    new_state = SyncState(
        last_sync="2025-01-02T00:00:00Z",
        json_file="test.json",
        regulations={"A": "hash-a", "B": "hash-b"},
    )

    loader = FakeLoader(
        state=new_state,
        chunks_by_code={"A": [make_chunk("A", "1")], "B": [make_chunk("B", "2")]},
    )
    store = FakeStore(codes={"A"}, count_value=1)
    usecase = SyncUseCase(loader, store, state_path=str(state_path))

    result = usecase.incremental_sync("dummy.json")

    assert loader.load_chunks_called_with == {"B"}
    assert "B" in store.codes
    assert result.added == 1
