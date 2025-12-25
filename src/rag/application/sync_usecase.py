"""
Sync Use Case for Regulation RAG System.

Handles synchronization between JSON files and the vector store.
Supports both full sync and incremental sync.
"""

import json
from pathlib import Path
from typing import Optional

from ..domain.repositories import IDocumentLoader, IVectorStore
from ..domain.value_objects import SyncResult, SyncState


class SyncUseCase:
    """
    Use case for synchronizing regulations with the vector store.
    
    Supports:
    - Full sync: Reload all regulations from JSON
    - Incremental sync: Only update changed regulations
    """

    def __init__(
        self,
        loader: IDocumentLoader,
        store: IVectorStore,
        state_path: str = "data/sync_state.json",
    ):
        """
        Initialize sync use case.

        Args:
            loader: Document loader implementation.
            store: Vector store implementation.
            state_path: Path to save sync state for incremental updates.
        """
        self.loader = loader
        self.store = store
        self.state_path = Path(state_path)

    def full_sync(self, json_path: str) -> SyncResult:
        """
        Perform full synchronization.

        Clears the vector store and reloads all regulations.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            SyncResult with statistics.
        """
        errors = []

        try:
            # Clear existing data
            self.store.clear_all()

            # Load all chunks
            chunks = self.loader.load_all_chunks(json_path)

            # Add to store
            added = self.store.add_chunks(chunks)

            # Compute and save state
            state = self.loader.compute_state(json_path)
            self._save_state(state)

            return SyncResult(
                added=len(state.regulations),
                modified=0,
                removed=0,
                unchanged=0,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            return SyncResult(errors=errors)

    def incremental_sync(self, json_path: str) -> SyncResult:
        """
        Perform incremental synchronization.

        Only updates regulations that have changed since last sync.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            SyncResult with statistics.
        """
        errors = []

        try:
            # Load previous state
            old_state = self._load_state()

            # Compute current state
            new_state = self.loader.compute_state(json_path)

            # If no previous state, do full sync
            if not old_state.regulations:
                return self.full_sync(json_path)

            # If store is empty but state exists, rebuild from scratch
            if self.store.count() == 0:
                return self.full_sync(json_path)

            # Compute diff
            old_codes = set(old_state.regulations.keys())
            new_codes = set(new_state.regulations.keys())

            added_codes = new_codes - old_codes
            removed_codes = old_codes - new_codes
            common_codes = old_codes & new_codes

            # Find modified regulations (hash changed)
            modified_codes = {
                code for code in common_codes
                if old_state.regulations[code] != new_state.regulations[code]
            }

            unchanged_count = len(common_codes) - len(modified_codes)

            # Reconcile with actual store state
            store_codes = self.store.get_all_rule_codes()
            missing_in_store = new_codes - store_codes
            extra_in_store = store_codes - new_codes

            if extra_in_store:
                removed_codes |= extra_in_store

            # Delete removed regulations
            if removed_codes:
                self.store.delete_by_rule_codes(list(removed_codes))

            # Update modified, added, and missing regulations
            codes_to_update = added_codes | modified_codes | missing_in_store
            if codes_to_update:
                # Delete existing chunks for modified regulations
                if modified_codes:
                    self.store.delete_by_rule_codes(list(modified_codes))

                # Load and add new chunks
                chunks = self.loader.load_chunks_by_rule_codes(
                    json_path, codes_to_update
                )
                self.store.add_chunks(chunks)

            # Save new state
            self._save_state(new_state)

            return SyncResult(
                added=len(added_codes | missing_in_store),
                modified=len(modified_codes),
                removed=len(removed_codes),
                unchanged=unchanged_count,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            return SyncResult(errors=errors)

    def get_sync_status(self) -> dict:
        """
        Get current sync status.

        Returns:
            Dict with sync status information.
        """
        state = self._load_state()
        store_count = self.store.count()
        store_codes = self.store.get_all_rule_codes()

        return {
            "last_sync": state.last_sync,
            "json_file": state.json_file,
            "state_regulations": len(state.regulations),
            "store_chunks": store_count,
            "store_regulations": len(store_codes),
        }

    def _load_state(self) -> SyncState:
        """Load sync state from file."""
        if not self.state_path.exists():
            return SyncState.empty()

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SyncState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return SyncState.empty()

    def _save_state(self, state: SyncState) -> None:
        """Save sync state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    def reset_state(self) -> None:
        """Reset sync state by deleting the state file."""
        if self.state_path.exists():
            self.state_path.unlink()
