import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

class CacheManager:
    """
    Manages caching for file states and LLM responses to optimize processing.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_state_path = self.cache_dir / "file_state_cache.json"
        self.llm_cache_path = self.cache_dir / "llm_response_cache.json"
        
        self.file_states = self._load_json(self.file_state_path)
        self.llm_cache = self._load_json(self.llm_cache_path)
        self.llm_cache_ttl_days = self._get_env_int("LLM_CACHE_TTL_DAYS")
        self.llm_cache_max_entries = self._get_env_int("LLM_CACHE_MAX_ENTRIES")

    def _get_env_int(self, name: str) -> Optional[int]:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_json(self, path: Path, data: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_all(self):
        """Persist all caches to disk."""
        self._prune_llm_cache()
        self._save_json(self.file_state_path, self.file_states)
        self._save_json(self.llm_cache_path, self.llm_cache)

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_text_hash(self, text: str) -> str:
        """Compute SHA256 hash of a string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str, namespace: Optional[str]) -> str:
        payload = f"{namespace or ''}\n{text.strip()}"
        return self.compute_text_hash(payload)

    def _normalize_llm_entry(self, entry: Any) -> Dict[str, Any]:
        if isinstance(entry, dict):
            return entry
        return {"response": entry, "ts": 0}

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        if not self.llm_cache_ttl_days:
            return False
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)) or ts <= 0:
            return False
        max_age_seconds = self.llm_cache_ttl_days * 24 * 60 * 60
        return (time.time() - ts) > max_age_seconds

    def _prune_llm_cache(self):
        if not self.llm_cache:
            return

        # TTL prune
        if self.llm_cache_ttl_days:
            expired_keys = []
            for key, entry in self.llm_cache.items():
                normalized = self._normalize_llm_entry(entry)
                if self._is_expired(normalized):
                    expired_keys.append(key)
            for key in expired_keys:
                self.llm_cache.pop(key, None)

        # Size prune (oldest first)
        if self.llm_cache_max_entries and len(self.llm_cache) > self.llm_cache_max_entries:
            entries = []
            for key, entry in self.llm_cache.items():
                normalized = self._normalize_llm_entry(entry)
                entries.append((normalized.get("ts", 0), key))
            entries.sort()
            excess = len(entries) - self.llm_cache_max_entries
            for _, key in entries[:excess]:
                self.llm_cache.pop(key, None)

    # --- File State Methods ---

    def get_file_state(self, file_path: str) -> Optional[Dict[str, str]]:
        return self.file_states.get(str(file_path))

    def update_file_state(
        self,
        file_path: str,
        hwp_hash: str = None,
        raw_md_hash: str = None,
        pipeline_signature: str = None,
        final_json_hash: str = None,
        metadata_hash: str = None,
    ):
        key = str(file_path)
        if key not in self.file_states:
            self.file_states[key] = {}
        
        if hwp_hash:
            self.file_states[key]["hwp_hash"] = hwp_hash
        if raw_md_hash:
            self.file_states[key]["raw_md_hash"] = raw_md_hash
        if pipeline_signature:
            self.file_states[key]["pipeline_signature"] = pipeline_signature
        if final_json_hash:
            self.file_states[key]["final_json_hash"] = final_json_hash
        if metadata_hash:
            self.file_states[key]["metadata_hash"] = metadata_hash

    # --- LLM Cache Methods ---

    def get_cached_llm_response(self, text_chunk: str, namespace: Optional[str] = None) -> Optional[str]:
        chunk_hash = self._cache_key(text_chunk, namespace)
        entry = self.llm_cache.get(chunk_hash)
        if entry is None:
            return None
        normalized = self._normalize_llm_entry(entry)
        if self._is_expired(normalized):
            self.llm_cache.pop(chunk_hash, None)
            return None
        return normalized.get("response")

    def cache_llm_response(self, text_chunk: str, response: str, namespace: Optional[str] = None):
        chunk_hash = self._cache_key(text_chunk, namespace)
        self.llm_cache[chunk_hash] = {"response": response, "ts": int(time.time())}
