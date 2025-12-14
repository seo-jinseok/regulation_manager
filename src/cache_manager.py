import hashlib
import json
import os
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

    # --- File State Methods ---

    def get_file_state(self, file_path: str) -> Optional[Dict[str, str]]:
        return self.file_states.get(str(file_path))

    def update_file_state(self, file_path: str, hwp_hash: str = None, raw_md_hash: str = None):
        key = str(file_path)
        if key not in self.file_states:
            self.file_states[key] = {}
        
        if hwp_hash:
            self.file_states[key]["hwp_hash"] = hwp_hash
        if raw_md_hash:
            self.file_states[key]["raw_md_hash"] = raw_md_hash

    # --- LLM Cache Methods ---

    def get_cached_llm_response(self, text_chunk: str) -> Optional[str]:
        chunk_hash = self.compute_text_hash(text_chunk.strip())
        return self.llm_cache.get(chunk_hash)

    def cache_llm_response(self, text_chunk: str, response: str):
        chunk_hash = self.compute_text_hash(text_chunk.strip())
        self.llm_cache[chunk_hash] = response
