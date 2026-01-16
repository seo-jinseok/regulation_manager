"""
Configuration module for Regulation RAG System.

Centralizes all configuration settings to avoid hardcoded values
scattered across the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RAGConfig:
    """Configuration for RAG system.

    All paths are relative to project root unless absolute.
    Environment variables override defaults.
    """

    # Database
    db_path: str = field(
        default_factory=lambda: os.getenv("RAG_DB_PATH", "data/chroma_db")
    )

    # JSON data
    json_path: str = field(
        default_factory=lambda: os.getenv(
            "RAG_JSON_PATH", "data/output/규정집-test01.json"
        )
    )
    sync_state_path: str = "data/sync_state.json"

    # LLM settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    llm_model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )

    # Search settings
    use_reranker: bool = True
    use_hybrid: bool = True
    default_top_k: int = 5
    synonyms_path: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "RAG_SYNONYMS_PATH", "data/config/synonyms.json"
        )
    )
    intents_path: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "RAG_INTENTS_PATH", "data/config/intents.json"
        )
    )

    # Additional data paths
    feedback_log_path: str = "data/feedback_log.jsonl"
    evaluation_dataset_path: str = "data/config/evaluation_dataset.json"
    regulation_keywords_path: str = "data/config/regulation_keywords.json"
    bm25_index_cache_path: Optional[str] = field(
        default_factory=lambda: os.getenv("BM25_INDEX_CACHE_PATH")
    )

    # Advanced RAG settings
    enable_self_rag: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SELF_RAG", "true").lower() == "true"
    )
    enable_hyde: bool = field(
        default_factory=lambda: os.getenv("ENABLE_HYDE", "true").lower() == "true"
    )
    bm25_tokenize_mode: str = field(
        default_factory=lambda: os.getenv("BM25_TOKENIZE_MODE", "konlpy")
    )
    corrective_rag_thresholds: dict = field(
        default_factory=lambda: {
            "simple": 0.3,   # 단순 쿼리는 낮은 임계값
            "medium": 0.4,   # 기본 임계값
            "complex": 0.5,  # 복잡 쿼리는 더 엄격
        }
    )
    hyde_cache_dir: str = field(
        default_factory=lambda: os.getenv("HYDE_CACHE_DIR", "data/cache/hyde")
    )
    hyde_cache_enabled: bool = field(
        default_factory=lambda: os.getenv("HYDE_CACHE_ENABLED", "true").lower() == "true"
    )

    # Fact check settings
    enable_fact_check: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true"
    )
    fact_check_max_retries: int = field(
        default_factory=lambda: int(os.getenv("FACT_CHECK_MAX_RETRIES", "2"))
    )

    # Supported LLM providers
    llm_providers: List[str] = field(
        default_factory=lambda: [
            "ollama",
            "lmstudio",
            "mlx",
            "local",
            "openai",
            "gemini",
            "openrouter",
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.llm_provider not in self.llm_providers:
            self.llm_provider = "ollama"

    @property
    def db_path_resolved(self) -> Path:
        """Get absolute path to database directory."""
        return Path(self.db_path).resolve()

    @property
    def json_path_resolved(self) -> Path:
        """Get absolute path to JSON file."""
        return Path(self.json_path).resolve()

    @property
    def sync_state_path_resolved(self) -> Path:
        """Get absolute path to sync state file."""
        return Path(self.sync_state_path).resolve()

    @property
    def synonyms_path_resolved(self) -> Optional[Path]:
        """Get absolute path to synonyms file if configured."""
        if not self.synonyms_path:
            return None
        return Path(self.synonyms_path).resolve()

    @property
    def intents_path_resolved(self) -> Optional[Path]:
        """Get absolute path to intents file if configured."""
        if not self.intents_path:
            return None
        return Path(self.intents_path).resolve()

    @property
    def feedback_log_path_resolved(self) -> Path:
        """Get absolute path to feedback log file."""
        return Path(self.feedback_log_path).resolve()

    @property
    def evaluation_dataset_path_resolved(self) -> Path:
        """Get absolute path to evaluation dataset file."""
        return Path(self.evaluation_dataset_path).resolve()

    @property
    def regulation_keywords_path_resolved(self) -> Path:
        """Get absolute path to regulation keywords file."""
        return Path(self.regulation_keywords_path).resolve()

    @property
    def bm25_index_cache_path_resolved(self) -> Optional[Path]:
        """Get absolute path to BM25 index cache file if configured."""
        if not self.bm25_index_cache_path:
            return None
        return Path(self.bm25_index_cache_path).resolve()

    @property
    def hyde_cache_dir_resolved(self) -> Path:
        """Get absolute path to HyDE cache directory."""
        return Path(self.hyde_cache_dir).resolve()


# Global configuration instance (singleton)
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """
    Get the global configuration instance.

    Returns:
        RAGConfig instance with current settings.
    """
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reset_config() -> None:
    """Reset configuration to defaults (useful for testing)."""
    global _config
    _config = None
