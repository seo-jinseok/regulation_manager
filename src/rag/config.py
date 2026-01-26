"""
Configuration module for Regulation RAG System.

Centralizes all configuration settings to avoid hardcoded values
scattered across the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""

    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    priority: int = 0  # Lower number = higher priority


@dataclass
class FallbackConfig:
    """Configuration for LLM provider fallback behavior."""

    enabled: bool = True
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 32.0
    backoff_multiplier: float = 2.0
    log_fallback_events: bool = True
    cache_failures: bool = True
    failure_cache_ttl_seconds: int = 300  # 5 minutes

    # Fallback chain: providers tried in order
    # Each entry defines a provider configuration
    provider_chain: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "provider": "openrouter",
            "model": "google/gemini-2.0-flash-exp:free",
            "api_key_env_var": "OPENROUTER_API_KEY",
            "priority": 1,
        },
        {
            "provider": "lmstudio",
            "model": None,  # Uses LM Studio default
            "base_url": "http://localhost:1234",
            "api_key_env_var": None,
            "priority": 2,
        },
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-r1:free",
            "api_key_env_var": "OPENROUTER_API_KEY",
            "priority": 3,
        },
    ])

    # Graceful degradation settings
    allow_partial_results: bool = True
    partial_result_fallback_message: str = (
        "[Note: Response generated using fallback provider due to primary provider unavailability.]"
    )



@dataclass
class RerankerConfig:
    """Configuration for reranker models (Cycle 5: Korean Reranker Integration)."""

    # Primary reranker model (multilingual)
    primary_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )

    # Korean-specific reranker models for A/B testing
    korean_models: List[str] = field(
        default_factory=lambda: os.getenv(
            "KOREAN_RERANKER_MODELS",
            "Dongjin-kr/kr-reranker,NLPai/ko-reranker"
        ).split(",")
        if os.getenv("KOREAN_RERANKER_MODELS")
        else ["Dongjin-kr/kr-reranker", "NLPai/ko-reranker"]
    )

    # A/B testing configuration
    enable_ab_testing: bool = field(
        default_factory=lambda: os.getenv("RERANKER_AB_TESTING", "true").lower() == "true"
    )
    ab_test_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANKER_AB_RATIO", "0.5"))
    )  # 50% traffic to korean model

    # Model selection strategy: "ab_test", "korean_only", "multilingual_only"
    model_selection_strategy: str = field(
        default_factory=lambda: os.getenv("RERANKER_STRATEGY", "ab_test")
    )

    # Performance settings
    use_fp16: bool = field(
        default_factory=lambda: os.getenv("RERANKER_FP16", "true").lower() == "true"
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("RERANKER_BATCH_SIZE", "32"))
    )

    # Fallback to multilingual model if Korean model fails
    fallback_to_multilingual: bool = True

    # Warmup on startup
    warmup_on_init: bool = field(
        default_factory=lambda: os.getenv("RERANKER_WARMUP", "true").lower() == "true"
    )

    # Metrics tracking for A/B testing
    track_model_performance: bool = True
    metrics_storage_dir: str = ".metrics/reranker_ab"


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

    # LLM settings - Primary provider (legacy, for backward compatibility)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    llm_model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )

    # Fallback configuration
    llm_fallback: FallbackConfig = field(default_factory=FallbackConfig)

    # Reranker configuration
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

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
        default_factory=lambda: os.getenv("BM25_TOKENIZE_MODE", "kiwi")
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

    # Dynamic Query Expansion settings (Phase 3)
    enable_query_expansion: bool = field(
        default_factory=lambda: os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    )
    query_expansion_cache_dir: str = field(
        default_factory=lambda: os.getenv("QUERY_EXPANSION_CACHE_DIR", "data/cache/query_expansion")
    )

    # Fact check settings
    enable_fact_check: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true"
    )
    fact_check_max_retries: int = field(
        default_factory=lambda: int(os.getenv("FACT_CHECK_MAX_RETRIES", "2"))
    )

    # RAG Query Cache settings
    enable_cache: bool = field(
        default_factory=lambda: os.getenv("RAG_ENABLE_CACHE", "true").lower() == "true"
    )
    cache_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("RAG_CACHE_TTL_HOURS", "24"))
    )
    cache_dir: str = field(
        default_factory=lambda: os.getenv("RAG_CACHE_DIR", "data/cache/rag")
    )
    redis_host: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_REDIS_HOST")
    )
    redis_port: int = field(
        default_factory=lambda: int(os.getenv("RAG_REDIS_PORT", "6379"))
    )
    redis_password: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_REDIS_PASSWORD")
    )
    cache_warm_queries_path: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_CACHE_WARM_QUERIES_PATH", "data/config/warm_queries.json")
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

        # Load fallback config from environment if provided
        if os.getenv("LLM_FALLBACK_ENABLED"):
            self.llm_fallback.enabled = os.getenv("LLM_FALLBACK_ENABLED").lower() == "true"
        if os.getenv("LLM_FALLBACK_MAX_RETRIES"):
            self.llm_fallback.max_retries = int(os.getenv("LLM_FALLBACK_MAX_RETRIES"))
        if os.getenv("LLM_FALLBACK_BACKOFF_SECONDS"):
            self.llm_fallback.initial_backoff_seconds = float(os.getenv("LLM_FALLBACK_BACKOFF_SECONDS"))

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

    @property
    def query_expansion_cache_dir_resolved(self) -> Path:
        """Get absolute path to query expansion cache directory."""
        return Path(self.query_expansion_cache_dir).resolve()

    @property
    def cache_dir_resolved(self) -> Path:
        """Get absolute path to RAG query cache directory."""
        return Path(self.cache_dir).resolve()

    @property
    def cache_warm_queries_path_resolved(self) -> Optional[Path]:
        """Get absolute path to cache warm queries file if configured."""
        if not self.cache_warm_queries_path:
            return None
        return Path(self.cache_warm_queries_path).resolve()


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
