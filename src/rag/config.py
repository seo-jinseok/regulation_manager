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
        default_factory=lambda: os.getenv("RAG_JSON_PATH", "data/output/규정집-test01.json")
    )
    sync_state_path: str = "data/sync_state.json"
    
    # LLM settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    llm_model: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_MODEL")
    )
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )
    
    # Search settings
    use_reranker: bool = True
    use_hybrid: bool = True
    default_top_k: int = 5
    synonyms_path: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_SYNONYMS_PATH")
    )
    
    # Supported LLM providers
    llm_providers: List[str] = field(
        default_factory=lambda: [
            "ollama", "lmstudio", "mlx", "local", 
            "openai", "gemini", "openrouter"
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
