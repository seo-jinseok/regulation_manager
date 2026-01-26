"""Tests for RAGConfig advanced RAG settings."""


from src.rag.config import RAGConfig, reset_config


class TestAdvancedRAGSettings:
    """Tests for advanced RAG configuration settings."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_enable_self_rag_default_true(self):
        """Self-RAG는 기본적으로 활성화되어야 함."""
        config = RAGConfig()
        assert config.enable_self_rag is True

    def test_enable_hyde_default_true(self):
        """HyDE는 기본적으로 활성화되어야 함."""
        config = RAGConfig()
        assert config.enable_hyde is True

    def test_bm25_tokenize_mode_default_konlpy(self):
        """BM25 토크나이저는 기본적으로 konlpy 모드여야 함."""
        config = RAGConfig()
        assert config.bm25_tokenize_mode == "konlpy"

    def test_corrective_rag_thresholds_default(self):
        """Corrective RAG 임계값은 쿼리 유형별 딕셔너리여야 함."""
        config = RAGConfig()
        thresholds = config.corrective_rag_thresholds

        assert isinstance(thresholds, dict)
        assert "simple" in thresholds
        assert "medium" in thresholds
        assert "complex" in thresholds
        # Simple 쿼리는 더 낮은 임계값 (재검색 덜 필요)
        assert thresholds["simple"] < thresholds["complex"]

    def test_hyde_cache_dir_default(self):
        """HyDE 캐시 디렉토리 기본값 확인."""
        config = RAGConfig()
        assert config.hyde_cache_dir == "data/cache/hyde"

    def test_hyde_cache_enabled_default_true(self):
        """HyDE 캐시는 기본적으로 활성화 (영구 저장)."""
        config = RAGConfig()
        assert config.hyde_cache_enabled is True

    def test_settings_can_be_overridden(self):
        """설정 값들을 오버라이드할 수 있어야 함."""
        config = RAGConfig(
            enable_self_rag=False,
            enable_hyde=False,
            bm25_tokenize_mode="simple",
        )
        assert config.enable_self_rag is False
        assert config.enable_hyde is False
        assert config.bm25_tokenize_mode == "simple"

    def test_corrective_rag_thresholds_override(self):
        """Corrective RAG 임계값을 커스텀할 수 있어야 함."""
        custom_thresholds = {"simple": 0.3, "medium": 0.5, "complex": 0.7}
        config = RAGConfig(corrective_rag_thresholds=custom_thresholds)
        assert config.corrective_rag_thresholds == custom_thresholds

    def test_bm25_tokenize_mode_validates(self):
        """bm25_tokenize_mode는 유효한 값만 허용해야 함."""
        # 유효한 값들
        for mode in ["simple", "morpheme", "konlpy"]:
            config = RAGConfig(bm25_tokenize_mode=mode)
            assert config.bm25_tokenize_mode == mode
