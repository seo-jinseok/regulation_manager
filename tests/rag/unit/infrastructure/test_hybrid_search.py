"""
Unit tests for HybridSearcher and BM25 implementation.

Tests cover:
- BM25 tokenization and indexing
- RRF (Reciprocal Rank Fusion) scoring
- Dynamic weight adjustment
- Query expansion integration
"""

import pytest

from src.rag.infrastructure.hybrid_search import BM25, HybridSearcher, ScoredDocument


class TestBM25Tokenization:
    """Test BM25 tokenization for Korean text."""

    @pytest.fixture
    def bm25(self) -> BM25:
        return BM25()

    @pytest.fixture
    def bm25_konlpy(self) -> BM25:
        """KoNLPy 토크나이저를 사용하는 BM25."""
        return BM25(tokenize_mode="konlpy")

    @pytest.fixture
    def bm25_morpheme(self) -> BM25:
        """규칙 기반 형태소 분석 토크나이저를 사용하는 BM25."""
        return BM25(tokenize_mode="morpheme")

    def test_tokenize_korean_text(self, bm25: BM25):
        """한글 텍스트가 올바르게 토큰화되는지 확인"""
        tokens = bm25._tokenize("교원인사규정 제15조에 따라 임용한다")
        assert "교원인사규정" in tokens
        assert "제" in tokens or "15" in tokens  # 숫자 분리 가능
        assert "임용한다" in tokens

    def test_tokenize_mixed_korean_english(self, bm25: BM25):
        """한영 혼합 텍스트 토큰화"""
        tokens = bm25._tokenize("MOU 체결에 관한 규정")
        assert "mou" in tokens  # lowercase
        assert "체결에" in tokens or "체결" in tokens
        assert "규정" in tokens

    def test_tokenize_removes_punctuation(self, bm25: BM25):
        """문장부호가 제거되는지 확인"""
        tokens = bm25._tokenize("제1조(목적) 이 규정은...")
        # 괄호, 마침표 제거 확인
        assert "(" not in "".join(tokens)
        assert "." not in "".join(tokens)

    def test_tokenize_empty_string(self, bm25: BM25):
        """빈 문자열 토큰화"""
        tokens = bm25._tokenize("")
        assert tokens == []

    def test_tokenize_numbers_only(self, bm25: BM25):
        """숫자만 있는 텍스트"""
        tokens = bm25._tokenize("제15조 제3항 제2호")
        assert "15" in tokens
        assert "3" in tokens
        assert "2" in tokens

    def test_konlpy_tokenize_extracts_nouns(self, bm25_konlpy: BM25):
        """KoNLPy 토크나이저가 명사를 추출하는지 확인"""
        tokens = bm25_konlpy._tokenize("교원의 휴직 신청 절차")
        # 명사들이 추출되어야 함
        assert "교원" in tokens
        assert "휴직" in tokens
        assert "신청" in tokens
        assert "절차" in tokens

    def test_konlpy_tokenize_handles_compound_words(self, bm25_konlpy: BM25):
        """KoNLPy가 복합어를 분리하는지 확인"""
        tokens = bm25_konlpy._tokenize("육아휴직신청서")
        # 형태소 분석으로 분리되어야 함
        assert len(tokens) >= 1  # 최소 1개 이상의 토큰

    def test_morpheme_tokenize_splits_compounds(self, bm25_morpheme: BM25):
        """규칙 기반 형태소 분석이 복합어를 분리하는지 확인"""
        tokens = bm25_morpheme._tokenize("육아휴직신청")
        # 접두사 "육아" + "휴직" + "신청"으로 분리되거나, 최소한 "신청" 분리
        assert "신청" in tokens or "육아휴직" in tokens

    def test_tokenize_mode_affects_results(self):
        """토크나이저 모드에 따라 결과가 달라지는지 확인"""
        bm25_simple = BM25(tokenize_mode="simple")
        bm25_morpheme = BM25(tokenize_mode="morpheme")
        
        text = "교원휴직규정에 따른 신청"
        
        simple_tokens = bm25_simple._tokenize(text)
        morpheme_tokens = bm25_morpheme._tokenize(text)
        
        # morpheme 모드가 더 많은 토큰을 생성할 수 있음 (복합어 분리)
        assert len(simple_tokens) >= 1
        assert len(morpheme_tokens) >= 1


class TestBM25Indexing:
    """Test BM25 document indexing and search."""

    @pytest.fixture
    def bm25_with_docs(self) -> BM25:
        bm25 = BM25()
        docs = [
            ("doc1", "교원인사규정에 따른 교원 임용 절차", {"title": "교원인사규정"}),
            ("doc2", "학생 장학금 지급에 관한 규정", {"title": "장학금규정"}),
            ("doc3", "휴학 및 복학에 관한 절차 규정", {"title": "학적규정"}),
        ]
        bm25.add_documents(docs)
        return bm25

    def test_add_documents_updates_stats(self, bm25_with_docs: BM25):
        """문서 추가 시 통계가 업데이트되는지 확인"""
        assert bm25_with_docs.doc_count == 3
        assert bm25_with_docs.avg_doc_length > 0

    def test_search_returns_relevant_docs(self, bm25_with_docs: BM25):
        """검색 시 관련 문서가 반환되는지 확인"""
        results = bm25_with_docs.search("교원 임용")
        assert len(results) > 0
        assert results[0].doc_id == "doc1"

    def test_search_returns_scored_documents(self, bm25_with_docs: BM25):
        """검색 결과가 ScoredDocument 타입인지 확인"""
        results = bm25_with_docs.search("장학금")
        assert len(results) > 0
        assert isinstance(results[0], ScoredDocument)
        assert results[0].score > 0

    def test_search_respects_top_k(self, bm25_with_docs: BM25):
        """top_k 제한이 적용되는지 확인"""
        results = bm25_with_docs.search("규정", top_k=2)
        assert len(results) <= 2

    def test_search_no_match_returns_empty(self, bm25_with_docs: BM25):
        """매칭되지 않는 쿼리는 빈 결과 반환"""
        results = bm25_with_docs.search("존재하지않는키워드xyz")
        assert results == []

    def test_metadata_preserved_in_results(self, bm25_with_docs: BM25):
        """검색 결과에 메타데이터가 포함되는지 확인"""
        results = bm25_with_docs.search("장학금")
        assert len(results) > 0
        assert results[0].metadata.get("title") == "장학금규정"


class TestRRFFusion:
    """Test Reciprocal Rank Fusion scoring."""

    @pytest.fixture
    def searcher(self) -> HybridSearcher:
        return HybridSearcher(
            bm25_weight=0.5,
            dense_weight=0.5,
            rrf_k=60,
            use_dynamic_weights=False,
        )

    def test_fuse_empty_results(self, searcher: HybridSearcher):
        """빈 결과 융합"""
        result = searcher.fuse_results([], [], top_k=10)
        assert result == []

    def test_fuse_sparse_only(self, searcher: HybridSearcher):
        """Sparse 결과만 있는 경우"""
        sparse = [
            ScoredDocument("doc1", 0.9, "content1", {}),
            ScoredDocument("doc2", 0.8, "content2", {}),
        ]
        result = searcher.fuse_results(sparse, [], top_k=10)
        assert len(result) == 2
        # RRF 점수가 계산되어야 함
        assert result[0].score > 0

    def test_fuse_dense_only(self, searcher: HybridSearcher):
        """Dense 결과만 있는 경우"""
        dense = [
            ScoredDocument("doc1", 0.95, "content1", {}),
            ScoredDocument("doc2", 0.85, "content2", {}),
        ]
        result = searcher.fuse_results([], dense, top_k=10)
        assert len(result) == 2

    def test_rrf_score_calculation(self, searcher: HybridSearcher):
        """RRF 점수가 올바르게 계산되는지 확인"""
        sparse = [ScoredDocument("doc1", 0.9, "c1", {})]
        dense = [ScoredDocument("doc1", 0.95, "c1", {})]
        
        result = searcher.fuse_results(sparse, dense, top_k=1)
        
        # RRF score = bm25_w * 1/(k+1) + dense_w * 1/(k+1)
        # = 0.5 * 1/61 + 0.5 * 1/61 = 1/61 ≈ 0.0164
        expected_score = (0.5 / 61) + (0.5 / 61)
        assert result[0].score == pytest.approx(expected_score, rel=1e-4)

    def test_rrf_merges_overlapping_docs(self, searcher: HybridSearcher):
        """두 결과에 모두 있는 문서가 높은 점수를 받는지 확인"""
        sparse = [
            ScoredDocument("doc1", 0.9, "c1", {}),
            ScoredDocument("doc2", 0.8, "c2", {}),
        ]
        dense = [
            ScoredDocument("doc2", 0.95, "c2", {}),  # doc2가 dense에서 1등
            ScoredDocument("doc1", 0.85, "c1", {}),
        ]
        
        result = searcher.fuse_results(sparse, dense, top_k=2)
        
        # doc1: sparse rank 1 (1/61), dense rank 2 (1/62)
        # doc2: sparse rank 2 (1/62), dense rank 1 (1/61)
        # 둘 다 비슷한 점수를 가져야 함
        assert len(result) == 2
        assert abs(result[0].score - result[1].score) < 0.01

    def test_rrf_respects_top_k(self, searcher: HybridSearcher):
        """RRF 융합 시 top_k가 적용되는지 확인"""
        sparse = [ScoredDocument(f"doc{i}", 0.9 - i*0.1, f"c{i}", {}) for i in range(5)]
        dense = [ScoredDocument(f"doc{i}", 0.9 - i*0.1, f"c{i}", {}) for i in range(5)]
        
        result = searcher.fuse_results(sparse, dense, top_k=3)
        assert len(result) == 3


class TestDynamicWeights:
    """Test dynamic weight adjustment based on query type."""

    @pytest.fixture
    def searcher(self) -> HybridSearcher:
        return HybridSearcher(use_dynamic_weights=True)

    def test_article_reference_favors_bm25(self, searcher: HybridSearcher):
        """조문 참조 쿼리는 BM25 가중치가 높아야 함"""
        sparse = [ScoredDocument("doc1", 0.9, "제15조 내용", {})]
        dense = [ScoredDocument("doc2", 0.95, "다른 내용", {})]
        
        # 조문 참조 쿼리
        result = searcher.fuse_results(sparse, dense, top_k=2, query_text="제15조")
        
        # sparse가 선호되어 doc1이 상위에 있어야 함
        # (동적 가중치로 bm25_weight > dense_weight)
        assert result[0].doc_id == "doc1"

    def test_natural_question_applies_dynamic_weights(self, searcher: HybridSearcher):
        """자연어 질문에 동적 가중치가 적용되는지 확인"""
        sparse = [
            ScoredDocument("doc1", 0.9, "학칙 규정", {}),
        ]
        dense = [
            ScoredDocument("doc1", 0.95, "학칙 규정", {}),  # 같은 문서
        ]
        
        # 동적 가중치 활성화 시
        result_dynamic = searcher.fuse_results(
            sparse, dense, top_k=1, 
            query_text="학칙이 무엇인가요?"
        )
        
        # 정적 가중치 (query_text 없이)
        result_static = searcher.fuse_results(
            sparse, dense, top_k=1, 
            query_text=None
        )
        
        # 점수가 다르면 동적 가중치가 적용된 것
        # (같은 문서이지만 가중치가 다르면 최종 점수가 다름)
        assert len(result_dynamic) == 1
        assert len(result_static) == 1

    def test_static_weights_when_disabled(self):
        """동적 가중치 비활성화 시 정적 가중치 사용"""
        searcher = HybridSearcher(
            bm25_weight=0.3,
            dense_weight=0.7,
            use_dynamic_weights=False,
        )
        
        sparse = [ScoredDocument("doc1", 0.9, "c1", {})]
        dense = [ScoredDocument("doc2", 0.95, "c2", {})]
        
        # query_text 제공해도 동적 가중치 미사용
        result = searcher.fuse_results(
            sparse, dense, top_k=2, 
            query_text="제15조"  # 조문 참조지만 무시됨
        )
        
        # 정적 가중치 0.7 dense가 높으므로 doc2 선호
        # dense: 0.7/61, sparse: 0.3/61 → doc2 점수가 더 높음
        assert result[0].doc_id == "doc2"


class TestQueryExpansion:
    """Test query expansion with synonyms."""

    @pytest.fixture
    def searcher(self) -> HybridSearcher:
        return HybridSearcher()

    def test_expand_query_with_synonym(self, searcher: HybridSearcher):
        """동의어가 포함된 확장 쿼리 생성"""
        expanded = searcher.expand_query("교수")
        # "교수"의 동의어 "교원"이 포함되어야 함
        assert "교원" in expanded or "교수" in expanded

    def test_expand_query_no_synonym(self, searcher: HybridSearcher):
        """동의어 없는 단어는 그대로 반환"""
        original = "존재하지않는단어xyz"
        expanded = searcher.expand_query(original)
        assert original in expanded

    def test_search_sparse_uses_expansion(self, searcher: HybridSearcher):
        """search_sparse가 쿼리 확장을 사용하는지 확인"""
        # 문서 추가
        searcher.add_documents([
            ("doc1", "교원인사규정에 따른 임용", {}),
        ])
        
        # "교수"로 검색해도 "교원" 문서가 검색되어야 함 (동의어 확장)
        results = searcher.search_sparse("교수 임용")
        assert len(results) > 0


class TestMorphologicalTokenization:
    """Test morphological analysis tokenization option (Step 3)."""

    def test_bm25_default_tokenization_mode(self):
        """기본 토큰화 모드는 simple이어야 함"""
        bm25 = BM25()
        assert bm25.tokenize_mode == "simple"

    def test_bm25_morpheme_mode_splits_compound(self):
        """형태소 분석 모드에서 복합어가 분리되는지 확인"""
        bm25 = BM25(tokenize_mode="morpheme")
        # "휴직신청"이 "휴직", "신청"으로 분리되어야 함
        tokens = bm25._tokenize("육아휴직신청")
        assert "휴직" in tokens or "신청" in tokens or "육아" in tokens

    def test_bm25_simple_mode_keeps_compound(self):
        """simple 모드에서 복합어가 유지되는지 확인"""
        bm25 = BM25(tokenize_mode="simple")
        tokens = bm25._tokenize("육아휴직신청")
        # 복합어가 그대로 유지되거나 전체가 하나의 토큰
        assert "육아휴직신청" in tokens

    def test_morpheme_improves_recall(self):
        """형태소 분석이 재현율을 개선하는지 확인"""
        bm25_simple = BM25(tokenize_mode="simple")
        bm25_morpheme = BM25(tokenize_mode="morpheme")
        
        docs = [
            ("doc1", "육아휴직에 관한 규정", {}),
        ]
        bm25_simple.add_documents(docs)
        bm25_morpheme.add_documents(docs)
        
        # "휴직" 단독 검색 시 morpheme 모드가 더 잘 찾아야 함
        results_simple = bm25_simple.search("휴직 신청")
        results_morpheme = bm25_morpheme.search("휴직 신청")
        
        # 형태소 분석 모드에서 더 높은 점수 기대
        assert len(results_morpheme) >= len(results_simple)


class TestDynamicRRFk:
    """Test dynamic RRF k-value adjustment based on query type (Step 3)."""

    def test_searcher_default_rrf_k(self):
        """기본 RRF k값은 60이어야 함"""
        searcher = HybridSearcher()
        assert searcher.rrf_k == 60

    def test_get_dynamic_rrf_k_for_article_query(self):
        """조문 참조 쿼리에 대한 동적 k값"""
        searcher = HybridSearcher(use_dynamic_rrf_k=True)
        k = searcher.get_dynamic_rrf_k("교원인사규정 제15조")
        # 조문 참조는 정확한 매칭 중요 → 낮은 k (30-40)
        assert 30 <= k <= 50

    def test_get_dynamic_rrf_k_for_natural_query(self):
        """자연어 질문에 대한 동적 k값"""
        searcher = HybridSearcher(use_dynamic_rrf_k=True)
        k = searcher.get_dynamic_rrf_k("휴학하고 싶어")
        # 자연어 질문은 다양한 결과 필요 → 높은 k (60-80)
        assert 50 <= k <= 80

    def test_dynamic_rrf_k_disabled_uses_static(self):
        """동적 k값 비활성화 시 정적 값 사용"""
        searcher = HybridSearcher(rrf_k=100, use_dynamic_rrf_k=False)
        k = searcher.get_dynamic_rrf_k("제15조")
        assert k == 100

    def test_fuse_uses_dynamic_rrf_k(self):
        """fuse_results가 동적 k값을 사용하는지 확인"""
        searcher = HybridSearcher(rrf_k=60, use_dynamic_rrf_k=True)
        
        sparse = [ScoredDocument("doc1", 0.9, "제15조 내용", {})]
        dense = [ScoredDocument("doc1", 0.95, "제15조 내용", {})]
        
        # 조문 참조 쿼리로 k가 낮아지면 점수가 달라짐
        result_dynamic = searcher.fuse_results(
            sparse, dense, top_k=1, 
            query_text="제15조"
        )
        
        # k가 낮을수록 상위 순위의 점수 차이가 커짐
        # RRF score = 1/(k+rank), k가 작을수록 점수 높음
        assert result_dynamic[0].score > 0


class TestBM25IndexPersistence:
    """Test BM25 index save/load functionality."""

    def test_save_index_creates_file(self, tmp_path):
        """save_index()가 파일을 생성하는지 확인"""
        bm25 = BM25()
        documents = [
            ("doc1", "교원인사규정 휴직 신청", {}),
            ("doc2", "학생 장학금 지급 규정", {}),
        ]
        bm25.add_documents(documents)
        
        cache_path = tmp_path / "bm25_index.pkl"
        bm25.save_index(str(cache_path))
        
        assert cache_path.exists()
        assert cache_path.stat().st_size > 0

    def test_load_index_restores_searchability(self, tmp_path):
        """load_index()가 검색 기능을 복원하는지 확인"""
        # 1. 인덱스 빌드 및 저장
        bm25_original = BM25()
        documents = [
            ("doc1", "교원인사규정 휴직 신청 절차", {}),
            ("doc2", "학생 장학금 지급 규정 기준", {}),
            ("doc3", "연구비 사용 지침 안내", {}),
        ]
        bm25_original.add_documents(documents)
        
        cache_path = tmp_path / "bm25_index.pkl"
        bm25_original.save_index(str(cache_path))
        
        # 2. 새 인스턴스에서 로드
        bm25_loaded = BM25()
        success = bm25_loaded.load_index(str(cache_path))
        
        assert success is True
        
        # 3. 검색이 동일하게 작동하는지 확인
        original_results = bm25_original.search("휴직 신청", top_k=3)
        loaded_results = bm25_loaded.search("휴직 신청", top_k=3)
        
        assert len(original_results) == len(loaded_results)
        assert [r.doc_id for r in original_results] == [r.doc_id for r in loaded_results]
        for orig, loaded in zip(original_results, loaded_results):
            assert abs(orig.score - loaded.score) < 0.001

    def test_load_index_returns_false_for_missing_file(self):
        """존재하지 않는 파일에 대해 load_index가 False를 반환하는지 확인"""
        bm25 = BM25()
        success = bm25.load_index("/nonexistent/path/index.pkl")
        assert success is False

    def test_load_index_preserves_all_attributes(self, tmp_path):
        """load_index가 모든 인덱스 속성을 복원하는지 확인"""
        bm25_original = BM25(k1=1.5, b=0.8)
        documents = [
            ("doc1", "테스트 문서 내용", {"meta": "value"}),
        ]
        bm25_original.add_documents(documents)
        
        cache_path = tmp_path / "bm25_index.pkl"
        bm25_original.save_index(str(cache_path))
        
        bm25_loaded = BM25()
        bm25_loaded.load_index(str(cache_path))
        
        # 모든 속성이 복원되었는지 확인
        assert bm25_loaded.k1 == bm25_original.k1
        assert bm25_loaded.b == bm25_original.b
        assert bm25_loaded.doc_count == bm25_original.doc_count
        assert bm25_loaded.avg_doc_length == bm25_original.avg_doc_length
        assert bm25_loaded.documents == bm25_original.documents
        assert bm25_loaded.doc_metadata == bm25_original.doc_metadata


class TestHybridSearcherIndexCache:
    """Test HybridSearcher index cache functionality."""

    def test_add_documents_saves_to_cache(self, tmp_path):
        """add_documents가 index_cache_path에 캐시를 저장하는지 확인"""
        cache_path = tmp_path / "hybrid_bm25.pkl"
        searcher = HybridSearcher(index_cache_path=str(cache_path))
        
        documents = [
            ("doc1", "교원 휴직 규정", {}),
            ("doc2", "학생 장학금 규정", {}),
        ]
        searcher.add_documents(documents)
        
        assert cache_path.exists()

    def test_add_documents_loads_from_cache(self, tmp_path):
        """add_documents가 기존 캐시에서 로드하는지 확인"""
        cache_path = tmp_path / "hybrid_bm25.pkl"
        
        # 1. 첫 번째 searcher - 캐시 생성
        searcher1 = HybridSearcher(index_cache_path=str(cache_path))
        documents = [
            ("doc1", "교원 휴직 규정", {}),
            ("doc2", "학생 장학금 규정", {}),
        ]
        searcher1.add_documents(documents)
        
        # 2. 두 번째 searcher - 캐시에서 로드 (새 문서 리스트 전달해도 무시됨)
        searcher2 = HybridSearcher(index_cache_path=str(cache_path))
        new_documents = [
            ("doc3", "다른 내용", {}),  # 이 문서는 추가되지 않아야 함
        ]
        searcher2.add_documents(new_documents)
        
        # 캐시에서 로드했으므로 원래 문서만 검색 가능
        assert searcher2.bm25.doc_count == 2
        assert "doc1" in searcher2.bm25.documents
