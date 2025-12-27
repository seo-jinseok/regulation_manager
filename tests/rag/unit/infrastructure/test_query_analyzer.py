"""
Unit tests for QueryAnalyzer and dynamic hybrid search weights.
"""

import json

import pytest

from src.rag.infrastructure.hybrid_search import (
    QueryAnalyzer,
    QueryType,
    HybridSearcher,
    ScoredDocument,
)


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer query pattern detection."""

    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        return QueryAnalyzer()

    # --- Article Reference Detection ---

    def test_detects_article_number(self, analyzer: QueryAnalyzer):
        """제N조 패턴을 ARTICLE_REFERENCE로 분류"""
        assert analyzer.analyze("제15조") == QueryType.ARTICLE_REFERENCE
        assert analyzer.analyze("학칙 제3조") == QueryType.ARTICLE_REFERENCE

    def test_detects_article_with_subsection(self, analyzer: QueryAnalyzer):
        """제N조의N 패턴을 ARTICLE_REFERENCE로 분류"""
        assert analyzer.analyze("제3조의2") == QueryType.ARTICLE_REFERENCE
        assert analyzer.analyze("제 15 조 의 3") == QueryType.ARTICLE_REFERENCE

    def test_detects_paragraph_and_item(self, analyzer: QueryAnalyzer):
        """제N항, 제N호 패턴을 ARTICLE_REFERENCE로 분류"""
        assert analyzer.analyze("제2항") == QueryType.ARTICLE_REFERENCE
        assert analyzer.analyze("제1호") == QueryType.ARTICLE_REFERENCE

    # --- Regulation Name Detection ---

    def test_detects_regulation_name(self, analyzer: QueryAnalyzer):
        """OO규정 패턴을 REGULATION_NAME으로 분류"""
        assert analyzer.analyze("장학금규정") == QueryType.REGULATION_NAME
        assert analyzer.analyze("등록금환불규정") == QueryType.REGULATION_NAME

    def test_detects_school_rules(self, analyzer: QueryAnalyzer):
        """OO학칙 패턴을 REGULATION_NAME으로 분류"""
        assert analyzer.analyze("학칙") == QueryType.REGULATION_NAME
        assert analyzer.analyze("대학원학칙") == QueryType.REGULATION_NAME

    def test_detects_other_rule_types(self, analyzer: QueryAnalyzer):
        """내규, 세칙, 지침 패턴도 REGULATION_NAME으로 분류"""
        assert analyzer.analyze("인사내규") == QueryType.REGULATION_NAME
        assert analyzer.analyze("학위수여세칙") == QueryType.REGULATION_NAME
        assert analyzer.analyze("연구윤리지침") == QueryType.REGULATION_NAME

    # --- Natural Question Detection ---

    def test_detects_natural_question_markers(self, analyzer: QueryAnalyzer):
        """자연어 질문 마커를 NATURAL_QUESTION으로 분류"""
        assert analyzer.analyze("어떻게 해야 하나요") == QueryType.NATURAL_QUESTION
        assert analyzer.analyze("무엇인가요") == QueryType.NATURAL_QUESTION
        assert analyzer.analyze("왜 그런가요") == QueryType.NATURAL_QUESTION

    def test_detects_question_mark(self, analyzer: QueryAnalyzer):
        """물음표를 NATURAL_QUESTION으로 분류 (단, 학사 키워드가 없는 경우)"""
        # 학사 키워드가 포함된 경우 REGULATION_NAME이 우선
        assert analyzer.analyze("휴학하려면?") == QueryType.REGULATION_NAME
        # 학사 키워드 없으면 물음표로 NATURAL_QUESTION
        assert analyzer.analyze("이게 뭔가요?") == QueryType.NATURAL_QUESTION

    # --- Intent Detection ---

    def test_detects_intent_queries(self, analyzer: QueryAnalyzer):
        """의도 표현은 INTENT로 분류"""
        assert analyzer.analyze("학교에 가기 싫어") == QueryType.INTENT
        assert analyzer.analyze("그만두고 싶어") == QueryType.INTENT

    # --- Academic Keywords Detection ---

    def test_detects_academic_keywords(self, analyzer: QueryAnalyzer):
        """학사 키워드는 REGULATION_NAME으로 분류"""
        assert analyzer.analyze("휴학") == QueryType.REGULATION_NAME
        assert analyzer.analyze("복학 절차") == QueryType.REGULATION_NAME
        assert analyzer.analyze("장학금 신청") == QueryType.REGULATION_NAME

    # --- General Queries ---

    def test_general_query(self, analyzer: QueryAnalyzer):
        """특정 패턴 없으면 GENERAL로 분류"""
        assert analyzer.analyze("일반 검색어") == QueryType.GENERAL
        assert analyzer.analyze("도서관 이용") == QueryType.GENERAL

    # --- Priority: Article > Question > Regulation ---

    def test_article_takes_priority_over_question(self, analyzer: QueryAnalyzer):
        """조문 번호가 질문 마커보다 우선"""
        # Contains both article reference and question mark
        assert analyzer.analyze("학칙 제15조가 뭔가요?") == QueryType.ARTICLE_REFERENCE

    # --- Weight Calculation ---

    def test_get_weights_article(self, analyzer: QueryAnalyzer):
        """ARTICLE_REFERENCE는 BM25 가중치가 높아야 함"""
        bm25_w, dense_w = analyzer.get_weights("제15조")
        assert bm25_w == 0.6
        assert dense_w == 0.4

    def test_get_weights_regulation(self, analyzer: QueryAnalyzer):
        """REGULATION_NAME은 균형 잡힌 가중치 (동의어 없을 때)"""
        # "인사규정"은 동의어가 없어서 기본 가중치 적용
        bm25_w, dense_w = analyzer.get_weights("인사규정")
        assert bm25_w == 0.5
        assert dense_w == 0.5

    def test_get_weights_question(self, analyzer: QueryAnalyzer):
        """NATURAL_QUESTION은 약간 Dense 가중치가 높음 (0.4, 0.6)"""
        bm25_w, dense_w = analyzer.get_weights("이건 뭔가요?")
        assert bm25_w == 0.4
        assert dense_w == 0.6

    def test_get_weights_intent(self, analyzer: QueryAnalyzer):
        """INTENT는 Dense 가중치가 더 높음"""
        bm25_w, dense_w = analyzer.get_weights("학교에 가기 싫어")
        assert bm25_w < dense_w

    def test_get_weights_general(self, analyzer: QueryAnalyzer):
        """GENERAL은 균형 가중치 (0.5, 0.5)"""
        bm25_w, dense_w = analyzer.get_weights("일반 검색어")
        assert bm25_w == 0.5
        assert dense_w == 0.5


class TestHybridSearcherDynamicWeights:
    """Tests for HybridSearcher dynamic weight integration."""

    def _make_doc(self, doc_id: str, score: float) -> ScoredDocument:
        return ScoredDocument(doc_id=doc_id, score=score, content="test", metadata={})

    def test_dynamic_weights_enabled_by_default(self):
        """동적 가중치가 기본적으로 활성화"""
        searcher = HybridSearcher()
        assert searcher.use_dynamic_weights is True

    def test_fuse_with_dynamic_weights(self):
        """query_text 제공 시 동적 가중치 적용"""
        searcher = HybridSearcher(use_dynamic_weights=True)

        sparse = [self._make_doc("doc1", 0.9)]
        dense = [self._make_doc("doc2", 0.9)]

        # Article query → BM25 weighted higher (0.6)
        results = searcher.fuse_results(sparse, dense, query_text="제15조")

        # doc1 (sparse) should rank higher due to BM25 weight 0.6 > 0.4
        assert results[0].doc_id == "doc1"

    def test_fuse_without_query_uses_static_weights(self):
        """query_text 미제공 시 고정 가중치 사용"""
        searcher = HybridSearcher(bm25_weight=0.3, dense_weight=0.7)

        sparse = [self._make_doc("doc1", 0.9)]
        dense = [self._make_doc("doc2", 0.9)]

        # No query_text → static weights (0.3, 0.7)
        results = searcher.fuse_results(sparse, dense)

        # doc2 (dense) should rank higher due to dense weight 0.7 > 0.3
        assert results[0].doc_id == "doc2"

    def test_dynamic_weights_disabled(self):
        """use_dynamic_weights=False면 항상 고정 가중치"""
        searcher = HybridSearcher(
            bm25_weight=0.3, dense_weight=0.7, use_dynamic_weights=False
        )

        sparse = [self._make_doc("doc1", 0.9)]
        dense = [self._make_doc("doc2", 0.9)]

        # Even with article query, uses static weights
        results = searcher.fuse_results(sparse, dense, query_text="제15조")

        # doc2 should rank higher (static 0.7 > 0.3)
        assert results[0].doc_id == "doc2"


class TestQueryRewriting:
    """LLM 기반 쿼리 리라이팅 테스트."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client for testing."""
        from unittest.mock import Mock
        from src.rag.domain.repositories import ILLMClient
        
        mock = Mock(spec=ILLMClient)
        return mock

    def test_rewrite_with_llm(self, mock_llm):
        """LLM을 사용한 쿼리 리라이팅 테스트."""
        mock_llm.generate.return_value = "휴직 휴가 연구년 안식년"
        
        analyzer = QueryAnalyzer(llm_client=mock_llm)
        result = analyzer.rewrite_query("학교에 가기 싫어")
        
        assert "휴직" in result
        assert "연구년" in result

    def test_rewrite_info_reports_llm_source(self, mock_llm):
        """리라이팅 정보에 LLM 사용 여부가 포함됨."""
        mock_llm.generate.return_value = "휴직 휴가"

        analyzer = QueryAnalyzer(llm_client=mock_llm)
        info = analyzer.rewrite_query_with_info("학교에 가기 싫어")

        assert info.used_llm is True
        assert info.method == "llm"
        assert info.from_cache is False
        assert info.fallback is False

    def test_rewrite_info_includes_matched_intents(self, tmp_path, mock_llm):
        """리라이팅 정보에 매칭된 의도 레이블 포함."""
        mock_llm.generate.return_value = "휴직"
        data = {
            "intents": [
                {
                    "id": "work_avoid",
                    "label": "근무 회피",
                    "triggers": ["야근 싫어"],
                    "keywords": ["휴직"],
                }
            ]
        }
        path = tmp_path / "intents.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        analyzer = QueryAnalyzer(llm_client=mock_llm, intents_path=str(path))
        info = analyzer.rewrite_query_with_info("야근 싫어")

        assert "근무 회피" in info.matched_intents

    def test_rewrite_without_llm_uses_expand_query(self):
        """LLM 미설정 시 기존 expand_query 사용."""
        analyzer = QueryAnalyzer()  # LLM 없음
        result = analyzer.rewrite_query("휴학")
        
        # 기존 동의어 확장 결과 반환
        assert "휴학" in result
        assert "휴학 신청" in result  # 동의어 확장됨

    def test_rewrite_info_reports_rules_source(self):
        """LLM 미설정 시 규칙 기반으로 표시됨."""
        analyzer = QueryAnalyzer()
        info = analyzer.rewrite_query_with_info("휴학")

        assert info.used_llm is False
        assert info.method == "rules"

    def test_expand_query_adds_professor_synonyms(self):
        """교수 키워드는 교원/교직원으로 확장."""
        analyzer = QueryAnalyzer()
        result = analyzer.expand_query("교수")

        assert "교원" in result
        assert "교직원" in result

    def test_expand_query_adds_intent_keywords(self):
        """자연어 의도 표현은 관련 키워드로 보강."""
        analyzer = QueryAnalyzer()
        result = analyzer.expand_query("나는 교수인데 학교에 가기 싫어")

        assert "휴직" in result
        assert "연구년" in result

    def test_external_intents_loaded(self, tmp_path):
        """외부 의도 파일을 로드해 확장에 반영."""
        data = {
            "intents": [
                {
                    "id": "work_avoid",
                    "label": "근무 회피",
                    "triggers": ["야근 싫어"],
                    "keywords": ["휴직", "휴가"],
                }
            ]
        }
        path = tmp_path / "intents.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        analyzer = QueryAnalyzer(intents_path=str(path))
        result = analyzer.expand_query("야근 싫어")

        assert "휴직" in result
        assert analyzer.analyze("야근 싫어") == QueryType.INTENT

    def test_external_synonyms_loaded(self, tmp_path):
        """외부 동의어 파일을 로드해 확장에 반영."""
        data = {
            "terms": {
                "복학": ["학교로 돌아가기", "복학 신청"],
            }
        }
        path = tmp_path / "synonyms.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        analyzer = QueryAnalyzer(synonyms_path=str(path))
        result = analyzer.expand_query("복학")

        assert "학교로 돌아가기" in result
        assert "복학 신청" in result

    def test_external_synonyms_merge_with_builtin(self, tmp_path):
        """외부 동의어가 기본 사전과 병합됨."""
        data = {
            "terms": {
                "교수": ["교수님"],
            }
        }
        path = tmp_path / "synonyms.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        analyzer = QueryAnalyzer(synonyms_path=str(path))
        result = analyzer.expand_query("교수")

        assert "교원" in result
        assert "교수님" in result

    def test_rewrite_fallback_on_llm_failure(self, mock_llm):
        """LLM 실패 시 기존 방식으로 폴백."""
        mock_llm.generate.side_effect = Exception("LLM connection error")
        
        analyzer = QueryAnalyzer(llm_client=mock_llm)
        info = analyzer.rewrite_query_with_info("폐과")
        
        # 기존 동의어 확장 결과 반환
        assert "폐과" in info.rewritten or "학과 폐지" in info.rewritten
        assert info.used_llm is False
        assert info.fallback is True

    def test_rewrite_caching(self, mock_llm):
        """동일 쿼리 캐싱 확인."""
        mock_llm.generate.return_value = "휴직 휴가"
        
        analyzer = QueryAnalyzer(llm_client=mock_llm)
        
        # 첫 번째 호출
        result1 = analyzer.rewrite_query_with_info("학교에 가기 싫어")
        # 두 번째 호출 (캐시 히트)
        result2 = analyzer.rewrite_query_with_info("학교에 가기 싫어")
        
        # LLM은 한 번만 호출되어야 함
        assert mock_llm.generate.call_count == 1
        assert result1.rewritten == result2.rewritten
        assert result2.from_cache is True

    def test_rewrite_merges_intent_keywords_with_llm(self, mock_llm):
        """의도 키워드는 LLM 출력과 병합되어 유지."""
        mock_llm.generate.return_value = "교원 복무"

        analyzer = QueryAnalyzer(llm_client=mock_llm)
        result = analyzer.rewrite_query("나는 교수인데 학교에 가기 싫어")

        assert "휴직" in result
        assert "연구년" in result

    def test_rewrite_clears_cache_for_different_queries(self, mock_llm):
        """다른 쿼리는 캐시 미스."""
        mock_llm.generate.side_effect = ["휴직 휴가", "퇴직 사직"]
        
        analyzer = QueryAnalyzer(llm_client=mock_llm)
        
        result1 = analyzer.rewrite_query("학교에 가기 싫어")
        result2 = analyzer.rewrite_query("그만두고 싶어")
        
        # 각각 LLM 호출
        assert mock_llm.generate.call_count == 2
        assert "휴직" in result1
        assert "퇴직" in result2
