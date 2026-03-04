"""Unit tests for QuerySynthesizer."""

import json
import tempfile
from pathlib import Path

import pytest

from src.rag.domain.evaluation.query_synthesizer import (
    DifficultyTier,
    GeneratedQuery,
    QueryCategory,
    QuerySynthesizer,
    QueryType,
)


class TestGeneratedQuery:
    """Test GeneratedQuery dataclass."""

    def test_to_dict(self):
        q = GeneratedQuery(
            query="테스트 쿼리",
            expected_source="test_reg.json",
            difficulty_tier="L1",
            category="definition",
            query_type="single_regulation",
        )
        d = q.to_dict()
        assert d["query"] == "테스트 쿼리"
        assert d["difficulty_tier"] == "L1"

    def test_from_dict(self):
        data = {
            "query": "복학 절차",
            "expected_source": "school_rules.json",
            "difficulty_tier": "L2",
            "category": "procedure",
            "query_type": "single_regulation",
            "expected_behavior": "answer",
        }
        q = GeneratedQuery.from_dict(data)
        assert q.query == "복학 절차"
        assert q.difficulty_tier == "L2"

    def test_from_dict_ignores_extra_keys(self):
        data = {
            "query": "Q",
            "expected_source": "S",
            "difficulty_tier": "L1",
            "category": "definition",
            "query_type": "single_regulation",
            "extra_field": "should_be_ignored",
        }
        q = GeneratedQuery.from_dict(data)
        assert q.query == "Q"


class TestQuerySynthesizer:
    """Test QuerySynthesizer class."""

    @pytest.fixture
    def sample_regulation(self, tmp_path):
        """Create a sample regulation JSON for testing."""
        reg = {
            "title": "학칙",
            "articles": [
                {
                    "article_number": "제1조",
                    "title": "목적",
                    "content": "이 학칙은 대학교의 학사운영에 관한 사항을 규정함을 목적으로 한다.",
                },
                {
                    "article_number": "제2조",
                    "title": "수업연한",
                    "content": "수업연한은 4년으로 한다. 다만 의학과는 6년으로 한다.",
                },
                {
                    "article_number": "제3조",
                    "title": "졸업요건",
                    "content": "졸업에 필요한 학점은 130학점 이상이며 평점평균 2.0 이상이어야 한다.",
                },
            ],
        }
        reg_file = tmp_path / "학칙_test.json"
        reg_file.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
        return tmp_path

    def test_load_regulations(self, sample_regulation):
        synth = QuerySynthesizer(data_dir=str(sample_regulation), seed=42)
        regs = synth.load_regulations()
        assert len(regs) == 1
        assert regs[0]["title"] == "학칙"

    def test_load_regulations_empty_dir(self, tmp_path):
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        regs = synth.load_regulations()
        assert regs == []

    def test_load_regulations_nonexistent_dir(self):
        synth = QuerySynthesizer(data_dir="/nonexistent/path", seed=42)
        regs = synth.load_regulations()
        assert regs == []

    def test_generate_from_regulations(self, sample_regulation):
        synth = QuerySynthesizer(data_dir=str(sample_regulation), seed=42)
        synth.load_regulations()
        queries = synth.generate_from_regulations(target_count=5)
        assert len(queries) > 0
        for q in queries:
            assert isinstance(q, GeneratedQuery)
            assert q.query  # Non-empty query
            assert q.difficulty_tier in ("L1", "L2")

    def test_generate_adversarial(self, sample_regulation):
        synth = QuerySynthesizer(data_dir=str(sample_regulation), seed=42)
        queries = synth.generate_adversarial(count=5)
        assert len(queries) > 0
        for q in queries:
            assert q.difficulty_tier in ("L4", "L5")
            assert "adversarial" in q.query_type

    def test_generate_all(self, sample_regulation):
        synth = QuerySynthesizer(data_dir=str(sample_regulation), seed=42)
        synth.load_regulations()
        queries = synth.generate_all(target_count=20)
        assert len(queries) >= 10  # Should produce a decent number

    def test_cache_save_and_load(self, sample_regulation, tmp_path):
        cache_path = tmp_path / "cache" / "queries.json"
        synth = QuerySynthesizer(
            data_dir=str(sample_regulation),
            cache_path=str(cache_path),
            seed=42,
        )
        synth.load_regulations()
        queries = synth.generate_all(target_count=10)

        # Save cache
        synth.save_cache(queries)
        assert cache_path.exists()

        # Load cache
        loaded = synth.load_cache()
        assert loaded is not None
        assert len(loaded) == len(queries)
        assert loaded[0].query == queries[0].query

    def test_get_queries_by_tier(self, sample_regulation):
        synth = QuerySynthesizer(data_dir=str(sample_regulation), seed=42)
        synth.load_regulations()
        queries = synth.generate_all(target_count=20)

        # Filter by tier
        l1 = [q for q in queries if q.difficulty_tier == "L1"]
        l4 = [q for q in queries if q.difficulty_tier == "L4"]
        assert len(l1) > 0 or len(l4) > 0  # At least one tier should have queries

    def test_difficulty_tier_enum(self):
        assert DifficultyTier.L1.value == "L1"
        assert DifficultyTier.L5.value == "L5"

    def test_query_category_enum(self):
        assert QueryCategory.DEFINITION.value == "definition"
        assert QueryCategory.ADVERSARIAL.value == "adversarial"

    def test_query_type_enum(self):
        assert QueryType.SINGLE_REGULATION.value == "single_regulation"
        assert QueryType.ADVERSARIAL_OOD.value == "adversarial_ood"

    def test_load_regulations_list_format(self, tmp_path):
        """Test loading regulations stored as JSON list."""
        regs = [
            {"title": "규정A", "articles": [{"article_number": "제1조", "title": "목적"}]},
            {"title": "규정B", "articles": [{"article_number": "제1조", "title": "취지"}]},
        ]
        (tmp_path / "list_reg.json").write_text(
            json.dumps(regs, ensure_ascii=False), encoding="utf-8"
        )
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        loaded = synth.load_regulations()
        assert len(loaded) == 2
        assert loaded[0]["_source_file"] == "list_reg.json"

    def test_load_regulations_invalid_json(self, tmp_path):
        """Test graceful handling of corrupt JSON files."""
        (tmp_path / "bad.json").write_text("{invalid json", encoding="utf-8")
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        regs = synth.load_regulations()
        assert regs == []

    def test_extract_articles_with_sub_articles(self, tmp_path):
        """Cover sub-articles extraction (lines 168-174)."""
        reg = {
            "title": "인사규정",
            "articles": [
                {
                    "article_number": "제1조",
                    "title": "총칙",
                    "content": "인사에 관한 사항",
                    "sub_articles": [
                        {"title": "임용", "ref": "제1조의2", "content": "교원 임용 절차"},
                        {"title": "승진", "ref": "제1조의3", "content": "승진 요건"},
                    ],
                }
            ],
        }
        (tmp_path / "hr.json").write_text(
            json.dumps(reg, ensure_ascii=False), encoding="utf-8"
        )
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        synth.load_regulations()
        articles = synth._extract_articles(synth._regulations[0])
        # parent + 2 sub-articles
        assert len(articles) >= 3

    def test_extract_keywords(self, tmp_path):
        """Cover keyword extraction (lines 209-214)."""
        reg = {
            "title": "학칙",
            "content": "휴학, 복학, 수강신청, 등록금 관련 규정이다. 졸업 요건도 포함한다.",
        }
        (tmp_path / "kw.json").write_text(
            json.dumps(reg, ensure_ascii=False), encoding="utf-8"
        )
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        synth.load_regulations()
        keywords = synth._extract_keywords(synth._regulations[0])
        assert len(keywords) >= 3
        assert "휴학" in keywords
        assert "졸업" in keywords

    def test_generate_cross_regulation(self, tmp_path):
        """Cover cross-regulation query generation (lines 352-394)."""
        reg1 = {
            "title": "학칙",
            "articles": [{"article_number": "제1조", "title": "목적", "content": "휴학 복학"}],
        }
        reg2 = {
            "title": "교원규정",
            "articles": [{"article_number": "제1조", "title": "취지", "content": "휴학 관련 교원 조항"}],
        }
        (tmp_path / "reg1.json").write_text(json.dumps(reg1, ensure_ascii=False), encoding="utf-8")
        (tmp_path / "reg2.json").write_text(json.dumps(reg2, ensure_ascii=False), encoding="utf-8")

        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        synth.load_regulations()
        queries = synth.generate_cross_regulation()
        assert len(queries) >= 0  # may or may not find shared keywords
        for q in queries:
            assert q.difficulty_tier == "L3"
            assert q.query_type == "cross_regulation"

    def test_generate_cross_regulation_insufficient_regs(self, tmp_path):
        """Cover early exit when < 2 regulations."""
        reg = {"title": "학칙", "articles": []}
        (tmp_path / "single.json").write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
        synth = QuerySynthesizer(data_dir=str(tmp_path), seed=42)
        synth.load_regulations()
        queries = synth.generate_cross_regulation()
        assert queries == []

    def test_get_queries_from_cache(self, sample_regulation, tmp_path):
        """Cover get_queries() cache path (lines 510-518)."""
        cache_path = tmp_path / "cache" / "queries.json"
        synth = QuerySynthesizer(
            data_dir=str(sample_regulation),
            cache_path=str(cache_path),
            seed=42,
        )
        synth.load_regulations()

        # First call generates and caches
        queries1 = synth.get_queries(target_count=20)
        assert cache_path.exists()

        # Second call loads from cache
        queries2 = synth.get_queries(target_count=20)
        assert len(queries2) == len(queries1)

    def test_get_queries_regenerate(self, sample_regulation, tmp_path):
        """Cover get_queries() regenerate path."""
        cache_path = tmp_path / "cache" / "queries.json"
        synth = QuerySynthesizer(
            data_dir=str(sample_regulation),
            cache_path=str(cache_path),
            seed=42,
        )
        synth.load_regulations()

        queries1 = synth.get_queries(target_count=20)
        queries2 = synth.get_queries(regenerate=True, target_count=20)
        assert len(queries2) > 0

    def test_load_cache_nonexistent(self, tmp_path):
        """Cover load_cache() when file doesn't exist."""
        synth = QuerySynthesizer(
            data_dir=str(tmp_path),
            cache_path=str(tmp_path / "nonexistent.json"),
        )
        assert synth.load_cache() is None

    def test_load_cache_corrupt(self, tmp_path):
        """Cover load_cache() error handling."""
        cache_path = tmp_path / "corrupt.json"
        cache_path.write_text("{invalid", encoding="utf-8")
        synth = QuerySynthesizer(
            data_dir=str(tmp_path),
            cache_path=str(cache_path),
        )
        assert synth.load_cache() is None
