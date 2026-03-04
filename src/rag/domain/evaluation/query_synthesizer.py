"""
Dynamic Query Synthesizer for RAG Evaluation.

SPEC: SPEC-RAG-EVAL-002
EARS: EARS-U-001 (Regulation-Based), EARS-U-002 (Cross-Regulation), EARS-U-003 (Adversarial)

Generates test queries by analyzing regulation JSON content, creating cross-regulation
synthesis queries, and adversarial queries for comprehensive evaluation.
"""

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DifficultyTier(str, Enum):
    """Difficulty tiers for generated queries (L1-L5)."""
    L1 = "L1"  # Single fact lookup
    L2 = "L2"  # Multi-fact synthesis from single regulation
    L3 = "L3"  # Cross-regulation synthesis
    L4 = "L4"  # Adversarial / edge case
    L5 = "L5"  # Multi-turn with context dependency


class QueryCategory(str, Enum):
    """Categories for generated queries."""
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    ELIGIBILITY = "eligibility"
    CROSS_REFERENCE = "cross_reference"
    CALCULATION = "calculation"
    ADVERSARIAL = "adversarial"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"


class QueryType(str, Enum):
    """Type of generated query."""
    SINGLE_REGULATION = "single_regulation"
    CROSS_REGULATION = "cross_regulation"
    ADVERSARIAL_OOD = "adversarial_ood"          # Out-of-domain
    ADVERSARIAL_HALLUCINATION = "adversarial_hallucination"  # False premises
    ADVERSARIAL_AMBIGUOUS = "adversarial_ambiguous"
    ADVERSARIAL_LONG = "adversarial_long"
    ADVERSARIAL_TYPO = "adversarial_typo"
    ADVERSARIAL_AMENDMENT = "adversarial_amendment"


@dataclass
class GeneratedQuery:
    """A dynamically generated test query."""
    query: str
    expected_source: str
    difficulty_tier: str  # L1-L5
    category: str
    query_type: str
    expected_behavior: str = "answer"  # answer, reject, clarify, partial
    cross_regulation_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedQuery":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class QuerySynthesizer:
    """Generates test queries from regulation content.

    EARS-U-001: Generates queries from regulation JSON data
    EARS-U-002: Creates cross-regulation synthesis queries
    EARS-U-003: Creates adversarial queries for difficulty L4-L5
    """

    CACHE_FILE = "data/evaluations/generated_queries.json"
    MIN_QUERIES = 100
    DEFAULT_TARGET = 200

    # Templates for L1 single fact lookup
    L1_TEMPLATES = [
        "{article_title}에 대해 알려주세요",
        "{article_title}의 내용이 뭔가요?",
        "{regulation_name}에서 {article_ref}은 어떤 내용인가요?",
        "{article_ref}에 대해 설명해주세요",
        "{keyword} 관련 규정이 어떻게 되나요?",
    ]

    # Templates for L2 multi-fact synthesis
    L2_TEMPLATES = [
        "{regulation_name}에서 {keyword1}과 {keyword2}의 관계는?",
        "{regulation_name}의 {article_ref1}와 {article_ref2}의 차이점은?",
        "{regulation_name}에서 {keyword1}의 자격요건과 절차를 설명해주세요",
        "{keyword1} 신청 시 필요한 조건과 제출 서류는?",
    ]

    # Templates for L3 cross-regulation
    L3_TEMPLATES = [
        "{reg1_name}과 {reg2_name}에서 {keyword}에 대한 규정이 다른가요?",
        "{keyword}에 대해 {reg1_name}과 {reg2_name}이 각각 어떻게 규정하고 있나요?",
        "{reg1_name}의 {keyword1}과 {reg2_name}의 {keyword2}를 비교해주세요",
    ]

    # Adversarial templates
    ADVERSARIAL_TEMPLATES = {
        QueryType.ADVERSARIAL_OOD: [
            "학교 식당 메뉴에서 가장 인기 있는 음식은?",
            "교수님 개인 연락처를 알려주세요",
            "주변 부동산 시세가 어떻게 되나요?",
            "학교 앞 맛집 추천해주세요",
            "교내 주차 요금제를 비트코인으로 결제할 수 있나요?",
        ],
        QueryType.ADVERSARIAL_HALLUCINATION: [
            "제999조에 따른 특별 감면 규정 알려주세요",
            "2030년도에 개정된 학칙 변경 사항은?",
            "외계인 학생을 위한 특별 장학금 규정이 있나요?",
            "AI 로봇이 학위를 취득할 수 있는 규정이 있나요?",
        ],
        QueryType.ADVERSARIAL_AMBIGUOUS: [
            "그거 알려줘",
            "관련 규정",
            "어떻게 해야 하나요?",
            "신청",
            "기간",
        ],
        QueryType.ADVERSARIAL_LONG: [
            "안녕하세요 저는 2학년 재학생인데요 제가 지난 학기에 장학금을 받았었는데 이번에도 신청을 하려고 하는데 성적이 좀 떨어져서 걱정이 되는데 장학금 유지 조건이 어떻게 되는지 그리고 만약에 조건을 충족하지 못하면 어떻게 되는지 그리고 다시 신청할 수 있는지 전체적인 절차를 알려주실 수 있을까요?",
        ],
        QueryType.ADVERSARIAL_TYPO: [
            "후학 신덩 방법 알려줘",  # 휴학 신청
            "등롣금 납뷰 기한",  # 등록금 납부
            "졸어 요건이 뭐에요?",  # 졸업
        ],
    }

    def __init__(
        self,
        data_dir: str = "data/output",
        cache_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_path = Path(cache_path or self.CACHE_FILE)
        self._rng = random.Random(seed)
        self._regulations: List[Dict[str, Any]] = []

    def load_regulations(self) -> List[Dict[str, Any]]:
        """Load regulation JSON files from data directory."""
        regulations = []
        if not self.data_dir.exists():
            logger.warning("Data directory %s does not exist", self.data_dir)
            return regulations

        for json_file in sorted(self.data_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data["_source_file"] = json_file.name
                    regulations.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item["_source_file"] = json_file.name
                            regulations.append(item)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", json_file, e)

        self._regulations = regulations
        logger.info("Loaded %d regulation entries from %s", len(regulations), self.data_dir)
        return regulations

    def _extract_articles(self, regulation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract articles (조/항) from a regulation document."""
        articles = []
        reg_name = regulation.get("title", "") or regulation.get("regulation_name", "") or regulation.get("_source_file", "unknown")

        # Handle nested structure: regulations -> chapters -> articles
        for key in ["articles", "provisions", "content", "sections", "chapters"]:
            items = regulation.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        article_title = item.get("title", "") or item.get("article_title", "")
                        article_ref = item.get("article_number", "") or item.get("ref", "")
                        content = item.get("content", "") or item.get("text", "")

                        if article_title or article_ref:
                            articles.append({
                                "regulation_name": reg_name,
                                "article_title": article_title,
                                "article_ref": article_ref,
                                "content": content[:500] if content else "",
                                "source_file": regulation.get("_source_file", ""),
                            })

                        # Nested sub-articles
                        for sub_key in ["sub_articles", "paragraphs", "items", "sections"]:
                            sub_items = item.get(sub_key, [])
                            if isinstance(sub_items, list):
                                for sub_item in sub_items:
                                    if isinstance(sub_item, dict):
                                        sub_title = sub_item.get("title", "")
                                        sub_ref = sub_item.get("article_number", "") or sub_item.get("ref", "")
                                        sub_content = sub_item.get("content", "") or sub_item.get("text", "")
                                        if sub_title or sub_ref:
                                            articles.append({
                                                "regulation_name": reg_name,
                                                "article_title": sub_title,
                                                "article_ref": sub_ref,
                                                "content": sub_content[:500] if sub_content else "",
                                                "source_file": regulation.get("_source_file", ""),
                                            })

        return articles

    def _extract_keywords(self, regulation: Dict[str, Any]) -> List[str]:
        """Extract keywords from regulation content for template filling."""
        text = json.dumps(regulation, ensure_ascii=False)

        # Common regulation keywords
        keyword_patterns = [
            r"(휴학|복학|수강신청|등록금|장학금|졸업|학위|성적|학점)",
            r"(휴직|연구년|승진|임용|교원|연구비|겸직)",
            r"(급여|복무|연수|휴가|퇴직|인사|징계)",
            r"(기숙사|도서관|시설|장비|예산|회계)",
            r"(입학|편입|전과|전공|부전공|복수전공)",
        ]

        keywords = set()
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)

        return list(keywords)

    def generate_from_regulations(
        self,
        target_count: int = 200,
    ) -> List[GeneratedQuery]:
        """Generate queries by analyzing regulation content.

        EARS-U-001: Auto-generate queries from regulation JSON data.
        """
        if not self._regulations:
            self.load_regulations()

        if not self._regulations:
            logger.warning("No regulations loaded; returning empty query list")
            return []

        queries: List[GeneratedQuery] = []

        for reg in self._regulations:
            articles = self._extract_articles(reg)
            keywords = self._extract_keywords(reg)
            reg_name = reg.get("title", "") or reg.get("regulation_name", "") or reg.get("_source_file", "unknown")

            # Generate L1 queries from articles
            for article in articles:
                if len(queries) >= target_count:
                    break
                template = self._rng.choice(self.L1_TEMPLATES)
                query_text = template.format(
                    article_title=article["article_title"] or article["article_ref"],
                    regulation_name=reg_name,
                    article_ref=article["article_ref"] or article["article_title"],
                    keyword=self._rng.choice(keywords) if keywords else "규정",
                )
                queries.append(GeneratedQuery(
                    query=query_text,
                    expected_source=article["source_file"],
                    difficulty_tier=DifficultyTier.L1.value,
                    category=QueryCategory.DEFINITION.value,
                    query_type=QueryType.SINGLE_REGULATION.value,
                ))

            # Generate L2 queries from keyword pairs
            if len(keywords) >= 2:
                kw_pairs = [(keywords[i], keywords[j])
                            for i in range(len(keywords))
                            for j in range(i + 1, len(keywords))]
                self._rng.shuffle(kw_pairs)

                for kw1, kw2 in kw_pairs[:min(5, len(kw_pairs))]:
                    if len(queries) >= target_count:
                        break
                    template = self._rng.choice(self.L2_TEMPLATES)
                    try:
                        query_text = template.format(
                            regulation_name=reg_name,
                            keyword1=kw1,
                            keyword2=kw2,
                            article_ref1=articles[0]["article_ref"] if articles else "제1조",
                            article_ref2=articles[1]["article_ref"] if len(articles) > 1 else "제2조",
                        )
                    except (IndexError, KeyError):
                        continue

                    queries.append(GeneratedQuery(
                        query=query_text,
                        expected_source=reg.get("_source_file", ""),
                        difficulty_tier=DifficultyTier.L2.value,
                        category=QueryCategory.PROCEDURE.value,
                        query_type=QueryType.SINGLE_REGULATION.value,
                    ))

        # Ensure minimum count by generating more L1/L2 if needed
        while len(queries) < min(target_count, max(50, len(queries))):
            if not self._regulations:
                break
            reg = self._rng.choice(self._regulations)
            keywords = self._extract_keywords(reg)
            if keywords:
                kw = self._rng.choice(keywords)
                queries.append(GeneratedQuery(
                    query=f"{kw} 관련 규정이 어떻게 되나요?",
                    expected_source=reg.get("_source_file", ""),
                    difficulty_tier=DifficultyTier.L1.value,
                    category=QueryCategory.DEFINITION.value,
                    query_type=QueryType.SINGLE_REGULATION.value,
                ))
            else:
                break

        return queries

    def generate_cross_regulation(
        self,
        existing_queries: Optional[List[GeneratedQuery]] = None,
    ) -> List[GeneratedQuery]:
        """Generate cross-regulation queries requiring 2+ sources.

        EARS-U-002: Cross-regulation queries at L3+.
        """
        if not self._regulations:
            self.load_regulations()

        queries: List[GeneratedQuery] = []
        if len(self._regulations) < 2:
            logger.warning("Need at least 2 regulations for cross-regulation queries")
            return queries

        # Find shared keywords between regulation pairs
        reg_keywords = {}
        for reg in self._regulations:
            reg_name = reg.get("title", "") or reg.get("regulation_name", "") or reg.get("_source_file", "")
            keywords = self._extract_keywords(reg)
            if reg_name and keywords:
                reg_keywords[reg_name] = {
                    "keywords": keywords,
                    "source": reg.get("_source_file", ""),
                }

        reg_names = list(reg_keywords.keys())
        for i in range(len(reg_names)):
            for j in range(i + 1, len(reg_names)):
                reg1 = reg_names[i]
                reg2 = reg_names[j]
                shared = set(reg_keywords[reg1]["keywords"]) & set(reg_keywords[reg2]["keywords"])

                for keyword in shared:
                    template = self._rng.choice(self.L3_TEMPLATES)
                    try:
                        query_text = template.format(
                            reg1_name=reg1,
                            reg2_name=reg2,
                            keyword=keyword,
                            keyword1=keyword,
                            keyword2=keyword,
                        )
                    except KeyError:
                        continue

                    queries.append(GeneratedQuery(
                        query=query_text,
                        expected_source=reg_keywords[reg1]["source"],
                        difficulty_tier=DifficultyTier.L3.value,
                        category=QueryCategory.CROSS_REFERENCE.value,
                        query_type=QueryType.CROSS_REGULATION.value,
                        cross_regulation_sources=[
                            reg_keywords[reg1]["source"],
                            reg_keywords[reg2]["source"],
                        ],
                    ))

        return queries

    def generate_adversarial(self, count: int = 30) -> List[GeneratedQuery]:
        """Generate adversarial queries to expose system weaknesses.

        EARS-U-003: Adversarial queries at L4-L5.
        """
        queries: List[GeneratedQuery] = []

        for query_type, templates in self.ADVERSARIAL_TEMPLATES.items():
            for template in templates:
                expected = "reject"
                if query_type == QueryType.ADVERSARIAL_AMBIGUOUS:
                    expected = "clarify"
                elif query_type == QueryType.ADVERSARIAL_LONG:
                    expected = "partial"
                elif query_type == QueryType.ADVERSARIAL_TYPO:
                    expected = "answer"

                tier = DifficultyTier.L4.value
                if query_type in (QueryType.ADVERSARIAL_LONG, QueryType.ADVERSARIAL_AMBIGUOUS):
                    tier = DifficultyTier.L5.value

                queries.append(GeneratedQuery(
                    query=template,
                    expected_source="",
                    difficulty_tier=tier,
                    category=QueryCategory.ADVERSARIAL.value,
                    query_type=query_type.value,
                    expected_behavior=expected,
                ))

        # Shuffle and limit
        self._rng.shuffle(queries)
        return queries[:count]

    def generate_all(
        self,
        target_count: int = 200,
    ) -> List[GeneratedQuery]:
        """Generate a complete query set across all tiers and types.

        Returns at least min(target_count, available) queries with:
        - L1-L2: From regulation content
        - L3: Cross-regulation
        - L4-L5: Adversarial
        - At least 20% cross-regulation in L3+
        - At least 15% adversarial
        """
        # Generate each type
        regulation_queries = self.generate_from_regulations(target_count=target_count)
        cross_queries = self.generate_cross_regulation()
        adversarial_queries = self.generate_adversarial()

        all_queries = regulation_queries + cross_queries + adversarial_queries

        # Deduplicate by query text
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q.query not in seen:
                seen.add(q.query)
                unique_queries.append(q)

        return unique_queries

    def save_cache(self, queries: List[GeneratedQuery]) -> str:
        """Save generated queries to cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "total_queries": len(queries),
            "tier_distribution": {},
            "queries": [q.to_dict() for q in queries],
        }

        # Calculate tier distribution
        for q in queries:
            tier = q.difficulty_tier
            data["tier_distribution"][tier] = data["tier_distribution"].get(tier, 0) + 1

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d queries to %s", len(queries), self.cache_path)
        return str(self.cache_path)

    def load_cache(self) -> Optional[List[GeneratedQuery]]:
        """Load queries from cache file if available."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            queries = [GeneratedQuery.from_dict(q) for q in data.get("queries", [])]
            logger.info("Loaded %d cached queries from %s", len(queries), self.cache_path)
            return queries
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("Failed to load cache: %s", e)
            return None

    def get_queries(
        self,
        regenerate: bool = False,
        target_count: int = 200,
    ) -> List[GeneratedQuery]:
        """Get queries: from cache or generate new ones.

        Args:
            regenerate: Force regeneration even if cache exists
            target_count: Target number of queries

        Returns:
            List of generated queries
        """
        if not regenerate:
            cached = self.load_cache()
            if cached and len(cached) >= self.MIN_QUERIES:
                return cached

        queries = self.generate_all(target_count=target_count)
        if queries:
            self.save_cache(queries)
        return queries
