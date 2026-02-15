"""
Regulation-Specific Query Generator for RAG Quality Evaluation.

Generates test queries from actual regulation documents as specified in SPEC-RAG-EVAL-001.

Features:
- Parses regulation JSON for query topic extraction
- Generates queries about specific articles (제N조)
- Creates cross-reference queries (multiple regulations)
- Includes temporal queries (deadlines, periods)
- Supports procedure-based queries (step-by-step)
- Balanced difficulty distribution (easy 30%, medium 40%, hard 30%)
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .personas import PERSONAS, PersonaManager
from .parallel_evaluator import PersonaQuery

logger = logging.getLogger(__name__)


@dataclass
class RegulationArticle:
    """Represents an article from a regulation."""

    regulation_title: str
    article_no: str
    article_title: str
    content: str
    rule_code: str
    paragraphs: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)


@dataclass
class QueryTemplate:
    """Template for generating queries."""

    pattern: str
    difficulty: str  # "easy", "medium", "hard"
    query_type: str  # "article", "procedure", "temporal", "cross_reference"
    required_info: List[str] = field(default_factory=list)


class RegulationQueryGenerator:
    """
    Generates regulation-specific test queries.

    Parses actual regulation documents to create realistic test queries
    that reflect real-world usage patterns.
    """

    # Difficulty distribution: easy 30%, medium 40%, hard 30%
    DIFFICULTY_DISTRIBUTION = {
        "easy": 0.30,
        "medium": 0.40,
        "hard": 0.30,
    }

    # Query templates by difficulty
    EASY_TEMPLATES = [
        QueryTemplate(
            pattern="{regulation} {article}의 내용은 무엇인가요?",
            difficulty="easy",
            query_type="article",
            required_info=["article_content"],
        ),
        QueryTemplate(
            pattern="{topic} 규정이 궁금합니다",
            difficulty="easy",
            query_type="article",
            required_info=["basic_info"],
        ),
        QueryTemplate(
            pattern="{regulation}에서 {topic}에 대해 어떻게 규정하고 있나요?",
            difficulty="easy",
            query_type="article",
            required_info=["regulation_reference"],
        ),
    ]

    MEDIUM_TEMPLATES = [
        QueryTemplate(
            pattern="{regulation} {article}에 따르면 {topic}의 절차는 어떻게 되나요?",
            difficulty="medium",
            query_type="procedure",
            required_info=["procedure_steps", "article_reference"],
        ),
        QueryTemplate(
            pattern="{topic} 신청 기한과 필요 서류를 알려주세요",
            difficulty="medium",
            query_type="temporal",
            required_info=["deadline", "required_documents"],
        ),
        QueryTemplate(
            pattern="{regulation}과 {related_regulation}에서 {topic} 규정의 차이점은?",
            difficulty="medium",
            query_type="cross_reference",
            required_info=["comparison", "regulation_references"],
        ),
    ]

    HARD_TEMPLATES = [
        QueryTemplate(
            pattern="{regulation} {article}의 예외 사항과 특례 규정을 모두 설명해 주세요",
            difficulty="hard",
            query_type="article",
            required_info=["exceptions", "special_cases", "article_reference"],
        ),
        QueryTemplate(
            pattern="{topic} 관련하여 {regulation1}, {regulation2}, {regulation3}의 관계를 설명해 주세요",
            difficulty="hard",
            query_type="cross_reference",
            required_info=["regulation_relationships", "hierarchy"],
        ),
        QueryTemplate(
            pattern="{regulation} 개정 전후 {topic} 규정의 변화와 적용 시기를 알려주세요",
            difficulty="hard",
            query_type="temporal",
            required_info=["revision_history", "effective_dates", "changes"],
        ),
    ]

    def __init__(
        self,
        regulation_json_path: Optional[str] = None,
        persona_manager: Optional[PersonaManager] = None,
    ):
        """
        Initialize the regulation query generator.

        Args:
            regulation_json_path: Path to the regulation JSON file
            persona_manager: Optional persona manager for persona-specific queries
        """
        self.persona_manager = persona_manager or PersonaManager()
        self.regulations: List[Dict[str, Any]] = []
        self.articles: List[RegulationArticle] = []

        if regulation_json_path:
            self.load_regulations(regulation_json_path)

    def load_regulations(self, json_path: str) -> None:
        """
        Load regulations from JSON file.

        Args:
            json_path: Path to the regulation JSON file
        """
        path = Path(json_path)
        if not path.exists():
            logger.warning(f"Regulation file not found: {json_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                self.regulations = data.get("docs", [])
            elif isinstance(data, list):
                self.regulations = data
            else:
                logger.warning(f"Unexpected JSON structure in {json_path}")
                return

            # Extract articles
            self._extract_articles()
            logger.info(
                f"Loaded {len(self.regulations)} regulations with "
                f"{len(self.articles)} articles"
            )

        except Exception as e:
            logger.error(f"Error loading regulations: {e}")

    def _extract_articles(self) -> None:
        """Extract all articles from loaded regulations."""
        self.articles = []

        for reg in self.regulations:
            reg_title = reg.get("title", "")
            rule_code = reg.get("rule_code", "")
            articles_list = reg.get("articles", [])

            for article in articles_list:
                article_no = article.get("article_no", "")
                article_title = article.get("title", "")
                content = article.get("content", "")

                # Skip empty or very short articles
                if len(content.strip()) < 20:
                    continue

                reg_article = RegulationArticle(
                    regulation_title=reg_title,
                    article_no=article_no,
                    article_title=article_title,
                    content=content,
                    rule_code=rule_code,
                    paragraphs=article.get("paragraphs", []),
                    items=article.get("items", []),
                )
                self.articles.append(reg_article)

    def generate_queries(
        self,
        count: int = 25,
        persona: Optional[str] = None,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> List[PersonaQuery]:
        """
        Generate regulation-specific test queries.

        Args:
            count: Number of queries to generate
            persona: Optional persona to target
            difficulty_distribution: Override default difficulty distribution

        Returns:
            List of PersonaQuery objects
        """
        if not self.articles:
            logger.warning("No articles loaded, returning empty list")
            return []

        distribution = difficulty_distribution or self.DIFFICULTY_DISTRIBUTION
        queries = []

        # Calculate counts per difficulty
        easy_count = int(count * distribution.get("easy", 0.30))
        medium_count = int(count * distribution.get("medium", 0.40))
        hard_count = count - easy_count - medium_count  # Remainder goes to hard

        # Generate queries for each difficulty
        queries.extend(self._generate_queries_for_difficulty("easy", easy_count, persona))
        queries.extend(self._generate_queries_for_difficulty("medium", medium_count, persona))
        queries.extend(self._generate_queries_for_difficulty("hard", hard_count, persona))

        # Shuffle to mix difficulties
        random.shuffle(queries)

        logger.info(f"Generated {len(queries)} regulation-specific queries")
        return queries[:count]

    def _generate_queries_for_difficulty(
        self,
        difficulty: str,
        count: int,
        persona: Optional[str] = None,
    ) -> List[PersonaQuery]:
        """Generate queries for a specific difficulty level."""
        queries = []

        if difficulty == "easy":
            templates = self.EASY_TEMPLATES
        elif difficulty == "medium":
            templates = self.MEDIUM_TEMPLATES
        else:
            templates = self.HARD_TEMPLATES

        for _ in range(count):
            template = random.choice(templates)
            article = random.choice(self.articles)

            query_text = self._render_template(template, article)

            # Apply persona styling if specified
            if persona:
                query_text = self._apply_persona_style(query_text, persona)

            query = PersonaQuery(
                query=query_text,
                persona=persona or "general",
                category=template.query_type,
                difficulty=difficulty,
                expected_intent=self._infer_intent(article),
                expected_info=template.required_info.copy(),
            )
            queries.append(query)

        return queries

    def _render_template(self, template: QueryTemplate, article: RegulationArticle) -> str:
        """Render a query template with article data."""
        # Extract topic from article
        topic = self._extract_topic(article)

        # Build template variables
        variables = {
            "regulation": article.regulation_title,
            "article": f"제{article.article_no}조",
            "topic": topic,
            "related_regulation": self._get_related_regulation(article),
            "regulation1": article.regulation_title,
            "regulation2": self._get_related_regulation(article),
            "regulation3": self._get_related_regulation(article, exclude_first=True),
        }

        try:
            return template.pattern.format(**variables)
        except KeyError:
            # Fallback if template has unmatched variables
            return f"{article.regulation_title} {article.article_no}에 대해 알려주세요"

    def _extract_topic(self, article: RegulationArticle) -> str:
        """Extract a topic keyword from article content."""
        # Combine title and first part of content
        text = f"{article.article_title} {article.content[:200]}"

        # Common topic keywords in Korean regulations
        topic_patterns = [
            r"(휴학|복학|자퇴|제적)",
            r"(등록금|장학금|납부)",
            r"(성적|학점|이수)",
            r"(졸업|학위)",
            r"(휴가|복무|연차)",
            r"(승진|임용|보직)",
            r"(연구|논문)",
            r"(시험|평가)",
            r"(수강|강의)",
            r"(기숙사|생활)",
        ]

        for pattern in topic_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # Default to article title if no pattern matched
        return article.article_title or "이 규정"

    def _get_related_regulation(
        self,
        article: RegulationArticle,
        exclude_first: bool = False,
    ) -> str:
        """Get a related regulation name."""
        # Filter regulations with similar titles
        related = []
        for reg in self.regulations:
            reg_title = reg.get("title", "")
            if reg_title and reg_title != article.regulation_title:
                # Check for common keywords
                if any(
                    kw in reg_title
                    for kw in ["학칙", "규정", "지침", "세칙"]
                ):
                    related.append(reg_title)

        if exclude_first and related:
            related = related[1:] if len(related) > 1 else related

        return random.choice(related) if related else "학칙"

    def _apply_persona_style(self, query: str, persona: str) -> str:
        """Apply persona-specific style to query."""
        persona_profile = PERSONAS.get(persona)
        if not persona_profile:
            return query

        # Adjust based on vocabulary style
        style = persona_profile.vocabulary_style

        if style == "simple":
            # Simplify for beginners
            query = query.replace("규정하고 있나요?", "어떻게 되나요?")
            query = query.replace("설명해 주세요", "알려주세요")
        elif style == "academic":
            # Make more formal
            query = query.replace("알려주세요", "설명해 주시겠습니까?")
            query = query.replace("궁금합니다", "확인 부탁드립니다")

        return query

    def _infer_intent(self, article: RegulationArticle) -> str:
        """Infer the intent of a query about an article."""
        content = article.content.lower()

        if any(kw in content for kw in ["절차", "신청", "제출", "승인"]):
            return "procedural"
        elif any(kw in content for kw in ["기간", "마감", "까지", "일 전"]):
            return "temporal"
        elif any(kw in content for kw in ["자격", "요건", "대상"]):
            return "eligibility"
        else:
            return "informational"

    def generate_all_personas_queries(
        self,
        queries_per_persona: int = 25,
    ) -> Dict[str, List[PersonaQuery]]:
        """
        Generate regulation-specific queries for all personas.

        Args:
            queries_per_persona: Number of queries per persona

        Returns:
            Dictionary mapping persona names to query lists
        """
        all_queries = {}

        for persona_name in self.persona_manager.list_personas():
            queries = self.generate_queries(
                count=queries_per_persona,
                persona=persona_name,
            )
            all_queries[persona_name] = queries

        total = sum(len(q) for q in all_queries.values())
        logger.info(f"Generated {total} total queries for {len(all_queries)} personas")

        return all_queries

    def get_article_queries(self, article_no: str) -> List[PersonaQuery]:
        """
        Generate queries specifically about a single article.

        Args:
            article_no: Article number (e.g., "15")

        Returns:
            List of queries about the specified article
        """
        matching_articles = [
            a
            for a in self.articles
            if article_no in a.article_no
        ]

        if not matching_articles:
            logger.warning(f"No articles found matching: {article_no}")
            return []

        queries = []
        for article in matching_articles[:5]:  # Limit to 5 articles
            # Generate a query for each template type
            for templates in [self.EASY_TEMPLATES, self.MEDIUM_TEMPLATES, self.HARD_TEMPLATES]:
                template = random.choice(templates)
                query_text = self._render_template(template, article)

                queries.append(
                    PersonaQuery(
                        query=query_text,
                        persona="general",
                        category=template.query_type,
                        difficulty=template.difficulty,
                        expected_intent=self._infer_intent(article),
                        expected_info=template.required_info.copy(),
                    )
                )

        return queries

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded regulations."""
        return {
            "total_regulations": len(self.regulations),
            "total_articles": len(self.articles),
            "avg_article_length": (
                sum(len(a.content) for a in self.articles) / len(self.articles)
                if self.articles
                else 0
            ),
            "difficulty_distribution": self.DIFFICULTY_DISTRIBUTION,
        }
