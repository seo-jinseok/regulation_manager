"""
Keyword Extractor for Regulation Documents.

Automatically extracts key terms from regulation JSON files
to improve search accuracy and build regulation_keywords.json.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RegulationKeywords:
    """Keywords extracted from a single regulation."""

    rule_code: str
    name: str
    keywords: List[str] = field(default_factory=list)
    context: str = "general"  # student, employee, general
    chapter_keywords: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of keyword extraction."""

    total_regulations: int
    total_keywords: int
    regulations: Dict[str, RegulationKeywords] = field(default_factory=dict)


class KeywordExtractor:
    """
    Extracts keywords from regulation JSON files.

    Identifies important terms from regulation names, chapter titles,
    and article titles to improve search accuracy.
    """

    # Context detection patterns
    STUDENT_PATTERNS = [
        r"학생",
        r"학칙",
        r"학사",
        r"등록",
        r"졸업",
        r"휴학",
        r"장학",
        r"수강",
        r"성적",
        r"학위",
        r"입학",
        r"재학",
        r"학년",
    ]
    EMPLOYEE_PATTERNS = [
        r"교원",
        r"직원",
        r"인사",
        r"보수",
        r"급여",
        r"퇴직",
        r"복무",
        r"연구년",
        r"승진",
        r"호봉",
        r"근로",
        r"노사",
    ]

    # Stopwords to exclude
    STOPWORDS = {
        "제",
        "조",
        "항",
        "호",
        "목",
        "다음",
        "각",
        "해당",
        "경우",
        "규정",
        "관한",
        "위한",
        "따른",
        "대한",
        "의한",
        "있는",
        "하는",
        "되는",
        "한다",
        "있다",
        "된다",
        "수",
        "것",
        "등",
        "및",
        "또는",
        "이",
        "그",
        "저",
        "위",
        "아래",
        "기타",
    }

    def __init__(
        self,
        json_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize keyword extractor.

        Args:
            json_path: Path to regulation JSON file.
            output_path: Path to save extracted keywords.
        """
        self._json_path = json_path or self._default_json_path()
        self._output_path = output_path or self._default_output_path()

    def _default_json_path(self) -> str:
        """Get default regulation JSON path."""
        from ..config import get_config
        return str(get_config().json_path_resolved)

    def _default_output_path(self) -> str:
        """Get default output path for extracted keywords."""
        from ..config import get_config
        return str(get_config().regulation_keywords_path_resolved)

    def extract_keywords(self) -> ExtractionResult:
        """
        Extract keywords from regulation JSON.

        Returns:
            ExtractionResult with extracted keywords.
        """
        path = Path(self._json_path)
        if not path.exists():
            raise FileNotFoundError(f"Regulation JSON not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        # SCHEMA_REFERENCE.md에 따르면 최상위 리스트는 'docs'
        docs = data.get("docs", [])

        result = ExtractionResult(
            total_regulations=len(
                [d for d in docs if d.get("doc_type") == "regulation"]
            ),
            total_keywords=0,
        )

        for doc in docs:
            # 규정 타입이 아닌 것은 건너뜀 (목차, 색인 등)
            if doc.get("doc_type") != "regulation":
                continue

            # metadata에서 rule_code 추출
            metadata = doc.get("metadata", {})
            rule_code = metadata.get("rule_code", "")
            title = doc.get("title", "")

            if not rule_code or not title:
                continue

            # Extract keywords from regulation
            keywords = self._extract_from_regulation(doc)
            context = self._detect_context(title, keywords)

            reg_keywords = RegulationKeywords(
                rule_code=rule_code,
                name=title,
                keywords=keywords,
                context=context,
            )

            result.regulations[rule_code] = reg_keywords
            result.total_keywords += len(keywords)

        return result

    def _extract_from_regulation(self, doc: dict) -> List[str]:
        """Extract keywords from a single regulation (Document object)."""
        all_terms: List[str] = []

        # From title
        title = doc.get("title", "")
        all_terms.extend(self._extract_nouns(title))

        # Recursive extraction from nodes in 'content' and 'addenda'
        def extract_from_nodes(nodes: List[dict]):
            for node in nodes:
                node_title = node.get("title", "")
                if node_title:
                    all_terms.extend(self._extract_nouns(node_title))

                # Also consider text if it's short (like a summary)
                text = node.get("text", "")
                if text and len(text) < 100:
                    all_terms.extend(self._extract_nouns(text))

                children = node.get("children", [])
                if children:
                    extract_from_nodes(children)

        extract_from_nodes(doc.get("content", []))
        extract_from_nodes(doc.get("addenda", []))

        # Count and filter
        counter = Counter(all_terms)
        keywords = [
            term
            for term, count in counter.most_common(30)
            if count >= 1 and len(term) >= 2
        ]

        return keywords[:20]  # Top 20 keywords

    def _extract_nouns(self, text: str) -> List[str]:
        """Extract noun-like terms from text."""
        if not text:
            return []

        # Remove parentheses content
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"「[^」]*」", "", text)
        text = re.sub(r"제\d+조(의\d+)?", "", text)
        text = re.sub(r"제\d+[항호목장편절]", "", text)

        # Split by non-Korean characters
        tokens = re.findall(r"[가-힣]+", text)

        # Filter stopwords and short tokens
        filtered = [t for t in tokens if t not in self.STOPWORDS and len(t) >= 2]

        return filtered

    def _detect_context(self, name: str, keywords: List[str]) -> str:
        """Detect context (student/employee/general) from keywords."""
        text = name + " " + " ".join(keywords)

        student_score = sum(
            1 for pattern in self.STUDENT_PATTERNS if re.search(pattern, text)
        )
        employee_score = sum(
            1 for pattern in self.EMPLOYEE_PATTERNS if re.search(pattern, text)
        )

        if student_score > employee_score:
            return "student"
        elif employee_score > student_score:
            return "employee"
        return "general"

    def save_keywords(self, result: ExtractionResult) -> str:
        """
        Save extracted keywords to JSON file.

        Args:
            result: ExtractionResult to save.

        Returns:
            Path to saved file.
        """
        output = {
            "version": "1.0.0",
            "total_regulations": result.total_regulations,
            "total_keywords": result.total_keywords,
            "regulations": {
                rule_code: {
                    "name": reg.name,
                    "keywords": reg.keywords,
                    "context": reg.context,
                }
                for rule_code, reg in result.regulations.items()
            },
        }

        path = Path(self._output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return str(path)

    def format_summary(self, result: ExtractionResult) -> str:
        """Format extraction result as readable string."""
        lines = [
            "=" * 60,
            "규정 키워드 추출 결과",
            "=" * 60,
            f"총 규정 수: {result.total_regulations}",
            f"총 키워드 수: {result.total_keywords}",
            "-" * 60,
        ]

        # Context breakdown
        contexts = {"student": 0, "employee": 0, "general": 0}
        for reg in result.regulations.values():
            contexts[reg.context] = contexts.get(reg.context, 0) + 1

        lines.append(f"학생 관련: {contexts['student']}개 규정")
        lines.append(f"교직원 관련: {contexts['employee']}개 규정")
        lines.append(f"일반: {contexts['general']}개 규정")
        lines.append("=" * 60)

        return "\n".join(lines)

    def format_details(self, result: ExtractionResult, limit: int = 10) -> str:
        """Format detailed keyword list."""
        lines = []
        for i, (rule_code, reg) in enumerate(result.regulations.items()):
            if i >= limit:
                lines.append(f"\n... and {len(result.regulations) - limit} more")
                break
            context_icon = {"student": "🎓", "employee": "👔", "general": "📋"}.get(
                reg.context, "📋"
            )
            lines.append(f"\n{context_icon} [{rule_code}] {reg.name}")
            lines.append(f"   키워드: {', '.join(reg.keywords[:10])}")
        return "\n".join(lines)
