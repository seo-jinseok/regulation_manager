"""
Full-view Use Case for Regulation RAG System.

Provides regulation-level retrieval for "전문/전체" requests.
"""

from dataclasses import dataclass
import re
from typing import List, Optional

from ..config import get_config
from ..domain.repositories import IDocumentLoader


FULL_VIEW_MARKERS = ["전문", "전체", "원문", "全文", "full text", "fullview"]


@dataclass(frozen=True)
class RegulationMatch:
    title: str
    rule_code: str
    score: int


@dataclass(frozen=True)
class RegulationView:
    title: str
    rule_code: str
    toc: List[str]
    content: List[dict]
    addenda: List[dict]


class FullViewUseCase:
    """Regulation full-view retrieval use case."""

    def __init__(self, loader: IDocumentLoader, json_path: Optional[str] = None):
        config = get_config()
        self.loader = loader
        self.json_path = json_path or config.json_path

    def find_matches(self, query: str) -> List[RegulationMatch]:
        """Find regulation matches for a query."""
        try:
            titles = self.loader.get_regulation_titles(self.json_path)
        except Exception:
            return []
        if not titles:
            return []

        term = self._strip_full_view_markers(query)
        term_norm = self._normalize(term)
        tokens = self._tokenize(term)

        matches: List[RegulationMatch] = []
        for rule_code, title in titles.items():
            title_norm = self._normalize(title)
            score = 0
            if term_norm and term_norm == title_norm:
                score = 3
            elif term_norm and term_norm in title_norm:
                score = 3
            elif tokens and all(t in title for t in tokens):
                score = 2
            elif tokens and any(t in title for t in tokens):
                score = 1

            if score > 0:
                matches.append(RegulationMatch(title=title, rule_code=rule_code, score=score))

        matches.sort(key=lambda m: (-m.score, m.title))
        return matches

    def get_full_view(self, identifier: str) -> Optional[RegulationView]:
        """Return regulation view by rule_code or title."""
        try:
            doc = self.loader.get_regulation_doc(self.json_path, identifier)
        except Exception:
            return None
        if not doc:
            return None

        rule_code = doc.get("metadata", {}).get("rule_code", "") or self._infer_rule_code(doc)
        title = doc.get("title", "")
        content = doc.get("content", []) or []
        addenda = doc.get("addenda", []) or []
        toc = self._build_toc(content, addenda)
        return RegulationView(
            title=title,
            rule_code=rule_code,
            toc=toc,
            content=content,
            addenda=addenda,
        )

    def _strip_full_view_markers(self, query: str) -> str:
        cleaned = query
        for marker in FULL_VIEW_MARKERS:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(str(text).split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[가-힣]+", text) if t]

    def _build_toc(self, content: List[dict], addenda: List[dict]) -> List[str]:
        toc: List[str] = []
        self._collect_toc(content, toc)
        self._collect_toc(addenda, toc)
        return toc

    def _collect_toc(self, nodes: List[dict], toc: List[str]) -> None:
        for node in nodes:
            label = self._format_label(node)
            if label:
                toc.append(label)
            children = node.get("children", []) or []
            if children:
                self._collect_toc(children, toc)

    @staticmethod
    def _format_label(node: dict) -> str:
        display_no = node.get("display_no", "")
        title = node.get("title", "")
        if display_no and title:
            return f"{display_no} {title}"
        return display_no or title

    @staticmethod
    def _infer_rule_code(doc: dict) -> str:
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("rule_code"):
            return metadata["rule_code"]
        for node in doc.get("content", []):
            node_meta = node.get("metadata", {})
            if isinstance(node_meta, dict) and node_meta.get("rule_code"):
                return node_meta["rule_code"]
        return ""
