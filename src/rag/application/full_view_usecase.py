"""
Full-view Use Case for Regulation RAG System.

Provides regulation-level retrieval for "전문/전체" requests.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import List, Optional

from ..config import get_config
from ..domain.repositories import IDocumentLoader


FULL_VIEW_MARKERS = ["전문", "전체", "원문", "全文", "full text", "fullview"]
TOC_ALLOWED_TYPES = {"chapter", "section", "subsection", "article", "addendum"}
TOC_SKIP_RECURSION_TYPES = {"paragraph", "item", "subitem", "text", "addendum_item", "addendum"}


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


@dataclass(frozen=True)
class TableMatch:
    path: List[str]
    display_no: str
    title: str
    text: str
    markdown: str
    table_index: int


class FullViewUseCase:
    """Regulation full-view retrieval use case."""

    def __init__(self, loader: IDocumentLoader, json_path: Optional[str] = None):
        config = get_config()
        self.loader = loader
        self.json_path = json_path or config.json_path

    def find_matches(self, query: str) -> List[RegulationMatch]:
        """Find regulation matches for a query."""
        json_path = self._resolve_json_path()
        if not json_path:
            return []
        try:
            titles = self.loader.get_regulation_titles(json_path)
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
        json_path = self._resolve_json_path()
        if not json_path:
            return None
        try:
            doc = self.loader.get_regulation_doc(json_path, identifier)
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

    def find_tables(
        self,
        identifier: str,
        table_no: Optional[int] = None,
        label_variants: Optional[List[str]] = None,
    ) -> List[TableMatch]:
        """Find tables for a regulation, optionally filtered by table number."""
        json_path = self._resolve_json_path()
        if not json_path:
            return []
        try:
            doc = self.loader.get_regulation_doc(json_path, identifier)
        except Exception:
            return []
        if not doc:
            return []

        labeled_matches: List[TableMatch] = []
        placeholder_matches: List[TableMatch] = []
        label_pattern = None
        variants = [v for v in (label_variants or ["별표"]) if v]
        if table_no is not None:
            escaped = "|".join(re.escape(v) for v in variants)
            if escaped:
                label_pattern = re.compile(rf"(?:{escaped})\s*{table_no}")

        def walk(nodes: List[dict], path_stack: List[str]) -> None:
            for node in nodes:
                text = str(node.get("text") or "")
                title = str(node.get("title") or "")
                display_no = str(node.get("display_no") or "")
                metadata = node.get("metadata") or {}
                tables = metadata.get("tables") if isinstance(metadata, dict) else None

                current_path = node.get("parent_path") or path_stack
                if not current_path:
                    label = f"{display_no} {title}".strip()
                    current_path = path_stack + ([label] if label else [])

                if tables and text:
                    placeholders = re.findall(r"\[TABLE:(\d+)\]", text)
                    if placeholders:
                        context = re.sub(r"\[TABLE:\d+\]", "", text).strip()
                        for placeholder in placeholders:
                            index = int(placeholder)
                            if index <= 0 or index > len(tables):
                                continue
                            table = tables[index - 1]
                            if isinstance(table, dict):
                                markdown = table.get("markdown") or ""
                            else:
                                markdown = ""
                            if not markdown:
                                continue
                            match = TableMatch(
                                path=list(current_path),
                                display_no=display_no,
                                title=title,
                                text=context,
                                markdown=markdown,
                                table_index=index,
                            )
                            if table_no is None:
                                placeholder_matches.append(match)
                            else:
                                target_text = " ".join([display_no, title, text]).strip()
                                if label_pattern and label_pattern.search(target_text):
                                    labeled_matches.append(match)
                                elif index == table_no:
                                    placeholder_matches.append(match)

                children = node.get("children") or []
                if children:
                    next_path = list(current_path)
                    walk(children, next_path)

        walk(doc.get("content", []) or [], [])
        walk(doc.get("addenda", []) or [], [])

        if table_no is not None and labeled_matches:
            return labeled_matches
        if table_no is None and variants:
            labeled = []
            for match in placeholder_matches:
                target_text = " ".join([match.display_no, match.title, match.text]).strip()
                if any(variant in target_text for variant in variants):
                    labeled.append(match)
            if labeled:
                return labeled
        return placeholder_matches

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
            node_type = node.get("type")
            if node_type in TOC_ALLOWED_TYPES:
                label = self._format_label(node)
                if label:
                    toc.append(label)
            if node_type in TOC_SKIP_RECURSION_TYPES:
                continue
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

    def _resolve_json_path(self) -> Optional[str]:
        config = get_config()
        candidates = []
        if self.json_path:
            candidates.append(self.json_path)
        if config.json_path and config.json_path not in candidates:
            candidates.append(config.json_path)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                self.json_path = candidate
                return candidate

        sync_path = Path(config.sync_state_path)
        if sync_path.exists():
            try:
                data = json.loads(sync_path.read_text(encoding="utf-8"))
                json_file = data.get("json_file", "")
            except Exception:
                json_file = ""

            if json_file:
                file_path = Path(json_file)
                if file_path.is_absolute() and file_path.exists():
                    self.json_path = str(file_path)
                    return self.json_path
                if file_path.exists():
                    self.json_path = str(file_path)
                    return self.json_path
                output_candidate = sync_path.parent / "output" / json_file
                if output_candidate.exists():
                    self.json_path = str(output_candidate)
                    return self.json_path

        output_dir = Path("data/output")
        if output_dir.exists():
            json_files = [
                p for p in output_dir.rglob("*.json")
                if not p.name.endswith("_metadata.json")
            ]
            if json_files:
                latest = max(json_files, key=lambda p: p.stat().st_mtime)
                self.json_path = str(latest)
                return self.json_path

        return None
