"""
Reference resolution for regulation cross-references.

Resolves internal cross-references (e.g., "제5조제2항") to their
target node IDs within the document structure.
"""

import re
from typing import Dict, List, Any, Optional


class ReferenceResolver:
    """
    Resolves cross-references between regulation nodes.
    
    Handles references like "제5조", "제10조제1항", "제3호" and
    links them to their corresponding node IDs.
    """

    def resolve_all(self, docs: List[Dict[str, Any]]) -> None:
        """
        Resolve references in all documents.
        
        Args:
            docs: List of document dictionaries to process.
        """
        for doc in docs:
            rule_code = (doc.get("metadata") or {}).get("rule_code")
            if not rule_code:
                continue

            doc_index = self._build_reference_index(doc)
            for section in ("content", "addenda"):
                nodes = doc.get(section) or []
                self._resolve_references_in_nodes(
                    nodes,
                    doc_rule_code=rule_code,
                    doc_index=doc_index,
                    current_article=None,
                    current_paragraph=None,
                    current_item=None,
                )

    def extract_references(self, text: str) -> List[Dict[str, str]]:
        """
        Extract cross-references from text.
        
        Args:
            text: The text to extract references from.
            
        Returns:
            List of reference dictionaries with 'text' and 'target' keys.
        """
        if not text:
            return []

        # Pattern to match "제N조", "제N조의M", followed by optional "제K항", "제L호", etc.
        pattern = (
            r"제\s*\d+\s*조(?:의\s*\d+)?(?:제\s*\d+\s*[항호목])*|제\s*\d+\s*[항호목]"
        )

        matches = re.finditer(pattern, text)
        refs = []
        for m in matches:
            t = m.group(0).strip()
            refs.append({"text": t, "target": t})
        return refs

    def _build_reference_index(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Build an index of articles and their sub-elements for reference resolution."""
        articles: Dict[str, Dict[str, Any]] = {}

        def walk(nodes: List[Dict[str, Any]]) -> None:
            for node in nodes or []:
                if node.get("type") == "article":
                    key = self._normalize_token(node.get("display_no") or "")
                    if key:
                        articles[key] = self._build_article_reference_index(node)
                walk(node.get("children") or [])

        walk(doc.get("content") or [])
        return {"articles": articles}

    def _build_article_reference_index(
        self, article_node: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build an index for a single article's paragraphs and items."""
        paragraphs: Dict[int, str] = {}
        items_by_paragraph: Dict[int, Dict[int, str]] = {}
        items_direct: Dict[int, str] = {}

        for child in article_node.get("children") or []:
            if child.get("type") == "paragraph":
                para_no = int((child.get("sort_no") or {}).get("main", 0) or 0)
                if para_no > 0:
                    paragraphs[para_no] = child.get("id")
                items_by_paragraph[para_no] = {}
                for item in child.get("children") or []:
                    if item.get("type") != "item":
                        continue
                    item_no = int((item.get("sort_no") or {}).get("main", 0) or 0)
                    if item_no > 0:
                        items_by_paragraph[para_no][item_no] = item.get("id")
            elif child.get("type") == "item":
                item_no = int((child.get("sort_no") or {}).get("main", 0) or 0)
                if item_no > 0:
                    items_direct[item_no] = child.get("id")

        return {
            "id": article_node.get("id"),
            "paragraphs": paragraphs,
            "items_by_paragraph": items_by_paragraph,
            "items_direct": items_direct,
        }

    def _resolve_references_in_nodes(
        self,
        nodes: List[Dict[str, Any]],
        *,
        doc_rule_code: str,
        doc_index: Dict[str, Any],
        current_article: Optional[Dict[str, Any]],
        current_paragraph: Optional[Dict[str, Any]],
        current_item: Optional[Dict[str, Any]],
    ) -> None:
        """Recursively resolve references in nodes."""
        for node in nodes or []:
            node_type = node.get("type")
            if node_type == "article":
                current_article = node
                current_paragraph = None
                current_item = None
            elif node_type == "paragraph":
                current_paragraph = node
                current_item = None
            elif node_type == "item":
                current_item = node

            for ref in node.get("references") or []:
                target_text = (ref.get("target") or ref.get("text") or "").strip()
                if not target_text:
                    continue
                target_id = self._resolve_reference_target(
                    target_text,
                    doc_index=doc_index,
                    current_article=current_article,
                    current_paragraph=current_paragraph,
                    current_item=current_item,
                )
                if target_id:
                    ref["target_node_id"] = target_id
                    ref["target_doc_rule_code"] = doc_rule_code

            self._resolve_references_in_nodes(
                node.get("children") or [],
                doc_rule_code=doc_rule_code,
                doc_index=doc_index,
                current_article=current_article,
                current_paragraph=current_paragraph,
                current_item=current_item,
            )

    def _resolve_reference_target(
        self,
        target_text: str,
        *,
        doc_index: Dict[str, Any],
        current_article: Optional[Dict[str, Any]],
        current_paragraph: Optional[Dict[str, Any]],
        current_item: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Resolve a reference target to a node ID."""
        parsed = self._parse_reference_token(target_text)

        article_key = None
        if parsed.get("article_main") is not None:
            article_key = self._format_article_key(
                parsed["article_main"], parsed.get("article_sub") or 0
            )
        elif current_article:
            article_key = self._normalize_token(current_article.get("display_no") or "")

        if not article_key:
            return None

        article_entry = (doc_index.get("articles") or {}).get(article_key)
        if not article_entry:
            return None

        para_no = parsed.get("paragraph_no")
        item_no = parsed.get("item_no")

        if para_no is None and item_no is None:
            return article_entry.get("id")

        resolved_para_no = None
        if para_no is not None:
            resolved_para_no = int(para_no)
        elif item_no is not None and current_paragraph and current_article:
            current_article_key = self._normalize_token(
                current_article.get("display_no") or ""
            )
            if current_article_key == article_key:
                resolved_para_no = (
                    int((current_paragraph.get("sort_no") or {}).get("main", 0) or 0)
                    or None
                )

        if item_no is None:
            if resolved_para_no is None or resolved_para_no <= 0:
                return None
            return article_entry.get("paragraphs", {}).get(resolved_para_no)

        item_no_int = int(item_no)
        if item_no_int <= 0:
            return None

        if resolved_para_no is not None and resolved_para_no >= 0:
            return (
                article_entry.get("items_by_paragraph", {})
                .get(resolved_para_no, {})
                .get(item_no_int)
            )

        return article_entry.get("items_direct", {}).get(item_no_int)

    def _parse_reference_token(self, token: str) -> Dict[str, Optional[int]]:
        """
        Parse a reference token like:
        - 제6조
        - 제6조의2
        - 제6조제2항
        - 제6조제2항제3호
        """
        value = self._normalize_token(token)
        result: Dict[str, Optional[int]] = {
            "article_main": None,
            "article_sub": None,
            "paragraph_no": None,
            "item_no": None,
        }

        m = re.match(r"^제(\d+)조(?:의(\d+))?", value)
        if m:
            result["article_main"] = int(m.group(1))
            result["article_sub"] = int(m.group(2)) if m.group(2) else 0
            value = value[m.end() :]

        while value:
            m = re.match(r"^제(\d+)([항호])", value)
            if not m:
                break
            num = int(m.group(1))
            kind = m.group(2)
            if kind == "항" and result["paragraph_no"] is None:
                result["paragraph_no"] = num
            elif kind == "호" and result["item_no"] is None:
                result["item_no"] = num
            value = value[m.end() :]

        return result

    def _format_article_key(self, main_no: int, sub_no: int) -> str:
        """Format article key from main and sub numbers."""
        if sub_no and int(sub_no) > 0:
            return f"제{int(main_no)}조의{int(sub_no)}"
        return f"제{int(main_no)}조"

    def _normalize_token(self, value: str) -> str:
        """Normalize token by removing all whitespace."""
        return re.sub(r"\s+", "", str(value)).strip()
