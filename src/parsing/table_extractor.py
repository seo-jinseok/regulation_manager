"""
Table extraction utilities for regulation text.

Extracts markdown tables from text content and structures them
as metadata for further processing.
"""

import re
from typing import Any, Dict, List, Tuple


class TableExtractor:
    """
    Extracts and structures markdown tables from regulation text.

    Tables are extracted from text, replaced with placeholders,
    and stored as structured metadata.
    """

    def extract_tables(self, docs: List[Dict[str, Any]]) -> None:
        """
        Extract tables from all documents.

        Args:
            docs: List of document dictionaries to process.
        """
        for doc in docs:
            for section in ("content", "addenda"):
                self._extract_tables_in_nodes(doc.get(section) or [])

    def _extract_tables_in_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Recursively extract tables from nodes.

        Args:
            nodes: List of node dictionaries.
        """
        for node in nodes or []:
            text = node.get("text") or ""
            updated_text, tables = self.split_markdown_tables(text)
            if tables:
                metadata = node.setdefault("metadata", {})
                metadata["tables"] = [
                    {"format": "markdown", "markdown": t} for t in tables
                ]
                node["text"] = updated_text
            self._extract_tables_in_nodes(node.get("children") or [])

    def split_markdown_tables(self, text: str) -> Tuple[str, List[str]]:
        """
        Split text into non-table content and table blocks.

        Args:
            text: The text to process.

        Returns:
            Tuple of (updated text with placeholders, list of table markdown).
        """
        if not text or "|" not in text:
            return text, []

        lines = text.splitlines()
        out_lines: List[str] = []
        tables: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if not self._is_table_row(line):
                out_lines.append(line)
                i += 1
                continue

            block: List[str] = []
            while i < len(lines) and self._is_table_row(lines[i]):
                block.append(lines[i])
                i += 1

            if len(block) >= 2 and any(
                self._is_table_separator_row(row) for row in block
            ):
                tables.append("\n".join(block).strip())
                out_lines.append(f"[TABLE:{len(tables)}]")
                continue

            out_lines.extend(block)

        return "\n".join(out_lines).strip(), tables

    def _is_table_row(self, line: str) -> bool:
        """Check if a line is a table row."""
        stripped = (line or "").strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    def _is_table_separator_row(self, line: str) -> bool:
        """Check if a line is a table separator row (|---|---|)."""
        stripped = (line or "").strip()
        if not stripped.startswith("|") or "---" not in stripped:
            return False

        core = stripped.strip("|").strip()
        if not core:
            return False

        cells = [c.strip() for c in core.split("|") if c.strip()]
        if not cells:
            return False

        return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)
