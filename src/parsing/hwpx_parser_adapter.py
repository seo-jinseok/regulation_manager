"""
HWPX Parser Adapter - Parses HWPX files and converts to regulation JSON format.

This adapter uses the python-hwpx library to extract text and structure
from HWPX files and converts them to the standard regulation JSON format
compatible with the existing codebase.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from hwpx import TextExtractor
except ImportError:
    raise ImportError(
        "python-hwpx is required. Install with: pip install python-hwpx"
    )


@dataclass
class RegulationArticle:
    """Represents a single article (조) within a regulation."""

    article_no: str  # e.g., "제1조"
    title: str = ""
    content: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    display_no: str = ""
    sort_no: Dict[str, int] = field(default_factory=lambda: {"main": 0, "sub": 0})


@dataclass
class RegulationDocument:
    """Represents a single regulation document."""

    id: str
    title: str
    rule_code: str
    part: str
    doc_type: str = "regulation"
    status: str = "effective"
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: List[Dict[str, Any]] = field(default_factory=list)
    addenda: List[Dict[str, Any]] = field(default_factory=list)
    preamble: Optional[str] = None
    attached_files: List[str] = field(default_factory=list)
    is_index_duplicate: bool = False


class HwpxParserAdapter:
    """
    Adapter for parsing HWPX files into regulation JSON format.

    This parser:
    1. Extracts plain text from HWPX files using python-hwpx
    2. Parses the table of contents to identify regulations
    3. Detects article structure (조, 항, 호) within each regulation
    4. Converts to the standard JSON format used by the RAG system
    """

    # Regex patterns for parsing regulation structure
    # TOC pattern: title followed by code (e.g., "학교법인동의학원정관1-0-1")
    TOC_PATTERN = re.compile(r"^([^\d\n]+?)(\d+-\d+-\d+)$")
    ARTICLE_PATTERN = re.compile(r"^제(\d+)조\s*(.+)?$")  # e.g., "제1조 목적"
    SECTION_PATTERN = re.compile(r"^제(\d+)장\s*(.+)$")  # e.g., "제1장 총칙"
    ITEM_PATTERN = re.compile(r"^(\d+)\.\s*(.+)$")  # e.g., "1. ..."
    SUBITEM_PATTERN = re.compile(r"^가?\s?나?\s?다?\s?라?\s?[).\-\s]\s*(.+)$")
    NUMBERED_ITEM = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩]|[ㄱㄲㄴㄵㄷㄹㅁㅂㅅㅇ]|[가-힣])\.\s*(.+)$")

    # Keywords that indicate sections
    SECTION_KEYWORDS = ["총칙", "목적", "정의", "설치", "조직", "직무", "운영", "심의", "회의", "재정"]

    def __init__(self, source_file: Path):
        """
        Initialize the adapter with a source HWPX file.

        Args:
            source_file: Path to the HWPX file to parse
        """
        self.source_file = Path(source_file)
        if not self.source_file.exists():
            raise FileNotFoundError(f"HWPX file not found: {source_file}")

        self.raw_text: str = ""
        self.lines: List[str] = []
        self.regulations: List[RegulationDocument] = []

    def parse(self) -> Dict[str, Any]:
        """
        Parse the HWPX file and return the regulation JSON structure.

        Returns:
            Dictionary containing metadata, toc, and docs in the standard format
        """
        # Step 1: Extract text from HWPX (with progress indication)
        print("Extracting text from HWPX file...")
        self._extract_text()
        print(f"Extracted {len(self.lines)} lines")

        # Step 2: Parse table of contents
        print("Parsing table of contents...")
        toc_entries = self._parse_table_of_contents()
        print(f"Found {len(toc_entries)} TOC entries")

        # Step 3: Parse individual regulations
        print("Parsing regulations...")
        docs = []
        for i, entry in enumerate(toc_entries):
            print(f"  [{i+1}/{len(toc_entries)}] {entry['title']} ({entry['rule_code']})")
            regulation = self._parse_regulation(entry)
            if regulation:
                docs.append(self._regulation_to_dict(regulation))

        # Step 4: Build result structure
        result = {
            "schema_version": "v4",
            "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "pipeline_signature": f"hwpx_adapter|{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "file_name": self.source_file.name,
            "source_serial": None,
            "source_date": self._extract_date_from_filename(),
            "toc": [{"title": e["title"], "rule_code": e["rule_code"]} for e in toc_entries],
            "docs": docs,
        }

        return result

    def _extract_text(self) -> None:
        """Extract plain text from the HWPX file."""
        with TextExtractor(self.source_file) as extractor:
            self.raw_text = extractor.extract_text()
            self.lines = self.raw_text.split("\n")

    def _parse_table_of_contents(self) -> List[Dict[str, str]]:
        """
        Parse the table of contents from the extracted text.

        The TOC typically appears near the beginning and contains entries like:
        "학교법인동의학원정관1-0-1"
        "동의대학교학칙2-1-1"

        Also handles cases where the title and code are on separate lines.
        """
        toc_entries = []
        in_toc_section = False

        # Pattern to match regulation codes like "1-0-1", "2-1-1"
        code_pattern = re.compile(r"^(\d+)-(\d+)-(\d+)$")
        # Combined pattern: title followed immediately by code
        combined_pattern = re.compile(r"^([^\d\n]+?)(\d+-\d+-\d+)$")

        for i, line in enumerate(self.lines):
            line = line.strip()
            if not line:
                continue

            # Detect start of TOC (contains "차례" or similar keywords)
            if "차례" in line or "목차" in line:
                in_toc_section = True
                continue

            # Detect end of TOC (when we see actual article content)
            # Articles start with "제X조" pattern
            if in_toc_section and re.match(r"^제\d+조", line):
                # We've hit actual article content, TOC is done
                break

            # Detect section headers (like "제1편", "제2장") - skip these
            if re.match(r"^제\d+[편장]", line):
                continue

            # Try to match combined pattern: "규정명코드" (most common)
            match = combined_pattern.match(line)
            if match:
                title = match.group(1).strip()
                code = match.group(2)
                # Skip if already added (avoid duplicates)
                if not any(e["rule_code"] == code for e in toc_entries):
                    toc_entries.append({"title": title, "rule_code": code})
                continue

            # Check if this line is a standalone regulation code (e.g., "1-0-1")
            # This is less common but possible
            if code_pattern.match(line) and in_toc_section:
                # This would be a code on its own line
                # We'd need to look at previous line for title
                # For now, skip this edge case
                continue

        return toc_entries

    def _parse_regulation(self, toc_entry: Dict[str, str]) -> Optional[RegulationDocument]:
        """
        Parse a single regulation from the text.

        Args:
            toc_entry: Dictionary containing title and rule_code

        Returns:
            RegulationDocument or None if parsing fails
        """
        title = toc_entry["title"]
        rule_code = toc_entry["rule_code"]

        # Find where this regulation starts in the text
        start_idx = self._find_regulation_start(title, rule_code)
        if start_idx == -1:
            return None

        # Find where this regulation ends (next regulation or EOF)
        end_idx = self._find_regulation_end(start_idx)

        # Extract lines for this regulation
        reg_lines = self.lines[start_idx:end_idx]

        # Parse articles within the regulation
        articles = self._parse_articles(reg_lines)

        # Extract preamble (text before first article)
        preamble_lines = []
        for line in reg_lines:
            if self.ARTICLE_PATTERN.match(line.strip()):
                break
            if line.strip():
                preamble_lines.append(line.strip())
        preamble = " ".join(preamble_lines) if preamble_lines else None

        # Determine part from rule code
        part = self._extract_part(rule_code)

        # Create regulation document
        doc = RegulationDocument(
            id=str(uuid.uuid4()),
            title=title,
            rule_code=rule_code,
            part=part,
            metadata={
                "rule_code": rule_code,
                "part": part,
                "is废止": "【폐지】" in title,
            },
            content=[self._article_to_dict(art, title) for art in articles],
            preamble=preamble,
            status="revoked" if "【폐지】" in title else "effective",
            is_index_duplicate=False,
        )

        return doc

    def _find_regulation_start(self, title: str, rule_code: str) -> int:
        """Find the starting line index for a regulation."""
        # Search for the title or rule_code pattern
        search_patterns = [
            f"{title}{rule_code}",
            f"{title} {rule_code}",
            title,
        ]

        for i, line in enumerate(self.lines):
            for pattern in search_patterns:
                if pattern in line:
                    return i

        return -1

    def _find_regulation_end(self, start_idx: int) -> int:
        """Find the ending line index for a regulation."""
        # Look for the next regulation's TOC entry or end of file
        for i in range(start_idx + 1, len(self.lines)):
            line = self.lines[i].strip()
            # If we hit another TOC-style entry, that's the end
            if self.TOC_PATTERN.match(line):
                return i
            # If we hit a major section boundary (like "제X편")
            if re.match(r"^제\d+편", line):
                return i

        return len(self.lines)

    def _parse_articles(self, reg_lines: List[str]) -> List[RegulationArticle]:
        """Parse articles from regulation lines."""
        articles = []
        current_article: Optional[RegulationArticle] = None
        article_counter = 0

        for line in reg_lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a new article
            article_match = self.ARTICLE_PATTERN.match(line)
            if article_match:
                # Save previous article
                if current_article:
                    articles.append(current_article)

                # Start new article
                article_counter += 1
                article_no = f"제{article_match.group(1)}조"
                title = article_match.group(2) or ""

                current_article = RegulationArticle(
                    article_no=article_no,
                    title=title.strip(),
                    content="",
                    display_no=article_no,
                    sort_no={"main": article_counter, "sub": 0},
                )
            elif current_article:
                # Add content to current article
                if current_article.content:
                    current_article.content += " "
                current_article.content += line

                # Try to parse as items
                item_match = self.ITEM_PATTERN.match(line)
                if item_match:
                    item_num = item_match.group(1)
                    item_text = item_match.group(2)
                    current_article.items.append({
                        "type": "항",
                        "number": item_num,
                        "text": item_text,
                    })

        # Add last article
        if current_article:
            articles.append(current_article)

        return articles

    def _article_to_dict(self, article: RegulationArticle, doc_title: str) -> Dict[str, Any]:
        """Convert an article to dictionary format."""
        return {
            "id": str(uuid.uuid4()),
            "type": "article",
            "display_no": article.display_no,
            "sort_no": article.sort_no,
            "title": article.title,
            "text": f"{article.article_no} {article.title}",
            "full_text": f"[{doc_title}] {article.article_no} {article.title} {article.content}",
            "content": article.content,
            "confidence_score": 1.0,
            "references": [],
            "metadata": {
                "article_no": article.article_no,
                "items": article.items,
            },
            "children": [],
            "parent_path": [doc_title],
            "embedding_text": f"{doc_title}: {article.article_no} {article.title}",
            "chunk_level": "article",
            "is_searchable": True,
            "token_count": len(article.content.split()),
        }

    def _regulation_to_dict(self, regulation: RegulationDocument) -> Dict[str, Any]:
        """Convert a regulation document to dictionary format."""
        return {
            "id": regulation.id,
            "part": regulation.part,
            "title": regulation.title,
            "metadata": regulation.metadata,
            "preamble": regulation.preamble,
            "content": regulation.content,
            "addenda": regulation.addenda,
            "attached_files": regulation.attached_files,
            "doc_type": regulation.doc_type,
            "status": regulation.status,
            "is_index_duplicate": regulation.is_index_duplicate,
        }

    def _extract_part(self, rule_code: str) -> str:
        """Extract part number from rule code (e.g., "1-0-1" -> "제1편")."""
        parts = rule_code.split("-")
        if parts:
            return f"제{parts[0]}편"
        return ""

    def _extract_date_from_filename(self) -> Optional[str]:
        """Extract date from filename like '규정집9-343(20250909).hwpx'."""
        match = re.search(r"\((\d{8})\)", self.source_file.name)
        if match:
            date_str = match.group(1)
            try:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except ValueError:
                pass
        return None

    def save_json(self, output_path: Path) -> None:
        """
        Parse the HWPX file and save to JSON.

        Args:
            output_path: Path where JSON output will be saved
        """
        result = self.parse()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("\n=== Parsing Complete ===")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"Regulations found: {len(result['docs'])}")
        print(f"TOC entries: {len(result['toc'])}")
        if result['toc']:
            print("Sample TOC entries:")
            for entry in result['toc'][:5]:
                print(f"  - {entry['title']} ({entry['rule_code']})")


def main():
    """Command-line interface for testing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hwpx_parser_adapter.py <hwpx_file> [output_file]")
        sys.exit(1)

    hwpx_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path("data/output/규정집_hwpx_parsed.json")

    adapter = HwpxParserAdapter(hwpx_file)
    adapter.save_json(output_file)


if __name__ == "__main__":
    main()
