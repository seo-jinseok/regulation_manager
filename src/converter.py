"""
HWPX to HTML/Markdown Converter

Parses HWPX files (ZIP+XML format) and converts to HTML for further processing.
Preserves table structures including merged cells (rowspan, colspan).
"""
import logging
import re
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import xml.etree.ElementTree as ET

from llama_index.core.readers.base import BaseReader  # type: ignore[import]
from llama_index.core.schema import Document  # type: ignore[import]

logger = logging.getLogger(__name__)

# HWPX XML namespace mapping
HWPX_NS = {
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp10': 'http://www.hancom.co.kr/hwpml/2016/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
}


class HWPXTableParser:
    """Parse HWPX table structures and convert to HTML."""

    def __init__(self, ns: dict):
        self.ns = ns

    def parse_table(self, tbl_elem: ET.Element) -> str:
        """
        Parse HWPX table element and convert to HTML table.

        Args:
            tbl_elem: Table XML element.

        Returns:
            HTML table string.
        """
        row_cnt = int(tbl_elem.get('rowCnt', '0'))
        col_cnt = int(tbl_elem.get('colCnt', '0'))

        if row_cnt == 0 or col_cnt == 0:
            return ''

        html_parts = ['<table border="1" cellpadding="4" cellspacing="0">']

        rows = tbl_elem.findall('./hp:tr', self.ns)
        for tr_elem in rows:
            html_parts.append('<tr>')

            cells = tr_elem.findall('./hp:tc', self.ns)
            for tc_elem in cells:
                # Get cell span info
                cell_span = tc_elem.find('./hp:cellSpan', self.ns)
                colspan = 1
                rowspan = 1
                if cell_span is not None:
                    colspan = int(cell_span.get('colSpan', '1'))
                    rowspan = int(cell_span.get('rowSpan', '1'))

                # Extract cell content
                cell_text = self._extract_cell_text(tc_elem)

                # Build TD tag
                attrs = []
                if colspan > 1:
                    attrs.append(f'colspan="{colspan}"')
                if rowspan > 1:
                    attrs.append(f'rowspan="{rowspan}"')

                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                html_parts.append(f'<td{attr_str}>{cell_text}</td>')

            html_parts.append('</tr>')

        html_parts.append('</table>')
        return '\n'.join(html_parts)

    def _extract_cell_text(self, tc_elem: ET.Element) -> str:
        """Extract text content from a table cell."""
        text_parts = []

        # Find all paragraphs in cell
        sub_list = tc_elem.find('./hp:subList', self.ns)
        if sub_list is None:
            return ''

        for p in sub_list.findall('./hp:p', self.ns):
            # Extract text from runs
            for run in p.findall('.//hp:run', self.ns):
                for t in run.findall('./hp:t', self.ns):
                    if t.text:
                        text_parts.append(t.text)

        return '<br>'.join(text_parts).strip()


class HWPXToHTMLConverter:
    """Convert HWPX file to HTML preserving structure and tables."""

    def __init__(self):
        self.ns = HWPX_NS
        self.table_parser = HWPXTableParser(self.ns)

    def convert(self, file_path: Path, status_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Convert HWPX file to HTML.

        Args:
            file_path: Path to HWPX file.
            status_callback: Optional callback for progress updates.

        Returns:
            HTML string.
        """
        if status_callback:
            status_callback("[dim]HWPX 파일 파싱 중...[/dim]")

        html_parts = ['<html><body>']

        with zipfile.ZipFile(file_path, 'r') as zf:
            # Find all section XML files
            section_files = [f for f in zf.namelist() if f.startswith('Contents/section') and f.endswith('.xml')]

            if not section_files:
                section_files = [f for f in zf.namelist() if 'Contents' in f and f.endswith('.xml')]

            for idx, section_file in enumerate(sorted(section_files)):
                if status_callback:
                    status_callback(f"[dim]섹션 파싱 중 ({idx+1}/{len(section_files)}): {section_file}[/dim]")

                try:
                    with zf.open(section_file) as f:
                        content = f.read().decode('utf-8')
                        section_html = self._parse_section_to_html(content)
                        html_parts.append(section_html)
                except Exception as e:
                    logger.warning(f"Failed to parse {section_file}: {e}")
                    continue

        html_parts.append('</body></html>')
        return '\n'.join(html_parts)

    def _parse_section_to_html(self, xml_content: str) -> str:
        """Parse section XML and convert to HTML."""
        root = ET.fromstring(xml_content)
        html_parts = []

        # Iterate through all elements in document order
        for elem in root.iter():
            # Handle paragraphs
            if elem.tag == f'{{{self.ns["hp"]}}}p':
                para_html = self._parse_paragraph(elem)
                if para_html:
                    html_parts.append(para_html)

            # Handle tables
            elif elem.tag == f'{{{self.ns["hp"]}}}tbl':
                table_html = self.table_parser.parse_table(elem)
                if table_html:
                    html_parts.append(table_html)

        return '\n'.join(html_parts)

    def _parse_paragraph(self, p_elem: ET.Element) -> str:
        """Parse a paragraph element to HTML."""
        text_parts = []

        for run in p_elem.findall('.//hp:run', self.ns):
            for t in run.findall('./hp:t', self.ns):
                if t.text:
                    text_parts.append(t.text)

        if not text_parts:
            return ''

        text = ''.join(text_parts).strip()

        # Detect headers
        header_pattern = r'^제\s*\d+\s*[조장절편]'
        if re.match(header_pattern, text):
            return f'<h2>{text}</h2>'

        return f'<p>{text}</p>'


class HwpToMarkdownReader(BaseReader):
    """
    Custom LlamaIndex Reader for HWPX files (HWP XML format).

    Note: This system only accepts .hwpx files (HWP XML format).
    Uses direct HWPX parsing (ZIP + XML) to convert to HTML,
    then converts HTML to Markdown while preserving table structures.

    Table structures are fully preserved including merged cells.
    """

    def __init__(self, keep_html: bool = False):
        self.keep_html = keep_html

    def load_data(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> List[Document]:
        """
        Load data from HWPX file (HWP XML format).

        Args:
            file: Path to the HWPX file (.hwpx extension required).
            extra_info: Optional extra metadata to include.
            status_callback: Optional function that receives status updates.
            verbose: Whether to print verbose output.

        Returns:
            List of Document objects containing the converted content.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a .hwpx file.
        """
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

        # Validate file extension - only .hwpx files are supported
        if file.suffix.lower() != ".hwpx":
            raise ValueError(
                f"Unsupported file format: {file.suffix}. "
                f"This system only supports .hwpx files (HWP XML format). "
                f"If you have .hwp files (legacy binary format), "
                f"please convert them to .hwpx first using Hangul (Hwp): "
                f"File > Save As > HWPX file format."
            )

        # Convert HWPX to HTML
        converter = HWPXToHTMLConverter()
        html_content = converter.convert(file, status_callback)

        # Convert HTML to Markdown using markdownify
        try:
            from markdownify import markdownify as md
            markdown_content = md(html_content, heading_style="ATX")
        except ImportError:
            # Fallback: use HTML content as-is
            markdown_content = html_content

        # Create Document with both Markdown and HTML
        # HTML is kept for table preservation in downstream processing
        return [
            Document(
                text=markdown_content,  # Markdown content for processing
                metadata={
                    "file_name": file.name,
                    "source": str(file),
                    "html_content": html_content,  # Raw HTML for table preservation
                    **(extra_info or {}),
                },
            )
        ]


if __name__ == "__main__":
    # Test block
    import sys

    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) > 1:
        reader = HwpToMarkdownReader(keep_html=True)
        docs = reader.load_data(Path(sys.argv[1]))
        logger.debug(f"Converted HTML preview: {docs[0].text[:1000]}")
