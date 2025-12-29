"""
HTML table conversion utilities.

Converts HTML tables with rowspan/colspan into expanded markdown tables.
"""

import re
from typing import List, Tuple

from bs4 import BeautifulSoup

from .table_extractor import TableExtractor


def convert_html_tables_to_markdown(html: str) -> List[str]:
    """Convert HTML tables to markdown tables with expanded spans."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    markdown_tables: List[str] = []

    for table in tables:
        grid, has_header = _table_to_grid(table)
        if not grid:
            continue
        markdown_tables.append(_grid_to_markdown(grid, has_header))

    return markdown_tables


def replace_markdown_tables_with_html(markdown: str, html: str) -> str:
    """Replace markdown tables with expanded tables parsed from HTML."""
    if not markdown or not html:
        return markdown

    html_tables = convert_html_tables_to_markdown(html)
    if not html_tables:
        return markdown

    extractor = TableExtractor()
    updated_text, markdown_tables = extractor.split_markdown_tables(markdown)
    if not markdown_tables:
        return markdown

    replacement_tables: List[str] = []
    for idx, original in enumerate(markdown_tables):
        if idx < len(html_tables):
            replacement_tables.append(html_tables[idx])
        else:
            replacement_tables.append(original)

    result = updated_text
    for idx, table in enumerate(replacement_tables, 1):
        result = result.replace(f"[TABLE:{idx}]", table)

    return result


def _table_to_grid(table) -> Tuple[List[List[str]], bool]:
    rows = table.find_all("tr")
    grid: List[List[str]] = []

    for r_index, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        while len(grid) <= r_index:
            grid.append([])

        col_index = 0
        for cell in cells:
            while col_index < len(grid[r_index]) and grid[r_index][col_index] is not None:
                col_index += 1

            text = _normalize_cell_text(cell.get_text(" ", strip=True))
            rowspan = _to_int(cell.get("rowspan"), default=1)
            colspan = _to_int(cell.get("colspan"), default=1)

            for r_offset in range(rowspan):
                target_row_index = r_index + r_offset
                while len(grid) <= target_row_index:
                    grid.append([])
                target_row = grid[target_row_index]
                if len(target_row) < col_index + colspan:
                    target_row.extend([None] * (col_index + colspan - len(target_row)))
                for c_offset in range(colspan):
                    # For colspan > 1, only put text in the first column
                    # For rowspan > 1, text is repeated in each row (markdown limitation)
                    if c_offset == 0:
                        target_row[col_index + c_offset] = text
                    else:
                        target_row[col_index + c_offset] = ""

            col_index += colspan

    max_cols = max((len(row) for row in grid), default=0)
    normalized_grid: List[List[str]] = []
    for row in grid:
        row_extended = row + [None] * (max_cols - len(row))
        normalized_grid.append([(cell or "") for cell in row_extended])

    has_header = bool(rows and rows[0].find_all("th"))
    return normalized_grid, has_header


def _grid_to_markdown(grid: List[List[str]], has_header: bool) -> str:
    if not grid:
        return ""

    column_count = len(grid[0])
    if not column_count:
        return ""

    if has_header:
        header = grid[0]
        body = grid[1:]
    else:
        header = [""] * column_count
        body = grid

    header_line = _format_markdown_row(header)
    separator_line = "| " + " | ".join(["---"] * column_count) + " |"
    body_lines = [_format_markdown_row(row) for row in body]

    lines = [header_line, separator_line] + body_lines
    return "\n".join(lines).strip()


def _format_markdown_row(row: List[str]) -> str:
    escaped = [cell.replace("|", "\\|") for cell in row]
    return "| " + " | ".join(escaped) + " |"


def _normalize_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _to_int(value, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
