"""
Unit tests for HTML table conversion utilities.
"""

from src.parsing.html_table_converter import (
    convert_html_tables_to_markdown,
    replace_markdown_tables_with_html,
)


def test_convert_html_tables_expands_rowspan_and_colspan():
    html = """
    <table>
      <tr>
        <th>구분</th>
        <th>인정기준</th>
        <th>인정률</th>
      </tr>
      <tr>
        <td rowspan="2">논문</td>
        <td>국제전문학술지</td>
        <td>250</td>
      </tr>
      <tr>
        <td>SCOPUS학술지</td>
        <td>150</td>
      </tr>
    </table>
    """
    tables = convert_html_tables_to_markdown(html)

    assert len(tables) == 1
    assert tables[0].strip() == (
        "| 구분 | 인정기준 | 인정률 |\n"
        "| --- | --- | --- |\n"
        "| 논문 | 국제전문학술지 | 250 |\n"
        "| 논문 | SCOPUS학술지 | 150 |"
    )


def test_replace_markdown_tables_with_html_uses_expanded_cells():
    html = """
    <table>
      <tr><th>구분</th><th>인정기준</th><th>인정률</th></tr>
      <tr><td rowspan="2">논문</td><td>국제전문학술지</td><td>250</td></tr>
      <tr><td>SCOPUS학술지</td><td>150</td></tr>
    </table>
    """
    markdown = (
        "본문\n"
        "| 구분 | 인정기준 | 인정률 |\n"
        "| --- | --- | --- |\n"
        "| 논문 | 국제전문학술지 | 250 |\n"
        "|  | SCOPUS학술지 | 150 |\n"
    )

    replaced = replace_markdown_tables_with_html(markdown, html)

    assert "| 논문 | SCOPUS학술지 | 150 |" in replaced
    assert "|  | SCOPUS학술지 | 150 |" not in replaced
