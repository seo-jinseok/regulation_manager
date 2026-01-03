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
    # 병합된 셀(rowspan/colspan)은 첫 번째 셀에만 내용을 넣고,
    # 나머지 병합 영역은 빈 셀로 처리하여 시각적으로 깔끔하게 표현
    assert tables[0].strip() == (
        "| 구분 | 인정기준 | 인정률 |\n"
        "| --- | --- | --- |\n"
        "| 논문 | 국제전문학술지 | 250 |\n"
        "|  | SCOPUS학술지 | 150 |"
    )


def test_replace_markdown_tables_with_html_uses_expanded_cells():
    """replace_markdown_tables_with_html은 마크다운 테이블을 HTML에서 재생성된 
    마크다운으로 대체한다. 병합 셀 영역은 빈 셀로 유지된다."""
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

    # 병합 영역은 빈 셀로 유지됨 (시각적 깔끔함을 위해)
    assert "|  | SCOPUS학술지 | 150 |" in replaced
    assert "본문" in replaced
