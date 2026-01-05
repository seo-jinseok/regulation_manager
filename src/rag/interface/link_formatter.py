"""
Link Formatter for Regulation References.

Extracts regulation references from text and formats them as
clickable links for different output formats (Markdown, CLI numbered list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RegulationRef:
    """A reference to a regulation found in text."""

    original_text: str  # The matched text (e.g., "교원인사규정 제8조")
    regulation_name: Optional[str]  # Regulation name (e.g., "교원인사규정")
    article: Optional[str]  # Article reference (e.g., "제8조")
    rule_code: Optional[str]  # Rule code if found (e.g., "3-1-24")
    start: int  # Start position in original text
    end: int  # End position in original text


# Patterns for extracting regulation references
# Pattern: 규정명 + 제N조 (optional 제N항, 제N호)
REGULATION_ARTICLE_PATTERN = re.compile(
    r"([가-힣]*(?:규정|학칙|내규|세칙|지침))\s*(제\d+조(?:\s*제?\d+항)?(?:\s*제?\d+호)?)"
)

# Pattern: Rule code (e.g., "3-1-24", "(3-1-24)")
RULE_CODE_PATTERN = re.compile(r"\(?(\d{1,2}-\d{1,2}-\d{1,3})\)?")

# Pattern: Standalone article (제N조)
STANDALONE_ARTICLE_PATTERN = re.compile(
    r"(?<![가-힣])(제\d+조(?:\s*제?\d+항)?(?:\s*제?\d+호)?)"
)


def extract_regulation_references(text: str) -> List[RegulationRef]:
    """
    Extract regulation references from text.

    Finds patterns like:
    - "교원인사규정 제8조"
    - "제15조 제2항"
    - "(규정번호: 3-1-24)"

    Args:
        text: The text to search for references.

    Returns:
        List of RegulationRef objects found in the text.
    """
    refs: List[RegulationRef] = []
    seen_spans: set[Tuple[int, int]] = set()

    # 1. Find regulation name + article patterns
    for match in REGULATION_ARTICLE_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)

        refs.append(
            RegulationRef(
                original_text=match.group(0),
                regulation_name=match.group(1),
                article=match.group(2).strip(),
                rule_code=None,
                start=match.start(),
                end=match.end(),
            )
        )

    # 2. Find rule codes and attach to nearby refs or create new ones
    for match in RULE_CODE_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if span in seen_spans:
            continue

        # Check if this is within 20 chars of an existing ref
        # Prioritize closest PRECEDING reference
        rule_code = match.group(1)
        best_ref = None
        min_dist = 20  # Max allowed distance

        for ref in refs:
            # Only consider refs that appear before or slightly overlapping (unlikely)
            # strictly: ref.end <= match.start()
            if ref.end <= match.start():
                dist = match.start() - ref.end
                if dist < min_dist:
                    min_dist = dist
                    best_ref = ref
            
        if best_ref:
            best_ref.rule_code = rule_code
            attached = True

        if not attached:
            seen_spans.add(span)
            refs.append(
                RegulationRef(
                    original_text=match.group(0),
                    regulation_name=None,
                    article=None,
                    rule_code=rule_code,
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Sort by position
    refs.sort(key=lambda r: r.start)
    return refs


def format_as_markdown_links(
    text: str,
    refs: List[RegulationRef],
    link_template: str = "javascript:void(0)",
) -> str:
    """
    Convert regulation references in text to Markdown links.

    Args:
        text: The original text.
        refs: List of RegulationRef objects.
        link_template: URL template. Use {rule_code}, {article} placeholders.

    Returns:
        Text with references converted to Markdown links.
    """
    if not refs:
        return text

    # Process in reverse order to preserve positions
    result = text
    for ref in sorted(refs, key=lambda r: r.start, reverse=True):
        link_url = link_template
        if "{rule_code}" in link_template and ref.rule_code:
            link_url = link_url.replace("{rule_code}", ref.rule_code)
        if "{article}" in link_template and ref.article:
            link_url = link_url.replace("{article}", ref.article)

        # Create markdown link
        link_text = f"[{ref.original_text}]({link_url})"
        result = result[: ref.start] + link_text + result[ref.end :]

    return result


def format_as_numbered_list(
    refs: List[RegulationRef],
    deduplicate: bool = True,
) -> str:
    """
    Format regulation references as a numbered list for CLI.

    Args:
        refs: List of RegulationRef objects.
        deduplicate: Whether to remove duplicate references.

    Returns:
        Formatted numbered list string.
    """
    if not refs:
        return ""

    # Deduplicate by (regulation_name, article, rule_code)
    seen = set()
    unique_refs = []
    for ref in refs:
        key = (ref.regulation_name, ref.article, ref.rule_code)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        unique_refs.append(ref)

    lines = []
    for i, ref in enumerate(unique_refs, 1):
        parts = []
        if ref.regulation_name:
            parts.append(ref.regulation_name)
        if ref.article:
            parts.append(ref.article)

        ref_text = " ".join(parts) if parts else ref.original_text

        if ref.rule_code:
            lines.append(f"  [{i}] {ref_text} ({ref.rule_code})")
        else:
            lines.append(f"  [{i}] {ref_text}")

    return "\n".join(lines)


def extract_and_format_references(
    text: str,
    format_type: str = "numbered",
) -> Tuple[List[RegulationRef], str]:
    """
    Extract references and format them.

    Args:
        text: The text to process.
        format_type: "numbered" for CLI, "markdown" for Web UI.

    Returns:
        Tuple of (refs, formatted_output).
    """
    refs = extract_regulation_references(text)

    if format_type == "numbered":
        formatted = format_as_numbered_list(refs)
    elif format_type == "markdown":
        formatted = format_as_markdown_links(text, refs)
    else:
        formatted = ""

    return refs, formatted
