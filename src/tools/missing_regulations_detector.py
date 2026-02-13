#!/usr/bin/env python3
"""
Missing Regulations Detector and Recovery

This tool analyzes the JSON output and identifies missing regulations,
then attempts to recover them from the raw markdown.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_json(json_path: str) -> dict:
    """Load JSON file."""
    # Try the path as-is first, then try relative path
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Try relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, '..', '..', json_path)
        json_path = os.path.normpath(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def find_missing_regulations(data: dict) -> List[dict]:
    """
    Find regulations that are in TOC but not in parsed docs.

    Returns list of missing regulation entries from TOC.
    """
    toc = data.get('toc', [])
    docs = data.get('docs', [])

    # Get all regulation titles from docs (excluding TOC and indices)
    doc_titles = set()
    for doc in docs:
        kind = doc.get('kind', '')
        if kind not in ['toc', 'index_alpha', 'index_dept']:
            title = doc.get('title', '')
            if title:
                doc_titles.add(title)

    # Find TOC entries not in docs
    missing = []
    for toc_entry in toc:
        toc_title = toc_entry.get('title', '')
        # Check if any doc title contains this TOC title
        found = any(toc_title in doc_title for doc_title in doc_titles)
        if not found:
            missing.append(toc_entry)

    return missing


def extract_regulation_from_markdown(
    markdown_path: str,
    regulation_title: str,
    rule_code: str
) -> Optional[dict]:
    """
    Try to extract a regulation from raw markdown.

    Returns a dict with 'title', 'rule_code', and 'content' if found.
    """
    try:
        # Try the path as-is first
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            # Try relative path from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            markdown_path = os.path.join(script_dir, '..', '..', markdown_path)
            markdown_path = os.path.normpath(markdown_path)
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except FileNotFoundError:
        return None

    # Pattern to find the regulation section
    # Look for the title followed by content until next regulation or EOF
    lines = content.split('\n')

    # Find the title line (can have page headers attached)
    # First try to find the title with page numbers (content section)
    # Then fall back to plain title (TOC section)
    title_idx = None
    title_pattern_with_page = re.compile(rf'^{re.escape(regulation_title)}\s+\d+[—－]\d+[—－]\d+[~～]')
    title_pattern_plain = re.compile(rf'^{re.escape(regulation_title)}$')

    # First pass: look for title with page numbers (actual content)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if title_pattern_with_page.match(stripped):
            title_idx = i
            break

    # Second pass: if not found, try plain title
    if title_idx is None:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if title_pattern_plain.match(stripped):
                title_idx = i
                break

    if title_idx is None:
        return None

    # Extract content from title until next regulation title or EOF
    content_lines = []

    # Pattern to detect next regulation (ends with 규정, 세칙, etc. on its own line)
    # Must NOT start with ## (those are article headers)
    next_reg_pattern = re.compile(r'^[^#\s\d].*?(규정|세칙|지침|요령|강령|내규|학칙|헌장|기준|수칙|준칙|요강|운영|정관)(?:\s|$)')

    # Also stop at major section headers (## indicates articles, # would be major sections)
    section_patterns = [
        re.compile(r'^#\s*제\s*\d+\s*편'),  # Part header (single #) - NOT ## (articles)
        re.compile(r'^#*\s*차\s*례'),  # TOC
        re.compile(r'^#*\s*찾아보기'),  # Index
        re.compile(r'^제\s*\d+\s*편\s*$'),  # Another part pattern (only if it's the entire content)
    ]

    # Skip empty lines and artifacts after title to find actual article content
    # We need to skip "## 제N편" headers (page headers) until we find "## 제N조" (articles)
    content_start = title_idx + 1
    article_start_pattern = re.compile(r'^##\s*제\s*\d+조')  # Start of actual articles

    # Find the first article line
    while content_start < len(lines):
        stripped = lines[content_start].strip()
        if article_start_pattern.match(stripped):
            break
        content_start += 1
        # Safety: if we've gone too far without finding content, stop
        if content_start - title_idx > 100:
            content_start = title_idx + 1
            break

    # Now extract content from content_start until next regulation
    for i in range(content_start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Stop if we hit another regulation title (not starting with ##)
        # But make sure it's actually a regulation title, not a reference
        if next_reg_pattern.match(stripped):
            if not any(word in stripped for word in ['규정에', '규정제', '규정에따라', '에따라', '규정의']):
                break

        # Stop at major sections (only if NOT followed by article content)
        if any(pattern.match(stripped) for pattern in section_patterns):
            # Check if the next few lines have actual article content
            has_articles_after = False
            for j in range(i+1, min(i+10, len(lines))):
                if article_start_pattern.match(lines[j].strip()):
                    has_articles_after = True
                    break
            if not has_articles_after:
                break

        content_lines.append(line)

    if not content_lines:
        return None

    # Try to find articles in the content
    # Handle both formats: "## 제1조 (목적)" and "## 제1조"
    articles = []
    current_article = None

    # Pattern for "## 제1조 (목적) 내용" or "## 제1조의2 (목적) 내용" format
    article_pattern = re.compile(r'^##\s*제(\d+)조(?:의\s*(\d+))?\s*(?:\(([^)]+)\))?\s*(.*)')
    # Pattern for "## 제1조목적" format (no parentheses, title directly attached)
    article_pattern_no_paren = re.compile(r'^##\s*제(\d+)조(?:의\s*(\d+))?\s*([^\d\n].*)')
    # Simple pattern for non-markdown format
    simple_article_pattern = re.compile(r'^제(\d+)조(?:의\s*(\d+))?\s*(?:\(([^)]+)\))?\s*(.*)')

    for line in content_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Try ## format first (markdown headers)
        match = article_pattern.match(stripped)
        if not match:
            # Try format without parentheses: "## 제1조목적"
            match = article_pattern_no_paren.match(stripped)
        if not match:
            # Try simple format (non-markdown)
            match = simple_article_pattern.match(stripped)

        if match:
            if current_article:
                articles.append(current_article)

            article_no = match.group(1)
            sub_no = match.group(2) if match.group(2) else ''
            display_no = f"제{article_no}조"
            if sub_no:
                display_no += f"의{sub_no}"

            # Extract title from different groups depending on which pattern matched
            # Group 3 is parenthesized title, group 4 is non-parenthesized title
            title = ''
            if match.lastindex:
                # article_pattern: groups are (1:no, 2:sub, 3:paren_title, 4:rest)
                if len(match.groups()) >= 4 and match.group(3):
                    title = match.group(3).strip()
                elif len(match.groups()) >= 4 and match.group(4):
                    title = match.group(4).strip()
                elif len(match.groups()) >= 3 and match.group(3):
                    # article_pattern_no_paren: groups are (1:no, 2:sub, 3:title)
                    title = match.group(3).strip()

            current_article = {
                'display_no': display_no,
                'title': title,
                'content': [],
                'children': []
            }
        elif current_article:
            # Add content to current article
            # Handle numbered items (1., 2., etc.) as list items
            item_match = re.match(r'^(\d+)\.\s*(.*)', stripped)
            if item_match:
                current_article['children'].append({
                    'type': 'item',
                    'number': item_match.group(1),
                    'text': item_match.group(2)
                })
            # Handle sub-items (가., 나., etc.)
            elif re.match(r'^[가-하]\.', stripped):
                current_article['children'].append({
                    'type': 'subitem',
                    'text': stripped
                })
            # Handle circled numbers (①, ②, etc.) as paragraphs
            elif re.match(r'^[①-⑮]', stripped):
                current_article['content'].append(stripped)
            else:
                current_article['content'].append(stripped)

    if current_article:
        articles.append(current_article)

    # If no articles found, just return the raw content
    if not articles:
        return {
            'title': regulation_title,
            'rule_code': rule_code,
            'articles': [],
            'raw_content': '\n'.join(content_lines[:100])
        }

    return {
        'title': regulation_title,
        'rule_code': rule_code,
        'articles': articles,
        'raw_content': '\n'.join(content_lines[:500])  # First 500 lines for preview
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python missing_regulations_detector.py <json_file> [raw_markdown_file]")
        sys.exit(1)

    json_path = sys.argv[1]
    md_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing {json_path}...")

    # Load JSON
    data = load_json(json_path)

    # Get statistics
    toc_count = len(data.get('toc', []))
    docs_count = len(data.get('docs', []))

    print(f"\n=== Statistics ===")
    print(f"TOC items: {toc_count}")
    print(f"Parsed docs: {docs_count}")

    # Find missing regulations
    missing = find_missing_regulations(data)

    print(f"\n=== Missing Regulations ===")
    print(f"Total missing: {len(missing)}")

    if missing:
        print(f"\nFirst 20 missing regulations:")
        for i, reg in enumerate(missing[:20]):
            title = reg.get('title', 'Unknown')
            code = reg.get('rule_code', 'N/A')
            print(f"  {i+1}. {title} ({code})")

        # Try to extract from markdown if provided
        if md_path:
            print(f"\n=== Attempting Recovery from {md_path} ===")
            recovered = []

            for i, reg in enumerate(missing):  # Try all missing regulations
                # Progress indicator every 20 regulations
                if i > 0 and i % 20 == 0:
                    print(f"  Progress: {i}/{len(missing)} processed...", flush=True)

                title = reg.get('title', '')
                code = reg.get('rule_code', '')

                extracted = extract_regulation_from_markdown(md_path, title, code)
                if extracted:
                    recovered.append(extracted)
                    print(f"  ✓ Recovered: {title}", flush=True)

                    # Show preview
                    articles_count = len(extracted.get('articles', []))
                    print(f"    - {articles_count} articles found", flush=True)

            print(f"\n=== Recovery Summary ===")
            print(f"Attempted: {len(missing)}")
            print(f"Recovered: {len(recovered)}")

            if recovered:
                # Save recovered regulations
                output_path = json_path.replace('.json', '_recovered.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'missing_count': len(missing),
                        'recovered_count': len(recovered),
                        'recovered_regulations': recovered
                    }, f, ensure_ascii=False, indent=2)
                print(f"\nRecovered data saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
