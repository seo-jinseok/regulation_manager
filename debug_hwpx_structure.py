#!/usr/bin/env python3
"""Debug script to understand HWPX file structure."""

from hwpx import TextExtractor
import re

def main():
    hwpx_file = 'data/input/규정집9-343(20250909).hwpx'

    with TextExtractor(hwpx_file) as extractor:
        text = extractor.extract_text()

    lines = text.split('\n')
    print(f'Total lines: {len(lines)}')
    print(f'Total characters: {len(text)}')

    # Print first 200 lines
    print('\n=== Lines 0-200 ===')
    for i, line in enumerate(lines[:200]):
        line_clean = line.strip()
        if line_clean:
            print(f'{i:4d}: {line_clean[:100]}')

    # Find and print lines with regulation codes
    print('\n=== Lines with regulation codes (format: X-Y-Z) ===')
    for i, line in enumerate(lines):
        if re.search(r'\d+-\d+-\d+', line):
            print(f'{i:4d}: {line.strip()[:100]}')

    # Find sections
    print('\n=== Lines starting with "제" (section/article indicators) ===')
    for i, line in enumerate(lines):
        if line.strip().startswith('제'):
            print(f'{i:4d}: {line.strip()[:80]}')
            if i > 500:  # Limit output
                print('...')
                break

    # Find "차례" (table of contents)
    print('\n=== Context around "차례" (table of contents) ===')
    for i, line in enumerate(lines):
        if '차례' in line:
            start = max(0, i - 5)
            end = min(len(lines), i + 50)
            print(f'Found "차례" at line {i}:')
            for j in range(start, end):
                marker = ' >>>' if j == i else ''
                print(f'  {j:4d}{marker}: {lines[j][:100]}')
            print()

if __name__ == '__main__':
    main()
