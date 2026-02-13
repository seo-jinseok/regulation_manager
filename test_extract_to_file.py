#!/usr/bin/env python3
"""Simple test to extract HWPX content to a text file."""

from hwpx import TextExtractor

def main():
    hwpx_file = 'data/input/규정집9-343(20250909).hwpx'
    output_file = 'data/output/hwpx_extracted_text.txt'

    print(f"Extracting text from: {hwpx_file}")
    with TextExtractor(hwpx_file) as extractor:
        text = extractor.extract_text()

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Saved {len(text)} characters to: {output_file}")
    print(f"Lines: {len(text.split(chr(10)))}")

    # Show sample
    lines = text.split('\n')
    print("\n=== Sample: Lines 100-200 ===")
    for i, line in enumerate(lines[100:200], start=100):
        if line.strip():
            print(f"{i}: {line[:100]}")

if __name__ == '__main__':
    main()
