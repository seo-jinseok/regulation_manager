#!/usr/bin/env python3
"""Test script for HWPX text extraction."""

from hwpx import TextExtractor
from pathlib import Path

def main():
    hwpx_file = Path('data/input/규정집9-343(20250909).hwpx')

    if not hwpx_file.exists():
        print(f"File not found: {hwpx_file}")
        return

    print(f"Extracting text from: {hwpx_file}")

    try:
        with TextExtractor(hwpx_file) as extractor:
            text = extractor.extract_text()
            print(f"\nTotal characters: {len(text)}")
            print(f"Total lines: {len(text.split(chr(10)))}")
            print("\n--- First 1000 characters ---")
            print(text[:1000])
            print("\n--- Lines 100-200 ---")
            lines = text.split('\n')
            for i, line in enumerate(lines[100:200], start=100):
                print(f"{i}: {line}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
