#!/usr/bin/env python3
"""
Test script to verify the fixed HWPX parser achieves better detection rate.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from parsing.hwpx_direct_parser_v2 import HWPXDirectParser


def test_parser():
    """Test the fixed parser."""
    print("=" * 60)
    print("Testing Fixed HWPX Parser v2.1")
    print("=" * 60)

    hwpx_file = Path("/Users/truestone/Dropbox/repo/University/regulation_manager/data/input/규정집9-343(20250909).hwpx")

    if not hwpx_file.exists():
        print(f"ERROR: HWPX file not found: {hwpx_file}")
        return

    print(f"\nParsing: {hwpx_file.name}")
    print(f"File size: {hwpx_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Create parser
    parser = HWPXDirectParser()

    # Parse file
    print("\nParsing HWPX file...")
    result = parser.parse_file(hwpx_file)

    # Print results
    print("\n" + "=" * 60)
    print("PARSING RESULTS")
    print("=" * 60)

    metadata = result["metadata"]
    print(f"Parser Version: {metadata['parser_version']}")
    print(f"Parsing Time: {metadata['parsing_time_seconds']:.2f} seconds")
    print(f"\nTotal Regulations: {metadata['total_regulations']}")
    print(f"Successfully Parsed: {metadata['successfully_parsed']}")
    print(f"Success Rate: {metadata['success_rate']:.2f}%")

    # Print sample parsed regulations
    print("\n" + "=" * 60)
    print("SAMPLE PARSED REGULATIONS (First 30)")
    print("=" * 60)

    for i, doc in enumerate(result["docs"][:30], 1):
        print(f"{i:3}. {doc['title']}")

        if len(doc["articles"]) > 0:
            print(f"     Articles: {len(doc['articles'])} ({doc['articles'][0]['article_no']} - {doc['articles'][-1]['article_no']})")
        else:
            print(f"     Articles: 0")

    # Print TOC count
    print(f"\nTOC Entries: {len(result['toc'])}")

    # Save output
    output_file = Path("/Users/truestone/Dropbox/repo/University/regulation_manager") / "test_output_fixed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nOutput saved to: {output_file}")

    # Compare with expected
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    expected_min = 2000  # Expected minimum number of regulations
    actual = metadata['successfully_parsed']

    print(f"Expected Minimum: {expected_min}")
    print(f"Actual Parsed: {actual}")
    print(f"Achievement: {actual / expected_min * 100:.1f}%")

    if actual >= expected_min:
        print("\n✓ SUCCESS: Parser meets minimum requirement!")
    else:
        print(f"\n✗ NEEDS IMPROVEMENT: Parser found {expected_min - actual} fewer regulations than expected")

    # Check for duplicate titles
    titles = [doc['title'] for doc in result['docs']]
    unique_titles = set(titles)
    duplicates = len(titles) - len(unique_titles)

    print(f"\nTotal Titles: {len(titles)}")
    print(f"Unique Titles: {len(unique_titles)}")
    print(f"Duplicates: {duplicates}")

    if duplicates > 0:
        print("\n⚠ WARNING: Found duplicate titles (may indicate over-parsing)")
        # Find duplicates
        from collections import Counter
        dup_counts = Counter(titles)
        dup_titles = [t for t, c in dup_counts.items() if c > 1]
        print(f"Duplicate title examples (first 10):")
        for title in dup_titles[:10]:
            print(f"  - {title} ({dup_counts[title]}x)")


if __name__ == "__main__":
    test_parser()
