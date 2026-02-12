#!/usr/bin/env python3
"""
Debug script to analyze HWPX parser and identify why 67% of regulations fail.
"""
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

# HWPX XML namespace mapping
HWPX_NS = {
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp10': 'http://www.hancom.co.kr/hwpml/2016/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
}


def extract_paragraph_text(p_elem: ET.Element, ns: Dict[str, str]) -> str:
    """Extract text content from a paragraph element."""
    text_parts = []

    for run in p_elem.findall('.//hp:run', ns):
        for t in run.findall('./hp:t', ns):
            if t.text:
                text_parts.append(t.text)

    return ''.join(text_parts).strip()


def is_regulation_title(text: str) -> bool:
    """Detect if text is a regulation title."""
    # Skip non-regulation content
    skip_patterns = [
        r'^\d+\.',  # Numbered lists (1. 2. 3.)
        r'^[가-힣]{2,3}\.',  # Korean letter prefixes (가., 나., 다.)
        r'^이\s*규정집은',  # "이 규정집은..."
        r'^제\s*\d+\s*편',  # Part markers (handled separately)
        r'^제\s*\d+\s*장',  # Chapter markers (handled separately)
        r'^제\s*\d+\s*절',  # Section markers (handled separately)
        r'^[①-⑮]',  # Paragraph numbers
        r'^제\s*\d+조',  # Article markers (handled separately)
        r'^동의대학교\s*규정집',  # TOC title
        r'^편찬례',  # TOC elements
        r'^총\s*장',  # TOC elements
        r'^추록',  # TOC elements
        r'^부록',  # TOC elements
        r'^목\s*차',  # TOC elements
    ]

    for pattern in skip_patterns:
        if re.match(pattern, text):
            return False

    # Regulation titles are typically main headings without article numbers
    regulation_patterns = [
        r'^(.+[규정요령지침세칙])$',  # Ends with regulation keyword
        r'^(.+?)(규정|요령|지침|시행세칙|시행규칙|업무처리지침)(?=\s|$)',  # Contains keyword
        r'^(.+?)(에 관한|관련)(규정|요령|지침)',  # "에 관한 규정" pattern
    ]

    # Must be reasonable length
    if len(text) < 5 or len(text) > 200:
        return False

    for pattern in regulation_patterns:
        if re.search(pattern, text):
            return True

    return False


def analyze_hwpx_file(file_path: Path) -> Dict[str, Any]:
    """Analyze HWPX file to identify parsing issues."""
    print(f"Analyzing HWPX file: {file_path.name}")
    print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

    results = {
        "file_name": file_path.name,
        "file_size_mb": file_path.stat().st_size / 1024 / 1024,
        "sections": {},
        "sample_detected_titles": [],
        "sample_undetected_texts": [],
        "paragraph_samples": [],
        "namespace_info": {},
        "analysis": {}
    }

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # List all files in ZIP
            print(f"\nFiles in ZIP archive:")
            all_files = sorted(zf.namelist())
            for f in all_files[:50]:  # Show first 50 files
                print(f"  {f}")

            # Find section XML files
            section_files = [
                f for f in zf.namelist()
                if f.startswith('Contents/section') and f.endswith('.xml')
            ]

            if not section_files:
                section_files = [
                    f for f in zf.namelist()
                    if 'Contents' in f and f.endswith('.xml')
                ]

            print(f"\nFound {len(section_files)} section files")

            for section_file in sorted(section_files):
                print(f"\n--- Analyzing {section_file} ---")

                try:
                    with zf.open(section_file) as f:
                        content = f.read().decode('utf-8')

                    # Parse XML
                    root = ET.fromstring(content)

                    # Get namespace info
                    results["namespace_info"][section_file] = {
                        "root_tag": root.tag,
                        "root_attrib": root.attrib,
                        "children_count": len(root),
                    }

                    # Sample paragraph texts
                    paragraph_count = 0
                    detected_count = 0
                    undetected_samples = []
                    detected_samples = []

                    for elem in root.iter():
                        if elem.tag == f'''{{{HWPX_NS["hp"]}}}p''':
                            paragraph_count += 1
                            text = extract_paragraph_text(elem, HWPX_NS)

                            if text and paragraph_count <= 100:
                                results["paragraph_samples"].append({
                                    "paragraph_num": paragraph_count,
                                    "text": text,
                                    "is_detected": is_regulation_title(text),
                                })

                                if is_regulation_title(text):
                                    detected_count += 1
                                    if len(detected_samples) < 20:
                                        detected_samples.append(text)
                                else:
                                    if len(undetected_samples) < 20 and len(text) > 10:
                                        undetected_samples.append(text)

                    results["sections"][section_file] = {
                        "paragraph_count": paragraph_count,
                        "detected_regulations": detected_count,
                        "detection_rate": detected_count / paragraph_count * 100 if paragraph_count > 0 else 0,
                    }

                    print(f"  Total paragraphs: {paragraph_count}")
                    print(f"  Detected as regulation titles: {detected_count}")
                    print(f"  Detection rate: {detected_count / paragraph_count * 100 if paragraph_count > 0 else 0:.2f}%")

                    results["sample_detected_titles"] = detected_samples
                    results["sample_undetected_texts"] = undetected_samples[:50]

                except Exception as e:
                    print(f"  Error analyzing {section_file}: {e}")
                    results["sections"][section_file] = {"error": str(e)}

    except Exception as e:
        print(f"Error opening ZIP file: {e}")
        results["error"] = str(e)

    # Analysis
    total_paragraphs = sum(s.get("paragraph_count", 0) for s in results["sections"].values() if isinstance(s, dict))
    total_detected = sum(s.get("detected_regulations", 0) for s in results["sections"].values() if isinstance(s, dict))

    results["analysis"] = {
        "total_paragraphs": total_paragraphs,
        "total_detected": total_detected,
        "overall_detection_rate": total_detected / total_paragraphs * 100 if total_paragraphs > 0 else 0,
        "potential_issue": "Low detection rate" if total_detected / total_paragraphs < 0.5 else "OK"
    }

    print(f"\n=== Overall Analysis ===")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Detected as regulations: {total_detected}")
    print(f"Detection rate: {total_detected / total_paragraphs * 100 if total_paragraphs > 0 else 0:.2f}%")

    return results


def main():
    """Main entry point."""
    data_dir = Path("/Users/truestone/Dropbox/repo/University/regulation_manager/data/input")
    hwpx_files = list(data_dir.glob("*.hwpx"))

    if not hwpx_files:
        print("No HWPX files found in data/input directory")
        return

    print(f"Found {len(hwpx_files)} HWPX file(s)")

    for hwpx_file in hwpx_files:
        results = analyze_hwpx_file(hwpx_file)

        # Save results
        output_file = Path("/Users/truestone/Dropbox/repo/University/regulation_manager") / f"hwpx_debug_{hwpx_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nDebug results saved to: {output_file}")

        # Print samples
        print("\n=== Sample Detected Regulation Titles ===")
        for i, title in enumerate(results["sample_detected_titles"][:10], 1):
            print(f"{i}. {title}")

        print("\n=== Sample Undetected Texts (Potential Regulations) ===")
        for i, text in enumerate(results["sample_undetected_texts"][:20], 1):
            print(f"{i}. {text}")


if __name__ == "__main__":
    main()
