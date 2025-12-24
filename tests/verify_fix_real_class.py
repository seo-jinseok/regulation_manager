
import sys
import os
import json

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from metadata_extractor import MetadataExtractor

# Test data with table artifacts
text_data_table = """
규정집

찾아보기
<소관부서별>

학생군사교육단
학생군사교육단운영규정 5-1-24

| 제1편 |

| 학 교 법 인 |

학교법인동의학원정관 1-0-1
"""

print("--- Verifying Fix with Real Class ---")
extractor = MetadataExtractor()
result = extractor._extract_index_dept(text_data_table)

print(json.dumps(result, indent=2, ensure_ascii=False))

# Assertion
expected_count = 1  # Should only contain '학생군사교육단' with 1 item
if "학생군사교육단" in result and len(result["학생군사교육단"]) == 1:
    print("\n✅ Verification SUCCESS: '학교법인동의학원정관' was correctly excluded from '학생군사교육단'.")
else:
    print("\n❌ Verification FAILED: Parsing logic is still incorrect.")
    exit(1)
