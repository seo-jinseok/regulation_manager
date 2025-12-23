import sys
import re
from pathlib import Path
from src.preprocessor import Preprocessor

# Initialize Preprocessor
preprocessor = Preprocessor()

# Load raw file
raw_path = Path("data/output/규정집9-343(20250909)_raw.md")
if not raw_path.exists():
    legacy_path = Path("output/규정집9-343(20250909)_raw.md")
    if legacy_path.exists():
        raw_path = legacy_path
    else:
        print(f"File not found: {raw_path}")
        sys.exit(1)

text = raw_path.read_text(encoding='utf-8')
print(f"Original length: {len(text)}")

# Check for PUA in original
chars_to_check = ['\uf85e', '\uf0fc', '\uf09e']
for c in chars_to_check:
    count = text.count(c)
    print(f"Original content has '{c}' (U+{ord(c):04X}): {count} times")

# Run cleaning
cleaned_text = preprocessor.clean(text)
print(f"Cleaned length: {len(cleaned_text)}")

# Verify
cleaned_pua_count = 0
for char in cleaned_text:
    if 0xE000 <= ord(char) <= 0xF8FF:
        cleaned_pua_count += 1
        print(f"Found remaining PUA: U+{ord(char):04X}")

# Check specific replacements
if '·' in cleaned_text:
    print(f"Cleaned text contains '·' (Middle Dot): {cleaned_text.count('·')} times")
if '✓' in cleaned_text:
    print(f"Cleaned text contains '✓' (Check Mark): {cleaned_text.count('✓')} times")

if cleaned_pua_count == 0:
    print("SUCCESS: No PUA characters found in cleaned text.")
else:
    print(f"FAILURE: Found {cleaned_pua_count} PUA characters.")
