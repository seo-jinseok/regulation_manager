import inspect
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.converter import HwpToMarkdownReader

sig = inspect.signature(HwpToMarkdownReader.load_data)
print(f"Signature: {sig}")
if 'verbose' in sig.parameters:
    print("SUCCESS: verbose parameter found.")
else:
    print("FAILURE: verbose parameter MISSING.")
