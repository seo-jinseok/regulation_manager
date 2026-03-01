#!/usr/bin/env python3
"""Check if 복무 regulation exists in ChromaDB."""
import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
col = client.get_collection("regulations")

# Get all unique regulation titles
all_docs = col.get(include=["metadatas", "documents"])
titles = set()
for meta in all_docs["metadatas"]:
    t = meta.get("regulation_title", "")
    if t:
        titles.add(t)

# Find 복무 related
print("=== Regulations containing '복무' ===")
for t in sorted(titles):
    if "복무" in t:
        print(f"  {t}")

print(f"\n=== Regulations containing '근무' ===")
for t in sorted(titles):
    if "근무" in t:
        print(f"  {t}")

print(f"\n=== Regulations containing '급여' or '봉급' or '보수' ===")
for t in sorted(titles):
    if any(k in t for k in ["급여", "봉급", "보수", "수당"]):
        print(f"  {t}")

print(f"\n=== Total unique regulations: {len(titles)} ===")

# Search for 복무 in document content
print("\n=== Documents mentioning '복무' (first 5) ===")
count = 0
for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    if "복무" in doc and count < 5:
        count += 1
        title = meta.get("regulation_title", "?")
        code = meta.get("regulation_code", "?")
        idx = doc.index("복무")
        snippet = doc[max(0, idx - 30):idx + 50]
        print(f"  [{code}] {title}: ...{snippet}...")
print(f"  Total docs with '복무': (checking...)")

복무_count = sum(1 for doc in all_docs["documents"] if "복무" in doc)
print(f"  Total docs with '복무': {복무_count}")
