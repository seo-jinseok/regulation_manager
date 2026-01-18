#!/usr/bin/env python3
"""인텐트 우선순위 테스트"""
import json
import re
from pathlib import Path

intents_path = Path("data/config/intents.json")
data = json.loads(intents_path.read_text(encoding="utf-8"))
intents = data.get("intents", [])

query = "연구 부정행위 신고하고 싶어"
print(f"Query: {query}\n")

matches = []
for intent in intents:
    for trigger in intent.get("triggers", []):
        if trigger in query:
            matches.append((intent["id"], intent["label"], trigger, "trigger"))
            break
    
    for pattern in intent.get("patterns", []):
        if re.search(pattern, query):
            matches.append((intent["id"], intent["label"], pattern, "pattern"))

print("Matches found:")
for m in matches:
    print(f"  - {m[1]} (id={m[0]}, via {m[3]}: '{m[2]}')")
