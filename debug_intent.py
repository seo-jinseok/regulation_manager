import json
import re
from pathlib import Path

path = Path("data/config/intents.json")
try:
    data = json.loads(path.read_text())
    print("JSON is valid.")

    intents = data.get("intents", [])
    target = next((i for i in intents if i["id"] == "professor_complaint"), None)

    if target:
        print(f"Found intent: {target['label']}")
        query = "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래"

        # Check triggers
        matched_trigger = False
        for t in target["triggers"]:
            if t in query:
                print(f"Trigger match: '{t}'")
                matched_trigger = True

        # Check patterns
        matched_pattern = False
        for p in target["patterns"]:
            if re.search(p, query):
                print(f"Pattern match: '{p}'")
                matched_pattern = True

        if not matched_trigger and not matched_pattern:
            print("Failed to match triggers or patterns.")
    else:
        print("Intent 'professor_complaint' not found in JSON.")

except Exception as e:
    print(f"JSON Error: {e}")
