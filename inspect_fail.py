import json

def inspect():
    print("Start inspection...")
    path = "output/규정집9-350(20251209).json"
    enc = "utf-8"
    try:
        with open(path, 'r', encoding=enc) as f:
            print(f"Loading {path}...")
            data = json.load(f)
            print("Loaded.")
    except FileNotFoundError:
        # Fallback if filename encoding issue
        import glob
        print("File not found directly. Globbing...")
        files = glob.glob("output/*.json")
        if not files:
            print("No JSON found")
            return
        path = files[0]
        print(f"Loading fallback: {path}...")
        with open(path, 'r', encoding=enc) as f:
            data = json.load(f)
            print("Loaded.")

    docs = data['docs']
    if len(docs) > 348:
        doc = docs[348]
        print(f"--- Doc 348 ---")
        print(f"Title: '{doc.get('title')}'")
        print(f"Preamble:\n{doc.get('preamble')}")
        print("-" * 20)
        
        # Check previous doc's appendices to see if split failed
        prev = docs[347]
        print(f"--- Doc 347 (Previous) ---")
        print(f"Title: '{prev.get('title')}'")
        # Appendices might be popped now if I ran pipeline but I haven't run pipeline yet with popped appendices.
        # But 'addenda'/'attached_files' are there.
        # Check raw 'appendices' if present (it is currently present)
        # print(f"Appendices End:\n{prev.get('appendices', '')[-200:]}")
        # Actually I care about content
        pass
        
if __name__ == "__main__":
    inspect()
