import json

def inspect_tail():
    path = "dump.json"
    enc = "utf-8"
    try:
        with open(path, 'r', encoding=enc) as f:
            data = json.load(f)
    except Exception:
        import glob
        path = glob.glob("output/*.json")[0]
        with open(path, 'r', encoding=enc) as f:
            data = json.load(f)

    docs = data['docs']
    target_indices = [348] 
    
    for idx in target_indices:
        prev_idx = idx - 1
        if prev_idx < 0: continue
        
        prev = docs[prev_idx]
        print(f"=== Doc #{prev_idx} ({prev.get('title')}) Tail ===")
        
        # Check addenda/attached_files/appendices
        # We popped 'appendices' in refine_json... so it's gone.
        # But 'addenda' is there.
        # Wait, if I popped it, I can't see it to debug "what went wrong in parsing".
        # But I can see if the *Title* of the next doc ended up in the *Content* of the last addenda item?
        
        addenda = prev.get('addenda', [])
        if addenda:
            last_item = addenda[-1]
            content = last_item.get('content', '')
            if isinstance(content, list): # structured articles
                 # check last article content
                 if content:
                     last_art = content[-1]
                     print(f"[Last Addenda Article Content]:\n{last_art.get('content')}")
            else:
                 print(f"[Last Addenda Content]:\n{content[-300:]}")
        
        # Also check 'attached_files'
        atts = prev.get('attached_files', [])
        if atts:
            last_att = atts[-1]
            print(f"[Last Attachment Content]:\n{last_att.get('content', '')[-300:]}")

if __name__ == "__main__":
    inspect_tail()
