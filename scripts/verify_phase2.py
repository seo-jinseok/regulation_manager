import argparse
import glob
import json

def verify_phase2(input_dir: str):
    files = glob.glob(f"{input_dir}/*.json")
    if not files:
        print(f"FAIL: No JSON found in {input_dir}")
        exit(1)
    path = files[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    docs = data['docs']
    print(f"Total docs: {len(docs)}")
    
    total_refs = 0
    nodes_with_confidence = 0
    total_nodes = 0
    
    def check_nodes(nodes):
        nonlocal total_refs, nodes_with_confidence, total_nodes
        for node in nodes:
            total_nodes += 1
            if 'confidence_score' in node:
                nodes_with_confidence += 1
            if 'references' in node and node['references']:
                total_refs += len(node['references'])
            
            check_nodes(node.get('children', []))

    for doc in docs:
        check_nodes(doc.get('content', []))
        check_nodes(doc.get('addenda', []))

    print(f"Total nodes: {total_nodes}")
    print(f"Nodes with confidence_score: {nodes_with_confidence}")
    print(f"Total references extracted: {total_refs}")
    
    if total_nodes == 0:
        print("FAIL: No nodes found at all")
        exit(1)

    if nodes_with_confidence < total_nodes:
        print(f"FAIL: Only {nodes_with_confidence}/{total_nodes} nodes have confidence_score")
        exit(1)
        
    if total_refs == 0:
        print("FAIL: No cross-references extracted (expected at least some in university regulations)")
        exit(1)
        
    print("PASS: All Phase 2 elements present in output.")
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Phase 2 output.")
    parser.add_argument("--input_dir", default="data/output", help="Directory containing JSON outputs.")
    args = parser.parse_args()
    verify_phase2(args.input_dir)
