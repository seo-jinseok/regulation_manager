
import re
from src.rag.infrastructure.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()
queries = ["장학금규정", "휴학하려면?", "휴학"]

print("Analyzing queries against INTENT_PATTERNS...")
for query in queries:
    print(f"\nQuery: '{query}'")
    matched = False
    for i, (pattern, keywords) in enumerate(analyzer.INTENT_PATTERNS):
        if pattern.search(query):
            print(f"  Matched Pattern #{i}: {pattern.pattern}")
            print(f"  Keywords: {keywords}")
            matched = True
    
    if not matched:
        print("  No patterns matched.")
    
    query_type = analyzer.analyze(query)
    print(f"  Resulting QueryType: {query_type}")
