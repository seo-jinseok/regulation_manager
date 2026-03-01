#!/usr/bin/env python3
"""Compare two evaluation results."""
import json
import sys

old_path = "data/evaluations/rag_quality_eval_20260301_125101.json"
new_path = "data/evaluations/rag_quality_eval_20260301_170705.json"

with open(old_path) as f:
    old = json.load(f)
with open(new_path) as f:
    new = json.load(f)

print("=== METRIC COMPARISON ===")
fmt = "{:<25} {:>8} {:>8} {:>8}"
print(fmt.format("Metric", "12:51", "17:07", "Delta"))
print("-" * 55)
for k in [
    "pass_rate",
    "avg_overall_score",
    "avg_accuracy",
    "avg_completeness",
    "avg_citations",
    "avg_context_relevance",
]:
    o = old["summary"].get(k, 0)
    n = new["summary"].get(k, 0)
    d = n - o
    sign = "+" if d > 0 else ""
    print(fmt.format(k, f"{o:.3f}", f"{n:.3f}", f"{sign}{d:.3f}"))

print("\n=== PERSONA COMPARISON ===")
for persona in old["persona_results"]:
    o = old["persona_results"].get(persona, {})
    n = new["persona_results"].get(persona, {})
    o_avg = o.get("avg_score", 0)
    n_avg = n.get("avg_score", 0)
    o_pr = o.get("pass_rate", 0)
    n_pr = n.get("pass_rate", 0)
    d = n_avg - o_avg
    sign = "+" if d > 0 else ""
    print(
        f"  {persona:<25} {o_avg:.3f}({o_pr:.0%}) -> {n_avg:.3f}({n_pr:.0%})  {sign}{d:.3f}"
    )

print("\n=== FAILED QUERIES (12:51 OLD) ===")
for persona, data in old["persona_results"].items():
    for r in data.get("results", []):
        if not r["passed"]:
            print(f"  [{persona}] {r['query']}")
            print(
                f"    Score: {r['overall_score']:.3f} | Acc={r['accuracy']:.3f} Comp={r['completeness']:.3f} Cit={r['citations']:.3f} Ctx={r['context_relevance']:.3f}"
            )
            issues = ", ".join(r.get("issues", []))
            print(f"    Issues: {issues}")
            print()

print("=== FAILED QUERIES (17:07 NEW) ===")
for persona, data in new["persona_results"].items():
    for r in data.get("results", []):
        if not r["passed"]:
            print(f"  [{persona}] {r['query']}")
            print(
                f"    Score: {r['overall_score']:.3f} | Acc={r['accuracy']:.3f} Comp={r['completeness']:.3f} Cit={r['citations']:.3f} Ctx={r['context_relevance']:.3f}"
            )
            issues = ", ".join(r.get("issues", []))
            print(f"    Issues: {issues}")
            print()

print("=== LOWEST 10 SCORES (NEW) ===")
all_results = []
for persona, data in new["persona_results"].items():
    for r in data.get("results", []):
        all_results.append((persona, r))
all_results.sort(key=lambda x: x[1]["overall_score"])
for persona, r in all_results[:10]:
    status = "FAIL" if not r["passed"] else "PASS"
    print(f"  {r['overall_score']:.3f} ({status}) [{persona}] {r['query']}")
    print(
        f"    Acc={r['accuracy']:.3f} Comp={r['completeness']:.3f} Cit={r['citations']:.3f} Ctx={r['context_relevance']:.3f}"
    )

print("\n=== QUERY-LEVEL DIFF (same queries) ===")
old_map = {}
for persona, data in old["persona_results"].items():
    for r in data.get("results", []):
        old_map[(persona, r["query"])] = r

for persona, data in new["persona_results"].items():
    for r in data.get("results", []):
        key = (persona, r["query"])
        if key in old_map:
            o_score = old_map[key]["overall_score"]
            n_score = r["overall_score"]
            delta = n_score - o_score
            if abs(delta) > 0.01:
                sign = "+" if delta > 0 else ""
                emoji = "UP" if delta > 0 else "DN"
                print(
                    f"  [{emoji}] {sign}{delta:.3f} [{persona}] {r['query']}"
                )
                print(f"       {o_score:.3f} -> {n_score:.3f}")

print("\n=== EVALUATION METHOD ANALYSIS ===")
for label, data in [("12:51", old), ("17:07", new)]:
    rule_based = 0
    llm_judged = 0
    for persona, pdata in data["persona_results"].items():
        for r in pdata.get("results", []):
            method = r.get("evaluation_method", "unknown")
            if method == "rule_based" or "rule" in method.lower():
                rule_based += 1
            else:
                llm_judged += 1
    print(f"  {label}: rule_based={rule_based}, llm_judged={llm_judged}")
