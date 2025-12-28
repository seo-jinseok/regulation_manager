import re


def extract_titles(log_path):
    titles = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"분석 완료: '(.*?)'", line)
            if match:
                titles.add(match.group(1))
    return titles


titles1 = extract_titles("logs/1.log")
titles2 = extract_titles("logs/2.log")

print(f"Log 1 (AI?): {len(titles1)} unique titles")
print(f"Log 2 (Non-AI?): {len(titles2)} unique titles")

only_in_1 = titles1 - titles2
only_in_2 = titles2 - titles1

print(f"\nIn Log 1 but NOT in Log 2 ({len(only_in_1)}):")
for t in sorted(list(only_in_1)):
    print(f"  - {t}")

print(f"\nIn Log 2 but NOT in Log 1 ({len(only_in_2)}):")
for t in sorted(list(only_in_2)):
    print(f"  - {t}")
