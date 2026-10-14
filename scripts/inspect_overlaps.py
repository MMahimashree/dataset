
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
ISSUES_FILE = BASE_DIR / "results" / "dataset_quality_issues.json"

with open(ISSUES_FILE, "r", encoding="utf-8") as f:
    issues = json.load(f)

overlaps = issues.get("overlapping_entities", [])

print("Total overlapping records:", len(overlaps))
print("=" * 80)

shown = 0

for issue in overlaps:
    split = issue["split"]
    index = issue["index"]

    path = DATA_DIR / f"{split}.json"

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    record = records[index]
    text = record["text"]

    first = issue["first_span"]
    second = issue["overlapping_span"]

    print(f"\nSplit: {split}, record index: {index}")
    print("Text:", text)
    print(
        "First:",
        first["label"],
        repr(text[first["start"]:first["end"]]),
        f"[{first['start']}:{first['end']}]",
    )
    print(
        "Overlapping:",
        second["label"],
        repr(text[second["start"]:second["end"]]),
        f"[{second['start']}:{second['end']}]",
    )
    print("-" * 80)

    shown += 1
    if shown >= 10:
        break

print("\nDisplayed up to 10 examples.")
