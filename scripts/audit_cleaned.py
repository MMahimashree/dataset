
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "splits" / "cleaned"

SPLIT_NAMES = ["train", "validation", "test"]

LABELS = {
    "PERSON", "LOCATION", "PHONE", "EMAIL",
    "DOB", "PAN", "AADHAAR"
}

normalized_texts = {}
total_records = 0
total_entities = Counter()
total_invalid = 0
total_overlaps = 0

print("CLEANED DATASET AUDIT")
print("=" * 60)

for split_name in SPLIT_NAMES:
    file_path = DATA / f"{split_name}.json"

    if not file_path.exists():
        print(f"\nERROR: File not found: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    counts = Counter()
    invalid_spans = 0
    overlapping_records = 0
    normalized_texts[split_name] = set()

    for row in records:
        text = row.get("text", "")

        normalized_texts[split_name].add(
            " ".join(text.casefold().split())
        )

        entities = sorted(
            row.get("entities", []),
            key=lambda e: (e.get("start", -1), e.get("end", -1))
        )

        previous_end = -1
        has_overlap = False

        for entity in entities:
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label")

            # Validate entity structure and character offsets
            if (
                not isinstance(text, str)
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(text)
                or label not in LABELS
            ):
                invalid_spans += 1
                continue

            # Entity text is obtained from its character offsets.
            # No entity["text"] field is required.
            extracted_text = text[start:end]

            if not extracted_text.strip():
                invalid_spans += 1
                continue

            if start < previous_end:
                has_overlap = True

            previous_end = max(previous_end, end)
            counts[label] += 1

        if has_overlap:
            overlapping_records += 1

    total_records += len(records)
    total_entities.update(counts)
    total_invalid += invalid_spans
    total_overlaps += overlapping_records

    print(f"\n{split_name.upper()}: {len(records):,} records")
    print(f"Invalid annotations: {invalid_spans}")
    print(f"Records with overlapping annotations: {overlapping_records}")
    print("Entity instances:")

    for label in sorted(LABELS):
        print(f"  {label:10s} {counts[label]:8,}")

print("\n" + "=" * 60)
print("CROSS-SPLIT NORMALIZED DUPLICATE CHECK")

for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]:
    if a in normalized_texts and b in normalized_texts:
        overlap = normalized_texts[a] & normalized_texts[b]
        print(f"{a} vs {b}: {len(overlap):,} duplicate texts")

print("\n" + "=" * 60)
print(f"TOTAL RECORDS: {total_records:,}")
print(f"TOTAL INVALID ANNOTATIONS: {total_invalid:,}")
print(f"TOTAL RECORDS WITH OVERLAPS: {total_overlaps:,}")
print("TOTAL ENTITY INSTANCES:")

for label in sorted(LABELS):
    print(f"  {label:10s} {total_entities[label]:8,}")

print("\nAudit complete.")
print("Note: This checks annotation structure, not semantic correctness.")
