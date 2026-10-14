"""Prepare a reproducible, small BERT experiment from total_68000.

Keeps the existing 68,000-record split unchanged. Creates 1,500 train,
300 validation, and 700 test records in data/splits/cleaned/fast_2500.
It requires all seven entity types to appear in every subset and checks
for normalized exact-text leakage between the new splits.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "splits" / "cleaned" / "total_68000"
OUTPUT_DIR = ROOT / "data" / "splits" / "cleaned" / "fast_2500"
SEED = 42
ENTITY_TYPES = {
    "PERSON", "LOCATION", "PHONE", "EMAIL", "DOB", "PAN", "AADHAAR"
}
SIZES = {"train.json": 1500, "validation.json": 300, "test.json": 700}


def load_records(filename):
    path = SOURCE_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for i, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not isinstance(row.get("entities"), list):
            raise ValueError(f"Malformed record {i} in {path}")
    return rows


def text_key(row):
    return re.sub(r"\s+", " ", row["text"]).strip().casefold()


def labels_in(row):
    return {e.get("label") for e in row["entities"] if e.get("label") in ENTITY_TYPES}


def choose_with_label_coverage(rows, n, rng, split_name):
    if len(rows) < n:
        raise ValueError(f"{split_name} has {len(rows)} records; needs {n}")
    shuffled = list(rows)
    rng.shuffle(shuffled)
    chosen = []
    chosen_ids = set()
    covered = set()
    # First select examples that add missing entity categories.
    while covered != ENTITY_TYPES:
        candidate = next(
            (r for r in shuffled if id(r) not in chosen_ids and labels_in(r) - covered),
            None,
        )
        if candidate is None:
            missing = sorted(ENTITY_TYPES - covered)
            raise ValueError(f"{split_name} source cannot cover entity labels: {missing}")
        chosen.append(candidate)
        chosen_ids.add(id(candidate))
        covered |= labels_in(candidate)
        if len(chosen) > n:
            raise ValueError(f"Need more than {n} records to cover all labels in {split_name}")
    for row in shuffled:
        if len(chosen) >= n:
            break
        if id(row) not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(id(row))
    rng.shuffle(chosen)
    return chosen


def main():
    rng = random.Random(SEED)
    source = {name: load_records(name) for name in SIZES}
    print("Source counts:", {name: len(rows) for name, rows in source.items()})

    # Fail rather than silently allow identical normalized text across splits.
    seen = {}
    for split_name, rows in source.items():
        for row in rows:
            key = text_key(row)
            if not key:
                continue
            if key in seen and seen[key] != split_name:
                raise ValueError(
                    "Exact normalized-text leakage found between "
                    f"{seen[key]} and {split_name}. Resolve duplicates before sampling."
                )
            seen[key] = split_name

    selected = {
        name: choose_with_label_coverage(source[name], n, rng, name)
        for name, n in SIZES.items()
    }

    # A second, explicit leakage check on the selected output.
    seen_output = {}
    for split_name, rows in selected.items():
        for row in rows:
            key = text_key(row)
            if key in seen_output and seen_output[key] != split_name:
                raise ValueError(f"Leakage in selected output: {seen_output[key]} vs {split_name}")
            seen_output[key] = split_name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in selected.items():
        with (OUTPUT_DIR / name).open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nCreated dataset:", OUTPUT_DIR)
    total = 0
    for name, rows in selected.items():
        counts = {label: sum(label in labels_in(r) for r in rows) for label in sorted(ENTITY_TYPES)}
        print(f"{name}: {len(rows)} records | records containing each label: {counts}")
        total += len(rows)
    print("TOTAL:", total)
    print("Normalized exact-text overlap across splits: 0")
    print("Original total_68000 files were not modified.")


if __name__ == "__main__":
    main()
