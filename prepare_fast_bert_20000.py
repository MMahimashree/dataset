import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "splits" / "cleaned" / "total_68000"
OUTPUT = ROOT / "data" / "splits" / "cleaned" / "fast_20000"
SEED = 42
SIZES = {"train.json": 14000, "validation.json": 3000, "test.json": 3000}
ENTITY_TYPES = {"PERSON", "LOCATION", "PHONE", "EMAIL", "DOB", "PAN", "AADHAAR"}

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = {}
    for name, size in SIZES.items():
        path = SOURCE / name
        if not path.exists():
            raise FileNotFoundError(path)
        source[name] = read_rows(path)
        if len(source[name]) < size:
            raise ValueError(f"{name} has only {len(source[name])} rows; needs {size}")

    # Refuse to proceed if exact normalized text leaks across source splits.
    seen = {}
    for name, rows in source.items():
        for i, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Invalid text in {name}, row {i}")
            key = normalize(text)
            if key in seen and seen[key] != name:
                raise ValueError(f"Normalized exact-text overlap across source splits: {seen[key]} and {name}")
            seen[key] = name

    rng = random.Random(SEED)
    chosen_by_name = {}
    for name, size in SIZES.items():
        rows = list(source[name])
        rng.shuffle(rows)
        # First ensure every available entity label occurs in this subset.
        chosen = []
        chosen_ids = set()
        covered = set()
        for label in sorted(ENTITY_TYPES):
            candidate = next((r for r in rows
                              if label in {e.get("label") for e in r.get("entities", [])}
                              and id(r) not in chosen_ids), None)
            if candidate is not None:
                chosen.append(candidate)
                chosen_ids.add(id(candidate))
                covered.add(label)
        for row in rows:
            if len(chosen) >= size:
                break
            if id(row) not in chosen_ids:
                chosen.append(row)
                chosen_ids.add(id(row))
        if len(chosen) != size:
            raise RuntimeError(f"Could not select {size} records for {name}")
        rng.shuffle(chosen)
        chosen_by_name[name] = chosen

    selected_seen = {}
    for name, rows in chosen_by_name.items():
        for row in rows:
            key = normalize(row["text"])
            if key in selected_seen and selected_seen[key] != name:
                raise ValueError(f"Selected split overlap: {selected_seen[key]} and {name}")
            selected_seen[key] = name

    print("Source counts:", {k: len(v) for k, v in source.items()})
    print("Writing independent subset files to:", OUTPUT)
    for name, rows in chosen_by_name.items():
        with (OUTPUT / name).open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        counts = {label: sum(any(e.get("label") == label for e in r.get("entities", [])) for r in rows)
                  for label in sorted(ENTITY_TYPES)}
        print(f"{name}: {len(rows)} records | records containing labels: {counts}")
    print("TOTAL:", sum(len(v) for v in chosen_by_name.values()))
    print("Normalized exact-text overlap across selected splits: 0")
    print("Original total_68000 files were not modified.")

if __name__ == "__main__":
    main()
