
import json
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE / "data" / "processed"
OUTPUT_DIR = BASE / "data" / "splits" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = {
    "PERSON", "LOCATION", "PHONE", "EMAIL",
    "DOB", "PAN", "AADHAAR",
}

# Conservative patterns for structured PII.
PATTERNS = {
    "EMAIL": re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?:\s*(?:ext\.?|x)\s*\d+)?(?!\w)"
    ),
    "AADHAAR": re.compile(
        r"(?<!\d)(?:\d{4}[\s-]?\d{4}[\s-]?\d{4})(?!\d)"
    ),
    "PAN": re.compile(
        r"(?i)(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"
    ),
    "DOB": re.compile(
        r"(?i)\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|"
        r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b"
    ),
}

SPLITS = ["train", "validation", "test"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def valid_entity(text, entity):
    start = entity.get("start")
    end = entity.get("end")
    label = entity.get("label")

    if not isinstance(start, int) or not isinstance(end, int):
        return False

    if start < 0 or end > len(text) or start >= end:
        return False

    if label not in TARGET_LABELS:
        return False

    value = text[start:end]

    if not value.strip():
        return False

    # Check the actual text span for structured PII.
    if label in PATTERNS:
        return bool(PATTERNS[label].fullmatch(value.strip()))

    # PERSON and LOCATION need contextual validation;
    # don't pretend a simple regex can validate them.
    return True


def has_overlap(entities):
    ordered = sorted(
        entities,
        key=lambda e: (e["start"], e["end"])
    )

    previous_end = -1

    for entity in ordered:
        if entity["start"] < previous_end:
            return True

        previous_end = entity["end"]

    return False


def main():
    for split in SPLITS:
        input_path = INPUT_DIR / f"{split}.json"
        records = load_json(input_path)

        kept = []
        dropped_invalid = 0
        dropped_overlap = 0
        label_counts = Counter()

        seen_texts = set()

        for record in records:
            text = record.get("text")
            entities = record.get("entities")

            if not isinstance(text, str) or not text.strip():
                dropped_invalid += 1
                continue

            if not isinstance(entities, list) or not entities:
                dropped_invalid += 1
                continue

            # Drop the entire record if any annotation is invalid.
            # This avoids silently leaving potentially unlabelled PII
            # behind after removing a bad annotation.
            if not all(
                valid_entity(text, entity)
                for entity in entities
            ):
                dropped_invalid += 1
                continue

            if has_overlap(entities):
                dropped_overlap += 1
                continue

            normalized = " ".join(text.casefold().split())

            if normalized in seen_texts:
                continue

            seen_texts.add(normalized)

            kept.append({
                "text": text,
                "entities": entities,
                "source": record.get("source", "UNKNOWN"),
            })

            for entity in entities:
                label_counts[entity["label"]] += 1

        save_json(OUTPUT_DIR / f"{split}.json", kept)

        print(f"\n{split.upper()}")
        print("Input records:", len(records))
        print("Kept records:", len(kept))
        print("Dropped for invalid annotations:", dropped_invalid)
        print("Dropped for overlaps:", dropped_overlap)
        print("Entity counts:", dict(label_counts))

    print("\nFiltered candidate files saved to:")
    print(OUTPUT_DIR)
    print("\nRun a fresh audit before training.")
    print("These files are candidates, not yet a validated final dataset.")


if __name__ == "__main__":
    main()
