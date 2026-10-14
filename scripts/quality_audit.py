
import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": DATA_DIR / "train.json",
    "validation": DATA_DIR / "validation.json",
    "test": DATA_DIR / "test.json",
}

TARGET_LABELS = {
    "PERSON",
    "LOCATION",
    "PHONE",
    "EMAIL",
    "DOB",
    "PAN",
    "AADHAAR",
}

SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}"
        )

    return data


def normalize_text(text):
    """Normalize whitespace and case for duplicate auditing."""
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("=" * 72)
    print("SEVEN-PII DATASET QUALITY AUDIT")
    print("=" * 72)

    datasets = {
        split: load_json(path)
        for split, path in SPLIT_FILES.items()
    }

    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    print("\n1. DOCUMENT COUNTS")

    for split, records in datasets.items():
        print(f"{split:12s}: {len(records):,}")

    total_records = sum(
        len(records) for records in datasets.values()
    )

    print(f"{'TOTAL':12s}: {total_records:,}")

    # --------------------------------------------------------
    # VALIDATE RECORDS AND ENTITY SPANS
    # --------------------------------------------------------

    entity_counts = Counter()
    document_counts = Counter()
    source_counts = Counter()
    source_entity_counts = defaultdict(Counter)
    split_entity_counts = defaultdict(Counter)
    split_document_counts = defaultdict(Counter)

    invalid_records = []
    invalid_entities = []
    overlap_records = []
    empty_entity_documents = []

    all_record_texts = defaultdict(list)

    for split, records in datasets.items():

        for index, record in enumerate(records):

            if not isinstance(record, dict):
                invalid_records.append({
                    "split": split,
                    "index": index,
                    "reason": "Record is not a JSON object",
                })
                continue

            text = record.get("text")
            entities = record.get("entities")
            source = record.get("source", "UNKNOWN")

            if not isinstance(text, str) or not text.strip():
                invalid_records.append({
                    "split": split,
                    "index": index,
                    "reason": "Missing or empty text",
                })
                continue

            if not isinstance(entities, list):
                invalid_records.append({
                    "split": split,
                    "index": index,
                    "reason": "Entities field is not a list",
                })
                continue

            source_counts[source] += 1
            all_record_texts[split].append(text)

            if not entities:
                empty_entity_documents.append({
                    "split": split,
                    "index": index,
                })

            valid_spans = []
            labels_in_document = set()

            for entity in entities:

                if not isinstance(entity, dict):
                    invalid_entities.append({
                        "split": split,
                        "index": index,
                        "entity": str(entity),
                        "reason": "Entity is not an object",
                    })
                    continue

                start = entity.get("start")
                end = entity.get("end")
                label = entity.get("label")

                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                ):
                    reason = "Start/end must be integers"

                elif start < 0 or end > len(text) or start >= end:
                    reason = "Invalid character boundaries"

                elif label not in TARGET_LABELS:
                    reason = "Unsupported PII label"

                elif not text[start:end].strip():
                    reason = "Entity span contains only whitespace"

                else:
                    reason = None

                if reason:
                    invalid_entities.append({
                        "split": split,
                        "index": index,
                        "entity": entity,
                        "reason": reason,
                    })
                    continue

                valid_spans.append(entity)

                entity_counts[label] += 1
                source_entity_counts[source][label] += 1
                split_entity_counts[split][label] += 1
                labels_in_document.add(label)

            for label in labels_in_document:
                document_counts[label] += 1
                split_document_counts[split][label] += 1

            # Detect overlapping valid annotations.
            valid_spans.sort(
                key=lambda entity: (
                    entity["start"],
                    entity["end"],
                )
            )

            furthest_end = -1
            previous_span = None

            for span in valid_spans:

                if span["start"] < furthest_end:
                    overlap_records.append({
                        "split": split,
                        "index": index,
                        "first_span": previous_span,
                        "overlapping_span": span,
                    })

                if span["end"] > furthest_end:
                    furthest_end = span["end"]
                    previous_span = span

    # --------------------------------------------------------
    # ENTITY COUNTS
    # --------------------------------------------------------

    print("\n2. PII ENTITY AND DOCUMENT COUNTS")

    for label in sorted(TARGET_LABELS):
        print(
            f"{label:10s} "
            f"entities={entity_counts[label]:>9,} "
            f"documents={document_counts[label]:>9,}"
        )

    # --------------------------------------------------------
    # SOURCE DISTRIBUTION
    # --------------------------------------------------------

    print("\n3. SOURCE DISTRIBUTION")

    for source, count in sorted(source_counts.items()):

        print(f"\n{source}: {count:,} documents")

        for label in sorted(TARGET_LABELS):
            print(
                f"  {label:10s} "
                f"{source_entity_counts[source][label]:,}"
            )

    # --------------------------------------------------------
    # SPLIT DISTRIBUTION
    # --------------------------------------------------------

    print("\n4. PII DISTRIBUTION BY SPLIT")

    for split in SPLIT_FILES:

        print(f"\n{split.upper()}")

        for label in sorted(TARGET_LABELS):
            print(
                f"  {label:10s} "
                f"entities={split_entity_counts[split][label]:,} "
                f"documents={split_document_counts[split][label]:,}"
            )

    # --------------------------------------------------------
    # EXACT AND NORMALIZED DUPLICATES
    # --------------------------------------------------------

    print("\n5. CROSS-SPLIT DUPLICATE AUDIT")

    exact_sets = {}
    normalized_sets = {}

    for split, records in datasets.items():

        exact_sets[split] = {
            record["text"]
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("text"), str)
        }

        normalized_sets[split] = {
            normalize_text(record["text"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("text"), str)
        }

    duplicate_summary = {}

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:

        exact_overlap = exact_sets[first] & exact_sets[second]
        normalized_overlap = (
            normalized_sets[first] & normalized_sets[second]
        )

        key = f"{first}_vs_{second}"

        duplicate_summary[key] = {
            "exact_duplicate_texts": len(exact_overlap),
            "normalized_duplicate_texts": len(normalized_overlap),
        }

        print(
            f"{first} vs {second}: "
            f"exact={len(exact_overlap):,}, "
            f"normalized={len(normalized_overlap):,}"
        )

    # --------------------------------------------------------
    # WITHIN-SPLIT DUPLICATES
    # --------------------------------------------------------

    print("\n6. DUPLICATES WITHIN EACH SPLIT")

    within_split_duplicates = {}

    for split, records in datasets.items():

        texts = [
            record["text"]
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("text"), str)
        ]

        normalized = [
            normalize_text(text)
            for text in texts
        ]

        within_split_duplicates[split] = {
            "exact_duplicate_excess": (
                len(texts) - len(set(texts))
            ),
            "normalized_duplicate_excess": (
                len(normalized) - len(set(normalized))
            ),
        }

        print(
            f"{split}: "
            f"exact excess="
            f"{within_split_duplicates[split]['exact_duplicate_excess']:,}, "
            f"normalized excess="
            f"{within_split_duplicates[split]['normalized_duplicate_excess']:,}"
        )

    # --------------------------------------------------------
    # SAVE DETAILED AUDIT REPORT
    # --------------------------------------------------------

    summary = {
        "random_seed_used_in_preparation": SEED,
        "total_documents": total_records,
        "documents_by_split": {
            split: len(records)
            for split, records in datasets.items()
        },
        "target_labels": sorted(TARGET_LABELS),
        "entity_counts": dict(entity_counts),
        "document_counts_by_label": dict(document_counts),
        "source_document_counts": dict(source_counts),
        "source_entity_counts": {
            source: dict(counts)
            for source, counts in source_entity_counts.items()
        },
        "split_entity_counts": {
            split: dict(counts)
            for split, counts in split_entity_counts.items()
        },
        "split_document_counts": {
            split: dict(counts)
            for split, counts in split_document_counts.items()
        },
        "invalid_record_count": len(invalid_records),
        "invalid_entity_count": len(invalid_entities),
        "overlap_count": len(overlap_records),
        "empty_entity_document_count": len(empty_entity_documents),
        "cross_split_duplicates": duplicate_summary,
        "within_split_duplicates": within_split_duplicates,
    }

    report_path = RESULTS_DIR / "dataset_quality_audit.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # Save detailed issues for review.
    issue_path = RESULTS_DIR / "dataset_quality_issues.json"

    with open(issue_path, "w", encoding="utf-8") as file:
        json.dump({
            "invalid_records": invalid_records,
            "invalid_entities": invalid_entities,
            "overlapping_entities": overlap_records,
            "empty_entity_documents": empty_entity_documents,
        }, file, indent=2, ensure_ascii=False)

    # Save a convenient entity statistics CSV.
    csv_path = RESULTS_DIR / "pii_statistics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "PII_Label",
            "Entity_Count",
            "Document_Count",
        ])

        for label in sorted(TARGET_LABELS):
            writer.writerow([
                label,
                entity_counts[label],
                document_counts[label],
            ])

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print("Total documents:", f"{total_records:,}")
    print("Invalid records:", len(invalid_records))
    print("Invalid entities:", len(invalid_entities))
    print("Overlapping annotations:", len(overlap_records))
    print("Documents with no entities:", len(empty_entity_documents))

    print("\nReports saved:")
    print(report_path)
    print(issue_path)
    print(csv_path)

    print("\nAUDIT COMPLETE")
    print("Review duplicate counts and label distribution before training.")


if __name__ == "__main__":
    main()
