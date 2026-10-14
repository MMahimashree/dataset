
import json
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
ISSUES_PATH = BASE / "results" / "dataset_quality_issues.json"

TARGET_LABELS = {
    "PERSON", "LOCATION", "PHONE", "EMAIL",
    "DOB", "PAN", "AADHAAR",
}

AI_MAP = {
    "PERSON": "PERSON",
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "LOCATION": "LOCATION",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "DOB": "DOB",
}

MASKARA_MAP = {
    "PERSON_NAME": "PERSON",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "DATE_OF_BIRTH": "DOB",
    "AADHAAR": "AADHAAR",
    "PAN_CARD": "PAN",
    "LOCATION": "LOCATION",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading processed training data...")
    train = read_json(DATA_DIR / "train.json")

    print("Loading overlap report...")
    issues = read_json(ISSUES_PATH)
    overlaps = issues.get("overlapping_entities", [])

    # Find records in the current training split that have overlaps.
    bad_indices = sorted({
        item["index"]
        for item in overlaps
        if item["split"] == "train"
    })

    suspicious = [train[i] for i in bad_indices if i < len(train)]
    texts_needed = {
        record["text"] for record in suspicious
    }

    print("Overlapping training records:", len(suspicious))
    print("Loading source datasets...")

    ai = load_dataset(
        "ai4privacy/pii-masking-200k",
        data_files=["english_pii_43k.jsonl"],
        split="train",
    )

    maskara = load_dataset(
        "somukandula/maskara-indian-pii-200k",
        split="train",
    )

    # Index source records by their exact text.
    # Keep all matching rows in case source texts repeat.
    source_index = defaultdict(list)

    for row in ai:
        text = row.get("source_text", "")
        if text in texts_needed:
            entities = []

            for e in row.get("privacy_mask", []):
                label = AI_MAP.get(e.get("label"))
                if label in TARGET_LABELS:
                    entities.append({
                        "start": e["start"],
                        "end": e["end"],
                        "label": label,
                    })

            source_index[text].append({
                "source": "AI4Privacy",
                "entities": entities,
            })

    for row in maskara:
        text = row.get("text", "")
        if text in texts_needed:
            entities = []

            for e in row.get("entities", []):
                label = MASKARA_MAP.get(e.get("label"))
                if label in TARGET_LABELS:
                    entities.append({
                        "start": e["start"],
                        "end": e["end"],
                        "label": label,
                    })

            source_index[text].append({
                "source": "Maskara",
                "entities": entities,
            })

    print("\nSOURCE MATCH RESULTS")
    matched = 0
    unmatched = 0

    for record in suspicious[:20]:
        text = record["text"]
        matches = source_index.get(text, [])

        print("\nTEXT:", repr(text))
        print("Processed source:", record.get("source", "UNKNOWN"))

        if not matches:
            print("No exact source-text match found.")
            unmatched += 1
            continue

        matched += 1

        for source_record in matches:
            print("Matched source:", source_record["source"])
            print("Original mapped entities:")

            for e in source_record["entities"]:
                print(
                    e["label"],
                    repr(text[e["start"]:e["end"]]),
                    e["start"],
                    e["end"],
                )

        print("Processed entities:")

        for e in record.get("entities", []):
            print(
                e["label"],
                repr(text[e["start"]:e["end"]]),
                e["start"],
                e["end"],
            )

    print("\nSUMMARY")
    print("Matched examples:", matched)
    print("Unmatched examples:", unmatched)
    print("Displayed at most 20 examples.")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
