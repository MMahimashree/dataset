from datasets import load_dataset
from pathlib import Path
import json
import random
from collections import Counter

SEED = 42
random.seed(SEED)

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = {
    "PERSON",
    "LOCATION",
    "PHONE",
    "EMAIL",
    "DOB",
    "PAN",
    "AADHAAR",
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


def save_json(filename, data):
    path = PROCESSED / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {filename}: {len(data)} records")


# ============================================================
# AI4PRIVACY
# ============================================================

print("\n========== AI4PRIVACY ==========")

ai = load_dataset(
    "ai4privacy/pii-masking-200k",
    data_files=["english_pii_43k.jsonl"],
    split="train"
)

print("AI4Privacy total:", len(ai))

ai_records = []

for row in ai:
    entities = []

    for e in row["privacy_mask"]:
        old_label = e["label"]

        if old_label in AI_MAP:
            entities.append({
                "start": e["start"],
                "end": e["end"],
                "label": AI_MAP[old_label]
            })

    if entities:
        ai_records.append({
            "text": row["source_text"],
            "entities": entities,
            "source": "AI4Privacy"
        })

print("AI4Privacy usable:", len(ai_records))


# ============================================================
# MASKARA
# ============================================================

print("\n========== MASKARA ==========")

maskara = load_dataset(
    "somukandula/maskara-indian-pii-200k",
    split="train"
)

print("Maskara total:", len(maskara))

maskara_records = []

for row in maskara:
    entities = []

    for e in row["entities"]:
        old_label = e["label"]

        if old_label in MASKARA_MAP:
            entities.append({
                "start": e["start"],
                "end": e["end"],
                "label": MASKARA_MAP[old_label]
            })

    if entities:
        maskara_records.append({
            "text": row["text"],
            "entities": entities,
            "source": "Maskara"
        })

print("Maskara usable:", len(maskara_records))


# ============================================================
# COMBINE
# ============================================================

print("\n========== COMBINING ==========")

combined = ai_records + maskara_records
random.shuffle(combined)

print("Before duplicate removal:", len(combined))

# Exact text deduplication
unique = {}

for item in combined:
    unique[item["text"]] = item

combined = list(unique.values())

print("After duplicate removal:", len(combined))


# ============================================================
# ENTITY DISTRIBUTION
# ============================================================

counts = Counter()

for item in combined:
    for entity in item["entities"]:
        if entity["label"] in TARGET_LABELS:
            counts[entity["label"]] += 1

print("\n========== 7-PII DISTRIBUTION ==========")

for label in sorted(TARGET_LABELS):
    print(f"{label:10s}: {counts[label]}")


# ============================================================
# SPLIT 80 / 10 / 10
# ============================================================

random.shuffle(combined)

total = len(combined)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

train = combined[:train_end]
validation = combined[train_end:val_end]
test = combined[val_end:]

print("\n========== SPLIT ==========")
print("Total:", total)
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# SAVE
# ============================================================

save_json("train.json", train)
save_json("validation.json", validation)
save_json("test.json", test)

print("\n================================")
print("DATASET PREPARATION COMPLETE")
print("================================")