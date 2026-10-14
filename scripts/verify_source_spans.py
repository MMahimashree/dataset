
import json
from pathlib import Path
from datasets import load_dataset

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"

# Load the original source datasets.
print("Loading AI4Privacy...")
ai = load_dataset(
    "ai4privacy/pii-masking-200k",
    data_files=["english_pii_43k.jsonl"],
    split="train",
)

print("Loading Maskara...")
maskara = load_dataset(
    "somukandula/maskara-indian-pii-200k",
    split="train",
)

# Load current processed records.
with open(DATA_DIR / "train.json", encoding="utf-8") as f:
    train = json.load(f)

# Inspect records known to contain overlapping annotations.
indices = [25, 121, 123, 348, 485]

print("\nCURRENT PROCESSED EXAMPLES")
for idx in indices:
    if idx >= len(train):
        continue

    row = train[idx]
    print("\nText:", repr(row["text"]))
    for e in row["entities"]:
        print(
            e["label"],
            repr(row["text"][e["start"]:e["end"]]),
            e["start"],
            e["end"],
        )

print("\nSource datasets loaded successfully.")
print("AI4Privacy records:", len(ai))
print("Maskara records:", len(maskara))
print(
    "\nNext, we must match each processed record to its source "
    "using its text and source field before deciding how to fix offsets."
)
