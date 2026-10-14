

import json

import random

import time

from pathlib import Path



import numpy as np

import pandas as pd

import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from transformers import (

          AutoTokenizer,

          AutoModelForTokenClassification,

          get_linear_schedule_with_warmup,

)

from seqeval.metrics import (

          precision_score,

          recall_score,

          f1_score,

          classification_report,

)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





# ============================================================

# CONFIGURATION

# ============================================================



ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data/splits/cleaned/fast_2500/train.json"

VAL_FILE = ROOT / "data/splits/cleaned/fast_2500/validation.json"

TEST_FILE = ROOT / "data/splits/cleaned/fast_2500/test.json"



RESULTS = ROOT / "results/bert_fast2500"

CKPT_DIR = ROOT / "experiments/bert_fast2500/checkpoints"

BEST_DIR = CKPT_DIR / "best_model"

STATE_FILE = CKPT_DIR / "training_state.pt"



RESULTS.mkdir(parents=True, exist_ok=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)



MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 128

BATCH_SIZE = 4

GRAD_ACCUMULATION = 4

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.01

MAX_EPOCHS = 50

PATIENCE = 5

SAVE_EVERY_BATCHES = 500

SEED = 42



ENTITY_TYPES = [

          "PERSON", "LOCATION", "PHONE", "EMAIL",

          "DOB", "PAN", "AADHAAR",

]

LABELS = ["O"]

for entity in ENTITY_TYPES:

          LABELS.extend([f"B-{entity}", f"I-{entity}"])



LABEL2ID = {label: i for i, label in enumerate(LABELS)}

ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = DEVICE.type == "cuda"





# ============================================================

# REPRODUCIBILITY

# ============================================================



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

          torch.cuda.manual_seed_all(SEED)





def load_json(path):

          if not path.exists():

                    raise FileNotFoundError(path)

          with path.open("r", encoding="utf-8") as f:

                    records = json.load(f)

          if isinstance(records, dict):

                    records = records.get("data")

          if not isinstance(records, list):

                    raise ValueError(f"{path.name} must contain a JSON list.")

          for i, row in enumerate(records):

                    if not isinstance(row.get("text"), str):

                              raise ValueError(f"Invalid text in {path.name}, row {i}")

                    if not isinstance(row.get("entities"), list):

                              raise ValueError(f"Invalid entities in {path.name}, row {i}")

          return records





print("=" * 60)

print("BERT PII TOKEN CLASSIFICATION")

print("Device:", DEVICE)

if torch.cuda.is_available():

          print("GPU:", torch.cuda.get_device_name(0))

print("Maximum epochs:", MAX_EPOCHS)

print("Batch size:", BATCH_SIZE)

print("Effective batch size:", BATCH_SIZE * GRAD_ACCUMULATION)





# ============================================================

# DATASET AND TOKEN ALIGNMENT

# ============================================================



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)





class PIIDataset(Dataset):

          def __init__(self, records, name):

                    self.items = []

                    for row in tqdm(records, desc=f"Tokenizing {name}"):

                              text = row["text"]

                              enc = tokenizer(

                                        text,

                                        truncation=True,

                                        max_length=MAX_LENGTH,

                                        padding=False,

                                        return_offsets_mapping=True,

                                        return_special_tokens_mask=True,

                              )

                              offsets = enc["offset_mapping"]

                              special = enc["special_tokens_mask"]

                              tags = [None if special[i] else "O" for i in range(len(offsets))]



                              entities = []

                              for ent in row["entities"]:

                                        label, start, end = (

                                                  ent.get("label"), ent.get("start"), ent.get("end")

                                        )

                                        if label not in ENTITY_TYPES:

                                                  continue

                                        if not isinstance(start, int) or not isinstance(end, int):

                                                  continue

                                        if start < 0 or end <= start or end > len(text):

                                                  continue

                                        entities.append((start, end, label))



                              entities.sort()

                              previous_end = -1

                              for start, end, label in entities:

                                        if start < previous_end:

                                                  continue

                                        ids = [

                                                  i for i, (a, b) in enumerate(offsets)

                                                  if not special[i] and b > a and a < end and b > start

                                        ]

                                        if not ids or any(tags[i] != "O" for i in ids):

                                                  continue

                                        for j, token_id in enumerate(ids):

                                                  tags[token_id] = ("B-" if j == 0 else "I-") + label

                                        previous_end = end



                              self.items.append({

                                        "input_ids": enc["input_ids"],

                                        "attention_mask": enc["attention_mask"],

                                        "labels": [LABEL2ID[t] if t is not None else -100 for t in tags],

                              })



          def __len__(self):

                    return len(self.items)



          def __getitem__(self, index):

                    return self.items[index]





def collate(items):

          max_len = max(len(x["input_ids"]) for x in items)

          ids, masks, labels = [], [], []

          for x in items:

                    n = max_len - len(x["input_ids"])

                    ids.append(x["input_ids"] + [tokenizer.pad_token_id] * n)

                    masks.append(x["attention_mask"] + [0] * n)

                    labels.append(x["labels"] + [-100] * n)

          return {

                    "input_ids": torch.tensor(ids, dtype=torch.long),

                    "attention_mask": torch.tensor(masks, dtype=torch.long),

                    "labels": torch.tensor(labels, dtype=torch.long),

          }





train_records = load_json(TRAIN_FILE)

val_records = load_json(VAL_FILE)

test_records = load_json(TEST_FILE)



print("Records:", len(train_records), "train |",

               len(val_records), "validation |", len(test_records), "test")



train_ds = PIIDataset(train_records, "train")

val_ds = PIIDataset(val_records, "validation")

test_ds = PIIDataset(test_records, "test")



if not train_ds or not val_ds or not test_ds:

          raise ValueError("A dataset split is empty.")



# Fixed order makes mid-epoch resume deterministic.

# Training still uses all training records each epoch.

train_loader = DataLoader(

          train_ds, batch_size=BATCH_SIZE, shuffle=False,

          collate_fn=collate, num_workers=0,

)

val_loader = DataLoader(

          val_ds, batch_size=BATCH_SIZE, shuffle=False,

          collate_fn=collate, num_workers=0,

)

test_loader = DataLoader(

          test_ds, batch_size=BATCH_SIZE, shuffle=False,

          collate_fn=collate, num_workers=0,

)





# ============================================================

# MODEL, OPTIMIZER, SCHEDULER

# ============================================================



model = AutoModelForTokenClassification.from_pretrained(

          MODEL_NAME,

          num_labels=len(LABELS),

          id2label=ID2LABEL,

          label2id=LABEL2ID,

).to(DEVICE)



optimizer = torch.optim.AdamW(

          model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY

)



updates_per_epoch = (

          len(train_loader) + GRAD_ACCUMULATION - 1

) // GRAD_ACCUMULATION

total_updates = updates_per_epoch * MAX_EPOCHS

scheduler = get_linear_schedule_with_warmup(

          optimizer,

          num_warmup_steps=int(0.1 * total_updates),

          num_training_steps=total_updates,

)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)





# ============================================================

# CHECKPOINT SAVE / RESTORE

# ============================================================



def save_state(epoch, next_batch, best_f1, best_epoch,

                                       stale_epochs, history, epoch_loss_sum, batches_done):

          state = {

                    "epoch": epoch,

                    "next_batch": next_batch,

                    "model": model.state_dict(),

                    "optimizer": optimizer.state_dict(),

                    "scheduler": scheduler.state_dict(),

                    "scaler": scaler.state_dict(),

                    "best_f1": best_f1,

                    "best_epoch": best_epoch,

                    "stale_epochs": stale_epochs,

                    "history": history,

                    "epoch_loss_sum": epoch_loss_sum,

                    "batches_done": batches_done,

                    "python_rng": random.getstate(),

                    "numpy_rng": np.random.get_state(),

                    "torch_rng": torch.get_rng_state(),

                    "cuda_rng": torch.cuda.get_rng_state_all()

                                                  if torch.cuda.is_available() else None,

          }

          tmp = STATE_FILE.with_suffix(".tmp")

          torch.save(state, tmp)

          tmp.replace(STATE_FILE)





history = []

best_f1 = -1.0

best_epoch = 0

stale_epochs = 0

start_epoch = 1

resume_batch = 0

epoch_loss_sum = 0.0

batches_done = 0



if STATE_FILE.exists():

          print("\nFound checkpoint. Resuming saved training state.")

          state = torch.load(

                    STATE_FILE, map_location=DEVICE, weights_only=False

          )

          model.load_state_dict(state["model"])

          optimizer.load_state_dict(state["optimizer"])

          scheduler.load_state_dict(state["scheduler"])

          scaler.load_state_dict(state["scaler"])



          start_epoch = state["epoch"]

          resume_batch = state["next_batch"]

          best_f1 = state["best_f1"]

          best_epoch = state["best_epoch"]

          stale_epochs = state["stale_epochs"]

          history = state["history"]

          epoch_loss_sum = state["epoch_loss_sum"]

          batches_done = state["batches_done"]



          random.setstate(state["python_rng"])

          np.random.set_state(state["numpy_rng"])

          torch.set_rng_state(state["torch_rng"].cpu())

          if torch.cuda.is_available() and state["cuda_rng"] is not None:

                    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])



          if resume_batch >= len(train_loader):

                    start_epoch += 1

                    resume_batch = 0

                    epoch_loss_sum = 0.0

                    batches_done = 0



          print("Next epoch:", start_epoch)

          print("Next batch:", resume_batch + 1)





# ============================================================

# EVALUATION

# ============================================================



def decode(logits, gold):

          pred = logits.argmax(-1).detach().cpu().numpy()

          gold = gold.detach().cpu().numpy()

          true_seqs, pred_seqs = [], []

          for p_row, g_row in zip(pred, gold):

                    ts, ps = [], []

                    for p, g in zip(p_row, g_row):

                              if g == -100:

                                        continue

                              ts.append(ID2LABEL[int(g)])

                              ps.append(ID2LABEL[int(p)])

                    true_seqs.append(ts)

                    pred_seqs.append(ps)

          return true_seqs, pred_seqs





def evaluate(loader, name):

          model.eval()

          loss_sum = 0.0

          count = 0

          y_true, y_pred = [], []

          with torch.no_grad():

                    for batch in tqdm(loader, desc=name, unit="batch"):

                              batch = {k: v.to(DEVICE) for k, v in batch.items()}

                              with torch.autocast(

                                        device_type=DEVICE.type,

                                        dtype=torch.float16,

                                        enabled=USE_AMP,

                              ):

                                        out = model(**batch)

                              loss_sum += out.loss.item()

                              count += 1

                              a, b = decode(out.logits, batch["labels"])

                              y_true.extend(a)

                              y_pred.extend(b)

          metrics = {

                    "loss": loss_sum / max(count, 1),

                    "precision": precision_score(y_true, y_pred, zero_division=0),

                    "recall": recall_score(y_true, y_pred, zero_division=0),

                    "f1": f1_score(y_true, y_pred, zero_division=0),

          }

          return metrics, y_true, y_pred





# ============================================================

# TRAIN — RESUME AT SAVED BATCH

# ============================================================



try:

          for epoch in range(start_epoch, MAX_EPOCHS + 1):

                    model.train()

                    optimizer.zero_grad(set_to_none=True)



                    if epoch != start_epoch or resume_batch == 0:

                              epoch_loss_sum = 0.0

                              batches_done = 0



                    progress = tqdm(

                              enumerate(train_loader),

                              total=len(train_loader),

                              initial=resume_batch,

                              desc=f"Epoch {epoch}/{MAX_EPOCHS}",

                              unit="batch",

                    )



                    for step, batch in progress:

                              if step < resume_batch:

                                        continue



                              batch = {k: v.to(DEVICE) for k, v in batch.items()}

                              with torch.autocast(

                                        device_type=DEVICE.type,

                                        dtype=torch.float16,

                                        enabled=USE_AMP,

                              ):

                                        out = model(**batch)

                                        loss = out.loss / GRAD_ACCUMULATION



                              scaler.scale(loss).backward()

                              epoch_loss_sum += out.loss.item()

                              batches_done += 1



                              update = (

                                        (step + 1) % GRAD_ACCUMULATION == 0

                                        or step + 1 == len(train_loader)

                              )

                              if update:

                                        scaler.unscale_(optimizer)

                                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                                        scaler.step(optimizer)

                                        scaler.update()

                                        scheduler.step()

                                        optimizer.zero_grad(set_to_none=True)



                              progress.set_postfix(loss=f"{out.loss.item():.4f}")



                              if (step + 1) % SAVE_EVERY_BATCHES == 0:

                                        save_state(

                                                  epoch, step + 1, best_f1, best_epoch,

                                                  stale_epochs, history, epoch_loss_sum, batches_done,

                                        )

                                        progress.write(f"Checkpoint saved at batch {step + 1}")



                    train_loss = epoch_loss_sum / max(batches_done, 1)

                    val_metrics, _, _ = evaluate(val_loader, "Validation")



                    row = {

                              "epoch": epoch,

                              "train_loss": train_loss,

                              "val_loss": val_metrics["loss"],

                              "val_precision": val_metrics["precision"],

                              "val_recall": val_metrics["recall"],

                              "val_f1": val_metrics["f1"],

                    }

                    history.append(row)

                    pd.DataFrame(history).to_csv(

                              RESULTS / "training_history.csv", index=False

                    )



                    print("\nEpoch", epoch, row)



                    if val_metrics["f1"] > best_f1:

                              best_f1 = val_metrics["f1"]

                              best_epoch = epoch

                              stale_epochs = 0

                              model.save_pretrained(BEST_DIR)

                              tokenizer.save_pretrained(BEST_DIR)

                              print("Best validation checkpoint saved.")

                    else:

                              stale_epochs += 1



                    # Epoch is complete; next batch is zero of the next epoch.

                    save_state(

                              epoch + 1, 0, best_f1, best_epoch,

                              stale_epochs, history, 0.0, 0,

                    )

                    print("Full resumable checkpoint saved.")



                    resume_batch = 0

                    if stale_epochs >= PATIENCE:

                              print("Early stopping: validation F1 has plateaued.")

                              break



except KeyboardInterrupt:

          print("\nInterrupted. Saving the latest training state...")

          # Last periodic checkpoint is preserved if interruption occurs

          # between save points. The incomplete interval may need repeating.

          print("Use the last saved checkpoint to resume next time.")

          raise





# ============================================================

# GRAPHS

# ============================================================



if history:

          df = pd.DataFrame(history)



          plt.figure(figsize=(9, 5))

          plt.plot(df.epoch, df.train_loss, marker="o", label="Train loss")

          plt.plot(df.epoch, df.val_loss, marker="o", label="Validation loss")

          plt.xlabel("Epoch")

          plt.ylabel("Loss")

          plt.title("BERT Training and Validation Loss")

          plt.legend()

          plt.grid(True, alpha=0.3)

          plt.tight_layout()

          plt.savefig(RESULTS / "loss_curves.png", dpi=300)

          plt.close()



          plt.figure(figsize=(9, 5))

          for col in ["val_precision", "val_recall", "val_f1"]:

                    plt.plot(df.epoch, df[col], marker="o", label=col)

          plt.xlabel("Epoch")

          plt.ylabel("Score")

          plt.title("Validation Metrics")

          plt.ylim(0, 1.02)

          plt.legend()

          plt.grid(True, alpha=0.3)

          plt.tight_layout()

          plt.savefig(RESULTS / "validation_metrics.png", dpi=300)

          plt.close()





# ============================================================

# FINAL TEST — ONLY AFTER TRAINING COMPLETES

# ============================================================



if history and (
    start_epoch > MAX_EPOCHS
    or history[-1]["epoch"] >= MAX_EPOCHS
    or stale_epochs >= PATIENCE
):

          print("\nEvaluating best model on held-out test set.")

          model = AutoModelForTokenClassification.from_pretrained(

                    BEST_DIR

          ).to(DEVICE)



          test_metrics, y_true, y_pred = evaluate(test_loader, "Final test")

          print("Best epoch:", best_epoch)

          print("Best validation F1:", best_f1)

          print("Test metrics:", test_metrics)



          with (RESULTS / "test_metrics.json").open("w", encoding="utf-8") as f:

                    json.dump({

                              "model": MODEL_NAME,

                              "seed": SEED,

                              "best_epoch": best_epoch,

                              "best_validation_f1": best_f1,

                              "test_metrics": test_metrics,

                              "train_records": len(train_records),

                              "validation_records": len(val_records),

                              "test_records": len(test_records),

                    }, f, indent=2)



          report = classification_report(

                    y_true, y_pred, zero_division=0

          )

          (RESULTS / "classification_report.txt").write_text(

                    report, encoding="utf-8"

          )



          report_dict = classification_report(

                    y_true, y_pred, output_dict=True, zero_division=0

          )

          pd.DataFrame(report_dict).transpose().to_csv(

                    RESULTS / "per_label_metrics.csv"

          )



          flat_true = [x for seq in y_true for x in seq]

          flat_pred = [x for seq in y_pred for x in seq]

          cm = confusion_matrix(flat_true, flat_pred, labels=LABELS)



          fig, ax = plt.subplots(figsize=(15, 13))

          ConfusionMatrixDisplay(

                    confusion_matrix=cm, display_labels=LABELS

          ).plot(ax=ax, xticks_rotation=90, include_values=False)

          plt.title("Token-Level Confusion Matrix")

          plt.tight_layout()

          plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)

          plt.close()



          print("All results saved in:", RESULTS)

else:

          print("\nTraining paused before reaching the planned final epoch.")

          print("Resume with the same command when ready.")
