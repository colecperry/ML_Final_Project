import os
import random
import pandas as pd
import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    set_seed,
    DebertaTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Remove all sources of randomness
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# ------------------------------
# 1) Load our pre-split data files
# ------------------------------
train_df = pd.read_csv("train_split_bert.csv")
val_df   = pd.read_csv("val_split_bert.csv")
test_df  = pd.read_csv("test_split_bert.csv")

# ------------------------------
# 2) Turn labels into 0/1 integers
# ------------------------------
label_mapping = {"real": 0, "fake": 1}
train_labels  = train_df["label"].map(label_mapping).tolist()
val_labels    = val_df["label"].map(label_mapping).tolist()
test_labels   = test_df["label"].map(label_mapping).tolist()

# ------------------------------
# 3) Prepare a PyTorch Dataset
# ------------------------------
tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-large")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = FakeNewsDataset(train_df["combined_text"].tolist(), train_labels, tokenizer)
val_dataset   = FakeNewsDataset(val_df["combined_text"].tolist(),   val_labels,   tokenizer)
test_dataset  = FakeNewsDataset(test_df["combined_text"].tolist(),  test_labels,  tokenizer)

# ------------------------------
# 4) Define metric function
# ------------------------------
def compute_metrics(pred):
    true_labels = pred.label_ids
    pred_labels = pred.predictions.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro")
    return {"precision": p, "recall": r, "f1": f1}

# ------------------------------
# 5) Train a “base” DeBERTa and find its best threshold
# ------------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-large", num_labels=2
)
base_args  = TrainingArguments(
    output_dir="deberta_base_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.0,
    logging_dir="logs",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED
)
base_trainer = Trainer(
    model=base_model,
    args=base_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("=== Training base DeBERTa ===")
base_trainer.train()

print("=== Threshold sweep (0.45–0.65) for base model ===")
val_output = base_trainer.predict(val_dataset)
base_probs = softmax(val_output.predictions, axis=1)[:, 1]

best_threshold = 0.0
best_f1_score  = 0.0
for thr in np.arange(0.45, 0.651, 0.01):
    preds = (base_probs >= thr).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(val_labels, preds, average="macro")
    if f1 > best_f1_score:
        best_f1_score, best_threshold = f1, thr

print(f"Base model → Best val-F1 = {best_f1_score:.4f} at thr = {best_threshold:.2f}")

# ------------------------------
# 6) Hyperparameter & threshold sweep
# ------------------------------
learning_rates = [1.5e-5, 1.8e-5, 2e-5, 2.2e-5, 2.5e-5]
batch_sizes    = [8, 16]
weight_decays  = [0.0, 0.01]

best_overall = {"f1": 0.0, "lr": None, "bs": None, "wd": None, "thr": None}

for lr in learning_rates:
    for bs in batch_sizes:
        for wd in weight_decays:
            print(f"\n--- Trying LR={lr}, BS={bs}, WD={wd} ---")
            combo_args = TrainingArguments(
                output_dir=f"deberta_lr{lr}_bs{bs}_wd{wd}",
                num_train_epochs=3,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs * 2,
                learning_rate=lr,
                warmup_ratio=0.1,
                weight_decay=wd,
                logging_dir="logs",
                logging_steps=50,
                save_strategy="no",
                report_to="none",
                seed=SEED
            )
            combo_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large", num_labels=2
            )
            combo_trainer = Trainer(
                model=combo_model,
                args=combo_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            combo_trainer.train()

            out   = combo_trainer.predict(val_dataset)
            probs = softmax(out.predictions, axis=1)[:, 1]

            best_thr, best_f1 = 0.0, 0.0
            for thr in np.arange(0.10, 0.91, 0.01):
                preds = (probs >= thr).astype(int)
                _, _, f1s, _ = precision_recall_fscore_support(val_labels, preds, average="macro")
                if f1s > best_f1:
                    best_f1, best_thr = f1s, thr

            print(f"→ val-F1 = {best_f1:.4f} at thr = {best_thr:.2f}")
            if best_f1 > best_overall["f1"]:
                best_overall.update({"f1": best_f1, "lr": lr, "bs": bs, "wd": wd, "thr": best_thr})

print("\n*** Best Validation Combo ***")
print(f"LR={best_overall['lr']}, BS={best_overall['bs']}, WD={best_overall['wd']}, "
      f"thr={best_overall['thr']:.2f}, F1={best_overall['f1']:.4f}")

# ------------------------------
# 7) Final training & test evaluation
# ------------------------------
print("\n=== Retraining best combo on train+val, then testing ===")
combined_df     = pd.concat([train_df, val_df], ignore_index=True)
combined_labels = combined_df["label"].map(label_mapping).tolist()
combined_ds     = FakeNewsDataset(
    combined_df["combined_text"].tolist(),
    combined_labels,
    tokenizer
)

final_args = TrainingArguments(
    output_dir="deberta_final_output",
    num_train_epochs=3,
    per_device_train_batch_size=best_overall["bs"],
    per_device_eval_batch_size=best_overall["bs"] * 2,
    learning_rate=best_overall["lr"],
    warmup_ratio=0.1,
    weight_decay=best_overall["wd"],
    logging_dir="logs",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED
)
final_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-large", num_labels=2
)
final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=combined_ds,
    tokenizer=tokenizer
)
final_trainer.train()

# test set prediction with locked threshold
test_out   = final_trainer.predict(test_dataset)
test_probs = softmax(test_out.predictions, axis=1)[:, 1]
test_preds = (test_probs >= best_overall["thr"]).astype(int)

# **detailed terminal output**
print(f"\nTest set size: {len(test_labels)}")
labels, counts = np.unique(test_labels, return_counts=True)
print(f"Label distribution (real, fake): {dict(zip(labels, counts))}\n")

cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix Counts:")
print(f"  True Negative (real→real): {tn}")
print(f"  False Positive (real→fake): {fp}")
print(f"  False Negative (fake→real): {fn}")
print(f"  True Positive (fake→fake): {tp}\n")

acc = accuracy_score(test_labels, test_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="macro"
)
print("Final Test Metrics:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  Macro-F1 : {f1:.4f}\n")
