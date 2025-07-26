import os
import random
import pandas as pd
import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    set_seed,
    RobertaTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ── 0) Remove all sources of randomness for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# ── 1) Data loading
train_df = pd.read_csv("train_split_bert.csv")
val_df   = pd.read_csv("val_split_bert.csv")
test_df  = pd.read_csv("test_split_bert.csv")

# ── 2) Labels → 0/1
label_map    = {"real": 0, "fake": 1}
train_labels = train_df["label"].map(label_map).tolist()
val_labels   = val_df["label"].map(label_map).tolist()
test_labels  = test_df["label"].map(label_map).tolist()

# ── 3) Tokenizer & Dataset
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = FakeNewsDataset(train_df["combined_text"].tolist(), train_labels, tokenizer)
val_ds   = FakeNewsDataset(val_df["combined_text"].tolist(),   val_labels,   tokenizer)
test_ds  = FakeNewsDataset(test_df["combined_text"].tolist(),  test_labels,  tokenizer)

# ── 4) Metric fn
def compute_metrics(pred):
    labels     = pred.label_ids
    preds      = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ── 5) Base model setup & training
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
base_args  = TrainingArguments(
    output_dir="bert_base_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=50,
    report_to="none",
    seed=SEED
)
base_trainer = Trainer(
    model=base_model,
    args=base_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("=== Training base model ===")
base_trainer.train()

# ── 6) Lock in threshold on validation
threshold = 0.30
val_out    = base_trainer.predict(val_ds)
fake_probs = torch.softmax(torch.tensor(val_out.predictions), dim=1)[:, 1].numpy()
val_preds  = (fake_probs >= threshold).astype(int)

print(f"\nValidation @ thresh={threshold}")
print(classification_report(val_labels, val_preds, target_names=["real","fake"]))

# ── 7) Hyperparameter sweep on validation
best_val_f1 = 0.0
best_params = {}

for lr in [1e-5, 2e-5, 3e-5]:
    for bs in [8, 16]:
        print(f"\n--- Trying LR={lr}, BS={bs} ---")
        sweep_args = TrainingArguments(
            output_dir=f"bert_out_lr{lr}_bs{bs}",
            num_train_epochs=3,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs * 2,
            learning_rate=lr,
            logging_dir="logs",
            logging_steps=50,
            report_to="none",
            seed=SEED
        )
        sweep_trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2),
            args=sweep_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        sweep_trainer.train()
        sweep_out   = sweep_trainer.predict(val_ds)
        sweep_probs = torch.softmax(torch.tensor(sweep_out.predictions), dim=1)[:, 1].numpy()
        sweep_preds = (sweep_probs >= threshold).astype(int)
        _, _, f1_sweep, _ = precision_recall_fscore_support(val_labels, sweep_preds, average="macro")
        print(f"Val F1 = {f1_sweep:.4f} @ thresh={threshold}")
        if f1_sweep > best_val_f1:
            best_val_f1 = f1_sweep
            best_params = {"learning_rate": lr, "batch_size": bs}

print(f"\n*** Best hyperparams = {best_params}, Val F1={best_val_f1:.4f} ***")

# ── 8) Train with best params on train+val, then test
print("\n=== Final training on train+val and testing ===")
combined_df     = pd.concat([train_df, val_df], ignore_index=True)
combined_labels = combined_df["label"].map(label_map).tolist()
combined_ds     = FakeNewsDataset(combined_df["combined_text"].tolist(), combined_labels, tokenizer)

final_args = TrainingArguments(
    output_dir="bert_final_output",
    num_train_epochs=3,
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"] * 2,
    learning_rate=best_params["learning_rate"],
    logging_dir="logs",
    logging_steps=50,
    report_to="none",
    seed=SEED
)
final_trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2),
    args=final_args,
    train_dataset=combined_ds,
    tokenizer=tokenizer
)
final_trainer.train()

# ── 9) Predict & print detailed test metrics
test_out   = final_trainer.predict(test_ds)
test_probs = softmax(test_out.predictions, axis=1)[:, 1]
test_preds = (test_probs >= threshold).astype(int)

print(f"\nTest set size: {len(test_labels)}")
unique, counts = np.unique(test_labels, return_counts=True)
print(f"Label distribution on test set: {dict(zip(['real','fake'], counts))}\n")

cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix Counts:")
print(f"  True Negative (real→real): {tn}")
print(f"  False Positive (real→fake): {fp}")
print(f"  False Negative (fake→real): {fn}")
print(f"  True Positive (fake→fake): {tp}\n")

accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="macro"
)
print("Final test set metrics:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  Macro-F1 : {f1:.4f}\n")
