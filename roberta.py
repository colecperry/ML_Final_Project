import os
import pandas as pd                # data loading/manipulation
import torch                       # core PyTorch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizerFast,         # tokenizer for RoBERTa
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

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

train_ds = FakeNewsDataset(train_df["combined_text"], train_labels, tokenizer)
val_ds   = FakeNewsDataset(val_df["combined_text"],   val_labels,   tokenizer)
test_ds  = FakeNewsDataset(test_df["combined_text"],  test_labels,  tokenizer)

# ── 4) Metric fn
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ── 5) Base model setup
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

# ── 6) Train base to find threshold
base_args = TrainingArguments(
    output_dir="bert_base_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=50,
    report_to="none"
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

# ── 7) Lock in threshold on validation
threshold = 0.30
val_out    = base_trainer.predict(val_ds)
fake_probs = torch.softmax(torch.tensor(val_out.predictions), dim=1)[:, 1].numpy()
val_preds  = (fake_probs >= threshold).astype(int)
print(f"\nValidation @ thresh={threshold}")
print(classification_report(val_labels, val_preds, target_names=["real","fake"]))

# ── 8) Hyperparameter sweep on validation
best_val_f1 = 0.0
best_params = {}
for lr in [1e-5, 2e-5, 3e-5]:
    for bs in [8, 16]:
        print(f"\n--- Trying LR={lr}, BS={bs} ---")
        sweep_args = TrainingArguments(
            output_dir=f"bert_out_lr{lr}_bs{bs}",
            num_train_epochs=3,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs*2,
            learning_rate=lr,
            logging_dir="logs",
            logging_steps=50,
            report_to="none"
        )
        sweep_trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2),
            args=sweep_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer
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

# ── 9) FINAL: train with best params on train+val, then test
print("\n=== Final training on train+val and testing ===")
# 9a) Combine train+val
combined_df   = pd.concat([train_df, val_df], ignore_index=True)
combined_labels = combined_df["label"].map(label_map).tolist()
combined_ds   = FakeNewsDataset(combined_df["combined_text"], combined_labels, tokenizer)

# 9b) Re-train best model
final_args = TrainingArguments(
    output_dir="bert_final_output",
    num_train_epochs=3,
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"]*2,
    learning_rate=best_params["learning_rate"],
    logging_dir="logs",
    logging_steps=50,
    report_to="none"
)
final_trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2),
    args=final_args,
    train_dataset=combined_ds,
    tokenizer=tokenizer
)
final_trainer.train()

# 9c) Predict on test set with locked threshold
test_out   = final_trainer.predict(test_ds)
test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=1)[:, 1].numpy()
test_preds = (test_probs >= threshold).astype(int)

print(f"\nTest Set Report @ thresh={threshold}")
print(classification_report(test_labels, test_preds, target_names=["real","fake"]))
