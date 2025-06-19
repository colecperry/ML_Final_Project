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

# ── 1) Data loading (CSVs sit in …/final_project)
train_df = pd.read_csv("train_split_bert.csv")
val_df   = pd.read_csv("val_split_bert.csv")
test_df  = pd.read_csv("test_split_bert.csv")

# ── 2) Labels
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
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
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

# ── 5) Model + Trainer setup (NO unsupported args)
model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

training_args = TrainingArguments(
    output_dir="bert_fakenews_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ── 6) Train
print("=== Starting BERT fine-tuning ===")
trainer.train()

# ── 7) Lock-in decision threshold on VALIDATION set only
threshold = 0.30
print(f"\n=== Validation Report @ threshold = {threshold} ===")
val_out = trainer.predict(val_ds)
# convert logits → probabilities for the "fake" class
fake_probs = torch.softmax(torch.tensor(val_out.predictions), dim=1)[:, 1].numpy()
# apply the cutoff
val_preds = (fake_probs >= threshold).astype(int)
# print a real-vs-fake classification report on validation
print(classification_report(
    val_labels,
    val_preds,
    target_names=["real", "fake"]
))

# ── 8) Hyperparameter sweep (sequential approach)
best_val_f1 = 0.0
best_params = {}
for lr in [1e-5, 2e-5, 3e-5]:
    for bs in [8, 16]:
        print(f"\n--- Tuning LR={lr}, BatchSize={bs} ---")
        # update training arguments
        sweep_args = TrainingArguments(
            output_dir=f"bert_out_lr{lr}_bs{bs}",
            num_train_epochs=3,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs * 2,
            learning_rate=lr,
            logging_dir="logs",
            logging_steps=50,
            report_to="none"
        )
        # re-init trainer
        sweep_trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(
                "roberta-large", num_labels=2),
            args=sweep_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer
        )
        # train and predict on validation
        sweep_trainer.train()
        sweep_out = sweep_trainer.predict(val_ds)
        sweep_probs = torch.softmax(torch.tensor(sweep_out.predictions), dim=1)[:, 1].numpy()
        sweep_preds = (sweep_probs >= threshold).astype(int)
        _, _, f1_sweep, _ = precision_recall_fscore_support(
            val_labels, sweep_preds, average="macro"
        )
        print(f"Val F1 @ thresh={threshold}: {f1_sweep:.4f}")
        # update best
        if f1_sweep > best_val_f1:
            best_val_f1 = f1_sweep
            best_params = {"learning_rate": lr, "batch_size": bs}

print(f"\n*** Best hyperparameters: {best_params}, Val F1={best_val_f1:.4f} ***")
