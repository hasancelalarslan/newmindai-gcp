import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import yaml
from collections import Counter
import torch.nn as nn

# ====== Config ======
NUM_LABELS = 4
LABEL2ID = {"Claim": 0, "Evidence": 1, "Counterclaim": 2, "Rebuttal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ====================

# Detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")

# Load config
def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

project_root = Path(__file__).resolve().parents[1]
cfg = load_config(project_root / "configs" / "config.yaml")["arg_classifier"]

MODEL_NAME = cfg["model_name"]
BATCH_SIZE = cfg.get("batch_size", 16)
EPOCHS = cfg.get("epochs", 3)
LR = float(cfg.get("learning_rate", 2e-5))
MAX_LEN = cfg.get("max_len", 256)

print(f"📑 Config → model={MODEL_NAME}, batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, max_len={MAX_LEN}")


# ===== Dataset =====
class OpinionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===== Data =====
def load_data(project_root: Path):
    train = pd.read_csv(project_root / "data/processed/opinions_train.csv")
    val   = pd.read_csv(project_root / "data/processed/opinions_val.csv")
    test  = pd.read_csv(project_root / "data/processed/opinions_test.csv")
    return train, val, test

def encode_labels(df):
    labels = df["type"].map(LABEL2ID).fillna(-1).astype(int)
    return labels.values

# ===== Metrics =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_macro": f1_macro}

# ===== Custom Weighted Trainer =====
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ===== Main =====
def main():
    train_df, val_df, test_df = load_data(project_root)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    y_train = encode_labels(train_df)
    train_ds = OpinionsDataset(train_df["text"].tolist(), y_train, tokenizer)
    val_ds   = OpinionsDataset(val_df["text"].tolist(), encode_labels(val_df), tokenizer)
    test_ds  = OpinionsDataset(test_df["text"].tolist(), encode_labels(test_df), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Compute class weights
    counts = Counter(y_train)
    total = sum(counts.values())
    weights = [total / (len(LABEL2ID) * counts.get(i, 1)) for i in range(len(LABEL2ID))]
    class_weights = torch.tensor(weights, dtype=torch.float, device=DEVICE)
    print("⚖️ Class weights:", class_weights.tolist())

    # Results dir
    results_dir = project_root / "results" / "arg_classifier"
    results_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(results_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=str(project_root / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()

    # Final eval
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    print("Test Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL2ID.keys()))

    # Save predictions CSV
    out_df = test_df.copy()
    out_df["true_label"] = [ID2LABEL[i] for i in y_true]
    out_df["pred_label"] = [ID2LABEL[i] for i in y_pred]
    out_path = results_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"🗂 Saved predictions to {out_path}")

    # Save final fine-tuned model + tokenizer
    save_dir = results_dir / "final_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model + tokenizer saved to {save_dir}")

if __name__ == "__main__":
    main()
