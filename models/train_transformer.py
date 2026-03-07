import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/telemetry_logs.csv")


# Convert structured row → text
def row_to_text(row):
    return (
        f"latency {row.latency} "
        f"packet_loss {row.packet_loss} "
        f"jitter {row.jitter} "
        f"dns_error {row.dns_error} "
        f"route_change {row.route_change}"
    )


df["text"] = df.apply(row_to_text, axis=1)


# Encode labels
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["label"])


# Train test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["text", "labels"]])


# -------------------------
# Tokenizer
# -------------------------

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )


train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)


train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# -------------------------
# Model
# -------------------------

num_labels = len(label_encoder.classes_)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)


# -------------------------
# Training config
# -------------------------

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,   # keep small for laptop
    logging_steps=10,
    save_strategy="no"
)


# -------------------------
# Trainer
# -------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


# -------------------------
# Train
# -------------------------

trainer.train()


# -------------------------
# Evaluate
# -------------------------

metrics = trainer.evaluate()

print(metrics)