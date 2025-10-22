import os
import subprocess
import sys
import logging

required = [
    "optuna>=3.0.0",
    "transformers>=4.35.0",
    "datasets>=2.14.4",
    "torch>=2.1.0",
    "scikit-learn>=1.3.0"
]
for pkg in required:
    name = pkg.split(">=")[0]
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV   = PROJECT_ROOT / "data" / "processed" / "validation.csv"

def load_hf_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df)

train_ds = load_hf_dataset(TRAIN_CSV)
val_ds   = load_hf_dataset(VAL_CSV)

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def tokenize_batch(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

def objective(trial):
    model_name    = trial.suggest_categorical("model_name", ["bert-base-uncased", "roberta-base"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 5e-5)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs        = trial.suggest_int("epochs", 2, 5)
    max_length    = trial.suggest_categorical("max_length", [128, 256, 512])

    output_dir = PROJECT_ROOT / "results" / f"trial_{trial.number}"
    logger.info(f"Trial {trial.number}: model={model_name}, lr={learning_rate:.2e}, "
                f"batch={batch_size}, epochs={epochs}, max_len={max_length}")

    checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda d: int(d.name.split('-')[1])) \
                     if output_dir.exists() else []
    if checkpoint_dirs:
        last_ckpt = checkpoint_dirs[-1]
        logger.info(f"Found existing checkpoint {last_ckpt}, skipping training")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(last_ckpt)
        def tokenize_fn(ex): return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=max_length)
        val_ds_tok = val_ds.map(tokenize_fn, batched=True)
        val_ds_tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        trainer = Trainer(model=model, compute_metrics=compute_metrics)
        eval_metrics = trainer.evaluate(val_ds_tok)
        f1 = eval_metrics["eval_f1"]
        logger.info(f"Trial {trial.number} (existing) F1={f1:.4f}")
        return f1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    tokenized_train = train_ds.map(lambda x: tokenize_batch(x, tokenizer, max_length), batched=True)
    tokenized_val   = val_ds.map(lambda x: tokenize_batch(x, tokenizer, max_length), batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=42,
        disable_tqdm=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    logger.info(f"Trial {trial.number} Completed: F1={eval_metrics['eval_f1']:.4f}, Acc={eval_metrics['eval_accuracy']:.4f}")
    return eval_metrics["eval_f1"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    logger.info("Starting hyperparameter optimization with checkpoint reuse")
    study.optimize(objective, n_trials=12, timeout=3600)

    best = study.best_trial
    print("\nBest trial:")
    print(f"  F1: {best.value:.4f}")
    print("  Params:")
    for key, val in best.params.items():
        print(f"    {key}: {val}")
