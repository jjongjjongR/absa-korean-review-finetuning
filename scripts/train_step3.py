from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import ensure_src_on_path
except ImportError:
    from _bootstrap import ensure_src_on_path


ensure_src_on_path()

from absa.io import read_data_csv  # noqa: E402
from absa.step3_aspect_sentiment import Step3Config, build_input_texts  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step3: aspect sentiment (binary) fine-tuning.")
    p.add_argument("--data", type=Path, default=Path("data/raw/data.csv"))
    p.add_argument("--model-name", default="klue/bert-base")
    p.add_argument("--format", default="case1", choices=["case1", "case2"])
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step3"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = read_data_csv(args.data)

    cfg = Step3Config()
    input_texts = build_input_texts(df, cfg, fmt=args.format)
    labels = df[cfg.label_col].astype(int).tolist()

    # Heavy deps
    from datasets import Dataset  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
    import numpy as np  # type: ignore

    data = Dataset.from_dict({"text": input_texts, "label": labels})
    split = data.train_test_split(test_size=0.3, seed=10)
    train_ds, val_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, padding=True)
        enc["labels"] = batch["label"]
        return enc

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels_).mean().item()
        return {"accuracy": acc}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir / "model"))
    tokenizer.save_pretrained(str(args.out_dir / "model"))

    print(f"Saved: {args.out_dir / 'model'}")


if __name__ == "__main__":
    main()

