from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import ensure_src_on_path
except ImportError:
    from _bootstrap import ensure_src_on_path


ensure_src_on_path()

from absa.io import read_data_csv  # noqa: E402
from absa.step2_aspect_multilabel import Step2Config, to_multilabel_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step2: aspect detection (multi-label) fine-tuning.")
    p.add_argument("--data", type=Path, default=Path("data/raw/data.csv"))
    p.add_argument("--model-name", default="klue/bert-base")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step2"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = read_data_csv(args.data)
    wide = to_multilabel_table(df, Step2Config())

    # Heavy deps
    from datasets import Dataset  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    text_col = "text"
    label_cols = [c for c in wide.columns if c != text_col]
    n = len(label_cols)

    train_df, val_df = train_test_split(wide, test_size=0.3, random_state=16)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        enc = tokenizer(batch[text_col], truncation=True, padding=True)
        labels = np.stack([batch[c] for c in label_cols], axis=1).astype("float32")
        enc["labels"] = labels
        return enc

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=n, problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        exact_match = (preds == labels).all(axis=1).mean().item()
        return {"exact_match": exact_match}

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

