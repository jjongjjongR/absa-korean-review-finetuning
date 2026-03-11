from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts._bootstrap import ensure_src_on_path
except ImportError:
    from _bootstrap import ensure_src_on_path


ensure_src_on_path()

from absa.io import read_data_csv  # noqa: E402
from absa.step1_sentiment import Step1Config, prepare_binary_dataset  # noqa: E402
from absa.training import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step1: sentiment classification fine-tuning (notebook-based).")
    p.add_argument("--data", type=Path, default=Path("data/raw/data.csv"))
    p.add_argument("--model-name", default="klue/bert-base")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step1"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample-per-class", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--run-lora", action="store_true")
    p.add_argument("--lora-learning-rate", type=float, default=5e-5)
    return p.parse_args()


def _classification_summary(labels, preds) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, target_names=["negative", "positive"]),
    }


def _save_eval_outputs(out_dir: Path, trainer, val_ds, label_name: str = "label") -> None:
    import numpy as np  # type: ignore

    pred_out = trainer.predict(val_ds)
    preds = np.argmax(pred_out.predictions, axis=-1)
    labels = pred_out.label_ids
    summary = _classification_summary(labels, preds)
    save_json(out_dir / "eval_summary.json", summary)
    (out_dir / "eval_predictions.json").write_text(
        json.dumps({"labels": labels.tolist(), "predictions": preds.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    df = read_data_csv(args.data)
    df_s = prepare_binary_dataset(df, Step1Config(train_size_per_class=args.sample_per_class))

    # Heavy deps are imported only when this script is executed.
    from datasets import Dataset  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    import numpy as np  # type: ignore
    import torch  # type: ignore

    train_df, val_df = train_test_split(df_s, test_size=0.2, random_state=16)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
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
        learning_rate=args.learning_rate,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(args.out_dir / "model"))
    tokenizer.save_pretrained(str(args.out_dir / "model"))
    _save_eval_outputs(args.out_dir, trainer, val_ds)

    if args.run_lora:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

        base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key"],
            bias="none",
        )
        lora_model = get_peft_model(base_model, lora_config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        lora_args = TrainingArguments(
            output_dir=str(args.out_dir / "lora_runs"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=args.lora_learning_rate,
            report_to=[],
        )
        trainer_lora = Trainer(
            model=lora_model,
            args=lora_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer_lora.train()
        trainer_lora.save_model(str(args.out_dir / "lora_model"))
        tokenizer.save_pretrained(str(args.out_dir / "lora_model"))
        _save_eval_outputs(args.out_dir / "lora", trainer_lora, val_ds)

    print(f"Saved: {args.out_dir / 'model'}")


if __name__ == "__main__":
    main()

