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
from absa.step3_aspect_sentiment import Step3Config, build_input_texts  # noqa: E402
from absa.training import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step3: aspect sentiment (binary) fine-tuning.")
    p.add_argument("--data", type=Path, default=Path("data/raw/data.csv"))
    p.add_argument("--model-name", nargs="+", default=["klue/bert-base", "klue/roberta-base"])
    p.add_argument("--format", nargs="+", default=["case1", "case2"], choices=["case1", "case2"])
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step3"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample-size", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = read_data_csv(args.data)
    if len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=10).reset_index(drop=True)

    cfg = Step3Config()

    # Heavy deps
    from datasets import Dataset  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
    import numpy as np  # type: ignore
    summary: dict[str, dict[str, float | str]] = {}

    for model_name in args.model_name:
        for fmt in args.format:
            input_texts = build_input_texts(df, cfg, fmt=fmt)
            labels = df[cfg.label_col].astype(int).tolist()
            data = Dataset.from_dict({"text": input_texts, "label": labels})
            split = data.train_test_split(test_size=0.3, seed=10)
            train_ds, val_ds = split["train"], split["test"]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if fmt == "case1":
                tokenizer.add_special_tokens({"additional_special_tokens": ["[ASPECT]"]})

            def tokenize(batch):
                enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
                enc["labels"] = [int(v) for v in batch["label"]]
                return enc

            train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
            val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            if fmt == "case1":
                model.resize_token_embeddings(len(tokenizer))

            def compute_metrics(eval_pred):
                logits, labels_ = eval_pred
                preds = np.argmax(logits, axis=-1)
                acc = (preds == labels_).mean().item()
                return {"accuracy": acc}

            variant_name = f"{model_name.split('/')[-1]}_{fmt}"
            variant_out = args.out_dir / variant_name
            variant_out.mkdir(parents=True, exist_ok=True)
            training_args = TrainingArguments(
                output_dir=str(variant_out),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=2e-5,
                report_to=[],
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(str(variant_out / "model"))
            tokenizer.save_pretrained(str(variant_out / "model"))
            pred_out = trainer.predict(val_tok)
            preds = np.argmax(pred_out.predictions, axis=-1)
            accuracy = float((preds == pred_out.label_ids).mean())
            summary[variant_name] = {"accuracy": accuracy, "model": model_name, "format": fmt}
            (variant_out / "eval_predictions.json").write_text(
                json.dumps({"labels": pred_out.label_ids.tolist(), "predictions": preds.tolist()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    save_json(args.out_dir / "summary.json", summary)
    print(f"Saved: {args.out_dir}")


if __name__ == "__main__":
    main()

