from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

args = ArgumentParser()
args.add_argument("--train_file", type=str, default="public_data/train/track_a/sun.csv")
args.add_argument("--test_file", type=str, default="public_data/dev/track_a/sun_a.csv")
args.add_argument("--model_checkpoint", type=str, default="LazarusNLP/NusaBERT-base")
args.add_argument("--output_dir", type=str, default="models")
args.add_argument("--num_train_epochs", type=int, default=50)
args.add_argument("--optim", type=str, default="adamw_torch")
args.add_argument("--early_stopping_patience", type=int, default=5)
args.add_argument("--early_stopping_threshold", type=float, default=0.0)
args.add_argument("--learning_rate", type=float, default=1e-5)
args.add_argument("--warmup_ratio", type=float, default=0.1)
args.add_argument("--weight_decay", type=float, default=0.01)
args.add_argument("--per_device_train_batch_size", type=int, default=8)
args.add_argument("--per_device_eval_batch_size", type=int, default=16)
args.add_argument("--fp16", action="store_true")
args.add_argument("--bf16", action="store_true")
args.add_argument("--hub_model_id", type=str, default="LazarusNLP/NusaBERT-base-CASA")


def main(args):
    train_df = pd.read_csv(args.train_file)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    test_df = pd.read_csv(args.test_file)

    labels = sorted(set(train_df.columns) - set(["id", "text"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in enumerate(labels)}
    test_df.loc[:, labels] = 0

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df),
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    def preprocess_function(example):
        tokenized_input = tokenizer(example["text"], truncation=True, max_length=model.config.max_position_embeddings)
        tokenized_input["labels"] = [float(example[label]) for label in labels]
        return tokenized_input

    tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int)
        labels = labels.astype(int)
        return {"f1": f1_score(y_true=labels, y_pred=predictions, average="macro")}

    callbacks = [EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / f"{args.model_checkpoint.split('/')[-1]}-SemEval-sun"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="tensorboard",
        # push_to_hub=True,
        # hub_model_id=args.hub_model_id,
        # hub_private_repo=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.create_model_card()

    val_scores, *_ = trainer.predict(tokenized_dataset["validation"])
    val_scores = sigmoid(val_scores)
    val_labels = val_df[labels].astype(int).to_numpy()

    # find best threshold via validation set, apply best threshold to test set
    thresholds = np.arange(0.0, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        val_prediction = (val_scores > threshold).astype(int)
        metrics_result = f1_score(y_true=val_labels, y_pred=val_prediction, average="macro")
        scores.append((threshold, metrics_result))

    best_threshold, best_score = max(scores, key=lambda x: x[1])
    print(f"Best threshold: {best_threshold}, Best score: {best_score}")
    predictions, *_ = trainer.predict(tokenized_dataset["test"])
    predictions = sigmoid(predictions)
    predictions = (predictions > best_threshold).astype(int)

    test_df[labels] = predictions
    test_df.drop(["text"], axis=1).to_csv(output_dir / "pred_sun_a.csv", index=False)


if __name__ == "__main__":
    main(args.parse_args())
