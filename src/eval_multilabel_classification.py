from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

args = ArgumentParser()
args.add_argument("--train_file", type=str, default="public_data/train/track_a/sun.csv")
args.add_argument("--test_file", type=str, default="public_data/dev/track_a/sun_a.csv")
args.add_argument("--model_checkpoint", type=str, default="LazarusNLP/NusaBERT-large-SemEval-sun")
args.add_argument("--output_dir", type=str, default="outputs")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.model_checkpoint.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_file)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    test_df = pd.read_csv(args.test_file)

    labels = sorted(set(train_df.columns) - set(["id", "text"]))
    test_df.loc[:, labels] = 0

    dataset = DatasetDict(
        {
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint).to(device)
    model.eval()

    def preprocess_function(example):
        tokenized_input = tokenizer(example["text"], truncation=True, max_length=model.config.max_position_embeddings)
        tokenized_input["labels"] = [float(example[label]) for label in labels]
        return tokenized_input

    tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset["validation"].column_names)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(tokenized_inputs):
        device = model.device
        scores = []
        for tokenized_input in tqdm(tokenized_inputs):
            with torch.no_grad():
                tokenized_input = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in tokenized_input.items()}
                logits = model(**tokenized_input).logits.detach().cpu().numpy()
                scores.append(logits)

        scores = np.concatenate(scores, axis=0)
        return scores

    val_scores = predict(tokenized_dataset["validation"])
    val_scores = sigmoid(val_scores)
    val_labels = val_df[labels].astype(int).to_numpy()

    # find best threshold via validation set, apply best threshold to test set
    thresholds = np.arange(0.0, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        val_prediction = (val_scores > threshold).astype(int)
        metrics_result = f1_score(y_true=val_labels, y_pred=val_prediction, average="macro")
        scores.append((threshold, metrics_result))

    best_threshold, _ = max(scores, key=lambda x: x[1])

    predictions = predict(tokenized_dataset["test"])
    predictions = sigmoid(predictions)
    predictions = (predictions > best_threshold).astype(int)

    test_df[labels] = predictions
    test_df.drop(["text"], axis=1).to_csv(output_dir / "pred_sun_a.csv", index=False)


if __name__ == "__main__":
    main(args.parse_args())
