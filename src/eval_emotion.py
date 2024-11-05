from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from liger_kernel.transformers import apply_liger_kernel_to_gemma2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, default="public_data/dev/track_a/sun_a.csv")
    parser.add_argument("--model_checkpoint", type=str, default="models/gemma2-9b-cpt-sea-lionv3-base-SemEval-sun")
    parser.add_argument("--apply_liger_kernel_to_gemma2", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/")
    return parser.parse_args()


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def main(args):
    model_id = args.model_checkpoint.split("/")[-1]

    test_df = pd.read_csv(args.test_file)
    text2id = dict(zip(test_df["text"], test_df["id"]))

    test_df = pd.melt(test_df.drop(["id"], axis=1), id_vars=["text"], var_name="emotion", value_name="label")

    dataset = Dataset.from_pandas(test_df)

    def preprocess_function(example):
        example["prompt"] = f"### Text: {example['text']}\n### Emotion: {example['emotion']}\n### Label: "
        return example

    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    device_index = Accelerator().process_index

    peft_config = PeftConfig.from_pretrained(args.model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        attn_implementation="eager",  # alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": device_index},
    )

    if args.apply_liger_kernel_to_gemma2:
        apply_liger_kernel_to_gemma2()

    model = PeftModel.from_pretrained(model, args.model_checkpoint)
    model.eval()

    choices = ["no", "yes"]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices]

    predictions = []

    for prompt in tqdm(dataset["prompt"]):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().float().numpy()[0]
        prediction = np.argmax(choice_logits)
        predictions.append(prediction)

    # attach predictions
    test_df["label"] = predictions

    # unmelt dataframe
    test_df = test_df.pivot(index="text", columns="emotion", values="label").reset_index()
    test_df["id"] = test_df["text"].map(text2id)

    # sort by id
    test_df = test_df.sort_values("id")
    test_df = test_df[["id", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]]

    test_df.to_csv(f"{args.output_dir}/{model_id}/pred_sun_a.csv", index=False)


if __name__ == "__main__":
    main(parse_args())
