import json
from argparse import ArgumentParser

import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from liger_kernel.transformers import apply_liger_kernel_to_gemma2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, default="public_data/dev/track_a/sun_a.csv")
    parser.add_argument("--model_checkpoint", type=str, default="models/gemma2-9b-cpt-sea-lionv3-base-SemEval-sun")
    parser.add_argument("--apply_liger_kernel_to_gemma2", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    return parser.parse_args()


def main(args):
    model_id = args.model_checkpoint.split("/")[-1]

    test_df = pd.read_csv(args.test_file)
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

    generation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=None,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
    )

    predictions = []

    for prompt in tqdm(dataset["prompt"]):
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_length = prompt_input_ids.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(**prompt_input_ids, generation_config=generation_config)

        prediction = tokenizer.decode(outputs[0, prompt_token_length:], skip_special_tokens=True)
        predictions.append(prediction)

    result = {"model_id": model_id, "predictions": predictions}

    with open(f"results/{model_id}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
