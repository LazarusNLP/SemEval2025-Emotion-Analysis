from argparse import ArgumentParser

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from accelerate import Accelerator
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from liger_kernel.transformers import apply_liger_kernel_to_gemma2

"""
Usage:

accelerate launch run_sft_qlora.py \
    --model_checkpoint "aisingapore/gemma2-9b-cpt-sea-lionv3-base" \
    --max_length 128 \
    --batch_size 16 \
    --learning_rate 3e-2 \
    --num_epochs 20 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_gemma2
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, default="public_data/train/track_a/sun.csv")
    parser.add_argument("--model_checkpoint", type=str, default="aisingapore/gemma2-9b-cpt-sea-lionv3-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_gemma2", action="store_true")
    return parser.parse_args()


def main(args):
    model_id = f"{args.model_checkpoint.split('/')[-1]}-SemEval-sun"

    train_df = pd.read_csv(args.train_file)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df = pd.melt(train_df.drop(["id"], axis=1), id_vars=["text"], var_name="emotion", value_name="label")
    val_df = pd.melt(val_df.drop(["id"], axis=1), id_vars=["text"], var_name="emotion", value_name="label")

    dataset = DatasetDict({"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(val_df)})

    def preprocess_function(example):
        label = "yes" if example["label"] == 1 else "no"
        example["prompt"] = f"### Text: {example['text']}\n### Emotion: {example['emotion']}\n### Label: {label}"
        return example

    dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    response_template = "### Label:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    max_seq_length = args.max_length

    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    device_index = Accelerator().process_index

    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        use_cache=True if args.gradient_checkpointing else False,
        attn_implementation="eager",  # https://github.com/huggingface/transformers/issues/32390
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": device_index},
    )

    if args.apply_liger_kernel_to_gemma2:
        apply_liger_kernel_to_gemma2()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    output_dir = f"models/{model_id}"

    args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True} if args.gradient_checkpointing else None,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        dataloader_num_workers=16,
        num_train_epochs=args.num_epochs,
        optim="adamw_torch",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    trainer.train()

    trainer.save_model(output_dir)
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main(parse_args())
