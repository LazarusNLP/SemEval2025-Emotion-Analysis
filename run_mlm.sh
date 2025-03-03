python src/run_mlm.py \
    --model_name_or_path LazarusNLP/NusaBERT-large \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --do_train --do_eval \
    --max_steps 10000 \
    --warmup_steps 500 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --optim adamw_torch_fused \
    --bf16 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --save_steps 2000 --save_total_limit 3 \
    --output_dir models/NusaBERT-Sundanese-large \
    --overwrite_output_dir \
    --report_to tensorboard