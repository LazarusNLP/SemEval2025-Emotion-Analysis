accelerate launch src/run_sft_instruct_qlora.py \
    --model_checkpoint "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct" \
    --max_length 256 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_gemma2