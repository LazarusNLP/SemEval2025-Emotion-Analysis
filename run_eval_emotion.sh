accelerate launch src/eval_emotion.py \
    --model_checkpoint "models/gemma2-9b-cpt-sea-lionv3-base-SemEval-sun-5EPOCHS" \
    --apply_liger_kernel_to_gemma2 \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 0.92 \
    --typical_p 0.95 \
    --repetition_penalty 1.0