accelerate launch src/eval_emotion.py \
    --model_checkpoint "models/gemma2-9b-cpt-sea-lionv3-base-SemEval-sun" \
    --apply_liger_kernel_to_gemma2