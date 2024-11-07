accelerate launch src/eval_instruct_emotion.py \
    --model_checkpoint "models/gemma2-9b-cpt-sea-lionv3-instruct-SemEval-sun" \
    --apply_liger_kernel_to_gemma2