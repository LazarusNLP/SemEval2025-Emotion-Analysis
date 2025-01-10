accelerate launch src/eval_emotion.py \
    --test_file public_data_dev/track_a/dev/sun.csv \
    --model_checkpoint "models/gemma2-9b-cpt-sahabatai-v1-base-SemEval-sun" \
    --apply_liger_kernel_to_gemma2

accelerate launch src/eval_emotion.py \
    --test_file public_data_dev/track_c/dev/ind.csv \
    --model_checkpoint "models/gemma2-9b-cpt-sahabatai-v1-base-SemEval-sun" \
    --apply_liger_kernel_to_gemma2

accelerate launch src/eval_emotion.py \
    --test_file public_data_dev/track_c/dev/jav.csv \
    --model_checkpoint "models/gemma2-9b-cpt-sahabatai-v1-base-SemEval-sun" \
    --apply_liger_kernel_to_gemma2