# SemEval2025-Emotion-Analysis

| Language   |  Track A   |  Track C   |
| ---------- | :--------: | :--------: |
| Indonesian |     -      |    dev     |
| Javanese   |     -      |    dev     |
| Sundanese  | train, dev | train, dev |

## Sundanese Track A: Multi-label Emotion Classification

### NusaBERT Fine-tuning

```sh
accelerate launch src/run_multilabel_classification.py \
    --model_checkpoint LazarusNLP/NusaBERT-large \
    --num_train_epochs 100 \
    --optim adamw_torch_fused \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --early_stopping_patience 10 \
    --bf16
```

**Dev Acc**: 0.48-0.52

### Gemma2-9b-SEA-LION-v3 SFT

```sh
accelerate launch src/run_sft_qlora.py \
    --model_checkpoint "gemma2-9b-cpt-sea-lion-v3-base-SemEval-sun" \
    --max_length 128 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_gemma2
```

```sh
accelerate launch src/eval_emotion.py \
    --model_checkpoint "models/gemma2-9b-cpt-sea-lion-v3-base-SemEval-sun" \
    --apply_liger_kernel_to_gemma2
```

**Dev Acc**: 0.57

### Gemma2-9b-SahabatAI-v1 SFT

```sh
accelerate launch src/run_sft_qlora.py \
    --model_checkpoint "GoToCompany/gemma2-9b-cpt-sahabatai-v1-base" \
    --max_length 128 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_gemma2
```

```sh
accelerate launch src/eval_emotion.py \
    --model_checkpoint "models/gemma2-9b-cpt-sahabatai-v1-base" \
    --apply_liger_kernel_to_gemma2
```

**Dev Acc**: 0.61

## TODOs

- [x] SetFit
- [x] Classical models
- [x] Fine-tune NusaBERT to Sun
- [x] SpanEmo
- [x] Claude
- [ ] Ensemble
- [x] SEA-LION-v3 SFT
- [x] SEA-LION-v3-instruct SFT
- [x] Gemma2-Sahabat-AI-v1 SFT
- [ ] Merge track A and C data for Track A Sun
- [ ] Cross-lingual transfer from Track A + C Sun to Track C Ind + Jav