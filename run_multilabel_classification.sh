python src/run_multilabel_classification.py \
    --model_checkpoint LazarusNLP/NusaBERT-large \
    --num_train_epochs 50 \
    --optim adamw_torch_fused \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --bf16