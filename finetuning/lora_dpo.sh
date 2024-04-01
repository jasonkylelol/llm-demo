#!/bin/bash

MODEL_PATH=~/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2
DATASET_PATH=~/github/LLaMA-Factory/data
DATASET_NAME=comparison_gpt4_zh
OUTPUT_PATH=$MODEL_PATH/dpo

mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python ~/github/LLaMA-Factory/src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $MODEL_PATH/sft_checkpooint \
    --create_new_adapter \
    --dataset $DATASET_NAME \
    --dataset_dir $DATASET_PATH \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16