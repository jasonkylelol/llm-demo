#!/bin/bash

MODEL_PATH=~/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2
DATASET_PATH=~/github/LLaMA-Factory/data
DATASET_NAME=alpaca_gpt4_zh
OUTPUT_PATH=$MODEL_PATH/predict

mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python ~/github/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $MODEL_PATH/sft_checkpooint,$MODEL_PATH/dpo \
    --dataset $DATASET_NAME \
    --dataset_dir $DATASET_PATH \
    --template default \
    --finetuning_type lora \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate