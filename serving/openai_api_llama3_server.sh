#!/bin/bash

MODEL=llama3/llama-3-chinese-8b-instruct-v2
MODEL_PATH=/root/huggingface/models/$MODEL
PORT=8060

#CUDA_VISIBLE_DEVICES=1
vllm serve $MODEL_PATH --served-model-name $MODEL --port $PORT --trust-remote-code \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --enforce-eager
