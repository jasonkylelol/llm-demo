#!/bin/bash

image=vllm/vllm-openai:v0.4.2
gpus="device=3"
model=shenzhi-wang/Llama3-8B-Chinese-Chat

docker run --runtime nvidia --gpus $gpus \
    -v /data/lm/huggingface/models/$model:/models/$model \
    -p 38000:8000 \
    --ipc=host \
    $image \
    --model /models/$model --gpu-memory-utilization 0.8
