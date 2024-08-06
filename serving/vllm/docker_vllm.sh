#!/bin/bash

image=vllm/vllm-openai:v0.5.3-custom-fix
gpus='"device=1"'
model=THUDM/glm-4-9b-chat

docker run -it --rm --ipc host --runtime nvidia --gpus $gpus --name vllm-openai \
    -v /data/lm/huggingface/models/$model:/models/$model \
    -p 38000:8000 \
    $image \
    --model /models/$model --trust-remote-code --gpu-memory-utilization=0.95 --enable-chunked-prefill=False --max-model-len=16384
