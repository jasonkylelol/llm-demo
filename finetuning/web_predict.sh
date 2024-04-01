
MODEL_PATH=~/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2

CUDA_VISIBLE_DEVICES=0 python ~/github/LLaMA-Factory/src/web_demo.py \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $MODEL_PATH/sft_checkpooint,$MODEL_PATH/dpo \
    --template default \
    --finetuning_type lora