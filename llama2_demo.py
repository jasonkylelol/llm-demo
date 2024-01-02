import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/root/huggingface/models/meta-llama/Llama-2-7b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                             load_in_4bit=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


if __name__ == '__main__':
    prompt = 'how brush teeth with shampoo'
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**model_inputs)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)
