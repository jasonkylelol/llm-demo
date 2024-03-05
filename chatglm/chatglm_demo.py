from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

device = "cuda:1" # the device to load the model onto
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

model_path = "/root/huggingface/models/THUDM/chatglm3-6b/"

model = AutoModelForCausalLM.from_pretrained(model_path,
    device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,
    device_map=device, trust_remote_code=True)

user_input = "how brush teeth with shampoo?"

messages = [
    {
        "role": "user",
        "content": user_input,
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("=========================================")

response, history = model.chat(tokenizer, prompt, history=[])

print(f"{response}\n\n")
print(f"{history}\n\n")
