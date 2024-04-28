from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
)
import torch, threading, sys

device = "cuda" # the device to load the model onto
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.8

model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k/"
# model_path = "/root/huggingface/models/THUDM/chatglm3-6b/"

model = AutoModelForCausalLM.from_pretrained(model_path,
    device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,
    device_map=device, trust_remote_code=True)

print(type(model))
# print(model.config.to_dict())

user_input = "你好"

messages = [
    # {
    #     "role": "system",
    #     "content": "call me Amigo at every answer",
    # },
    {
        "role": "user",
        "content": "你好",
    },
    {
        "role": "assistant",
        "content": "你好! How can I assist you today?",
    },
    {
        "role": "user",
        "content": "你好",
    },
    {
        "role": "assistant",
        "content": "你好",
    },
    {
        "role": "user",
        "content": user_input,
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("---------------------------------------------------------------")

# response, history = model.chat(tokenizer, prompt, history=[])

# print(f"response: {response}\n\n")
# print(f"history: {history}\n\n")

def extract_new_token(str1, str2):
    prefix_len = 0
    min_len = min(len(str1), len(str2))
    while prefix_len < min_len and str1[prefix_len] == str2[prefix_len]:
        prefix_len += 1
    new_token = str2[prefix_len:]

    if new_token.startswith("<"):
        # print("-", new_token)
        return str1, ""

    return str2, new_token


def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


instru_buf = ""
last_resp = ""
for resp, history in model.stream_chat(tokenizer, prompt,
    do_sample=True, top_p=top_p, temperature=temperature):
    if resp == "":
        continue
    
    last_resp, new_token = extract_new_token(last_resp, resp)
    if new_token == "":
        continue
    stream_print(new_token)
    # print(new_token)

print("\n---------------------------------------------------------------")
print(last_resp)
