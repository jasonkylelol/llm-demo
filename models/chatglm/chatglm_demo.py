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

query = "解释一下这段话的意思：\n你有这么高速运转的机械进入中国，记住我给出的原理，小的时候就是研发人，就是研发这个东西的原理，是阴间政权管着，知道为什么有生灵给他运转先位，还有专门饲养这个，为什么地下产这种东西，它管着它说是五世同堂旗下子孙，你以为我跟你闹着玩呢，你不警察吗，黄龙江一派全都带蓝牙，黄龙江我告我告诉你，在阴间是那个化名，化名小舅，亲小舅，赵金兰的那个嫡子嫡孙"

history = [
    # {
    #     "role": "system",
    #     "content": "call me Amigo at every answer",
    # },
    {
        "role": "user",
        "content": "你好！",
    },
    {
        "role": "assistant",
        "content": "你好! 我是一个严谨认真的人工智能助手，对于不知道的内容从来不乱说。",
    }
]

# response, history = model.chat(tokenizer, query=query, history=history)
# print(f"response: {response}\n\n")
# print(f"history: {history}\n\n")

def extract_new_token(str1, str2):
    prefix_len = 0
    min_len = min(len(str1), len(str2))
    while prefix_len < min_len and str1[prefix_len] == str2[prefix_len]:
        prefix_len += 1
    new_token = str2[prefix_len:]
    return str2, new_token

def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()

model_resp = ""
for resp, model_history in model.stream_chat(tokenizer, query=query, history=history,
    do_sample=True, temperature=temperature):
    model_resp, new_token = extract_new_token(model_resp, resp)
    if new_token == "":
        continue
    stream_print(new_token)
    # print(new_token)

print("\n---------------------------------------------------------------")
print(model_resp)
