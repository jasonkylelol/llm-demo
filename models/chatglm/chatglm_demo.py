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
print(model.config.to_dict())

user_input = "what about gasoline?"

messages = [
    {
        "role": "system",
        "content": "call me Amigo at every answer",
    },
    {
        "role": "user",
        "content": "how brush teeth with shampoo?",
    },
    {
        "role": "assistant",
        "content": "Amigo. Brushing your teeth with shampoo is not recommended as it can damage your toothbrush and may not be effective in removing plaque and bacteria from your teeth.",
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


def extract_added_content(str1, str2):
    prefix_len = 0
    min_len = min(len(str1), len(str2))
    while prefix_len < min_len and str1[prefix_len] == str2[prefix_len]:
        prefix_len += 1
    added_content = str2[prefix_len:]
    return str2, added_content


def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


last_resp = ""
postfix_delimiter_filter = "<|user|>"
postfix_delimiter = ""
for resp, history in model.stream_chat(tokenizer, prompt,
    do_sample=True, top_p=top_p, temperature=temperature):
    if resp == "":
        continue
    last_resp, added_content = extract_added_content(last_resp, resp)
    if added_content == "":
        continue
    if added_content in postfix_delimiter_filter:
        postfix_delimiter += added_content
        if len(postfix_delimiter) >= len(postfix_delimiter_filter):
            if postfix_delimiter == postfix_delimiter_filter:
                # print(f"\n[generator] need skip {postfix_delimiter_filter} from:\n{resp}", flush=True)
                postfix_delimiter = ""
                continue
            else:
                stream_print(postfix_delimiter)
                postfix_delimiter = ""
    else:
        if len(postfix_delimiter) > 0:
            stream_print(postfix_delimiter)
            postfix_delimiter = ""
        stream_print(added_content)

print("\n---------------------------------------------------------------")
print(last_resp)
