from openai import OpenAI
import os, sys

api_key = "EMPTY"
model = "/models/shenzhi-wang/Llama3-8B-Chinese-Chat"

client = OpenAI(
    api_key=api_key,
    base_url="http://192.168.0.20:38000/v1",
)

response = client.chat.completions.create(
    model=model,
    messages=[ 
        {"role": "system", "content": "你总是使用嘻哈风格回答问题，回答中带有emoji表情，只使用简体中文进行回复"},
        # {"role": "user", "content": "如何使用洗发水刷牙？"},
        # {"role": "assistant", "content": "不建议使用洗发水刷牙！嘿哈！"},
        {"role": "user", "content": "详细介绍如何使用牙膏刷牙"},
    ],
    temperature=0.6,
    stream=True,
)

def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()

full_answer = ""
for idx, chunk in enumerate(response):
    chunk_message = chunk.choices[0].delta
    if not chunk_message.content:
        continue
    full_answer += chunk_message.content
    stream_print(chunk_message.content)
print("")