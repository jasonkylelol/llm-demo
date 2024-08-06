from openai import OpenAI
import os, sys, re
import threading
from datetime import datetime

api_key = "EMPTY"
# model = "THUDM/glm-4-9b-chat"
model = "llama3/llama-3-chinese-8b-instruct-v2"

def init_client():
    client = OpenAI(
        api_key=api_key,
        base_url="http://192.168.0.20:38061/v1",
    )
    return client


def generate_content(client):
    response = client.chat.completions.create(
        model=model,
        messages=[ 
            {"role": "system", "content": "你总是使用嘻哈风格回答问题，回答中带有emoji表情，只使用简体中文进行回复"},
            {"role": "user", "content": "详细介绍如何使用牙膏刷牙"},
        ],
        temperature=0.6,
        stream=True,
    )
    full_msg = ""
    for idx, chunk in enumerate(response):
        chunk_message = chunk.choices[0].delta
        if not chunk_message.content:
            continue
        stream_print(chunk_message.content)
        full_msg += chunk_message.content
    print("")
    return full_msg


def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def xprint(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

if __name__ == "__main__":
    thread_num = 1
    loop_num = 1
    threads = []

    def loop():
        client = init_client()
        for i in range(loop_num):
            # xprint(f"{threading.current_thread().name}: loop: {i}")
            full_msg = generate_content(client)
            # with open(f"serving/output/{threading.current_thread().name}_output.txt", "a+") as f:
            #     full_msg = re.sub(r'\n+', ' ', full_msg)
            #     f.write(f"{full_msg}\n\n\n")
            # xprint(f"{threading.current_thread().name}: msg len: {len(full_msg)}")

    xprint("start")
    stime = datetime.now()

    for i in range(thread_num):
        thread = threading.Thread(name=f"th-{i}", target=loop)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    xprint(f"finished cost: {datetime.now() - stime}")