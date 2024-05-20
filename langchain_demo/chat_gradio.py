import sys, os, logging
import re
from typing import Tuple, List
from threading import Thread

import torch
import gradio as gr
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AutoModelForCausalLM, TextIteratorStreamer)

logging.basicConfig(level=logging.INFO, encoding="utf-8")
logger = logging.getLogger()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.handlers[0].setFormatter(formatter)

device="cuda"

model_path = "/root/huggingface/models"
# model_name = "THUDM/chatglm3-6b"
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model_full = f"{model_path}/{model_name}"

max_new_tokens=8192
# top_p=0.1
temperature=0.9

model, tokenizer = None, None

def generate_prompt(chat_history) -> Tuple[str, List]:
    history = chat_history[:-1]
    query = chat_history[-1][0]

    model_history = [{"role":"system","content":"你是一个调皮的机器人，以诙谐幽默的方式回答问题，回答中会使用emoji表情"}]
    for hist in history:
        user_msg = hist[0]
        model_history.append({"role":"user","content":user_msg})
        assistant_msg = hist[1]
        model_history.append({"role":"assistant","content":assistant_msg})
        print(f"user: {user_msg}\nassistant: {assistant_msg}", flush=True)
    print(f"query: {query}", flush=True)
    return query, model_history


def chat_resp(chat_history, msg):
    chat_history[-1][1] = msg
    return chat_history


def handle_chat(chat_history):
    logger.info(f"temperature: {temperature}")
    query, history = generate_prompt(chat_history)

    if "chatglm3" in model_name:
        for resp, model_history in model.stream_chat(tokenizer, query=query, history=history,
            do_sample=True, temperature=temperature):
            yield chat_resp(chat_history, resp)
    else:
        streamer, thread = llama_stream_chat(query, history, temperature)
        generated_text = ""
        for new_text in streamer:
            # stream_print(new_text)
            generated_text += new_text
            yield chat_resp(chat_history, generated_text)
        thread.join()


def llama_stream_chat(query, history, temperature):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    # prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # print(f"{prompt}\n\n")
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    generation_kwargs = dict(
        inputs=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        # top_p=top_p,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer, thread


def handle_add_msg(query, chat_history):
    # logger.info(query, chat_history)
    return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def init_llm():
    global model, tokenizer

    logger.info(f"load from {model_full}")
    if "chatglm3" in model_name:
        model_config = AutoConfig.from_pretrained(model_full, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_full, device_map=device, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_full, config=model_config, trust_remote_code=True, device_map=device)
        model = model.eval()
    else:
        model_config = AutoConfig.from_pretrained(model_full)
        tokenizer = AutoTokenizer.from_pretrained(model_full, device_map=device)
        model = AutoModelForCausalLM.from_pretrained(model_full, config=model_config,
            torch_dtype=torch.bfloat16, device_map=device)


def init_blocks():
    with gr.Blocks(title="Shiba ChatBot") as app:
        gr.Markdown("# 已读乱回机器人🤖 v2.0  \n"
            # f"- llm: {model_name}  \n"
        )
        chatbot = gr.Chatbot(label="chatroom", show_label=False)
        with gr.Row():
            query = gr.Textbox(label="Say something", scale=4)
            clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot], scale=1)

        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
            handle_chat, inputs=[chatbot], outputs=[chatbot]).then(
            lambda: gr.Textbox(interactive=True), outputs=[query])

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False,
        share=False, favicon_path="langchain_demo/icons/shiba.svg")
