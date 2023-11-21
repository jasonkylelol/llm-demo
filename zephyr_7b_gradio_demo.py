import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


model_path = "/app/models/HuggingFaceH4/zephyr-7b-beta/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = []
    for item in history_transformer_format:
        messages.append({"role": "user", "content": item[0]})
        if item[1] != "":
            messages.append({"role": "assistant", "content": item[1]})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt, "==================================")
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    cnt =  0
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            #print(cnt, ":", partial_message)
            cnt+=1
            yield partial_message


if __name__ == '__main__':
    iface = gr.ChatInterface(fn=predict, title="HuggingFaceH4/zephyr-7b-beta Chat Bot",
        retry_btn=None, undo_btn=None, clear_btn=None)
    iface.queue().launch(server_name='0.0.0.0')

