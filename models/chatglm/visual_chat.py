import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer, AutoModel, BitsAndBytesConfig
)
from PIL import Image
import gradio as gr

device="auto"
max_new_tokens=8192
top_p=0.6
temperature=0.1

model_path = "/root/huggingface/models"
model_name = "THUDM/glm-4v-9b"
model_full = f"{model_path}/{model_name}"

model, tokenizer = None, None

def chat_resp(chat_history, msg):
    chat_history[-1][1] = msg
    return chat_history


def generate_query(chat_history):
    history = chat_history[:-1]
    query = chat_history[-1][0]

    pic_file = ""
    model_history = []
    for hist in history:
        user_msg = hist[0]
        assistant_msg = hist[1]
        if isinstance(user_msg, str):
            model_history.append({"role":"user","content":user_msg})
            # print(f"user: {user_msg}")
        elif isinstance(user_msg, tuple):
            pic_file = user_msg[0]
            # print(f"pic_file: {pic_file}")
        else:
            print(f"skip user: {user_msg}")
        if isinstance(assistant_msg, str):
            model_history.append({"role":"assistant","content":assistant_msg})
            # print(f"assistant: {assistant_msg}")
        else:
            pass
            # print(f"skip assistant: {assistant_msg}")
    # query = f"<指令>根据已知信息，简洁和专业的来回答问题。不允许在答案中添加编造成分</指令>\n<问题>{query}</问题>"
    # print(f"query: {query}")
    return query, model_history, pic_file


def glm4v_stream_chat(query, history, pic_file, model, tokenizer):
    print(f"{pic_file}\n{query}\n{history}\n")
    img = None
    try:
        img = Image.open(pic_file).convert("RGB")
    except Exception as e:
        print(f"Invalid image path. Continuing with text conversation: {e}")
    
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    if img:
        messages[-1].update({"image": img})

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = model.config.eos_token_id
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )
    stop = StopOnTokens()
    generate_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    return streamer, thread


def handle_chat(chat_history):
    query, history, pic_file = generate_query(chat_history)
    streamer, thread = glm4v_stream_chat(query, history, pic_file, model, tokenizer)
    generated_text = ""
    for new_token in streamer:
        generated_text += new_token
        yield chat_resp(chat_history, generated_text)
    thread.join()


def handle_add_msg(query, chat_history):
    # logger.info(query, chat_history)
    for x in query["files"]:
        chat_history.append(((x,), None))
    if query["text"] is not None:
        chat_history.append((query["text"], None))
    return gr.MultimodalTextbox(value=None, interactive=False), chat_history


def init_llm():
    global model, tokenizer
    print(f"load model from {model_full}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_full,
        trust_remote_code=True,
        encode_special_tokens=True
    )
    model = AutoModel.from_pretrained(
        model_full,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
    ).eval()
    # model = AutoModel.from_pretrained(
    #     model_full,
    #     trust_remote_code=True,
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    #     torch_dtype=torch.float16,
    #     # low_cpu_mem_usage=True,
    #     device_map=device,
    # ).eval()


def init_blocks():
    with gr.Blocks(title="Multimodal") as app:
        gr.Markdown("# 多模态 🤖  \n"
            f"- llm: {model_name}")
        chatbot = gr.Chatbot(label="chat", show_label=False, height=600)
        with gr.Row():
            query = gr.MultimodalTextbox(label="chat with picture", show_label=False,
                scale=4, file_types=["image"])
            clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot], scale=1)
        
        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
                handle_chat, inputs=[chatbot], outputs=[chatbot]).then(
                    lambda: gr.MultimodalTextbox(interactive=True), outputs=[query])

    return app


if __name__ == "__main__":
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False,
        share=False, favicon_path="langchain_demo/icons/shiba.svg")
