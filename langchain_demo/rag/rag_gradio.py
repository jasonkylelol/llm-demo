import sys, os
import gradio as gr

from transformers import AutoTokenizer, AutoModel, AutoConfig
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader


device="cuda"
model_path = "/root/huggingface/models"
model_name = "THUDM/chatglm3-6b-128k"
model_full = f"{model_path}/THUDM/chatglm3-6b-128k"
context_window=131072

top_p=0.65
temperature=0.1


model, tokenizer = None, None
documents = []
chat_file = ""


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


def handle_chat(chat_history):
    instru_buf = ""
    last_resp = ""
    query = chat_history[-1][0]
    knowledge = ""
    for document in documents:
        knowledge = f"{knowledge}\n{document.page_content}"
    content = f"根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识）等文案:\n{knowledge}\n问题: {query}"
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"{prompt}\n\n")

    chat_history[-1][1] = ""
    for resp, history in model.stream_chat(tokenizer, prompt,
        do_sample=True, top_p=top_p, temperature=temperature):
        if resp == "":
            continue
        last_resp, new_token = extract_new_token(last_resp, resp)
        if new_token == "":
            continue
        chat_history[-1][1] += new_token
        yield chat_history


def handle_add_msg(query, chat_history):
    # print(query, chat_history)
    return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def init_llm():
    global model
    global tokenizer
    print(f"load from {model_full}")
    model_config = AutoConfig.from_pretrained(
        model_full,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_full,
        device_map=device,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_full,
        config=model_config,
        trust_remote_code=True,
        device_map=device).eval()


def handle_upload_file(upload_file):
    global documents, chat_file

    if not upload_file:
        print("invalid upload_file")
        return
    
    print(f"handle file: {upload_file}")
    
    basename, ext = os.path.splitext(os.path.basename(upload_file))
    # print(basename, ext)
    if ext == '.pdf':
        loader = RapidOCRPDFLoader(upload_file)
        documents = loader.load()
    elif ext in ['.doc', '.docx']:
        loader = RapidOCRDocLoader(upload_file)
        documents = loader.load()
    elif ext == '.txt':
        loader = UnstructuredFileLoader(upload_file, autodetect_encoding=True)
        documents = loader.load()
    else:
        raise RuntimeError(f"invalid file: {upload_file}")
    chat_file = f"{basename}{ext}"
    print(f"init document: {basename}{ext} succeed")


def file_uploading():
    return gr.Markdown("正在处理文件...")

def file_loaded():
    return gr.Markdown(f"当前使用的文件: {chat_file}")


def init_blocks():
    with gr.Blocks() as app:
        gr.Markdown("# RAG\n"
            f"### 当前模型: {model_name}\n"
            f"### context-window: {context_window}")
        with gr.Row():
            with gr.Column(scale=1):
                upload_file = gr.File(file_types=[".text"], label="对话文件")
                file_status = file_loaded()
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="chatroom", show_label=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        query = gr.Textbox(label="Say something", scale=5)
                    with gr.Column(scale=1):
                        clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot],
                            size="sm", scale=1, variant="primary")
        query.submit(
            handle_add_msg, [query, chatbot], [query, chatbot]).then(
            handle_chat, chatbot, chatbot).then(
            lambda: gr.Textbox(interactive=True), None, [query])
        upload_file.upload(handle_upload_file, upload_file).then(file_loaded, outputs=file_status)
        upload_file.change(file_uploading, outputs=file_status)
        app.load(file_loaded, outputs=file_status)

    return app


# nohup CUDA_VISIBLE_DEVICES=1 python langchain_demo/rag/rag_gradio.py > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False, share=True)
