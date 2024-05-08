import sys, os, logging
import gradio as gr

from transformers import AutoTokenizer, AutoModel, AutoConfig
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO)

device="cuda"
model_path = "/root/huggingface/models"
model_name = "THUDM/chatglm3-6b-128k"
model_full = f"{model_path}/{model_name}"
context_window=131072

top_p=0.65
temperature=0.1


model, tokenizer = None, None
document_dict = {}
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


def handle_chat(chat_history, doc):
    if doc is None:
        err = f"必须选择一个文件"
        print(f"[handle_chat] err: {err}", flush=True)
        chat_history[-1][1] = err
        yield chat_history
        return

    if doc not in document_dict.keys():
        err = f"文件: {doc} 不存在"
        print(f"[handle_chat] err: {err}", flush=True)
        chat_history[-1][1] = err
        yield chat_history
        return

    last_resp = ""
    query = chat_history[-1][0]
    knowledge = ""
    documents = document_dict.get(doc)
    for document in documents:
        knowledge = f"{knowledge}\n{document.page_content}"
    content = ("根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识，根据 XX 文档）等文案，"
        "如果问题与背景知识不相关，或无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n"
        f"背景知识: \n{knowledge}\n问题: {query}")
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"{prompt}\n\n", flush=True)

    try:
        chat_history[-1][1] = ""
        # top_p=top_p,
        for resp, history in model.stream_chat(tokenizer, prompt,
            do_sample=True, temperature=temperature):
            if resp == "":
                continue
            last_resp, new_token = extract_new_token(last_resp, resp)
            if new_token == "":
                continue
            chat_history[-1][1] += new_token
            yield chat_history
    except Exception as e:
        err = f"[handle_chat] exception: {e}"
        print(f"{err}", flush=True)
        chat_history[-1][1] = err
        yield chat_history
        return


def handle_add_msg(query, chat_history):
    # print(query, chat_history)
    return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def init_llm():
    global model
    global tokenizer
    print(f"load from {model_full}", flush=True)
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
    global document_dict, chat_file

    if not upload_file:
        print("invalid upload_file", flush=True)
        return gr.Markdown("invalid upload file")
    
    print(f"handle file: {upload_file}", flush=True)
    file_basename = os.path.basename(upload_file)
    basename, ext = os.path.splitext(file_basename)
    # print(basename, ext)
    if ext == '.pdf':
        loader = RapidOCRPDFLoader(upload_file)
        documents = loader.load()
        document_dict[file_basename] = documents
    elif ext in ['.doc', '.docx']:
        loader = RapidOCRDocLoader(upload_file)
        documents = loader.load()
        document_dict[file_basename] = documents
    elif ext == '.txt':
        loader = UnstructuredFileLoader(upload_file, autodetect_encoding=True)
        documents = loader.load()
        document_dict[file_basename] = documents
    else:
        print(f"invalid upload file: {upload_file}", flush=True)
        return gr.Markdown(f"file: {file_basename} not support")
    print(f"init document: {file_basename} succeed", flush=True)


def doc_loaded():
    return gr.Radio(choices=document_dict.keys(), label="可供选择的文件:")


def init_blocks():
    with gr.Blocks() as app:
        gr.Markdown("# RAG\n"
            "将整个文件内容作为问题的上下文信息  \n"
            f"{model_name}  \n"
            f"context-window: {context_window}  \n"
            f"支持 txt, pdf, doc, docx")
        with gr.Row():
            with gr.Column(scale=1):
                upload_file = gr.File(file_types=[".text"], label="对话文件")
                upload_status = gr.Markdown()
                docs = doc_loaded()
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
            handle_chat, inputs=[chatbot, docs], outputs=chatbot).then(
            lambda: gr.Textbox(interactive=True), outputs=[query])
        upload_file.upload(handle_upload_file, inputs=upload_file, outputs=upload_status).then(
            doc_loaded, outputs=docs)
        app.load(doc_loaded, outputs=docs)

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False, share=False)
