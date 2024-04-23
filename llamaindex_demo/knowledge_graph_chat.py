import os
import logging
import sys
import threading, time, random

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

import torch
import gradio as gr
from transformers import BitsAndBytesConfig
from pyvis.network import Network
from fastapi import FastAPI, HTTPException, Response, Query
from starlette.responses import HTMLResponse
import uvicorn
from llamaindex_demo.chatglm_llm import ChatGLM3LLM
from llamaindex_demo.zephyr_llm import ZephyrLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter


model_path = "/root/huggingface/models"

model_name = f"{model_path}/THUDM/chatglm3-6b-128k"
context_window=131072

# model_name = f"{model_path}/HuggingFaceH4/zephyr-7b-beta"
# context_window=32768

embedding_model_name = f"{model_path}/BAAI/bge-large-zh-v1.5"

max_new_tokens=1024
top_k=50
top_p=0.5
temperature=0.1


query_engine = None
network_html_path = None

def init_embed_model():
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name,
        device="cuda")
    # embeddings = embed_model.get_text_embedding("Hello World!")
    # print(len(embeddings))
    # print(embeddings[:5])
    return embed_model


def init_documents(input_files):
    # input_files = [
    #     "/root/github/llm-demo/langchain_demo/rag/files/小米汽车发布会.txt"
    # ]
    documents = SimpleDirectoryReader(
        input_files=input_files
    ).load_data(show_progress=True)
    return documents


def init_graph_query_engine(basename, documents):
    persist_path = f"/root/github/llm-demo/cache/graph_store/{basename}_graph_store.json"
    if os.path.exists(persist_path):
        graph_store = SimpleGraphStore().from_persist_path(persist_path)
    else:
        graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents,
        show_progress = True,
        max_triplets_per_chunk = 10,
        storage_context = storage_context,
        include_embeddings = True,
    )
    engine = index.as_query_engine(
        streaming=True,
        include_text = True,
        response_mode = "tree_summarize",
        verbose = True,
        retriever_mode = "hybrid",
        similarity_top_k = 3,
        max_keywords_per_query = 3,
        num_chunks_per_query = 3,
        graph_store_query_depth = 2,
    )

    if not os.path.exists(persist_path):
        graph_store.persist(persist_path)

    save_graph_visualizing(basename, index)
    return engine


def init_llm():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"load model from: {model_name}")
    llm = ChatGLM3LLM(
        model_name=model_name,
        device_map="cuda",
        max_new_tokens=max_new_tokens,
        # model_kwargs={"quantization_config": quantization_config},
    )
    # llm = ZephyrLLM(
    #     model_name=model_name,
    #     tokenizer_name=model_name,
    #     device_map="cuda",
    #     max_new_tokens=max_new_tokens,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     generate_kwargs={"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p},
    # )
    embed_model = init_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=100)
    Settings.num_output = 1024
    Settings.context_window = context_window


def save_graph_visualizing(basename: str, index: KnowledgeGraphIndex):
    global network_html_path

    g = index.get_networkx_graph()
    net = Network(cdn_resources="remote", directed=True, notebook=False)
    net.from_nx(g)
    
    network_html_path = f"/root/github/llm-demo/cache/{basename}_knowledge_graph_network.html"
    net.write_html(name=network_html_path)

    # html_content = net.generate_html(name=network_html_path)
    # with open(network_html_path, "w", encoding="utf-8") as f:
        # f.write(html_content)


def handle_upload_file(upload_file: str):
    global query_engine

    if not upload_file:
        print("invalid upload_file")
        return
    
    basename, ext = os.path.splitext(os.path.basename(upload_file.name))
    documents = init_documents([upload_file.name])
    query_engine = None
    query_engine = init_graph_query_engine(basename, documents)

    link = f"http://192.168.0.20:38061/graph_visualizing"
    return gr.Button(f"知识图谱可视化: {basename}", link=link, variant="primary", interactive=True)


def handle_chat(chat_history):
    global query_engine

    if not query_engine:
        chat_history[-1][1] = "需要先上传文件初始化知识图谱"
        yield chat_history
        return

    query = chat_history[-1][0]
    streaming_resp = query_engine.query(query)
    # streaming_resp.print_response_stream()
    response_txt = ""
    chat_history[-1][1] = ""
    for text in streaming_resp.response_gen:
        # print(text, end="", flush=True)
        chat_history[-1][1] += text
        yield chat_history
        # response_txt += text

    # response = query_engine.query(query)
    # return response.response

    # print(chat_history)
    # print("---------------------------------------------------------------")

    # chat_history[-1][1] = ""
    # bot_message = random.choice([
    #     "你有这么高速运转的机械进入中国，记住我给出的原理，小的时候就是研发人，就是研发这个东西的原理是阴间政权管着",
    #     "知道为什么有生灵给他运转先位，还有专门饲养这个，为什么地下产这种东西，它管着它说是五世同堂旗下子孙，你以为我跟你闹着玩呢",
    #     "你不警察吗，黄龙江一派全都带蓝牙，黄龙江我告我告诉你，在阴间是是那个化名，化名我小舅，亲小舅，赵金兰的那个嫡子嫡孙"])
    # for char in bot_message:
    #     chat_history[-1][1] += char
    #     # print(chat_history)
    #     time.sleep(0.05)
    #     yield chat_history


def handle_add_msg(query, chat_history):
    # print(query, chat_history)
    return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def refresh_gv_button():
    if query_engine:
        print(f"network_html_path: {network_html_path}")
        basename, ext = os.path.splitext(os.path.basename(network_html_path))
        basename = basename.rstrip("_knowledge_graph_network")
        gv_btn_value = f"可视化知识图谱: {basename}"
        gv_btn_link = f"http://192.168.0.20:38061/graph_visualizing"
        graph_visualizing_btn = gr.Button(gv_btn_value, variant="primary", 
            interactive=True, link=gv_btn_link)
    else:
        print("graph is not initialized")
        graph_visualizing_btn = gr.Button("当前知识图谱: 无", variant="primary", interactive=False)
    return graph_visualizing_btn


def uploading_gv_button():
    return gr.Button("知识图谱初始化中...", variant="primary", interactive=False)


def init_blocks():
    with gr.Blocks() as app:
        gr.Markdown("# 知识图谱对话")
        with gr.Row():
            with gr.Column(scale=1):
                upload_file = gr.File(file_types=[".text"], label="知识图谱原始文件")
                graph_visualizing_btn = refresh_gv_button()
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
        upload_file.upload(
            handle_upload_file, upload_file, graph_visualizing_btn)
        upload_file.change(
            uploading_gv_button, None, graph_visualizing_btn)
        app.load(refresh_gv_button, None, graph_visualizing_btn)

    return app


def render_html_content() -> str:
    if not network_html_path:
        return f"<html><b>file not exist</b></html>"
    with open(network_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


app = FastAPI()
@app.get("/graph_visualizing")
async def graph_visualizing():
    html_content = render_html_content()
    return HTMLResponse(content=html_content, status_code=200)


def run_html_svr():
    uvicorn.run(app, host="0.0.0.0", port=8061)


if __name__ == "__main__":
    init_llm()
    # documents = init_documents()
    # query_engine = init_graph_query_engine(documents)
    # response = query_engine.query(
    #     "小米汽车售价",
    # )
    # print(type(response))

    th = threading.Thread(target=run_html_svr)
    th.daemon = True
    th.start()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False)
