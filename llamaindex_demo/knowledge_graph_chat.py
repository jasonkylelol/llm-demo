import os
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

import torch
import gradio as gr
from transformers import BitsAndBytesConfig
from pyvis.network import Network
from llamaindex_demo.chatglm_llm import ChatGLM3LLM
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


model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-zh-v1.5"
network_html_path = "cache/knowledge_graph_network.html"
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.7
context_window=131072

query_engine = None

def init_embed_model():
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_path,
        device="cuda")
    # embeddings = embed_model.get_text_embedding("Hello World!")
    # print(len(embeddings))
    # print(embeddings[:5])
    return embed_model


def init_documents():
    input_files = [
        "/root/github/llm-demo/langchain_demo/rag/files/小米汽车发布会.txt"
    ]
    documents = SimpleDirectoryReader(
        input_files=input_files
    ).load_data(show_progress=True)
    return documents


def init_graph_query_engine(documents):
    persist_path = "/root/github/llm-demo/cache/graph_store/graph_store.json"
    graph_store = SimpleGraphStore().from_persist_path(persist_path)
    # graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # NOTE: can take a while!
    index = KnowledgeGraphIndex.from_documents(
        documents,
        show_progress = True,
        max_triplets_per_chunk = 10,
        storage_context = storage_context,
        include_embeddings = True,
    )
    global query_engine
    query_engine = index.as_query_engine(
        include_text = True,
        response_mode = "tree_summarize",
        verbose = True,
        retriever_mode = "hybrid",
        similarity_top_k = 3,
        max_keywords_per_query = 3,
        num_chunks_per_query = 3,
        graph_store_query_depth = 2,
    )

    # query_engine = KnowledgeGraphQueryEngine(
    #     storage_context=storage_context,
    #     llm=llm,
    #     verbose=True,
    # )

    # graph_store.persist(persist_path)
    graph_visualizing(index)
    return query_engine


def init_llm():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"load model from: {model_path}")
    llm = ChatGLM3LLM(
        model_name=model_path,
        device_map="cuda",
        # model_kwargs={"quantization_config": quantization_config},
    )
    embed_model = init_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    Settings.num_output = 1024
    Settings.context_window = context_window


def graph_visualizing(index: KnowledgeGraphIndex):
    g = index.get_networkx_graph()
    net = Network(cdn_resources="remote", directed=True, notebook=False)
    net.from_nx(g)
    
    net.write_html(name=network_html_path)

    # html_content = net.generate_html(name=network_html_path)
    # with open(network_html_path, "w", encoding="utf-8") as f:
        # f.write(html_content)


def on_submit(query: str) -> str:
    global query_engine

    response = query_engine.query(query)
    return response.response


def render_html_content() -> str:
    with open(network_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


def init_blocks():
    with gr.Blocks() as app:
        gr.Markdown("# 知识图谱对话")
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(label="检索内容")
                submit_btn = gr.Button("提交", variant="primary")
                graph_visualizing_btn = gr.Button("知识图谱可视化", variant="secondary", link="http://192.168.0.20:38061/")
            with gr.Column(scale=4):
                result = gr.TextArea(label="检索结果")
        inputs = [query]
        submit_btn.click(fn=on_submit, inputs=inputs, outputs=[result])
    return app


if __name__ == "__main__":
    init_llm()
    documents = init_documents()
    query_engine = init_graph_query_engine(documents)

    # response = query_engine.query(
    #     "小米汽车售价",
    # )
    # print(type(response))

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False)
