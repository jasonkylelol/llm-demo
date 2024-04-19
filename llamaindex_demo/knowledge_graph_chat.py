import os
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

import torch
from transformers import BitsAndBytesConfig
from llamaindex_demo.chatglm_llm import ChatGLM3LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from llama_index.core.service_context import ServiceContext
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.core.query_engine import KnowledgeGraphQueryEngine

model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-zh-v1.5"
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.7
context_window=131072


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
    ).load_data()
    return documents


def init_graph_query_engine(service_context, documents):
    persist_path = "/root/github/llm-demo/cache/graph_store/graph_store.json"
    graph_store = SimpleGraphStore().from_persist_path(persist_path)
    # graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # NOTE: can take a while!
    index = KnowledgeGraphIndex.from_documents(
        documents,
        show_progress = True,
        max_triplets_per_chunk = 3,
        service_context = service_context,
        storage_context = storage_context,
        include_embeddings = True,
    )

    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        verbose=True,
        # embedding_mode="hybrid",
        retriever_mode="hybrid",
        similarity_top_k = 3,
    )

    # query_engine = KnowledgeGraphQueryEngine(
    #     storage_context=storage_context,
    #     llm=llm,
    #     verbose=True,
    # )

    # graph_store.persist(persist_path)

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

    return llm, embed_model


if __name__ == "__main__":
    llm, embed_model = init_llm()
    documents = init_documents()

    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model,
    )

    query_engine = init_graph_query_engine(service_context, documents)

    response = query_engine.query(
        "小米汽车售价",
    )
    print(response)
