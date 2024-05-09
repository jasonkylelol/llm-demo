import sys, os, logging
import re
from typing import Tuple, List

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter

logging.basicConfig(level=logging.INFO, encoding="utf-8")
logger = logging.getLogger()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.handlers[0].setFormatter(formatter)

device="cuda"

model_path = "/root/huggingface/models"
model_name = "THUDM/chatglm3-6b"
model_full = f"{model_path}/{model_name}"

embedding_model_name = "BAAI/bge-large-zh-v1.5"
embedding_model_full = f"{model_path}/{embedding_model_name}"

rerank_model_name = "BAAI/bge-reranker-large"
rerank_model_full = f"{model_path}/{rerank_model_name}"

top_p=0.65
temperature=0.1

model, tokenizer = None, None
embedding_model = None
rerank_model, rerank_tokenizer = None, None

vector_db_dict = {}
uploading_files = {}


def generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k) -> Tuple[str, List, List]:
    query = chat_history[-1][0]
    
    searched_docs = embedding_query(query, kb_file, embedding_top_k)

    rerank_docs = rerank_documents(query, searched_docs, rerank_top_k)

    knowledge = ""
    for idx, document in enumerate(rerank_docs):
        knowledge = f"{knowledge}\n\n{document.page_content}"
    knowledge = knowledge.strip()

    kb_query = ("根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识，根据文档）等文案，"
        "如果问题与背景知识不相关，或无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n"
        f"背景知识: \n\n{knowledge}\n\n问题: {query}")
    return kb_query, rerank_docs, []


def embedding_query(query, kb_file, embedding_top_k):
    vector_db = vector_db_dict.get(kb_file)

    # searched_docs = vector_db.similarity_search(query, k=3)
    embedding_vectors = embedding_model.embed_query(query)
    searched_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=embedding_top_k)
    return searched_docs


def rerank_documents(query, docs, rerank_top_k):
    pairs = []
    for idx, document in enumerate(docs):
        pairs.append([query, document.page_content])
    rerank_docs = []
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
        # print(scores)

    combined_list = list(zip(docs, scores))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
    for idx, item in enumerate(sorted_combined_list):
        if idx >= rerank_top_k:
            break
        document = item[0]
        rerank_docs.append(document)

    print(f"query: {query}")
    for idx, document in enumerate(rerank_docs):
        print(f"{document.page_content}\n")

    return rerank_docs


def generate_prompt(chat_history) -> Tuple[str, List]:
    history = chat_history[:-1]
    query = chat_history[-1][0]
    model_history = []
    for hist in history:
        user_msg = hist[0]
        model_history.append({"role":"user","content":user_msg})
        assistant_msg = hist[1]
        model_history.append({"role":"assistant","content":assistant_msg})
    return query, model_history


def chat_resp(chat_history, msg, searched_docs=[]):
    chat_history[-1][1] = msg
    knowledge = ""
    for idx, doc in enumerate(searched_docs):
        knowledge = f"{knowledge}{doc.page_content}\n\n"
    return chat_history, knowledge.strip()


def handle_chat(chat_history, kb_file, embedding_top_k, rerank_top_k):
    if kb_file is not None and kb_file not in vector_db_dict.keys():
        err = f"文件: {kb_file} 不存在"
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return
    logger.info(f"embedding_top_k: {embedding_top_k} rerank_top_k: {rerank_top_k}")
    if kb_file is None:
        # query, history = generate_prompt(chat_history)
        yield chat_resp(chat_history, "需要选择文件")
        return
    else:
        query, searched_docs, history = generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k)
    # logger.info(query)
    # if len(history) > 0:
    #     logger.info(history)

    for resp, model_history in model.stream_chat(tokenizer, query=query, history=history,
        do_sample=True, temperature=temperature):
        yield chat_resp(chat_history, resp, searched_docs)


def handle_add_msg(query, chat_history):
    # logger.info(query, chat_history)
    return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def init_llm():
    global model, tokenizer

    logger.info(f"load from {model_full}")
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
        device_map=device)
    model = model.eval()


def init_embeddings():
    global embedding_model

    logger.info(f"load from {embedding_model_full}")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_full,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_reranker():
    global rerank_model, rerank_tokenizer

    logger.info(f"load from {rerank_model_full}")
    rerank_tokenizer = AutoTokenizer.from_pretrained(
        rerank_model_full,
        device_map=device)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        rerank_model_full,
        device_map=device)
    rerank_model = rerank_model.eval()


def load_documents(upload_file: str):
    file_basename = os.path.basename(upload_file)
    basename, ext = os.path.splitext(file_basename)
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
        return "仅支持 txt pdf doc docx"
    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\s+', ' ', doc.page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]
    return documents


def split_documents(documents: list, chunk_size, chunk_overlap: int):
    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def human_readable_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_names = ('KB', 'MB', 'GB')
    i = 0
    while size_bytes >= 1024 and i < len(size_names):
        size_bytes /= 1024.0
        i += 1
    return '{:.2f} {}'.format(size_bytes, size_names[i-1])


def handle_upload_file(upload_file: str, chunk_size: int, chunk_overlap: int):
    global vector_db_dict, uploading_files

    if not upload_file:
        logger.error("invalid upload_file")
        return

    logger.info(f"handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)

    uploading_files[file_basename] = "正在加载文件..."
    documents = load_documents(upload_file)
    if isinstance(documents, str):
        logger.error(documents)
        uploading_files[file_basename] = documents
        return

    uploading_files[file_basename] = "正在拆分文件..."
    documents = split_documents(documents, chunk_size, chunk_overlap)
    logger.info(f"file: {file_basename} split to {len(documents)} chunks")

    uploading_files[file_basename] = f"拆分文件为{len(documents)}份，正在向量化..."
    vector_db = FAISS.from_documents(documents, embedding_model)

    vector_db_key = f"{file_basename}({human_readable_size(upload_file)})"
    if vector_db_key in vector_db_dict.keys():
        del vector_db_dict[vector_db_key]
    vector_db_dict[vector_db_key] = vector_db

    uploading_files[file_basename] = f"拆分文件为{len(documents)}份，向量化成功"

    logger.info(f"[chinese_rec_text_splitter] [{vector_db_key}] "
        f"chunk_size: {chunk_size} chunk_overlap: {chunk_overlap} chunks: {len(documents)}")
    return


def doc_loaded():
    return gr.Radio(choices=vector_db_dict.keys(), label="可供选择的文件:")


def uploading_status():
    global uploading_files

    # logging.info(uploading_files)
    if len(uploading_files) == 0:
        return ""
    status_info = ""
    for k in uploading_files.keys():
        v = uploading_files.get(k)
        status_info = f"{status_info}{k}:&emsp;&emsp;{v}  \n"
    return status_info.rstrip()


def init_blocks():
    with gr.Blocks(title="RAG") as app:
        gr.Markdown("# RAG  \n"
            f"- llm: {model_name}  \n"
            f"- embeddings: {embedding_model_name}  \n"
            f"- rerank: {rerank_model_name}  \n"
            f"- 支持 txt, pdf, doc, docx")
        with gr.Row():
            with gr.Column(scale=3):
                # upload_file = gr.File(file_types=[".text"], label="对话文件")
                upload_file = gr.UploadButton("对话文件上传", variant="primary")
                upload_stat = gr.Markdown(value=uploading_status, every=0.5)
                with gr.Row():
                    chunk_size = gr.Number(value=200, minimum=100, maximum=1000, label="chunk_size")
                    chunk_overlap = gr.Number(value=50, minimum=10, maximum=500, label="chunk_overlap")
                with gr.Row():
                    embedding_top_k = gr.Number(value=10, minimum=5, maximum=100, label="embedding_top_k")
                    rerank_top_k = gr.Number(value=3, minimum=1, maximum=5, label="rerank_top_k")
                searched_docs = gr.Textbox(label="检索到的文本", lines=10)
            with gr.Column(scale=5):
                query_doc = doc_loaded()
                chatbot = gr.Chatbot(label="chatroom", show_label=False)
                with gr.Row():
                    query = gr.Textbox(label="Say something", scale=4)
                    clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot, searched_docs], scale=1)
        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
            handle_chat, inputs=[chatbot, query_doc, embedding_top_k, rerank_top_k], outputs=[chatbot, searched_docs]).then(
            lambda: gr.Textbox(interactive=True), outputs=[query])
        upload_file.upload(
            handle_upload_file, inputs=[upload_file, chunk_size, chunk_overlap], show_progress="full").then(
            doc_loaded, outputs=query_doc)
        app.load(doc_loaded, outputs=query_doc)

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()
    init_embeddings()
    init_reranker()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False, share=False)
