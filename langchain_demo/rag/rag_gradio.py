import sys, os, logging
import re
from typing import Tuple, List

import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoConfig
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

top_p=0.65
temperature=0.1

model, tokenizer = None, None
embedding_model = None
vector_db_dict = {}


def generate_kb_prompt(chat_history, doc) -> Tuple[str, str, List]:
    query = chat_history[-1][0]
    vector_db = vector_db_dict.get(doc)

    # searched_docs = vector_db.similarity_search(query, k=3)
    embedding_vectors = embedding_model.embed_query(query)
    searched_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=3)

    knowledge = ""
    for idx, document in enumerate(searched_docs):
        knowledge = f"{knowledge}\n{document.page_content}"

    kb_query = ("根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识，根据文档）等文案，"
        "如果问题与背景知识不相关，或无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n"
        f"背景知识: \n{knowledge}\n\n问题: {query}")
    return kb_query, knowledge, []


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


def chat_resp(chat_history, msg, knowledge=""):
    chat_history[-1][1] = msg
    return chat_history, knowledge.strip()


def handle_chat(chat_history, doc):
    if doc is not None and doc not in vector_db_dict.keys():
        err = f"文件: {doc} 不存在"
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return

    if doc is None:
        # query, history = generate_prompt(chat_history)
        yield chat_resp(chat_history, "需要选择文件")
        return
    else:
        query, knowledge, history = generate_kb_prompt(chat_history, doc)
    logger.info(f"{query}\n{history}\n")

    for resp, model_history in model.stream_chat(tokenizer, query=query, history=history,
        do_sample=True, temperature=temperature):
        yield chat_resp(chat_history, resp, knowledge)


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


def handle_upload_file(upload_file: str, chunk_size: int, chunk_overlap: int):
    global vector_db_dict

    if not upload_file:
        logger.error("invalid upload_file")
        return gr.Markdown("invalid upload file", visible=True)

    logger.info(f"handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)

    documents = load_documents(upload_file)
    if isinstance(documents, str):
        logger.error(documents)
        return gr.Markdown(documents, visible=True)

    documents = split_documents(documents, chunk_size, chunk_overlap)

    vector_db = FAISS.from_documents(documents, embedding_model)
    if file_basename in vector_db_dict.keys():
        del vector_db_dict[file_basename]
    vector_db_dict[file_basename] = vector_db

    logger.info(f"[chinese_rec_text_splitter] [{file_basename}] [chunks] {len(documents)}")
    return gr.Markdown(visible=False)


def doc_loaded():
    return gr.Radio(choices=vector_db_dict.keys(), label="可供选择的文件:")


def init_blocks():
    with gr.Blocks(title="RAG") as app:
        gr.Markdown("# RAG  \n"
            f"- llm: {model_name}&emsp;&emsp;context-window: 8192  \n"
            f"- embeddings: {embedding_model_name}&emsp;&emsp;context-window: 512  \n"
            f"- 支持 txt, pdf, doc, docx")
        with gr.Row():
            with gr.Column(scale=2):
                # upload_file = gr.File(file_types=[".text"], label="对话文件")
                upload_file = gr.UploadButton("对话文件上传", variant="primary")
                upload_stat = gr.Markdown(visible=False)
                chunk_size = gr.Number(value=300, minimum=100, maximum=1000, label="chunk_size")
                chunk_overlap = gr.Number(value=50, minimum=10, maximum=500, label="chunk_overlap")
                searched_docs = gr.Textbox(label="检索到的文本", lines=10)
            with gr.Column(scale=5):
                docs = doc_loaded()
                chatbot = gr.Chatbot(label="chatroom", show_label=False)
                with gr.Row():
                    query = gr.Textbox(label="Say something", lines=1, scale=4)
                    clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot, searched_docs], scale=1)
        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
            handle_chat, inputs=[chatbot, docs], outputs=[chatbot, searched_docs]).then(
            lambda: gr.Textbox(interactive=True), outputs=[query])
        upload_file.upload(
            handle_upload_file, inputs=[upload_file, chunk_size, chunk_overlap], outputs=upload_stat).then(
            doc_loaded, outputs=docs)
        app.load(doc_loaded, outputs=docs)

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()
    init_embeddings()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False, share=False)
