import sys, os, logging
import re, math
from typing import Tuple, List
from threading import Thread

import torch
import gradio as gr
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification,
    AutoModelForCausalLM, TextIteratorStreamer)
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter
from langchain_demo.rag.markdown_splitter import split_markdown_documents, load_markdown

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

# embedding_model_name = "BAAI/bge-large-zh-v1.5"
embedding_model_name = "maidalun1020/bce-embedding-base_v1"
embedding_model_full = f"{model_path}/{embedding_model_name}"

# rerank_model_name = "BAAI/bge-reranker-large"
rerank_model_name = "maidalun1020/bce-reranker-base_v1"
rerank_model_full = f"{model_path}/{rerank_model_name}"

max_new_tokens=8192
# top_p=0.1
# temperature=0.1

model, tokenizer = None, None
embedding_model = None
embedding_score_threshold = 0.3
rerank_model, rerank_tokenizer = None, None

vector_db_dict = {}
uploading_files = {}


def generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k) -> Tuple[str, List, List]:
    query = chat_history[-1][0]
    
    searched_docs = embedding_query(query, kb_file, embedding_top_k)

    rerank_docs = rerank_documents(query, searched_docs, rerank_top_k)

    print(f"query: {query}\n", flush=True)
    knowledge = ""
    for idx, document in enumerate(rerank_docs):
        print(f"{document.page_content}\n", flush=True)
        knowledge = f"{knowledge}\n\n{document.page_content}"
    knowledge = knowledge.strip()

    kb_query = ("<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，"
        "请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。</指令>\n"
        f"<已知信息>{knowledge}</已知信息>\n<问题>{query}</问题>")
    # kb_query = ("根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识，根据文档）等文案，"
    #     "如果问题与背景知识不相关，或无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n"
    #     f"背景知识: \n\n{knowledge}\n\n问题: {query}")
    return kb_query, rerank_docs, []


def embedding_query(query, kb_file, embedding_top_k):
    vector_db = vector_db_dict.get(kb_file)

    # searched_docs = vector_db.similarity_search(query, k=embedding_top_k)
    searched_docs = vector_db.similarity_search_with_relevance_scores(query, k=embedding_top_k)
    # embedding_vectors = embedding_model.embed_query(query)
    # searched_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=embedding_top_k)
    # searched_docs = vector_db.similarity_search_with_score_by_vector(embedding_vectors, k=embedding_top_k)
    docs = []
    for searched_doc in searched_docs:
        doc = searched_doc[0]
        score = searched_doc[1]
        # print(f"{score} : {doc.page_content}")
        if score < embedding_score_threshold:
            continue
        docs.append(doc)
    return docs


def rerank_documents(query, docs, rerank_top_k):
    if len(docs) < 2:
        return docs
    pairs = []
    for idx, document in enumerate(docs):
        pairs.append([query, document.page_content])
    rerank_docs = []
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = torch.sigmoid(scores)
        scores = scores.tolist()
    
    # print(f"scores: {scores}")
    combined_list = list(zip(docs, scores))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
    for idx, item in enumerate(sorted_combined_list):
        if idx >= rerank_top_k:
            break
        document = item[0]
        rerank_docs.append(document)
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
        print(f"user: {user_msg}\nassistant: {assistant_msg}", flush=True)
    print(f"query: {query}", flush=True)
    return query, model_history


def chat_resp(chat_history, msg, searched_docs=[]):
    chat_history[-1][1] = msg
    knowledge = ""
    for idx, doc in enumerate(searched_docs):
        knowledge = f"{knowledge}{doc.page_content}\n\n"
    return chat_history, knowledge.strip()


def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def handle_chat(chat_history, kb_file, temperature, embedding_top_k, rerank_top_k):
    if kb_file is not None and kb_file not in vector_db_dict.keys():
        err = f"文件: {kb_file} 不存在"
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return

    logger.info(f"temperature: {temperature} embedding_top_k: {embedding_top_k} rerank_top_k: {rerank_top_k}")
    if kb_file is None:
        query, history = generate_prompt(chat_history)
        searched_docs = []
        # yield chat_resp(chat_history, "需要选择文件")
        # return
    else:
        query, searched_docs, history = generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k)

    if "chatglm3" in model_name:
        for resp, model_history in model.stream_chat(tokenizer, query=query, history=history,
            do_sample=True, temperature=temperature):
            yield chat_resp(chat_history, resp, searched_docs)
    else:
        streamer, thread = llama_stream_chat(query, history, temperature)
        generated_text = ""
        for new_text in streamer:
            # stream_print(new_text)
            generated_text += new_text
            yield chat_resp(chat_history, generated_text, searched_docs)
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


def init_embeddings():
    global embedding_model

    logger.info(f"load from {embedding_model_full}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_full,
        model_kwargs={"device": device},
        # encode_kwargs={'batch_size': 32, 'normalize_embeddings': True},
        encode_kwargs={'normalize_embeddings': True},
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
    elif ext == '.md':
        documents = load_markdown(upload_file)
        return documents
    else:
        return "支持 txt pdf doc docx markdown 文件"
    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\n+', ' ', doc.page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]
    return documents


def split_documents(documents: list, chunk_size, chunk_overlap: int):
    if chunk_size > 300:
        full_docs = []
        all_chunk_size = [chunk_size-100, chunk_size, chunk_size+100]
        for auto_chunk_size in all_chunk_size:
            auto_chunk_overlap = int(auto_chunk_size / 4)
            logger.info(f"[split_documents] auto_chunk_size:{auto_chunk_size} auto_chunk_overlap:{auto_chunk_overlap}")
            text_splitter = ChineseRecursiveTextSplitter(
                chunk_size=auto_chunk_size,
                chunk_overlap=auto_chunk_overlap,
            )
            docs = text_splitter.split_documents(documents)
            full_docs.extend(docs)
        return full_docs

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


def custom_relevance_score_fn(distance: float) -> float:
    score = 1.0 - distance / math.sqrt(2)
    score = 0 if score < 0 else score 
    return score
    # if distance > 0:
    #     return 1.0 - distance
    # return -1.0 * distance


def handle_upload_file(upload_file: str, chunk_size: int):
    global vector_db_dict, uploading_files

    if not upload_file:
        logger.error("invalid upload_file")
        return

    logger.info(f"handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)
    basename, ext = os.path.splitext(file_basename)

    uploading_files[file_basename] = "正在加载文件..."
    documents = load_documents(upload_file)
    if isinstance(documents, str):
        logger.error(documents)
        uploading_files[file_basename] = documents
        return

    uploading_files[file_basename] = "正在拆分文件..."
    chunk_overlap = int(chunk_size / 4)
    if ext == '.md':
        documents = split_markdown_documents(documents, chunk_size)
    else:
        documents = split_documents(documents, chunk_size, chunk_overlap)
    logger.info(f"file: {file_basename} split to {len(documents)} chunks")

    uploading_files[file_basename] = f"拆分文件为{len(documents)}份，正在向量化..."
    vector_db = FAISS.from_documents(documents, embedding_model,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE, relevance_score_fn=custom_relevance_score_fn)

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
        gr.Markdown("# RAG 🤖  \n"
            f"- llm: {model_name}  \n"
            f"- embeddings: {embedding_model_name}  \n"
            f"- rerank: {rerank_model_name}  \n"
            f"- 支持 txt, pdf, doc, docx, markdown")
        with gr.Row():
            with gr.Column(scale=3):
                # upload_file = gr.File(file_types=[".text"], label="对话文件")
                upload_file = gr.UploadButton("对话文件上传", variant="primary")
                upload_stat = gr.Markdown(value=uploading_status, every=0.5)
                with gr.Row():
                    chunk_size = gr.Number(value=300, minimum=100, maximum=1000, label="chunk_size")
                    # chunk_overlap = gr.Number(value=50, minimum=10, maximum=500, label="chunk_overlap")
                    temperature = gr.Number(value=0.1, minimum=0.01, maximum=0.99, label="temperature")
                with gr.Row():
                    embedding_top_k = gr.Number(value=15, minimum=5, maximum=100, label="embedding_top_k")
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
            handle_chat, inputs=[chatbot, query_doc, temperature, embedding_top_k, rerank_top_k], outputs=[chatbot, searched_docs]).then(
            lambda: gr.Textbox(interactive=True), outputs=[query])
        upload_file.upload(
            handle_upload_file, inputs=[upload_file, chunk_size], show_progress="full").then(
            doc_loaded, outputs=query_doc)
        app.load(doc_loaded, outputs=query_doc)

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_llm()
    init_embeddings()
    init_reranker()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False,
        share=False, favicon_path="langchain_demo/icons/shiba.svg")
