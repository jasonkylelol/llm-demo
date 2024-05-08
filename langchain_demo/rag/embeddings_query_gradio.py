import os,sys
sys.path.append(os.getcwd())

import re
import gradio as gr
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_core.documents import Document


embeddings_models = {}
separators=[
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200B",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
]

DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


def init_documents(upload_file):
    print(f"handle file: {upload_file}", flush=True)
    file_basename = os.path.basename(upload_file)
    basename, ext = os.path.splitext(file_basename)
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
        print(f"invalid upload file: {upload_file}", flush=True)
        return f"仅支持 txt pdf doc docx"

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\s+', ' ', doc.page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]

    return documents


def handle_rec_text_splitter(
    file, model_name, query, chunk_size, chunk_overlap) -> str:
    documents = init_documents(file.name)
    if isinstance(documents, str):
        return documents
    rec_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # add_start_index=add_start_index,
        length_function=len,
        is_separator_regex=False,
        separators=separators,
    )
    docs = rec_text_splitter.split_documents(documents)
    print(f"[rec_text_splitter] [{model_name}] [chunks] {len(docs)}")
    vector_db = FAISS.from_documents(docs, embeddings_models[model_name])
    resp_docs = vector_db.similarity_search(query, k=3)
    # embedding_vectors = embeddings_models[model_name].embed_query(query)
    # resp_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)
    # return resp_docs[0].page_content

    resp_content = ""
    for idx, doc in enumerate(resp_docs):
        if idx >= 3:
            break
        resp_content = f"{resp_content}{doc.page_content}\n\n"
    return resp_content


def handle_chinese_rec_text_splitter(
    file, model_name, query, chunk_size, chunk_overlap) -> str:
    documents = init_documents(file.name)
    if isinstance(documents, str):
        return documents
    rec_text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = rec_text_splitter.split_documents(documents)
    print(f"[chinese_rec_text_splitter] [{model_name}] [chunks] {len(docs)}")
    vector_db = FAISS.from_documents(docs, embeddings_models[model_name])
    # resp_docs = vector_db.similarity_search(query, k=3)
    embedding_vectors = embeddings_models[model_name].embed_query(query)
    resp_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=3)
    # return resp_docs[0].page_content

    resp_content = ""
    for idx, doc in enumerate(resp_docs):
        if idx >= 3:
            break
        resp_content = f"{resp_content}{doc.page_content}\n\n"
    return resp_content


def handle_semantic_chunker(file, model_name, query, breakpoint_threshold_type) -> str:
    text_splitter = SemanticChunker(embeddings_models[model_name],
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    with open(file.name) as f:
        content = f.read()
    docs = text_splitter.create_documents([content])
    print(f"[semantic_chunker] [{model_name}] [chunks] {len(docs)}")
    embedding_vectors = embeddings_models[model_name].embed_query(query)
    vector_db = FAISS.from_documents(docs, embeddings_models[model_name])
    resp_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)
    return resp_docs[0].page_content


def init_embeddings_models():
    global embeddings_models

    embedding_device = "cuda"

    embeddings_models["BAAI/bge-large-zh-v1.5"] = HuggingFaceBgeEmbeddings(
        model_name="/root/huggingface/models/BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    # embeddings_models["BAAI/bge-large-en-v1.5"] = HuggingFaceBgeEmbeddings(
    #     model_name="/root/huggingface/models/BAAI/bge-large-en-v1.5",
    #     model_kwargs=model_kwargs,
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    # embeddings_models["Alibaba-NLP/gte-Qwen1.5-7B-instruct"] = HuggingFaceBgeEmbeddings(
    #     model_name="/root/huggingface/models/Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    #     model_kwargs={
    #         "device": embedding_device,
    #         "trust_remote_code": True,
    #     },
    #     # query_instruction=DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    #     # embed_instruction="",
    #     # encode_kwargs={"normalize_embeddings": True},
    # )
    embeddings_models["maidalun1020/bce-embedding-base_v1"] = HuggingFaceEmbeddings(
        model_name="/root/huggingface/models/maidalun1020/bce-embedding-base_v1",
        model_kwargs={"device": embedding_device},
        encode_kwargs={'batch_size': 32, 'normalize_embeddings': True},
    )

    print("[init_embeddings_models]", embeddings_models.keys())


def on_submit(
    upload_file, splitter_radio, query: str,
    chunk_size, chunk_overlap: int,
    embeddings_model, breakpoint_threshold_type: str
    ) -> str:
    print("[parameters]", upload_file, splitter_radio, query,
        chunk_size, chunk_overlap,
        embeddings_model, breakpoint_threshold_type, flush=True)

    if not upload_file:
        return f"需要上传文件"
    if splitter_radio != "RecursiveCharacterTextSplitter" and splitter_radio != "SemanticChunker" \
        and splitter_radio != "ChineseRecursiveTextSplitter":
        return f"无效文本分割器"
    if query == "":
        return f"检索内容不能为空"
    if embeddings_model == "":
        return f"需要选择 embeddings_model"

    if splitter_radio == "RecursiveCharacterTextSplitter":
        return handle_rec_text_splitter(upload_file, embeddings_model, query, chunk_size, chunk_overlap)
    elif splitter_radio == "ChineseRecursiveTextSplitter":
        return handle_chinese_rec_text_splitter(upload_file, embeddings_model, query, chunk_size, chunk_overlap)
    else:
        return handle_semantic_chunker(upload_file, embeddings_model, query, breakpoint_threshold_type)


def on_splitter_radio_changed(choice):
    if choice == "RecursiveCharacterTextSplitter" or choice == "ChineseRecursiveTextSplitter":
        return gr.Group(visible=True), gr.Group(visible=False)
    else:
        return gr.Group(visible=False), gr.Group(visible=True)


def init_blocks():
    with gr.Blocks() as app:
        gr.Markdown("# embeddings 检索预览")
        with gr.Row():
            with gr.Column():
                upload_file = gr.File(file_types=[".text"], label="需要拆分的文件: [txt pdf doc docx]")
                splitter_radio = gr.Radio(label="文本分割器",
                    choices=[
                        "RecursiveCharacterTextSplitter",
                        "ChineseRecursiveTextSplitter"
                    ],
                    value="ChineseRecursiveTextSplitter")
                embeddings_model = gr.Dropdown(label="embeddings_model",
                    choices=embeddings_models.keys(),
                    value="BAAI/bge-large-zh-v1.5")
                with gr.Group() as recursive_character_params:
                    # recursive_character_params.visible = True
                    # chunk_size=1000, chunk_overlap=400, add_start_index=False
                    chunk_size = gr.Number(value=300, label="chunk_size")
                    chunk_overlap = gr.Number(value=50, label="chunk_overlap")
                with gr.Group() as semantic_params:
                    semantic_params.visible = False
                    # breakpoint_threshold_type ["percentile","standard_deviation","interquartile"]
                    breakpoint_threshold_type = gr.Radio(label="breakpoint_threshold_type",
                        choices=["percentile","standard_deviation","interquartile"],
                        value="percentile",
                        info="任意两个句子之间的差异阈值; percentile: 百分位数 standard_deviation: 标准差 interquartile: 四分位数距离")

                query = gr.Textbox(label="检索内容")
                submit_btn = gr.Button("提交", variant="primary")

                splitter_radio.change(fn=on_splitter_radio_changed, inputs=splitter_radio, outputs=[
                    recursive_character_params, semantic_params])
            with gr.Column():
                result = gr.TextArea(label="检索结果")
        inputs = [upload_file, splitter_radio, query, chunk_size, chunk_overlap,
            embeddings_model, breakpoint_threshold_type]
        submit_btn.click(fn=on_submit, inputs=inputs, outputs=[result])
    return app


if __name__ == '__main__':
    init_embeddings_models()
    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False)
