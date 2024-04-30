import os,sys
sys.path.append(os.getcwd())

import gradio as gr
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter

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


def handle_rec_text_splitter(
    file, model_name, query, chunk_size, chunk_overlap) -> str:
    # print("[handle_rec_text_splitter]", file, model_name, query, chunk_size, chunk_overlap)
    loader = TextLoader(file.name)
    documents = loader.load()
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
    resp_docs = vector_db.similarity_search(query)
    # embedding_vectors = embeddings_models[model_name].embed_query(query)
    # resp_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)
    return resp_docs[0].page_content


def handle_chinese_rec_text_splitter(
    file, model_name, query, chunk_size, chunk_overlap) -> str:
    loader = TextLoader(file.name)
    documents = loader.load()
    rec_text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = rec_text_splitter.split_documents(documents)
    print(f"[chinese_rec_text_splitter] [{model_name}] [chunks] {len(docs)}")
    vector_db = FAISS.from_documents(docs, embeddings_models[model_name])
    resp_docs = vector_db.similarity_search(query)
    # embedding_vectors = embeddings_models[model_name].embed_query(query)
    # resp_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=1)
    return resp_docs[0].page_content


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
        encode_kwargs={"normalize_embeddings": True}
    )
    # embeddings_models["infgrad/stella-large-zh-v3-1792d"] = HuggingFaceEmbeddings(
    #     model_name="/root/huggingface/models/infgrad/stella-large-zh-v3-1792d",
    #     model_kwargs=model_kwargs,
    # )
    # embeddings_models["BAAI/bge-large-en-v1.5"] = HuggingFaceBgeEmbeddings(
    #     model_name="/root/huggingface/models/BAAI/bge-large-en-v1.5",
    #     model_kwargs=model_kwargs,
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    embeddings_models["Alibaba-NLP/gte-Qwen1.5-7B-instruct"] = HuggingFaceBgeEmbeddings(
        model_name="/root/huggingface/models/Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        model_kwargs={
            "device": embedding_device,
            "trust_remote_code": True,
        },
        query_instruction=DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
        embed_instruction="",
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
        gr.Markdown("# 中/英 文本分割效果预览")
        with gr.Row():
            with gr.Column():
                upload_file = gr.File(file_types=[".text"], label="需要拆分的文件")

                splitter_radio = gr.Radio(label="文本分割器",
                    choices=["RecursiveCharacterTextSplitter", "SemanticChunker", "ChineseRecursiveTextSplitter"],
                    value="RecursiveCharacterTextSplitter")
                embeddings_model = gr.Dropdown(label="embeddings_model",
                        choices=["BAAI/bge-large-zh-v1.5", "infgrad/stella-large-zh-v3-1792d", "BAAI/bge-large-en-v1.5"],
                        value="BAAI/bge-large-zh-v1.5",
                        info="中文: bge-large-zh-v1.5, stella-large-zh-v3-1792d  英文: bge-large-en-v1.5")
                with gr.Group() as recursive_character_params:
                    # recursive_character_params.visible = True
                    # chunk_size=1000, chunk_overlap=400, add_start_index=False
                    chunk_size = gr.Number(value=1000, label="chunk_size")
                    chunk_overlap = gr.Number(value=400, label="chunk_overlap")
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


# nohup python langchain/rag/text_splitter_gradio.py > logs.txt 2>&1 &
if __name__ == '__main__':
    init_embeddings_models()
    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=8060, show_api=False)
