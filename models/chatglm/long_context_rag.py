import os, sys, json, re
sys.path.append(os.getcwd())

from typing import List, Optional
from chatglm3 import ChatGLM3, ResponseParser
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable, ConfigurableField
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_demo.custom.prompt_template.zephyr_prompt_template import CustomChatPromptTemplate


device = "cuda"
embedding_device = "cuda"

model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-zh-v1.5"

max_new_tokens=128000
top_k=50
top_p=0.65
temperature=0.2

chain, retriever_chain = None, None


def init_pdf_documents():
    loader = RapidOCRPDFLoader("chatglm/docs/科大讯飞股份有限公司2023年半年度报告摘要.pdf")
    documents = loader.load()

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\s+', ' ', doc.page_content)
        # emoji_pattern = re.compile("[\U00010000-\U0010ffff]")
        # cleaned_page_content = emoji_pattern.sub('', cleaned_page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]

    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        keep_separator=True,
        is_separator_regex=True,
        strip_whitespace=True,
    )
    documents = text_splitter.split_documents(documents)
    # print(documents)
    return documents


def init_word_documents():
    loader = RapidOCRDocLoader("chatglm/docs/平台接入手册v1.0.docx")
    documents = loader.load()

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\s+', ' ', doc.page_content)
        # emoji_pattern = re.compile("[\U00010000-\U0010ffff]")
        # cleaned_page_content = emoji_pattern.sub('', cleaned_page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]

    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        keep_separator=True,
        is_separator_regex=True,
        strip_whitespace=True,
    )
    documents = text_splitter.split_documents(documents)
    for idx, doc in enumerate(documents):
        if idx != 0 and idx != len(documents)-1:
            print("-------------------------------------------------------------")
        print(doc.page_content)
    return documents


def init_retriever():
    documents = init_pdf_documents()

    model_name = embedding_model_path
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_db = FAISS.from_documents(documents, embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={'k': 1,})
    return retriever


def init_llm():
    llm = ChatGLM3()
    llm.load_model(model_path)
    llm = llm.configurable_fields(
        temperature=ConfigurableField("temperature"),
        max_token=ConfigurableField("max_token"),
    )
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def init_chain():
    global chain, retriever_chain

    llm = init_llm()
    retriever = init_retriever()

    template = \
"""请根据以下背景知识:
{context}\n\n
回答问题: {question}"""
    template_messages = [
        HumanMessagePromptTemplate.from_template(template),
    ]
    prompt = CustomChatPromptTemplate.from_messages(template_messages)

    retriever_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt)

    chain = (retriever_chain | llm | ResponseParser())


def chain_invoke(question: str, temp: float):
    # print(retriever_chain.invoke(question).to_string())
    print("---------------------------------------------------------------------------")
    cfg = {
        "max_token": max_new_tokens,
        "temperature": temp,
    }
    print(chain.with_config(configurable=cfg).invoke(question), "\n\n")


if __name__ == '__main__':
    init_chain()

    chain_invoke("根据报告，科大讯飞公司上半年扣非净利润较上年同期有什么变化？发生变化的主要原因是？", 0.1)

    chain_invoke("报告中提到的股票回购数量和成交价格是多少？", 0.2)

    chain_invoke("讯飞星火认知大模型在报告期内取得的主要进展是?", 0.3)

    init_word_documents()
