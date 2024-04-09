import os, sys, json, re
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "langchain_demo/custom"))

from chatglm3 import ChatGLM3, ResponseParser
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.llms.base import LLM
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_demo.custom.text_splitter import ChineseRecursiveTextSplitter
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader
from langchain_demo.custom.prompt_template.zephyr_prompt_template import CustomChatPromptTemplate


device = "cuda"
embedding_device = "cuda"

model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-zh-v1.5"

max_new_tokens=128000
top_k=50
top_p=0.65
temperature=0.2


def init_retriever():
    model_name = embedding_model_path
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    loader = RapidOCRPDFLoader("chatglm/docs/科大讯飞股份有限公司2023年半年度报告摘要.pdf")
    documents = loader.load()

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\n+', '\n', doc.page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]

    # text_splitter = ChineseRecursiveTextSplitter(
        # chunk_size=102400, chunk_overlap=4096, add_start_index=False
    # )
    # docs = text_splitter.split_documents(documents)

    # print(documents)

    vector_db = FAISS.from_documents(documents, embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={'k': 1,})
    return retriever


def init_llm():
    llm = ChatGLM3()
    llm.load_model(model_path)
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == '__main__':
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
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt)
    # print("====================================")
    # print(retriever_chain.invoke(question).to_string())

    # sys.exit()

    llm = init_llm()
    chain = (retriever_chain | llm | ResponseParser())
    print("====================================")
    question = "根据报告，科大讯飞公司上半年扣非净利润较上年同期有什么变化？发生变化的主要原因是？"
    print(chain.invoke(question))

    print("====================================")
    question = "报告中提到的股票回购数量和成交价格是多少？"
    print(chain.invoke(question))

