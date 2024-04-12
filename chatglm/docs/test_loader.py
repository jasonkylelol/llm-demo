import sys,os
sys.path.append(os.getcwd())

from custom.text_splitter import ChineseRecursiveTextSplitter
from custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_core.documents import Document
import re


def init_pdf_documents():
    loader = RapidOCRPDFLoader("科大讯飞股份有限公司2023年半年度报告摘要.pdf")
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

    # print(doc_page_content)
    # print("====================================================================")

    chunk_size = 500
    chunk_overlap = chunk_size / 5
    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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


def init_word_documents():
    loader = RapidOCRDocLoader("平台接入手册v1.0.docx")
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

    chunk_size = 500
    chunk_overlap = chunk_size / 5
    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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


if __name__ == '__main__':
    # init_pdf_documents()
    # print("\n\n\n\n\n\n\n\n")
    init_word_documents()
