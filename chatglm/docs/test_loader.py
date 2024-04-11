import sys,os
sys.path.append(os.getcwd())

from custom.text_splitter import ChineseRecursiveTextSplitter
from custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_core.documents import Document
import re


def init_pdf_documents():
    loader = RapidOCRPDFLoader("平台对接说明.pdf")
    documents = loader.load()

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\n+', '\n', doc.page_content)
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


def init_word_documents():
    loader = RapidOCRDocLoader("平台对接说明.docx")
    documents = loader.load()

    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\n+', '\n', doc.page_content)
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


if __name__ == '__main__':
    init_pdf_documents()
    print("\n\n\n\n\n\n\n\n")
    init_word_documents()
