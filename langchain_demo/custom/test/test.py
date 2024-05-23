import os, sys
import re

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_demo.custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from langchain_core.documents import Document

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
        documents = load_txt_with_encodings(upload_file)
    else:
        return "仅支持 txt pdf doc docx"
    # doc_meta = None
    # doc_page_content = ""
    # for idx, doc in enumerate(documents):
    #     if idx == 0:
    #         doc_meta = doc.metadata
    #     cleaned_page_content = re.sub(r'\s+', ' ', doc.page_content)
    #     doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    # documents = [Document(page_content=doc_page_content, metadata=doc_meta)]
    return documents


def load_txt_with_encodings(file):
    encoding_options = [None, "gbk", "gb2312", "gb18030"]
    for encoding in encoding_options:
        try:
            loader = UnstructuredFileLoader(file, encoding=encoding)
            document = loader.load()
            print(f"use encoding {encoding} for {file} succeed", flush=True)
            return document
        except Exception as e:
            print(f"try use encoding {encoding} for {file} failed, exception: {e}", flush=True)


if __name__ == "__main__":
    # f = "cache/files/平台对接说明.pdf"
    f = "cache/files/红楼梦.txt"
    documents = load_documents(f)
    print(f"split {f} to {len(documents)} documents")

    # print("-----------------------------------------------------------")
    # for idx, doc in enumerate(documents):
    #     print(f"{doc.page_content}\n")