import os
import requests
from typing import Type, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS


class StateInput(BaseModel):
    content: str = Field(description="the query need to be searched for State Of The union")


class State(BaseTool):
    name = "state_of_the_union"
    description = "Use for searching any content about State Of The union"
    args_schema: Type[BaseModel] = StateInput
    vector_db: Optional[FAISS] = None


    def __init__(self):
        super().__init__()

        model_name = "/root/huggingface/models/BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda:2"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # loader = TextLoader("./state_of_the_union.txt")
        loader = WebBaseLoader(
            web_paths=("https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt",),
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=400, add_start_index=False
        )
        docs = text_splitter.split_documents(documents)
        self.vector_db = FAISS.from_documents(docs, embeddings_model)
        print(f"Tool [State] loaded, using {model_name}")


    def _run(self, content: str) -> dict[str, Any]:
        docs = self.vector_db.similarity_search(content, k=1)
        # print(f"\n[{self.name}] {docs[0].page_content}\n")
        return {"result": docs[0].page_content}
