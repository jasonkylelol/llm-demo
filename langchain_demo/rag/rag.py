import os,sys
sys.path.append(os.getcwd())

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
import json
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_demo.custom.prompt_template.llama2_prompt_template import (
    CustomChatPromptTemplate,
    CustomCallbkHandler,
)
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, time

device = "cuda:0"
embedding_device = "cuda:0"

model_path = "/root/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-en-v1.5"

max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

def init_retriever():
    model_name = embedding_model_path
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    loader = TextLoader("langchain_demo/rag/files/opanai_lawsuit.txt")
    # loader = WebBaseLoader(
    #     web_paths=("https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt",),
    # )
    documents = loader.load()

    # print(documents[0].page_content[:500])
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=400, add_start_index=False
    )
    docs = text_splitter.split_documents(documents)

    vector_db = FAISS.from_documents(docs, embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={'k': 1,})
    return retriever


def init_semantic_retriever():
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    text_splitter = SemanticChunker(embeddings_model,
        # breakpoint_threshold_type="percentile",
        # breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_type="interquartile",
    )

    with open("langchain_demo/rag/files/opanai_lawsuit.txt") as f:
        content = f.read()

    docs = text_splitter.create_documents([content])
    vector_db = FAISS.from_documents(docs, embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={'k': 1,})
    return retriever


def init_llm():
    model = AutoModelForCausalLM.from_pretrained(model_path,
        device_map=device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        device_map=device, use_fast=True)
    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        # clean_up_tokenization_spaces=True,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        # num_beams=1,
        top_p=top_p,
        top_k=top_k,
        # repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == '__main__':
    retriever = init_retriever()
    # retriever = init_semantic_retriever()
    llm = init_llm()

    # question = 'who is Joshua Davis and what happened to him?'
    question = "Why Musk is indispensable to the development of the OpenAI?"
    template = \
"""Answer the question based only on the following context:
{context}
Question: {question}"""
    template_messages = [
        HumanMessagePromptTemplate.from_template(template),
    ]
    prompt = CustomChatPromptTemplate.from_messages(template_messages)

    retriever_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt)
    print("====================================")
    print(retriever_chain.invoke(question).to_string())

    chain = (retriever_chain | llm | StrOutputParser())
    print("====================================")
    print(chain.invoke(question))

    # time.sleep(3600)
