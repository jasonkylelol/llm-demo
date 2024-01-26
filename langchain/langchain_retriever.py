from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import json
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from custom.zephyr_7b_custom_prompt_template import (
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


def init_retriver():
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

def init_llm():
    llm = HuggingFaceTextGenInference(
        inference_server_url="http://192.168.0.20:8080/",
        # max_new_tokens=256,
        top_k=50,
        top_p=0.05,
        #typical_p=0.95,
        temperature=0.7,
        repetition_penalty=1.05,
        # streaming=True,
        do_sample=True,
        # callbacks=[callbk_handler]
    )
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    retriever = init_retriver()
    llm = init_llm()

    question = 'who is Joshua Davis and what happend to him?'
    template = \
"""Answer the question based only on the following context:
{context}
Question: {question}"""
    template_messages = [
        HumanMessagePromptTemplate.from_template(template),
        AIMessagePromptTemplate.from_template(""),
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
