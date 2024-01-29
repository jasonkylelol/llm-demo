from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import json
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from custom.llama2_7b_custom_prompt_template import (
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
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.load_tools import get_all_tool_names
from langchain import hub
from langchain.tools.render import render_text_description
from operator import itemgetter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

device = "cuda:1" # the device to load the model onto
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

model_path = "/root/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_path,
    device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path,
    device_map=device, use_fast=True)
pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=True,
    return_full_text=False,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=1.1,
    pad_token_id=2,
)

def init_retriver():
    model_name = "/root/huggingface/models/BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": device}
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
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    retriever = init_retriver()
    llm = init_llm()

    # retriever_tool = create_retriever_tool(
    #     retriever,
    #     "embedding-retriever",
    #     "embedding-retriever(input: string) -> string"
    # )
    # tools = [retriever_tool]
    # # print(retriever_tool.name)
    # # print(retriever_tool.description)
    # # print(retriever_tool.args)

    # rendered_tools = render_text_description(tools)
    # print(rendered_tools)

    question = 'who is Joshua Davis and what happend to him?'
    input_template = f'''you always response with pure JSON blob with key: "input" with value '{{input}}', and put your answer as the string type value of key: "AI"'''
    template_messages = [
        HumanMessagePromptTemplate.from_template(input_template),
    ]
    prompt = CustomChatPromptTemplate.from_messages(template_messages)
    print(prompt.invoke({"input": question}).to_string())
    
    print("====================================================")

    chain = ({"input": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser())
    print(chain.invoke(question))

    print("====================================================")

    agent_chain = ({"input": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
        | itemgetter("input")
        | retriever
        | format_docs)
    print(agent_chain.invoke(question))

