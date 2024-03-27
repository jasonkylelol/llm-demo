from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from custom.llama2_custom_prompt_template import (
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
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
Always answer questions starting with "As Stephen Hawking told me". \
Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question. \
Always answer questions starting with "As Leibniz told me". \
Here is a question:
{query}"""

common_template = """You are a helpful AI assistant. \
Always answer questions starting with "As far as I know". \
Here is a question:
{query}"""


model_path = "/root/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2"
embedding_model_path = "/root/huggingface/models/BAAI/bge-large-en-v1.5"

device = "cuda:2"
embedding_device = "cuda:1"

max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

embeddings_model, prompt_embeddings = None, None
prompt_templates = [physics_template, math_template, common_template]


def init_embeddings():
    global embeddings_model, prompt_embeddings

    model_name = embedding_model_path
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    prompt_embeddings = embeddings_model.embed_documents(prompt_templates)


def init_llm():
    init_embeddings()

    model = AutoModelForCausalLM.from_pretrained(model_path,
        device_map=device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        device_map=device, padding_side="left")
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


def prompt_router(input):
    print("--------------------------------------------------------------")
    query_embedding = embeddings_model.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    print(similarity.argmax())
    if most_similar == math_template:
        print("Using Math")
    elif most_similar == physics_template:
        print("Using Physics")
    else:
        print("using Common")

    template_messages = [
        HumanMessagePromptTemplate.from_template(most_similar),
        AIMessagePromptTemplate.from_template(""),
    ]
    prompt = CustomChatPromptTemplate.from_messages(template_messages)

    # return PromptTemplate.from_template(most_similar)
    print(prompt.invoke(input).to_string())
    return prompt

def post_process(input):
    print(input)
    print("--------------------------------------------------------------")
    return input

if __name__ == '__main__':
    llm = init_llm()

    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | llm
        | StrOutputParser()
        | RunnableLambda(post_process)
    )

    chain.invoke("What's a black hole")

    chain.invoke("What's Trigonometric functions?")

    chain.invoke("Can I brush teeth with shampoo?")
