from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from custom.prompt_template.llama2_prompt_template import (
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
from langchain_core.runnables.history import RunnableWithMessageHistory
import torch
import time

model_path = "/root/huggingface/models/mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda:1" # the device to load the model onto
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

session_id = "langchain_msg_history_brush_teeth"
redis_url = "redis://:JuEeig2vMIfqbFB5@192.168.0.20:30005"


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


def init_chain(llm):
    input_template = "{input}"
    prompt = CustomChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(input_template),
        ]
    )
    chain = prompt | llm
    return chain


def init_memory_chain(chain, session_id):
    redis_chat_history_for_chain = RedisChatMessageHistory(
        url=redis_url,
        session_id=session_id,
    )
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: redis_chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_with_message_history


if __name__ == '__main__':
    llm = init_llm()
    chain = init_chain(llm)
    chain_with_message_history = init_memory_chain(chain, session_id)

    questions = [
        # "how brush teeth with shampoo?",
        # "what about gasoline?",
        "how about butter?"
    ]

    for question in questions:
        response = chain_with_message_history.invoke(
            {"input": question},
            {"configurable": {"session_id": session_id}},
        )
        print(response)
