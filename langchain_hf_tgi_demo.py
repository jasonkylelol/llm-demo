from langchain.llms import HuggingFaceTextGenInference
from langchain_experimental.chat_models import Llama2Chat
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
import time
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langserve import add_routes
from fastapi import FastAPI
import uvicorn
import gradio as gr


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}")


llm = HuggingFaceTextGenInference(
    inference_server_url="http://192.168.2.75:8080/",
    max_new_tokens=256,
    top_k=50,
    top_p=0.05,
    #typical_p=0.95,
    temperature=0.7,
    #repetition_penalty=1.05,
    streaming=True,
    do_sample=True,
)
model = Llama2Chat(llm=llm, callbacks=[StreamingStdOutCallbackHandler()])
# model = Llama2Chat(llm=llm, callbacks=[MyCustomHandler()])

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)


if __name__ == "__main__":
    print(chain.run(text="how brush teeth with shampoo?"))
    
    print(chain.run(text="how about mercury"))
