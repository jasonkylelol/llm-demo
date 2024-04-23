import os
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

import torch
from transformers import BitsAndBytesConfig
from llamaindex_demo.chatglm_llm import ChatGLM3LLM
from llamaindex_demo.mistral_llm import MistralLLM
from llamaindex_demo.zephyr_llm import ZephyrLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)

model_path = "/root/huggingface/models"

# model_path = "/root/huggingface/models/THUDM/chatglm3-6b-128k"
# context_window=131072
# model_name = f"{model_path}/mistralai/Mistral-7B-Instruct-v0.2"
model_name = f"{model_path}/HuggingFaceH4/zephyr-7b-beta"
context_window=32768

embedding_model_name = f"{model_path}/BAAI/bge-large-zh-v1.5"

max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.7


def init_embed_model():
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name,
        device="cuda")
    # embeddings = embed_model.get_text_embedding("Hello World!")
    # print(len(embeddings))
    # print(embeddings[:5])
    return embed_model

def init_llm():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"load model from: {model_name}")
    # llm = ChatGLM3LLM(
    #     model_name=model_path,
    #     device_map="cuda",
    # )

    llm = ZephyrLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        device_map="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    return llm

if __name__ == "__main__":
    llm = init_llm()

    user_input = "what about gasoline?"
    messages = [
        ChatMessage(role="system", content="call me Amigo at every answer"),
        ChatMessage(role="user", content="how brush teeth with shampoo?"),
        ChatMessage(role="assistant", content="Amigo. Brushing your teeth with shampoo is not recommended as it can damage your toothbrush and may not be effective in removing plaque and bacteria from your teeth."),
        ChatMessage(role="user", content=user_input),
    ]

    generate_cfg = {"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p}
    response = llm.chat(messages, **generate_cfg)
    print(response.message.content)

    print("-----------------------------------------------------------------------------")

    for resp in llm.stream_chat(messages, **generate_cfg):
        # print(f"{resp.message.content} : {resp.delta}")
        sys.stdout.write(resp.delta)
        sys.stdout.flush()

    print()
