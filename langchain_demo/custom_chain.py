from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from custom.prompt_template.zephyr_prompt_template import (
    CustomChatPromptTemplate,
    CustomCallbkHandler,
)
from langchain.schema import StrOutputParser
import sys, asyncio
from operator import itemgetter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

global chain, queue, memory


model_path = "/root/huggingface/models/HuggingFaceH4/zephyr-7b-beta/"
device = "cuda:2"
max_new_tokens=1024
top_k=50
top_p=0.65
temperature=0.95

def init_llm(callbk_handler):
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
        callbacks=[callbk_handler],
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def init_chain():
    global chain, queue, memory

    queue = asyncio.Queue()
    callbk_handler = CustomCallbkHandler(queue)

    llm = init_llm(callbk_handler)

    template_messages = [
        SystemMessagePromptTemplate.from_template("You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
        AIMessagePromptTemplate.from_template(""),
    ]
    prompt = CustomChatPromptTemplate.from_messages(template_messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    output_parser = StrOutputParser()

    # chain = prompt_template | llm | output_parser
    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
        ) 
        | prompt 
        | llm 
        | output_parser
    )

async def llm_call(message):
    # print(prompt_template.invoke({"text": message}).to_string())
    # print(llm.invoke(prompt_template.invoke({"text": message})))
    # print(memory.load_memory_variables({}))
    inputs = {"text": message}
    rsp = chain.invoke(inputs)
    memory.save_context(inputs=inputs, outputs={"output": rsp})

    await queue.join()
    print("\n-------------------------------------------------")
    # print(memory.load_memory_variables({}))

async def queue_task():
    while True:
        token = await queue.get()
        # print(f"get {token}")
        sys.stdout.write(token)
        sys.stdout.flush()
        await asyncio.sleep(0.03)
        queue.task_done()

async def main():
    init_chain()

    task = asyncio.create_task(queue_task())

    print("")
    await llm_call("how brush teeth with shampoo?")
    await llm_call("what about gasoline?")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError as e:
        # print("cancel error:", e)
        pass

if __name__ == "__main__":
    asyncio.run(main())
