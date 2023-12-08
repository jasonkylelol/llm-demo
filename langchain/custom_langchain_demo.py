from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from custom_langchain_hf_tgi_handler import (
    CustomChatPromptTemplate,
    CustomCallbkHandler,
)
from langchain.schema import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys, asyncio
from operator import itemgetter

global chain, queue, prompt_template, memory

def init_chain():
    global chain, queue, prompt_template, memory

    queue = asyncio.Queue()
    callbk_handler = CustomCallbkHandler(queue)

    llm = HuggingFaceTextGenInference(
        inference_server_url="http://192.168.2.75:8080/",
        max_new_tokens=256,
        top_k=50,
        top_p=0.05,
        #typical_p=0.95,
        temperature=0.7,
        repetition_penalty=1.05,
        streaming=True,
        do_sample=True,
        callbacks=[callbk_handler]
    )
    # model = Llama2Chat(llm=llm)

    template_messages = [
        SystemMessagePromptTemplate.from_template("You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
        AIMessagePromptTemplate.from_template(""),
    ]
    prompt_template = CustomChatPromptTemplate.from_messages(template_messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    output_parser = StrOutputParser()

    # chain = LLMChain(llm=model, prompt=prompt_template, verbose=True, memory=memory)
    # chain = prompt_template | model | output_parser
    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
        ) 
        | prompt_template 
        | llm 
        | output_parser
    )

async def llm_call(message):
    # print(chain.run(text=message))
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
