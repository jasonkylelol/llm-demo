from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_experimental.chat_models import Llama2Chat
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

llm = HuggingFaceTextGenInference(
    inference_server_url="http://192.168.2.75:8080/",
    max_new_tokens=256,
    top_k=50,
    top_p=0.05,
    #typical_p=0.95,
    temperature=0.7,
    #repetition_penalty=1.05,
    #streaming=True,
    do_sample=True,
)
model = Llama2Chat(llm=llm)

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

if __name__ == "__main__":
    print(chain.run(text="What can I see in Vienna? Propose 5 locations. Names only, no details."))
    print(chain.run(text="Tell me more about #2."))
