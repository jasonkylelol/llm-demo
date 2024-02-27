from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langserve import RemoteRunnable
import json
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint

openai_llm = RemoteRunnable("http://localhost:8060/openai/")
ernie_bot_llm = RemoteRunnable("http://localhost:8060/ernie-bot/")

messages = ChatPromptTemplate.from_messages([
    HumanMessage(content="How brush teeth with shampoo?"),
    AIMessage(content="Brushing teeth with shampoo is not recommended as it can cause oral health problems"),
    HumanMessage(content="Is mercury an option?"),
]).format_messages()

# response = ernie_bot_llm.invoke(messages)
# print(response.content)
# print(json.dumps(response.additional_kwargs))

# try:
#     for chunk in ernie_bot_llm.stream(messages):
#         print(chunk.content, end="", flush=True)
# except TypeError as e:
#     print("")

chat = QianfanChatEndpoint(streaming=True)

# response = chat.invoke(messages)
# print(response.content)
# print(json.dumps(response.additional_kwargs))

try:
    for chunk in chat.stream(messages):
        print(chunk.content, end="")
except TypeError as e:
    print("")