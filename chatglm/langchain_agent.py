import os
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_core.messages import AIMessage, HumanMessage
from chatglm3 import ChatGLM3
from tools.weather import Weather


model_path = "/root/huggingface/models/THUDM/chatglm3-6b/"


if __name__ == "__main__":
    llm = ChatGLM3()
    llm.load_model(model_path)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    print(prompt.messages)

    tools = [Weather()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    ans = agent_executor.invoke(
        {
            "input": "厦门比北京热吗?",
            "chat_history": [
                HumanMessage(content="北京温度多少度"),
                AIMessage(content="北京现在12度"),
            ],
        }
    )
    print(ans)