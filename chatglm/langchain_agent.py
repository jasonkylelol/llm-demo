import os
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_core.messages import AIMessage, HumanMessage
from chatglm3 import ChatGLM3
from tools.weather import Weather
from tools.state_of_the_union import State


model_path = "/root/huggingface/models/THUDM/chatglm3-6b/"


if __name__ == "__main__":
    llm = ChatGLM3()
    llm.load_model(model_path)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    # print(prompt.messages)

    tools = [
        # Weather(),
        State(),
        ]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # ans = agent_executor.invoke(
    #     {
    #         "input": "厦门比北京热吗?",
    #         "chat_history": [
    #             HumanMessage(content="北京温度多少度"),
    #             AIMessage(content="北京现在12度"),
    #         ],
    #     }
    # )

    ans = agent_executor.invoke(
        {
            "input": "In State Of The Union, who is Joshua Davis and what happened to him?"
        }
    )
    print(ans)
