import os
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_core.messages import AIMessage, HumanMessage
from chatglm3 import ChatGLM3
from tools.weather import Weather
from tools.state_of_the_union import State
from fastapi import FastAPI, HTTPException, Query, Request, Response
import uvicorn

app = FastAPI()
agent_executor = None

model_path = "/root/huggingface/models/THUDM/chatglm3-6b/"


@app.post("/invoke")
async def invoke(
    request: Request
):
    raw_body = await request.body()
    # print(f"{raw_body}")
    # print("---------------------------------------------------------------------------")
    
    try:
        ans = agent_executor.invoke(
            {
                "input": raw_body.decode('utf-8')
            }
        )
        # print(ans)
        # print(type(ans.get('output')))
        res = f"{ans.get('output')}\n"
        return Response(content=res, media_type="text/plain")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def init_agent():
    global agent_executor

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


if __name__ == "__main__":
    init_agent()

    # ans = agent_executor.invoke(
    #     {
    #         "input": "厦门比北京热吗?",
    #         "chat_history": [
    #             HumanMessage(content="北京温度多少度"),
    #             AIMessage(content="北京现在12度"),
    #         ],
    #     }
    # )

    # ans = agent_executor.invoke(
    #     {
    #         "input": "In State Of The Union, who is Joshua Davis and what happened to him?"
    #     }
    # )

    # print(ans)

    uvicorn.run(app, host="0.0.0.0", port=8060)
