#!/usr/bin/env python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    # description="A simple api server using Langchain's Runnable interfaces",
)

# add_routes(
#     app,
#     ChatOpenAI(model_name="gpt-3.5-turbo"),
#     path="/openai",
# )

add_routes(
    app,
    QianfanChatEndpoint(streaming=True),
    path="/ernie-bot",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8060)
