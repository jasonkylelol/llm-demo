import sys, os, signal
import uvicorn
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response
from graphrag.query.cli import run_global_search, run_local_search

DEFAULT_COMMUNITY_LEVEL = 2
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"

app = FastAPI()
svrs = []

class GraphRAGQueryRequest(BaseModel):
    root: str
    method: Literal["local", "global"]
    query: str


class GraphRAGQueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=GraphRAGQueryResponse)
def create_chat_completion(request: GraphRAGQueryRequest):
    if request.root.strip() == "" or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    print(f"[query] request: {request}")
    
    if request.method == "local":
        response = run_local_search(
            config_dir=None,
            data_dir=None,
            root_dir=request.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=request.query,
        )
    elif request.method == "global":
        response = run_global_search(
            config_dir=None,
            data_dir=None,
            root_dir=request.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=request.query,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    return GraphRAGQueryResponse(response=response)


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...")
        for svr in svrs:
            svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = uvicorn.Config(app=app, host='0.0.0.0', port=80)
    server = uvicorn.Server(config)
    svrs.append(server)
    print(f"[GraphRAG] start api server...")
    server.run()
