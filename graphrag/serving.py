import sys, os, signal
import uvicorn
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import FileResponse
from graphrag.query.cli import run_global_search, run_local_search
from datetime import datetime

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


def extract_dir_name_datetime(folder_name):
    try:
        return datetime.strptime(folder_name, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


# curl -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test1", "method":"local", "query":"why Musk is essential for OpenAI?"}' http://192.168.0.20:38062/query
@app.post("/query", response_model=GraphRAGQueryResponse)
def query(req: GraphRAGQueryRequest):
    if req.root.strip() == "" or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    print(f"[query] request: {req}")
    if req.method == "local":
        resp = run_local_search(
            config_dir=None,
            data_dir=None,
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=req.query,
        )
    elif req.method == "global":
        resp = run_global_search(
            config_dir=None,
            data_dir=None,
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=req.query,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid request")
    return GraphRAGQueryResponse(response=resp)


# curl 'http://192.168.0.20:38062/get-graphml?index=test1&filename=summarized_graph.graphml
@app.get("/get-graphml")
def get_graphml(
    index: str = Query(..., description="graph index root"),
    filename: str = Query("summarized_graph.graphml", description="filename to get")):
    output_path = os.path.join("/workspace", index, "output")
    if os.path.exists(output_path):
        subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
        latest_subfolder = max(
            (folder for folder in subfolders if extract_dir_name_datetime(os.path.basename(folder))),
            key=lambda folder: extract_dir_name_datetime(os.path.basename(folder))
        )
        target_graphml = os.path.join(latest_subfolder, f"artifacts/{filename}")
        print(f"target_graphml: {target_graphml}")
        if os.path.exists(target_graphml) and os.path.isfile(target_graphml):
            return FileResponse(target_graphml, media_type='application/xml')
        else:
            raise HTTPException(status_code=404, detail="File not found") 
    else:
        raise HTTPException(status_code=404, detail="File not found")


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

    print("[Main] exited")
