import sys, os, signal
from threading import Thread
import uvicorn
from vllm import AsyncEngineArgs

from openai_api_embedding_app import init_embeddings
from openai_api_embedding_app import router as embedding_router
from openai_api_server_glm4 import init_engine, lifespan
from openai_api_server_glm4 import router as llm_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/huggingface/models")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", f"{MODEL_ROOT}/THUDM/glm-4-9b-chat")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", f"{MODEL_ROOT}/maidalun1020/bce-embedding-base_v1")

try:
    SERVER_PORT = int(os.getenv("SERVER_PORT", 80))
except (ValueError, TypeError):
    e = os.getenv("SERVER_PORT")
    print(f"SERVER_PORT: {type(e)} : {e}", flush=True)
    SERVER_PORT = 80

MAX_MODEL_LENGTH = 1024 * 16

svrs = []

def llm_init():
    engine_args = AsyncEngineArgs(
        model=LLM_MODEL_PATH,
        tokenizer=LLM_MODEL_PATH,
        # 如果你有多张显卡，可以在这里设置成你的显卡数量
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        # 占用显存的比例，请根据你的显卡显存大小设置合适的值，例如，如果你的显卡有80G，您只想使用24G，请按照24/80=0.3设置
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_MODEL_LENGTH,
    )
    init_engine(engine_args)


def embedding_init():
    os.environ['MODEL'] = EMBEDDING_MODEL_PATH
    init_embeddings()


def handler():
    app = FastAPI(
        title="OpenAI Compatible API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(llm_router)
    app.include_router(embedding_router)

    host = "0.0.0.0"
    config = uvicorn.Config(app=app, host=host, port=SERVER_PORT)
    server = uvicorn.Server(config)
    svrs.append(server)
    print(f"[AllInOne] start api server on {host}:{SERVER_PORT}", flush=True)
    server.run()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...", flush=True)
        for svr in svrs:
            svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    llm_init()
    embedding_init()

    ths = [
        Thread(target=handler),
    ]

    for th in ths:
        th.start()

    for th in ths:
        th.join()

    print("[Main] exited", flush=True)