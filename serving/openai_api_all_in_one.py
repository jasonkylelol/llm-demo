import sys, os, signal
from threading import Thread
import uvicorn
from vllm import AsyncEngineArgs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/huggingface/models")

# from openai_api_glm4_app import init_engine, lifespan
# from openai_api_glm4_app import router as llm_router
# LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", f"{MODEL_ROOT}/THUDM/glm-4-9b-chat")

from openai_api_qwen2_app import init_engine, lifespan
from openai_api_qwen2_app import router as llm_router
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", f"{MODEL_ROOT}/Qwen/Qwen2.5-7B-Instruct")

from openai_api_embedding_app import init_embeddings
from openai_api_embedding_app import router as embedding_router
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
    tensor_parallel_size = torch.cuda.device_count()
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    engine_args = AsyncEngineArgs(
        model=LLM_MODEL_PATH,
        tokenizer=LLM_MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_MODEL_LENGTH,
        disable_custom_all_reduce=True,
    )
    print(f"[AsyncEngineArgs] tensor_parallel_size: {engine_args.tensor_parallel_size} "
        f"gpu_memory_utilization: {engine_args.gpu_memory_utilization} ")
    init_engine(engine_args)


def embedding_init():
    init_embeddings(EMBEDDING_MODEL_PATH)


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
    handler()

    print("[Main] exited", flush=True)
