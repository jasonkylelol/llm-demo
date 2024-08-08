import sys, os, signal
from threading import Thread
import uvicorn
from vllm import AsyncEngineArgs

from openai_api_embedding_app import create_app
from openai_api_server_glm4 import app, init_engine

MODEL_ROOT = "/root/huggingface/models"
LLM_MODEL_PATH = f"{MODEL_ROOT}/THUDM/glm-4-9b-chat"
MAX_MODEL_LENGTH = 1024 * 16
EMBEDDING_MODEL_PATH = f"{MODEL_ROOT}/maidalun1020/bce-embedding-base_v1"

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
        max_num_batched_tokens=8192
    )
    init_engine(engine_args)


def llm_handler():
    config = uvicorn.Config(app=app, host='0.0.0.0', port=8061)
    server = uvicorn.Server(config)
    svrs.append(server)
    print(f"[LLM] start api server...")
    server.run()


def embedding_handler():
    os.environ['MODEL'] = EMBEDDING_MODEL_PATH
    app = create_app()
    config = uvicorn.Config(app=app, host='0.0.0.0', port=8060)
    server = uvicorn.Server(config)
    svrs.append(server)
    print(f"[Embedding] start api server...")
    server.run()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...")
        for svr in svrs:
            svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    llm_init()

    ths = [
        Thread(target=llm_handler),
        Thread(target=embedding_handler),
    ]

    for th in ths:
        th.start()

    for th in ths:
        th.join()

    print("[Main] exited")