from typing import List, Optional, Union
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import torch

# from fastapi.middleware.gzip import GZipMiddleware
import pydantic
from transformers import AutoTokenizer

router = APIRouter()

DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
E5_EMBED_INSTRUCTION = "passage: "
E5_QUERY_INSTRUCTION = "query: "
BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def create_embedding_app():
    app = FastAPI(
        title="Open Text Embeddings API",
        version="1.0.4",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # app.add_middleware(GZipRequestMiddleware)

    # handling gzip response only
    # app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.include_router(router)

    return app


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(
        description="The model to use for generating embeddings.", default=None)
    input: Union[str, List[str]] = Field(description="The input to embed.")
    dimensions: Optional[int] = Field(
        description="The number of dimensions the resulting output embeddings should have.",
        default=None)
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


class Embedding(BaseModel):
    object: str
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(BaseModel):
    object: str
    data: List[Embedding]
    model: str
    usage: Usage


embeddings = None
tokenizer = None


def str_to_bool(s):
    map = {'true': True, 'false': False, '1': True, '0': False}
    if s.lower() not in map:
        raise ValueError("Cannot convert {} to a bool".format(s))
    return map[s.lower()]


def init_embeddings():
    global embeddings
    global tokenizer

    if "DEVICE" in os.environ:
        device = os.environ["DEVICE"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model_name = os.environ.get("MODEL")
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    print(f"Loading embedding model: {model_name}", flush=True)
    normalize_embeddings = str_to_bool(
        os.environ.get("NORMALIZE_EMBEDDINGS", "1"))
    encode_kwargs = {
        "normalize_embeddings": normalize_embeddings
    }
    print(f"Normalize embeddings: {normalize_embeddings}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "e5" in model_name:
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                   embed_instruction=E5_EMBED_INSTRUCTION,
                                                   query_instruction=E5_QUERY_INSTRUCTION,
                                                   encode_kwargs=encode_kwargs,
                                                   model_kwargs={"device": device})
    elif "bge-" in model_name and "-en" in model_name:
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                              query_instruction=BGE_EN_QUERY_INSTRUCTION,
                                              encode_kwargs=encode_kwargs,
                                              model_kwargs={"device": device})
    elif "bge-" in model_name and "-zh" in model_name:
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                              query_instruction=BGE_ZH_QUERY_INSTRUCTION,
                                              encode_kwargs=encode_kwargs,
                                              model_kwargs={"device": device})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                           encode_kwargs=encode_kwargs,
                                           model_kwargs={"device": device})


def _create_embedding(input: Union[str, List[str]]):
    global embeddings
    model_name = os.environ.get("MODEL")
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    model_name_short = model_name.split("/")[-1]
    if isinstance(input, str):
        tokens = tokenizer.tokenize(input)
        return CreateEmbeddingResponse(data=[Embedding(embedding=embeddings.embed_query(input),
                                                       object="embedding", index=0)],
                                       model=model_name_short, object='list',
                                       usage=Usage(prompt_tokens=len(tokens), total_tokens=len(tokens)))
    else:
        data = [Embedding(embedding=embedding, object="embedding", index=i)
                for i, embedding in enumerate(embeddings.embed_documents(input))]
        total_tokens = 0
        for text in input:
            total_tokens += len(tokenizer.tokenize(text))
        return CreateEmbeddingResponse(data=data, model=model_name_short, object='list',
                                       usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens))


@router.post(
    "/v1/embeddings",
    response_model=CreateEmbeddingResponse,
)
async def create_embedding(
        request: CreateEmbeddingRequest
):
    print(f"[Embeddings] request: {request}", flush=True)
    if pydantic.__version__ > '2.0.0':
        return await run_in_threadpool(
            _create_embedding, **request.model_dump(exclude={"user", "model", "model_config", "dimensions"})
        )
    else:
        return await run_in_threadpool(
            _create_embedding, **request.dict(exclude={"user", "model", "model_config", "dimensions"})
        )