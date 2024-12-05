import time
import torch
import random
import string
import gc
import json

from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

router = APIRouter()
engine = None
tokenizer = None

class ModelCard(BaseModel):
    id: str = ""
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = ["Qwen2.5"]

class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class ChoiceDeltaToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionMessageToolCall(BaseModel):
    index: Optional[int] = 0
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[Literal["function"]] = 'function'

class ChatMessage(BaseModel):
    # “function” 字段解释：
    # 使用较老的OpenAI API版本需要注意在这里添加 function 字段并在 process_messages函数中添加相应角色转换逻辑为 observation

    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]

class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int

class ChatCompletionResponse(BaseModel):
    model: str
    id: Optional[str] = Field(default_factory=lambda: generate_id('chatcmpl-', 29))
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    system_fingerprint: Optional[str] = Field(default_factory=lambda: generate_id('fp_', 9))
    usage: Optional[UsageInfo] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    repetition_penalty: Optional[float] = 1.1


def init_engine(engine_args):
    global tokenizer, engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(engine_args.tokenizer, trust_remote_code=True)


def generate_id(prefix: str, k=29) -> str:
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return f"{prefix}{suffix}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@router.get("/health")
async def health() -> Response:
    try:
        await engine.check_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail="model not ready") from e
    return Response(status_code=200)


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="Qwen2.5")
    return ModelList(data=[model_card])


@torch.inference_mode()
async def generate_stream(model_id, params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))
    created_time = int(time.time())
    response_id = generate_id('chatcmpl-', 29)
    system_fingerprint = generate_id('fp_', 9)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        # "stop_token_ids": [151645],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    generate_text = ""
    async for output in engine.generate(inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        finish_reason = output.outputs[0].finish_reason
        output_len = len(output.outputs[0].token_ids)
        input_len = len(output.prompt_token_ids)
        output_text = output.outputs[0].text

        delta_text = output_text[len(generate_text):]
        generate_text = output_text
        usage = UsageInfo()
        task_usage = UsageInfo.model_validate({
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            })
        for usage_key, usage_value in task_usage.model_dump().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        message = DeltaMessage(
                content=delta_text,
                role="assistant",
                function_call=None,
            )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id=response_id,
            choices=[choice_data],
            created=created_time,
            system_fingerprint=system_fingerprint,
            object="chat.completion.chunk",
            usage=usage,
        )
        yield chunk.model_dump_json(exclude_unset=True)

    print(f"----- response -----\n{generate_text}\n", flush=True)
    # gc.collect()
    # torch.cuda.empty_cache()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    print(f"----- request -----\n{gen_params}\n", flush=True)

    if request.stream:
        stream_generator = generate_stream(request.model, gen_params)
        return EventSourceResponse(stream_generator, media_type="text/event-stream")
    
    usage = UsageInfo()
    response_text, finish_reason = "", ""
    async for response in generate_stream(request.model, gen_params):
        chunk = ChatCompletionResponse.model_validate_json(response)
        usage = chunk.usage
        finish_reason = chunk.choices[0].finish_reason
        response_text += chunk.choices[0].delta.content

    message = ChatMessage(
        role="assistant",
        content=response_text,
    )
    if message.content and isinstance(message.content, str):
        message.content = message.content.strip()
        prefix = "```json"
        if message.content.startswith(prefix):
            message.content = message.content[len(prefix):]
            message.content = message.content.replace("\n", "")
            message.content = message.content.replace("```", "")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )