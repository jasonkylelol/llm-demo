from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
)


class ChatGLM3LLM(CustomLLM):
    model_name: str = Field(
        default="",
        description=(
            "The model name to use from HuggingFace. "
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=(
            LLMMetadata.__fields__["is_chat_model"].field_info.description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )
    max_new_tokens: int = Field(
        default=8192,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self,
        model_name: str = "",
        device_map: Optional[str] = "auto",
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = 8192,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        ):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, trust_remote_code=True, **model_kwargs)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, device_map=device_map, trust_remote_code=True, **tokenizer_kwargs)
        
        config_dict = self._model.config.to_dict()
        context_window = int(config_dict.get("seq_length")) or 8192

        # print(kwargs)
        super().__init__(
            context_window=context_window,
            model_name=model_name,
            device_map=device_map,
        )


    @classmethod
    def class_name(cls) -> str:
        return "ChatGLM3_LLM"


    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )


    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # print(f"ChatGLM3LLM.complete: {prompt}")
        return CompletionResponse(text="use chat", raw={"model_output": None})


    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        # print(f"ChatGLM3LLM.stream_complete: {prompt}")
        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text="use stream_chat", delta="use stream_chat")
        return gen()


    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # print(f"ChatGLM3LLM.chat: {messages}")
        messages = self.format_messages(messages)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response, history = self._model.chat(self._tokenizer, prompt, history=[], **kwargs)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.rstrip("<|user|>"),
            )
        )


    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # print(f"ChatGLM3LLM.stream_chat: {messages}")
        messages = self.format_messages(messages)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        max_length = self.context_window
        # create generator based off of streamer
        def gen() -> ChatResponseGen:
            last_resp = ""
            postfix_delimiter_filter = "<|user|>"
            postfix_delimiter = ""
            for resp, history in self._model.stream_chat(self._tokenizer, prompt, max_length=max_length, **kwargs):
                if resp == "":
                    continue
                last_resp, added_content = self.extract_added_content(last_resp, resp)
                if added_content == "":
                    continue
                if added_content in postfix_delimiter_filter:
                    postfix_delimiter += added_content
                    if len(postfix_delimiter) >= len(postfix_delimiter_filter):
                        if postfix_delimiter == postfix_delimiter_filter:
                            # print(f"\n[generator] need skip {postfix_delimiter_filter} from:\n{resp}", flush=True)
                            postfix_delimiter = ""
                            continue
                        else:
                            yield ChatResponse(
                                    message=ChatMessage(
                                        role=MessageRole.ASSISTANT,
                                        content=resp.rstrip(postfix_delimiter_filter),
                                    ),
                                    delta=postfix_delimiter,
                                )
                            postfix_delimiter = ""
                else:
                    if len(postfix_delimiter) > 0:
                        yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=resp.rstrip(postfix_delimiter_filter),
                                ),
                                delta=postfix_delimiter,
                            )
                        postfix_delimiter = ""
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=resp.rstrip(postfix_delimiter_filter),
                        ),
                        delta=added_content,
                    )
        return gen()


    def extract_added_content(self, str1, str2):
        prefix_len = 0
        min_len = min(len(str1), len(str2))
        while prefix_len < min_len and str1[prefix_len] == str2[prefix_len]:
            prefix_len += 1
        added_content = str2[prefix_len:]
        return str2, added_content
    

    def format_messages(self, messages: Sequence[ChatMessage]):
        msgs = []
        for chat_message in messages:
            msgs.append({
                "role": chat_message.role.value,
                "content": chat_message.content,
            })

        return msgs


def messages_to_prompt(messages):
    prompt = "[gMASK]sop"
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}"
    # ensure we start with a system prompt, insert blank if needed
    # if not prompt.startswith("<|system|>\n"):
    #     prompt = "[gMASK]sop<|system|>\n" + prompt
    # add final assistant prompt
    prompt = prompt + "<|assistant|>"
    print(prompt)
    return prompt


def completion_to_prompt(completion):
    return f"[gMASK]sop<|user|>\n{completion}<|assistant|>"
