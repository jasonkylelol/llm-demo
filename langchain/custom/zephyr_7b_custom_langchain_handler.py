from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Set,
)
from langchain_core.prompts.chat import (
    BaseMessagePromptTemplate,
    BaseChatPromptTemplate,
    MessagesPlaceholder,
    _convert_to_message,
    MessageLikeRepresentation,
    MessageLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.prompt_values import PromptValue
from langchain.callbacks.base import BaseCallbackHandler
import asyncio


'''
<|system|>:
You are a helpful assistant
<|user|>:
how brush teeth with shampoo?
<|assistant|>:
I'm sorry, but brushing your teeth with shampoo is not recommended as shampoo is designed to clean your hair and scalp, not your teeth and mouth.
<|user|>:
what about gasoline?
<|assistant|>:
'''
def custom_get_buffer_string(
    messages: Sequence[BaseMessage]
) -> str:
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"<|{role}|>:\n{m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)

class CustomChatPromptValue(PromptValue):
    messages: Sequence[BaseMessage]

    def to_string(self) -> str:
        buffer_str = custom_get_buffer_string(self.messages)
        # print(buffer_str)
        return buffer_str

    def to_messages(self) -> List[BaseMessage]:
        return list(self.messages)

class CustomChatPromptTemplate(BaseChatPromptTemplate):
    input_variables: List[str]
    messages: List[MessageLike]
    validate_template: bool = False

    @root_validator(pre=True)
    def validate_input_variables(cls, values: dict) -> dict:
        messages = values["messages"]
        input_vars = set()
        input_types: Dict[str, Any] = values.get("input_types", {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = List[AnyMessage]
        if "partial_variables" in values:
            input_vars = input_vars - set(values["partial_variables"])
        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                raise ValueError(
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
        else:
            values["input_variables"] = sorted(input_vars)
        values["input_types"] = input_types
        return values
    
    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
    ) -> 'CustomChatPromptTemplate':
        _messages = [_convert_to_message(message) for message in messages]

        # Automatically infer input variables from messages
        input_vars: Set[str] = set()
        for _message in _messages:
            if isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        return cls(input_variables=sorted(input_vars), messages=_messages)
    
    @property
    def _prompt_type(self) -> str:
        return "chat"
    
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        messages = self.format_messages(**kwargs)
        return CustomChatPromptValue(messages=messages)

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()
    
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                # print('1', type(message_template), message_template)
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                # print('2', type(message_template), message_template)
                rel_params = {
                    k: v
                    for k, v in kwargs.items()
                    if k in message_template.input_variables
                }
                message = message_template.format_messages(**rel_params)
                result.extend(message)
            else:
                raise ValueError(f"Unexpected input: {message_template}")
        return result


class CustomCallbkHandler(BaseCallbackHandler):
    queue: asyncio.Queue

    def __init__(self, q: asyncio.Queue):
        self.queue = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # print(f"put {token}")
        self.queue.put_nowait(token)
